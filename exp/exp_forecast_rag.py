"""
检索增强的时间序列预测实验类

支持三种模式:
1. 纯时间序列预测 (TS-only)
2. 时间序列+文本预测 (TS+Text)
3. 时间序列+文本+检索增强预测 (TS+Text+RAG)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pickle
import ast
import json
import re
import pandas as pd
import os
import math
from exp.exp_basic import Exp_Basic
from utils.augmentation import augment_time_series


class Exp_Forecast_RAG(Exp_Basic):
    """
    检索增强的预测实验类

    架构说明:
    1. 时序编码器: 从对比学习预训练模型加载
    2. 文本编码器: 冻结的 all-mpnet-base-v2
    3. 融合模块: 将时序特征和文本特征融合
    4. 检索模块: 从共性模式库检索相似样本
    5. 预测头: 基于融合特征预测未来时间序列
    """

    def __init__(self, args):
        super(Exp_Forecast_RAG, self).__init__(args)

        # 预测模式
        self.use_text = getattr(args, 'use_text', False)  # 是否使用文本
        self.use_rag = getattr(args, 'use_rag', False)    # 是否使用检索增强
        self.rag_lambda = getattr(args, 'rag_lambda', 0.3)
        self.freeze_text_encoder = getattr(args, 'freeze_text_encoder', True)
        self.train_pred_only = getattr(args, 'train_pred_only', False)

        # 时序编码器配置
        self.freeze_ts_encoder = getattr(args, 'freeze_ts_encoder', False)  # 是否冻结时序编码器
        if self.train_pred_only:
            self.freeze_ts_encoder = True
            self.freeze_text_encoder = True

        # 时序编码器 (可以从预训练模型初始化，但默认不冻结)
        self.ts_encoder = self._build_model()

        if self.freeze_ts_encoder:
            # 如果需要冻结，则设置为eval模式并禁用梯度
            for param in self.ts_encoder.parameters():
                param.requires_grad = False
            self.ts_encoder.eval()
            print("TS encoder frozen")
        else:
            self.ts_encoder.train()
            print("TS encoder trainable")

        # 文本编码器 (如果需要)
        if self.use_text or self.use_rag:
            self.text_encoder = self._load_llm()
            if self.freeze_text_encoder:
                self.text_encoder.eval()
            else:
                self.text_encoder.train()

        # 融合模块
        self.fusion_module = self._build_fusion_module()

        # 预测头
        self.prediction_head = self._build_prediction_head()

        # 文本门控，用于调节 RAG 残差的强度（初始偏保守）
        if self.use_text or self.use_rag:
            self.rag_text_gate = nn.Linear(self.args.llm_dim, 1).to(self.device)
            with torch.no_grad():
                self.rag_text_gate.bias.data.fill_(-1.0)

        if self.train_pred_only:
            for param in self.fusion_module.parameters():
                param.requires_grad = False
            if hasattr(self, "rag_text_gate"):
                for param in self.rag_text_gate.parameters():
                    param.requires_grad = False

        # 输出偏置校准，仅做平移（不缩放），用于抬高整体幅度
        self.output_bias = nn.Linear(self.args.d_model, self.args.pred_len * self.args.enc_in).to(self.device)
        with torch.no_grad():
            nn.init.zeros_(self.output_bias.weight)
            nn.init.zeros_(self.output_bias.bias)

        # 检索模块 (如果需要)
        if self.use_rag:
            self.retrieval_module = self._build_retrieval_module()
            neighbors_path = getattr(self.args, 'per_sample_neighbors_path', None)
            if neighbors_path and os.path.exists(neighbors_path):
                self._load_per_sample_neighbors()
            else:
                self._load_pattern_library()

        # 优化器 (仅优化未冻结的部分)
        trainable_params = []

        if self.train_pred_only:
            trainable_params += list(self.prediction_head.parameters())
            if hasattr(self, "output_bias"):
                trainable_params += list(self.output_bias.parameters())
        else:
            if not self.freeze_ts_encoder:
                trainable_params += list(self.ts_encoder.parameters())
            trainable_params += list(self.fusion_module.parameters())
            trainable_params += list(self.prediction_head.parameters())
            if hasattr(self, "rag_text_gate"):
                trainable_params += list(self.rag_text_gate.parameters())
            if hasattr(self, "output_bias"):
                trainable_params += list(self.output_bias.parameters())
            if (self.use_text or self.use_rag) and not self.freeze_text_encoder:
                trainable_params += list(self.text_encoder.parameters())

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        # 损失函数
        self.criterion = nn.MSELoss()

    def _load_llm(self):
        """Load text encoder (optionally frozen)."""
        from transformers import AutoModel, AutoTokenizer

        llm_path = getattr(self.args, 'llm_local_path', None) or 'sentence-transformers/all-mpnet-base-v2'

        status = "frozen" if self.freeze_text_encoder else "trainable"
        print(f"Loading LLM: {llm_path} ({status})")
        model = AutoModel.from_pretrained(llm_path, local_files_only=(self.args.llm_local_path is not None))
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=(self.args.llm_local_path is not None))

        if self.freeze_text_encoder:
            for param in model.parameters():
                param.requires_grad = False
            model.eval()
            print("LLM loaded and frozen")
        else:
            model.train()
            print("LLM loaded and trainable")

        return model



    def _build_fusion_module(self):
        """
        构建融合模块

        融合策略 (可通过 --fusion_type 选择):
        - 'simple': 简单拼接 + MLP (baseline)
        - 'gated': 自适应门控融合 (默认，推荐)
        - 'attention': 交叉注意力融合 (参数多)
        """
        fusion_type = getattr(self.args, 'fusion_type', 'gated')
        print(f"Using fusion type: {fusion_type}")

        if fusion_type == 'simple':
            return SimpleConcatFusion(
                d_model=self.args.d_model,
                d_text=self.args.llm_dim if (self.use_text or self.use_rag) else 0,
                use_text=self.use_text or self.use_rag,
                use_rag=self.use_rag,
                output_dim=getattr(self.args, 'fusion_output_dim', 256)
            )
        elif fusion_type == 'gated':
            return AdaptiveGatedFusionModule(
                d_model=self.args.d_model,
                d_text=self.args.llm_dim if (self.use_text or self.use_rag) else 0,
                use_text=self.use_text or self.use_rag,
                use_rag=self.use_rag,
                output_dim=getattr(self.args, 'fusion_output_dim', 256),
                gate_init_bias=getattr(self.args, 'gate_init_bias', -2.0)
            )
        elif fusion_type == 'attention':
            return ResidualCrossAttentionFusion(
                d_model=self.args.d_model,
                d_text=self.args.llm_dim if (self.use_text or self.use_rag) else 0,
                use_text=self.use_text or self.use_rag,
                use_rag=self.use_rag,
                output_dim=getattr(self.args, 'fusion_output_dim', 256)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")


    def _build_prediction_head(self):
        """
        构建预测头

        输入: 融合特征 [batch, fusion_output_dim]
        输出: 未来时间序列 [batch, pred_len, features]
        """
        fusion_dim = getattr(self.args, 'fusion_output_dim', 256)
        pred_len = self.args.pred_len
        features = self.args.enc_in  # 输入特征数

        return nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim * 2, pred_len * features),
            nn.Unflatten(1, (pred_len, features))
        )


    def _build_retrieval_module(self):
        """
        构建检索模块

        功能:
        1. 提取查询特征 (时序+文本)
        2. 计算与模式库的相似度
        3. 检索 top-k 相似模式
        4. 聚合检索结果
        """
        return RetrievalModule(
            d_model=self.args.d_model,
            llm_dim=self.args.llm_dim,
            top_k=getattr(self.args, 'retrieval_top_k', 3),
            aggregation=getattr(self.args, 'retrieval_aggregation', 'weighted')  # 'weighted' or 'mean'
        )


    def _load_pattern_library(self):
        """加载共性模式库"""
        pattern_lib_path = getattr(self.args, 'pattern_lib_path', './pattern_library/pattern_library.pkl')
        cluster_centers_path = getattr(self.args, 'cluster_centers_path', './pattern_library/cluster_centers.npy')

        print(f"Loading pattern library from {pattern_lib_path}")

        # 加载完整模式库
        with open(pattern_lib_path, 'rb') as f:
            self.pattern_library = pickle.load(f)

        # 加载聚类中心矩阵 [K, d_model + llm_dim]
        self.cluster_centers = torch.from_numpy(np.load(cluster_centers_path)).float()

        print(f"✓ Pattern library loaded: {len(self.pattern_library)} clusters, centers shape: {self.cluster_centers.shape}")

        # 提取每个簇的原型时间序列 (用于后续的增强)
        self.cluster_prototypes = {}
        for cluster_key, cluster_info in self.pattern_library.items():
            cluster_id = int(cluster_key.split('_')[1])
            proto_ts = cluster_info['prototype']['timeseries']  # [seq_len, 1]
            self.cluster_prototypes[cluster_id] = torch.from_numpy(proto_ts).float()

    def _apply_scaler(self, seq_list):
        """
        使用训练集的 scaler 对邻居序列做标准化，保持与模型输入一致。
        seq_list: list/array 形状 [L]
        """
        if not hasattr(self, "neighbor_scaler") or self.neighbor_scaler is None:
            return np.array(seq_list, dtype=np.float32)
        arr = np.array(seq_list, dtype=np.float32).reshape(-1, 1)
        scaled = self.neighbor_scaler.transform(arr).reshape(-1)
        return scaled

    def _load_per_sample_neighbors(self):
        """加载每样本邻居表和原始CSV，用于RAG按sample_id查邻居"""
        neighbors_path = getattr(self.args, 'per_sample_neighbors_path', './pattern_library/per_sample_neighbors.pkl')
        data_csv_path = os.path.join(getattr(self.args, 'root_path', './data/'), getattr(self.args, 'data_path', ''))
        print(f"Loading per-sample neighbors from {neighbors_path}")
        if neighbors_path.endswith('.pkl'):
            self.sample_neighbors_df = pd.read_pickle(neighbors_path)
        else:
            self.sample_neighbors_df = pd.read_csv(neighbors_path)
        # 邻居字段可能是字符串，需要转换（兼容 Timestamp('...')）
        def _parse_neighbors(x):
            if not isinstance(x, str):
                return x if x is not None else []
            s = x.strip()
            if s == "" or s.lower() in {"nan", "none"}:
                return []
            # ?? Timestamp('2020-01-01 00:00:00')
            if "Timestamp(" in s:
                s = re.sub(r"Timestamp\('([^']+)'\)", r'"\1"', s)
                s = re.sub(r'Timestamp\("([^"]+)"\)', r'"\1"', s)
            # ??? ast ????? dict/list?
            try:
                return ast.literal_eval(s)
            except Exception:
                try:
                    return json.loads(s.replace("'", '"'))
                except Exception:
                    return []
        if isinstance(self.sample_neighbors_df.iloc[0]['neighbors'], str):
            self.sample_neighbors_df['neighbors'] = self.sample_neighbors_df['neighbors'].apply(_parse_neighbors)
        # 加载原始CSV以便按sample_id取x_ot
        self.raw_df = pd.read_csv(data_csv_path)
        self.raw_df['x_ot_parsed'] = self.raw_df['x_ot'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        if 'y_ot' in self.raw_df.columns:
            self.raw_df['y_ot_parsed'] = self.raw_df['y_ot'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.sample_to_row = {int(r.sample_id): idx for idx, r in self.raw_df.iterrows()} if 'sample_id' in self.raw_df.columns else {}
        self.neighbor_ts_cache = {}  # sample_id -> ts_feature
        print(f"Per-sample neighbors loaded: {len(self.sample_neighbors_df)} entries")


    def encode_text(self, text_list):
        """编码文本"""
        inputs = self.tokenizer(
            text_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        if self.freeze_text_encoder:
            with torch.no_grad():
                outputs = self.text_encoder(**inputs)
        else:
            outputs = self.text_encoder(**inputs)

        # Mean pooling
        attention_mask = inputs['attention_mask']
        text_features = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
        text_features = text_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return text_features  # [batch, llm_dim]


    def _encode_timeseries(self, x):
        """
        编码时间序列

        Args:
            x: [batch, seq_len, features] 输入时间序列

        Returns:
            ts_features: [batch, d_model] 时序特征
        """
        # 使用预训练的时序编码器
        # 这里的逻辑与 exp_contrastive.py 中的 _encode_timeseries 一致
        if hasattr(self.ts_encoder, 'enc_embedding'):
            enc_out = self.ts_encoder.enc_embedding(x, None)  # [B, T, d_model]
            for i in range(self.ts_encoder.layer):
                enc_out = self.ts_encoder.layer_norm(self.ts_encoder.model[i](enc_out))
            return enc_out.mean(dim=1)  # [B, d_model]

        raise AttributeError(f"Unsupported ts_encoder structure: {type(self.ts_encoder)}")

    def _encode_neighbor_history(self, sample_id):
        """编码邻居历史序列，带缓存"""
        cache = getattr(self, "neighbor_ts_cache", {})
        if sample_id in cache:
            return cache[sample_id]
        row_idx = self.sample_to_row.get(int(sample_id), None)
        if row_idx is None:
            return None
        x = self.raw_df.loc[row_idx, 'x_ot_parsed']
        x_scaled = self._apply_scaler(x)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=self.device).view(1, -1, 1)
        with torch.no_grad():
            ts_hidden = self._encode_timeseries(x_tensor)  # [1, d_model]
        cache[sample_id] = ts_hidden.squeeze(0)
        self.neighbor_ts_cache = cache
        return cache[sample_id]

    def _encode_neighbor_future(self, sample_id):
        """编码邻居未来序列（y_ot），无缓存，长度固定 pred_len"""
        row_idx = self.sample_to_row.get(int(sample_id), None)
        if row_idx is None or 'y_ot_parsed' not in self.raw_df.columns:
            return None
        y = self.raw_df.loc[row_idx, 'y_ot_parsed']
        if not isinstance(y, list):
            return None
        y_scaled = self._apply_scaler(y)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32, device=self.device).view(1, -1, 1)
        with torch.no_grad():
            y_hidden = self._encode_timeseries(y_tensor)  # [1, d_model]
        return y_hidden.squeeze(0)

    def _build_rag_from_neighbors(self, sample_ids, ts_features):
        """根据 per-sample neighbors 构建 rag_features，保留K维度后用注意力聚合"""
        if sample_ids is None or not hasattr(self, "sample_neighbors_df"):
            return None
        debug_neighbors = getattr(self.args, "debug_neighbors", False)
        used_count = 0
        total_k = 0
        rag_list = []
        for b_idx, sid in enumerate(sample_ids):
            sid_int = int(sid)
            row = self.sample_neighbors_df[self.sample_neighbors_df['sample_id'] == sid_int]
            if row.empty:
                rag_list.append(ts_features[b_idx])
                continue
            neighbors = row.iloc[0]['neighbors']
            if not neighbors:
                rag_list.append(ts_features[b_idx])
                continue
            # 按 rank 截断为 retrieval_top_k（若有 rank 字段）
            if isinstance(neighbors, list) and neighbors:
                if any(isinstance(n, dict) and "rank" in n for n in neighbors):
                    neighbors = sorted(neighbors, key=lambda n: n.get("rank", float("inf")))
                top_k = getattr(self.args, "retrieval_top_k", None)
                if top_k is not None:
                    neighbors = neighbors[:top_k]
            k_list, v_list = [], []
            for nei in neighbors:
                nei_id = nei.get('neighbor_id')
                nei_k = self._encode_neighbor_history(nei_id)
                if nei_k is None:
                    continue
                nei_v = self._encode_neighbor_future(nei_id)
                if nei_v is None:
                    nei_v = nei_k  # 没有未来则退化为历史
                k_list.append(nei_k)
                v_list.append(nei_v)
            if not k_list:
                rag_list.append(ts_features[b_idx])
                continue
            used_count += 1
            total_k += len(k_list)
            K = torch.stack(k_list, dim=0)  # [K, d_model]
            V = torch.stack(v_list, dim=0)  # [K, d_model]
            q = ts_features[b_idx]  # [d_model]
            scores = torch.matmul(K, q.unsqueeze(-1)).squeeze(-1) / math.sqrt(self.args.d_model)
            attn = torch.softmax(scores, dim=0).unsqueeze(1)  # [K,1]
            rag_feat = (V * attn).sum(dim=0)
            rag_list.append(rag_feat)
        if debug_neighbors:
            batch_size = len(sample_ids)
            avg_k = (total_k / used_count) if used_count > 0 else 0.0
            print(f"[RAG] batch={batch_size} used={used_count} avg_k={avg_k:.2f}")
        return torch.stack(rag_list, dim=0)


    def forward(self, x_enc, x_mark_enc, context_text=None, sample_ids=None):
        """
        前向传播

        Args:
            x_enc: [batch, seq_len, features] 输入时间序列
            x_mark_enc: [batch, seq_len, mark_dim] 时间标记 (可选)
            context_text: List[str] 上下文文本 (可选)

        Returns:
            pred: [batch, pred_len, features] 预测的未来时间序列
        """
        batch_size = x_enc.shape[0]

        # 1. 编码时间序列
        ts_features = self._encode_timeseries(x_enc)  # [B, d_model]

        # 2. 编码文本 (如果需要)
        if self.use_text or self.use_rag:
            if context_text is None:
                raise ValueError("context_text is required when use_text or use_rag is True")
            text_features = self.encode_text(context_text)  # [B, llm_dim]
        else:
            text_features = None

        # 3. 检索增强 (如果需要)
        if self.use_rag:
            # 优先使用每样本邻居文件（离线检索 + normalize_ts）
            if sample_ids is not None and hasattr(self, "sample_neighbors_df"):
                rag_features = self._build_rag_from_neighbors(sample_ids, ts_features)
            else:
                rag_features = self.retrieval_module(
                    ts_features,
                    text_features,
                    self.cluster_centers.to(self.device),
                    self.cluster_prototypes
                )  # [B, d_model]
        else:
            rag_features = None

        # 4. 基线分支：ts + text
        fused_base = self.fusion_module(
            ts_features=ts_features,
            text_features=text_features,
            rag_features=None
        )  # [B, fusion_output_dim]
        base_pred = self.prediction_head(fused_base)  # [B, pred_len, features]

        # 无 RAG 或没有邻居时，直接用基线预测
        if not (self.use_rag and rag_features is not None):
            return base_pred

        # 5. RAG 残差分支：ts + rag
        fused_rag = self.fusion_module(
            ts_features=ts_features,
            text_features=None,
            rag_features=rag_features
        )  # [B, fusion_output_dim]
        rag_delta = self.prediction_head(fused_rag)  # [B, pred_len, features]

        # 文本生成门控，控制 rag_delta 贡献；无文本时默认为 1
        if text_features is not None and hasattr(self, "rag_text_gate"):
            gate = torch.sigmoid(self.rag_text_gate(text_features)).view(-1, 1, 1)  # [B,1,1]
        else:
            gate = 1.0

        pred = base_pred + self.rag_lambda * gate * rag_delta

        # 输出偏置校准，仅平移，不缩放
        bias = self.output_bias(ts_features).view(-1, self.args.pred_len, self.args.enc_in)
        pred = pred + bias
        return pred


    def   train_one_epoch(self, train_loader, epoch):
        """训练一个epoch"""
        # 根据冻结标志设置时序编码器模式
        if self.freeze_ts_encoder:
            self.ts_encoder.eval()  # 冻结时保持eval模式
        else:
            self.ts_encoder.train()  # 可训练时设置为train模式

        if any(p.requires_grad for p in self.fusion_module.parameters()):
            self.fusion_module.train()
        else:
            self.fusion_module.eval()
        self.prediction_head.train()

        if self.use_text or self.use_rag:
            if self.freeze_text_encoder:
                self.text_encoder.eval()
            else:
                self.text_encoder.train()

        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            x_enc = batch['x_enc'].to(self.device)      # [B, seq_len, features]
            x_dec = batch['x_dec'].to(self.device)      # [B, label_len+pred_len, features]
            y = batch['y'].to(self.device)              # [B, pred_len, features]
            x_mark_enc = batch.get('x_mark_enc', None)

            # 获取文本 (如果需要)
            context_text = batch.get('context_text', None) if (self.use_text or self.use_rag) else None
            sample_ids = batch.get('sample_id', None)

            # 前向传播
            pred = self.forward(x_enc, x_mark_enc, context_text, sample_ids)  # [B, pred_len, features]

            # 计算损失
            loss = self.criterion(pred, y)

            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (仅对可训练参数)
            if not self.freeze_ts_encoder:
                torch.nn.utils.clip_grad_norm_(self.ts_encoder.parameters(), max_norm=1.0)
            if any(p.requires_grad for p in self.fusion_module.parameters()):
                torch.nn.utils.clip_grad_norm_(self.fusion_module.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.prediction_head.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}')

        avg_loss = total_loss / len(train_loader)
        return avg_loss


    def validate(self, val_loader):
        """验证"""
        self.ts_encoder.eval()
        self.fusion_module.eval()
        self.prediction_head.eval()

        total_loss = 0
        total_mse = 0
        total_mae = 0

        with torch.no_grad():
            for batch in val_loader:
                x_enc = batch['x_enc'].to(self.device)
                x_dec = batch['x_dec'].to(self.device)
                y = batch['y'].to(self.device)
                x_mark_enc = batch.get('x_mark_enc', None)
                context_text = batch.get('context_text', None) if (self.use_text or self.use_rag) else None
                sample_ids = batch.get('sample_id', None)

                pred = self.forward(x_enc, x_mark_enc, context_text, sample_ids)

                loss = self.criterion(pred, y)
                mse = ((pred - y) ** 2).mean()
                mae = (pred - y).abs().mean()

                total_loss += loss.item()
                total_mse += mse.item()
                total_mae += mae.item()

        avg_loss = total_loss / len(val_loader)
        avg_mse = total_mse / len(val_loader)
        avg_mae = total_mae / len(val_loader)

        return {'loss': avg_loss, 'mse': avg_mse, 'mae': avg_mae}


    def load_pretrained_encoder(self, checkpoint_path):
        """
        从对比学习预训练模型加载时序编码器

        注意: 这只是初始化权重，不会冻结参数
        如果需要冻结，请在args中设置 freeze_ts_encoder=True
        """
        print(f"\nLoading pretrained encoder from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 提取时序编码器权重
        ts_encoder_state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith('ts_encoder.'):
                new_key = key.replace('ts_encoder.', '')
                ts_encoder_state_dict[new_key] = value

        if not ts_encoder_state_dict:
            print("  Warning: No 'ts_encoder.' keys found in checkpoint")
            print("  Trying to load entire checkpoint...")
            ts_encoder_state_dict = checkpoint

        # 加载权重 (strict=False 允许部分加载)
        missing_keys, unexpected_keys = self.ts_encoder.load_state_dict(ts_encoder_state_dict, strict=False)

        if missing_keys:
            print(f"  Warning: Missing keys: {missing_keys[:5]}..." if len(missing_keys) > 5 else missing_keys)
        if unexpected_keys:
            print(f"  Warning: Unexpected keys: {unexpected_keys[:5]}..." if len(unexpected_keys) > 5 else unexpected_keys)

        print("Pretrained encoder loaded successfully")
        if self.freeze_ts_encoder:
            print("  TS encoder weights frozen (will not update)")
        else:
            print("  TS encoder weights trainable (will continue training)")


class SimpleConcatFusion(nn.Module):
    """简单拼接融合 (baseline)"""
    def __init__(self, d_model, d_text, use_text, use_rag, output_dim=256):
        super().__init__()
        self.use_text = use_text
        self.use_rag = use_rag

        input_dim = d_model
        if use_text:
            input_dim += d_text
        if use_rag:
            input_dim += d_model

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, ts_features, text_features=None, rag_features=None):
        features = [ts_features]
        if self.use_text and text_features is not None:
            features.append(text_features)
        if self.use_rag and rag_features is not None:
            features.append(rag_features)
        return self.mlp(torch.cat(features, dim=1))


class AdaptiveGatedFusionModule(nn.Module):
    """
    自适应门控融合 - 样本自适应但参数少

    核心改进:
    1. 门控向量而非矩阵 - 减少参数量
    2. 初始化偏向TS - 保守融合
    3. 简单投影 - 避免过拟合

    gate_init_bias控制初始gate值:
    - bias = -2.0 → gate ≈ 0.12 (非常保守)
    - bias = -1.0 → gate ≈ 0.27 (中等)
    - bias = 0.0  → gate ≈ 0.50 (平衡)
    - bias = 1.0  → gate ≈ 0.73 (激进)
    """
    def __init__(self, d_model, d_text, use_text, use_rag, output_dim=256, gate_init_bias=-2.0):
        super().__init__()
        self.d_model = d_model
        self.use_text = use_text
        self.use_rag = use_rag
        self.gate_init_bias = gate_init_bias

        if use_text:
            self.text_proj = nn.Linear(d_text, d_model, bias=False)
            self.gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid()
            )
            # 初始化gate bias (控制初始gate值)
            with torch.no_grad():
                self.gate_net[-2].bias.data.fill_(gate_init_bias)
            print(f"Gate initialized with bias={gate_init_bias:.1f}, initial gate≈{torch.sigmoid(torch.tensor(gate_init_bias)).item():.3f}")

        if use_rag:
            self.rag_proj = nn.Linear(d_model, d_model, bias=False)
            self.rag_gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid()
            )
            with torch.no_grad():
                self.rag_gate_net[-2].bias.data.fill_(gate_init_bias)
            print(f"RAG Gate initialized with bias={gate_init_bias:.1f}")

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, ts_features, text_features=None, rag_features=None):
        fused = ts_features

        if self.use_text and text_features is not None:
            text_proj = self.text_proj(text_features)
            gate_input = torch.cat([ts_features, text_proj], dim=1)
            gate = self.gate_net(gate_input)
            fused = ts_features + gate * text_proj

        if self.use_rag and rag_features is not None:
            rag_proj = self.rag_proj(rag_features)
            gate_input = torch.cat([fused, rag_proj], dim=1)
            gate = self.rag_gate_net(gate_input)
            fused = fused + gate * rag_proj

        return self.output_proj(fused)


class ResidualCrossAttentionFusion(nn.Module):
    """
    改进的交叉注意力融合 - 针对小数据集优化

    改进点:
    1. 更少的注意力头 (2个而非4个)
    2. 去掉FFN (减少参数)
    3. 更强的正则化 (Dropout 0.3)
    4. 残差权重可学习
    """
    def __init__(self, d_model, d_text, use_text, use_rag, output_dim=256):
        super().__init__()
        self.d_model = d_model
        self.use_text = use_text
        self.use_rag = use_rag

        if use_text:
            self.text_proj = nn.Linear(d_text, d_model)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=2,
                dropout=0.3,
                batch_first=True
            )
            self.norm = nn.LayerNorm(d_model)
            self.residual_weight = nn.Parameter(torch.tensor(0.9))

        if use_rag:
            self.rag_proj = nn.Linear(d_model, d_model)
            self.rag_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=2,
                dropout=0.3,
                batch_first=True
            )
            self.norm_rag = nn.LayerNorm(d_model)
            self.rag_residual_weight = nn.Parameter(torch.tensor(0.9))

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, ts_features, text_features=None, rag_features=None):
        if not self.use_text:
            return self.output_proj(ts_features)

        fused = ts_features

        if text_features is not None:
            text_proj = self.text_proj(text_features).unsqueeze(1)
            ts_query = ts_features.unsqueeze(1)
            attn_out, _ = self.cross_attention(
                query=ts_query,
                key=text_proj,
                value=text_proj
            )
            attn_out = attn_out.squeeze(1)
            alpha = torch.sigmoid(self.residual_weight)
            fused = self.norm(alpha * ts_features + (1 - alpha) * attn_out)

        if self.use_rag and rag_features is not None:
            rag_proj = self.rag_proj(rag_features).unsqueeze(1)
            query = fused.unsqueeze(1)
            attn_out, _ = self.rag_attention(
                query=query,
                key=rag_proj,
                value=rag_proj
            )
            attn_out = attn_out.squeeze(1)
            beta = torch.sigmoid(self.rag_residual_weight)
            fused = self.norm_rag(beta * fused + (1 - beta) * attn_out)

        return self.output_proj(fused)


class MultiModalFusionModule(nn.Module):
    """
    多模态融合模块 - 使用交叉注意力和残差连接 (原始版本，参数较多)

    架构:
    1. TS-only模式: TS特征 -> MLP -> 输出
    2. TS+Text模式:
       - 交叉注意力: Text作为K,V, TS作为Q -> 提取文本中与时序相关的信息
       - 残差连接: 融合后的特征 + 原始TS特征
       - 门控机制: 动态控制文本信息的贡献
    3. TS+Text+RAG模式: 在TS+Text基础上再融合检索特征
    """

    def __init__(self, d_model, d_text, use_text, use_rag, num_heads=4, dropout=0.1, output_dim=256):
        super().__init__()
        self.d_model = d_model
        self.d_text = d_text
        self.use_text = use_text
        self.use_rag = use_rag
        self.output_dim = output_dim

        # 1. 投影层 - 将不同模态投影到统一维度
        if use_text:
            self.text_proj = nn.Linear(d_text, d_model)
            self.ts_query_proj = nn.Linear(d_model, d_model)

            # 2. 交叉注意力层 (Text -> TS)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )

            # 3. 层归一化
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            # 4. 门控机制 - 动态控制文本信息的贡献
            self.gate = nn.Sequential(
                nn.Linear(d_model * 2, d_model),
                nn.Sigmoid()
            )

            # 5. 前馈网络
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout)
            )

        # RAG特征融合
        if use_rag:
            self.rag_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm_rag = nn.LayerNorm(d_model)

        # 最终输出投影
        final_input_dim = d_model
        self.output_proj = nn.Sequential(
            nn.Linear(final_input_dim, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.Dropout(dropout)
        )

    def forward(self, ts_features, text_features=None, rag_features=None):
        """
        Args:
            ts_features: [B, d_model] 时序特征
            text_features: [B, d_text] 文本特征 (可选)
            rag_features: [B, d_model] 检索特征 (可选)

        Returns:
            fused_features: [B, output_dim] 融合后的特征
        """
        batch_size = ts_features.shape[0]

        # === 模式1: TS-only ===
        if not self.use_text:
            return self.output_proj(ts_features)

        # === 模式2 & 3: TS+Text (+ RAG) ===
        # Step 1: 投影文本特征到统一空间
        text_proj = self.text_proj(text_features)  # [B, d_model]
        ts_query = self.ts_query_proj(ts_features)  # [B, d_model]

        # Step 2: 交叉注意力 - TS作为Query, Text作为Key和Value
        # 扩展维度以适配MultiheadAttention (需要 [B, seq_len, d_model])
        ts_query_seq = ts_query.unsqueeze(1)  # [B, 1, d_model]
        text_kv_seq = text_proj.unsqueeze(1)   # [B, 1, d_model]

        attn_output, _ = self.cross_attention(
            query=ts_query_seq,
            key=text_kv_seq,
            value=text_kv_seq
        )  # [B, 1, d_model]

        attn_output = attn_output.squeeze(1)  # [B, d_model]

        # Step 3: 门控残差连接
        # 门控: 根据当前TS特征和注意力输出，决定接受多少文本信息
        gate_input = torch.cat([ts_features, attn_output], dim=1)  # [B, d_model*2]
        gate_values = self.gate(gate_input)  # [B, d_model]

        # 融合: 原始TS + 门控的文本信息
        fused = ts_features + gate_values * attn_output
        fused = self.norm1(fused)

        # Step 4: 前馈网络 + 残差
        ffn_output = self.ffn(fused)
        fused = self.norm2(fused + ffn_output)

        # === 模式3: 加入RAG特征 ===
        if self.use_rag and rag_features is not None:
            # 使用注意力融合RAG特征
            fused_seq = fused.unsqueeze(1)  # [B, 1, d_model]
            rag_seq = rag_features.unsqueeze(1)  # [B, 1, d_model]

            rag_attn_output, _ = self.rag_attention(
                query=fused_seq,
                key=rag_seq,
                value=rag_seq
            )  # [B, 1, d_model]

            rag_attn_output = rag_attn_output.squeeze(1)  # [B, d_model]
            fused = self.norm_rag(fused + rag_attn_output)

        # Step 5: 最终投影
        output = self.output_proj(fused)  # [B, output_dim]

        return output


class RetrievalModule(nn.Module):
    """
    检索模块

    功能:
    1. 计算查询特征 (TS + Text) 与模式库的相似度
    2. 检索 top-k 最相似的模式
    3. 聚合检索结果
    """

    def __init__(self, d_model, llm_dim, top_k=3, aggregation='weighted'):
        super().__init__()
        self.d_model = d_model
        self.llm_dim = llm_dim
        self.top_k = top_k
        self.aggregation = aggregation


    def forward(self, ts_features, text_features, cluster_centers, cluster_prototypes):
        """
        Args:
            ts_features: [B, d_model] 时序特征
            text_features: [B, llm_dim] 文本特征
            cluster_centers: [K, d_model + llm_dim] 聚类中心
            cluster_prototypes: dict {cluster_id: [seq_len, 1]} 原型时间序列

        Returns:
            rag_features: [B, d_model] 检索增强特征
        """
        batch_size = ts_features.shape[0]

        # 1. 拼接查询特征
        query_features = torch.cat([ts_features, text_features], dim=1)  # [B, d_model + llm_dim]

        # 2. 计算与聚类中心的余弦相似度
        query_norm = F.normalize(query_features, dim=1)
        centers_norm = F.normalize(cluster_centers, dim=1)
        similarities = torch.mm(query_norm, centers_norm.T)  # [B, K]

        # 3. 检索 top-k
        top_k_sims, top_k_indices = torch.topk(similarities, k=self.top_k, dim=1)  # [B, top_k]

        # 4. 聚合检索结果
        # 方法1: 加权聚合 (根据相似度加权)
        if self.aggregation == 'weighted':
            # 提取 top-k 簇中心的时序特征部分
            top_k_ts_features = cluster_centers[top_k_indices, :self.d_model]  # [B, top_k, d_model]

            # 相似度归一化为权重
            weights = F.softmax(top_k_sims, dim=1).unsqueeze(2)  # [B, top_k, 1]

            # 加权求和
            rag_features = (top_k_ts_features * weights).sum(dim=1)  # [B, d_model]

        # 方法2: 简单平均
        elif self.aggregation == 'mean':
            top_k_ts_features = cluster_centers[top_k_indices, :self.d_model]
            rag_features = top_k_ts_features.mean(dim=1)  # [B, d_model]

        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

        return rag_features
