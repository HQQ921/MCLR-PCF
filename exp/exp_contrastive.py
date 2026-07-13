import torch
import torch.nn as nn
import torch.nn.functional as F
from exp.exp_basic import Exp_Basic
from utils.augmentation import augment_time_series
from layers.projection_heads import ProjectionHead
from layers.contrastive_losses import NTXentLoss, CLIPLoss
import os
import numpy as np


class Exp_Contrastive(Exp_Basic):
    """
    纯对比学习实验类（无预测任务）

    主任务: 时序-时序对比学习（λ=0.8）
    辅助任务: 时序-文本对比学习（λ=0.2）

    关键特性:
    - 时序编码器: 可训练（TimesNet/iTransformer等）
    - 文本编码器: 冻结（BERT/LLAMA/GPT2）
    - 投影头: 可训练（将特征映射到对比学习空间）
    """
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        # ===== 编码器 =====
        # 时序编码器（可训练）
        self.ts_encoder = self._build_model()

        # 文本编码器（可选择冻结/微调）
        self.freeze_text_encoder = getattr(args, 'freeze_text_encoder', False)
        self.text_encoder = self._load_llm()

        # ===== 投影头（独立控制 TS/Text） =====
        base_use_proj = getattr(args, 'use_projection', True)
        self.use_ts_projection = getattr(args, 'use_ts_projection', None)
        self.use_text_projection = getattr(args, 'use_text_projection', None)
        if self.use_ts_projection is None:
            self.use_ts_projection = base_use_proj
        if self.use_text_projection is None:
            self.use_text_projection = base_use_proj
        # 兼容旧字段：仍保留 use_projection 以供其他代码使用
        self.use_projection = base_use_proj

        self.ts_projector = ProjectionHead(
            input_dim=args.d_model,
            hidden_dims=[512, 256],
            output_dim=args.proj_dim,
            use_bn=True,
            dropout=0.1
        )

        self.text_projector = ProjectionHead(
            input_dim=args.llm_dim,
            hidden_dims=[512, 256],
            output_dim=args.proj_dim,
            use_bn=True,
            dropout=0.1
        )

        # ===== 损失函数 =====
        self.ntxent_loss = NTXentLoss(temperature=args.temperature)
        self.clip_loss = CLIPLoss(temperature=args.temperature)

        # ===== 优化器 =====
        self.ts_encoder_optimizer = torch.optim.AdamW(
            self.ts_encoder.parameters(),
            lr=args.ts_encoder_lr,
            weight_decay=args.weight_decay
        )

        self.ts_proj_optimizer = torch.optim.AdamW(
            self.ts_projector.parameters(),
            lr=args.proj_lr,
            weight_decay=args.weight_decay
        )

        self.text_proj_optimizer = torch.optim.AdamW(
            self.text_projector.parameters(),
            lr=args.proj_lr,
            weight_decay=args.weight_decay
        )
        self.text_encoder_optimizer = None
        if not self.freeze_text_encoder:
            self.text_encoder_optimizer = torch.optim.AdamW(
                self.text_encoder.parameters(),
                lr=getattr(args, 'text_encoder_lr', 1e-5),
                weight_decay=args.weight_decay
            )

        # 是否冻结文本投影层
        self.freeze_text_projector = getattr(args, 'freeze_text_projector', False)
        if self.freeze_text_projector:
            for p in self.text_projector.parameters():
                p.requires_grad = False
            self.text_proj_optimizer = None

        # 近邻屏蔽范围（对比损失中忽略近邻作为负样本）
        self.neg_neighbor_exclude = getattr(args, 'neg_neighbor_exclude', 0)

        # ===== 超参数 =====
        self.lambda_ts_ts = args.lambda_ts_ts      # 时序-时序对比权重（主任务）
        self.lambda_ts_text = args.lambda_ts_text  # 时序-文本对比权重（辅助）

        # 设备
        self.device = args.device if hasattr(args, 'device') else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    def _load_llm(self):
        """
        加载LLM编码器（可冻结或微调）

        Returns:
            model: LLM模型
        """
        status = "frozen" if self.freeze_text_encoder else "trainable"
        print(f"Loading LLM: {self.args.llm_model} ({status})")

        # 确定模型路径（本地路径优先，否则使用HuggingFace hub）
        llm_local_path = getattr(self.args, 'llm_local_path', None)

        # 模型名称映射
        model_names = {
            'BERT': 'bert-base-uncased',
            'LLAMA2': 'huggyllama/llama-7b',
            'LLAMA3': 'meta-llama/Llama-3-8B',
            'GPT2': 'gpt2',
            'GPT2-M': 'gpt2-medium',
            'GPT2-L': 'gpt2-large',
            'all-mpnet-base-v2': 'sentence-transformers/all-mpnet-base-v2',
            'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2'
        }

        try:
            if self.args.llm_model == 'BERT':
                from transformers import BertModel, BertTokenizer
                model_path = llm_local_path if llm_local_path else model_names['BERT']
                print(f"  Loading from: {model_path}")
                model = BertModel.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = BertTokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))

            elif self.args.llm_model == 'LLAMA2':
                from transformers import LlamaModel, LlamaTokenizer
                model_path = llm_local_path if llm_local_path else model_names['LLAMA2']
                print(f"  Loading from: {model_path}")
                model = LlamaModel.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = LlamaTokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))

            elif self.args.llm_model == 'LLAMA3':
                from transformers import AutoModel, AutoTokenizer
                model_path = llm_local_path if llm_local_path else model_names['LLAMA3']
                print(f"  Loading from: {model_path}")
                model = AutoModel.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))

            elif self.args.llm_model == 'GPT2':
                from transformers import GPT2Model, GPT2Tokenizer
                model_path = llm_local_path if llm_local_path else model_names['GPT2']
                print(f"  Loading from: {model_path}")
                model = GPT2Model.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer.pad_token = self.tokenizer.eos_token

            elif self.args.llm_model == 'GPT2-M':
                from transformers import GPT2Model, GPT2Tokenizer
                model_path = llm_local_path if llm_local_path else model_names['GPT2-M']
                print(f"  Loading from: {model_path}")
                model = GPT2Model.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer.pad_token = self.tokenizer.eos_token

            elif self.args.llm_model == 'GPT2-L':
                from transformers import GPT2Model, GPT2Tokenizer
                model_path = llm_local_path if llm_local_path else model_names['GPT2-L']
                print(f"  Loading from: {model_path}")
                model = GPT2Model.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = GPT2Tokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            elif self.args.llm_model == 'all-mpnet-base-v2':
                from transformers import AutoModel, AutoTokenizer
                model_path = llm_local_path if llm_local_path else model_names['all-mpnet-base-v2']
                print(f"  Loading from: {model_path}")
                model = AutoModel.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))

            elif self.args.llm_model == 'all-MiniLM-L6-v2':
                from transformers import AutoModel, AutoTokenizer
                model_path = llm_local_path if llm_local_path else model_names['all-MiniLM-L6-v2']
                print(f"  Loading from: {model_path}")
                model = AutoModel.from_pretrained(model_path, local_files_only=(llm_local_path is not None))
                self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=(llm_local_path is not None))

            else:
                raise ValueError(f"Unsupported LLM model: {self.args.llm_model}")

        except OSError as e:
            print(f"\n{'='*80}")
            print(f"错误: 无法加载LLM模型 '{self.args.llm_model}'")
            print(f"{'='*80}")
            if llm_local_path:
                print(f"本地路径无效: {llm_local_path}")
                print(f"请确保路径正确且包含模型文件。")
            else:
                print(f"无法从HuggingFace下载模型。可能的原因:")
                print(f"  1. 网络连接问题")
                print(f"  2. 需要先手动下载模型")
                print(f"\n解决方案:")
                print(f"  方法1: 手动下载模型到本地，然后使用 --llm_local_path 参数")
                print(f"  方法2: 确保网络连接正常，脚本会自动下载到缓存目录")
                print(f"  方法3: 使用已缓存的模型（通常在 ~/.cache/huggingface/）")
                print(f"\n如何下载 {self.args.llm_model} 模型:")
                print(f"  from transformers import AutoModel, AutoTokenizer")
                print(f"  model = AutoModel.from_pretrained('{model_names.get(self.args.llm_model, 'N/A')}')")
                print(f"  tokenizer = AutoTokenizer.from_pretrained('{model_names.get(self.args.llm_model, 'N/A')}')")
            print(f"{'='*80}\n")
            raise

        # 控制是否冻结
        for param in model.parameters():
            param.requires_grad = not self.freeze_text_encoder

        if self.freeze_text_encoder:
            # 选项1: 解冻最后N层（推荐）
            text_unfreeze_layers = getattr(self.args, 'text_unfreeze_layers', 0)
            if text_unfreeze_layers > 0:
                unfrozen = self._unfreeze_last_n_layers(model, n_layers=text_unfreeze_layers)
                if unfrozen > 0:
                    model.train()
                    print(f"✓ LLM frozen except last {text_unfreeze_layers} layers ({unfrozen:,} params trainable)")
                else:
                    model.eval()
                    print("✓ LLM fully frozen (no layers found to unfreeze)")
            # 选项2: 仅解冻最后一层注意力（Q/K/V + output projection）
            elif getattr(self.args, 'text_unfreeze_last_attn', False):
                unfrozen = self._unfreeze_last_attn_params(model)
                if unfrozen > 0:
                    model.train()
                    print(f"✓ LLM frozen except last attention ({unfrozen:,} params trainable)")
                else:
                    model.eval()
                    print("✓ LLM frozen (no last-attn params found to unfreeze)")
            else:
                model.eval()  # 冻结时保持评估模式
                print("✓ LLM loaded and frozen successfully")
        else:
            model.train()  # 微调时开启训练模式
            print("✓ LLM loaded and set to trainable")

        return model

    def _unfreeze_last_attn_params(self, model):
        """
        仅解冻文本编码器最后一层 Multi-Head Attention 的 Q/K/V 投影和输出投影
        Returns:
            int: 解冻的参数个数
        """
        # 尝试获取 encoder.layer 序列（BERT/MPNet/MiniLM 均使用）
        encoder = getattr(model, 'encoder', None)
        if encoder is None and hasattr(model, 'transformer'):
            encoder = getattr(model.transformer, 'encoder', None)
        if encoder is None or not hasattr(encoder, 'layer') or len(encoder.layer) == 0:
            return 0

        last_layer = encoder.layer[-1]
        target_substrings = ['attention.self.query', 'attention.self.key', 'attention.self.value', 'attention.output.dense',
                             '.attn.q', '.attn.k', '.attn.v', '.attn.out_proj']

        unfrozen = 0
        for name, param in last_layer.named_parameters():
            if any(sub in name for sub in target_substrings):
                param.requires_grad = True
                unfrozen += param.numel()
            else:
                param.requires_grad = False
        return unfrozen

    def _unfreeze_last_n_layers(self, model, n_layers=3):
        """
        解冻文本编码器最后N层的所有参数

        Args:
            model: 文本编码器模型
            n_layers: 要解冻的层数（从后往前数）

        Returns:
            int: 解冻的参数个数
        """
        # 尝试获取 encoder.layer 序列（BERT/MPNet/MiniLM 均使用）
        encoder = getattr(model, 'encoder', None)
        if encoder is None and hasattr(model, 'transformer'):
            encoder = getattr(model.transformer, 'encoder', None)
        if encoder is None or not hasattr(encoder, 'layer') or len(encoder.layer) == 0:
            return 0

        total_layers = len(encoder.layer)
        unfreeze_from = max(0, total_layers - n_layers)

        unfrozen = 0
        for i in range(total_layers):
            for param in encoder.layer[i].parameters():
                if i >= unfreeze_from:
                    param.requires_grad = True
                    unfrozen += param.numel()
                else:
                    param.requires_grad = False

        print(f"  解冻最后 {n_layers} 层 (Layer {unfreeze_from}-{total_layers-1}), 共 {unfrozen:,} 参数")
        return unfrozen


    def encode_text(self, text_list):
        """
        文本编码（冻结模式，不计算梯度）

        Args:
            text_list: List[str], batch的文本列表

        Returns:
            text_features: [batch, d_llm] 文本特征

        """

        # 数据类型检查和处理
        if isinstance(text_list, str):
            # 单个字符串转为列表
            text_list = [text_list]
        elif isinstance(text_list, (list, tuple)):
            # 确保列表中每个元素都是字符串
            text_list = [str(t) if t is not None else "" for t in text_list]
        else:
            # 其他情况转为字符串再包装成列表
            text_list = [str(text_list)]

        grad_enabled = not self.freeze_text_encoder
        with torch.set_grad_enabled(grad_enabled):
            # Tokenization
            inputs = self.tokenizer(
                text_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # 编码
            outputs = self.text_encoder(**inputs)

            # 提取特征（不同模型的pooling策略）
            if self.args.llm_model in ['BERT']:
                # BERT使用[CLS] token
                text_features = outputs.last_hidden_state[:, 0, :]
            else:
                # 其他模型使用mean pooling
                attention_mask = inputs['attention_mask']
                text_features = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
                text_features = text_features.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        return text_features  # [batch, d_llm]


    def _encode_timeseries(self, x):
        """
        使用时序编码器提取特征

        Args:
            x: [batch, seq_len, features] 时间序列

        Returns:
            features: [batch, d_model] 编码后的特征
        """
        # PatchTST：复用 patch_embedding + encoder，做均值池化得到序列表征
        if getattr(self.args, 'model', '').lower() == 'patchtst':
            # x: [B, T, C] -> PatchEmbedding 期望 [B, C, T]
            x_enc = x.permute(0, 2, 1)
            enc_out, n_vars = self.ts_encoder.patch_embedding(x_enc)  # [B*n_vars, patch_num, d_model]
            enc_out, _ = self.ts_encoder.encoder(enc_out)             # [B*n_vars, patch_num, d_model]

            enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
            # 形状: [B, n_vars, patch_num, d_model] -> mean over (vars, patch_num)
            enc_out = enc_out.permute(0, 1, 3, 2)  # [B, n_vars, d_model, patch_num]
            features = enc_out.mean(dim=(1, 3))    # [B, d_model]
            return features

        # Autoformer: 使用 enc_embedding + encoder
        if getattr(self.args, 'model', '').lower() == 'autoformer':
            enc_out = self.ts_encoder.enc_embedding(x, None)  # [B, T, d_model]
            enc_out, _ = self.ts_encoder.encoder(enc_out, attn_mask=None)  # [B, T, d_model]
            return enc_out.mean(dim=1)  # [B, d_model] - 时间维度平均池化

        # TSMixer: 堆叠 ResBlock，时间维度平均后映射到 d_model
        if getattr(self.args, 'model', '').lower() == 'tsmixer':
            enc_out = x  # [B, T, C]
            for i in range(self.ts_encoder.layer):
                enc_out = self.ts_encoder.model[i](enc_out)  # 仍为 [B, T, C]
            enc_out = enc_out.mean(dim=1)  # [B, C]
            if hasattr(self.ts_encoder, 'contrastive_proj'):
                enc_out = self.ts_encoder.contrastive_proj(enc_out)  # [B, d_model]
            return enc_out

        # TimesNet 等：使用 enc_embedding + TimesBlock
        if hasattr(self.ts_encoder, 'enc_embedding'):
            enc_out = self.ts_encoder.enc_embedding(x, None)  # [B, T, d_model]
            for i in range(self.ts_encoder.layer):
                enc_out = self.ts_encoder.layer_norm(self.ts_encoder.model[i](enc_out))
            return enc_out.mean(dim=1)  # [B, d_model]

        raise AttributeError(f"Unsupported ts_encoder structure for contrastive encoding: {type(self.ts_encoder)}")

    def forward_ts_ts(self, x_ot):
        """
        时序-时序对比前向传播

        Args:
            x_ot: [batch, seq_len, features] 原始时间序列

        Returns:
            z1, z2: [batch, proj_dim] 两个增强视图的投影（已归一化）
        """
        # 数据增强（生成两个不同的视图）
        # x_aug1 = augment_time_series(x_ot, methods=['jitter', 'scaling'])
        # x_aug2 = augment_time_series(x_ot, methods=['time_warp', 'window_slice'])
        x_aug1 = augment_time_series(x_ot, methods=['jitter'])
        x_aug2 = augment_time_series(x_ot, methods=['scaling'])

        # 编码
        ts_hidden1 = self._encode_timeseries(x_aug1)  # [B, d_model]
        ts_hidden2 = self._encode_timeseries(x_aug2)  # [B, d_model]

        if self.use_ts_projection:
            # 投影
            z1 = self.ts_projector(ts_hidden1)  # [B, proj_dim]
            z2 = self.ts_projector(ts_hidden2)
        else:
            # 直接使用编码器输出
            z1, z2 = ts_hidden1, ts_hidden2

        # L2归一化（投影到单位球面）
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        return z1, z2


    def forward_ts_text(self, x_ot, text_list):
        """
        时序-文本跨模态对比前向传播

        Args:
            x_ot: [batch, seq_len, features] 原始时间序列
            text_list: List[str] 文本列表

        Returns:
            ts_proj, text_proj: [batch, proj_dim] 归一化后的投影（已归一化）
        """
        # 时序编码（不增强，使用原始数据）
        ts_input = x_ot
        if getattr(self.args, 'ts_text_aug', False):
            # 轻量增强：与TS-TS相同的一组增广
            ts_input = augment_time_series(x_ot, methods=['jitter', 'scaling'])

        ts_hidden = self._encode_timeseries(ts_input)  # [B, d_model]
        if self.use_ts_projection:
            ts_proj = self.ts_projector(ts_hidden)  # [B, proj_dim]
        else:
            ts_proj = ts_hidden  # 直接用编码器输出
        ts_proj = F.normalize(ts_proj, dim=-1)

        # 文本编码（冻结）
        text_hidden = self.encode_text(text_list)  # [B, d_llm]
        if self.use_text_projection:
            text_proj = self.text_projector(text_hidden)  # [B, proj_dim]
        else:
            # 关闭投影时要求维度一致，否则无法对比
            if text_hidden.shape[-1] != ts_proj.shape[-1]:
                raise ValueError(
                    f"关闭投影层时，TS与Text维度需一致。当前 ts_hidden={ts_proj.shape[-1]}, text_hidden={text_hidden.shape[-1]}。"
                )
            text_proj = text_hidden
        text_proj = F.normalize(text_proj, dim=-1)

        return ts_proj, text_proj


    def train_one_epoch(self, train_loader, epoch):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器
            epoch: 当前epoch数

        Returns:
            avg_loss: 平均总损失
            avg_loss_ts_ts: 平均TS-TS对比损失
            avg_loss_ts_text: 平均TS-Text对比损失
        """
        self.ts_encoder.train()
        self.ts_projector.train()
        self.text_projector.train()
        if self.freeze_text_encoder:
            self.text_encoder.eval()  # 冻结时保持评估模式
        else:
            self.text_encoder.train()

        total_loss = 0
        total_loss_ts_ts = 0
        total_loss_ts_text = 0

        for batch_idx, batch in enumerate(train_loader):
            x_ot = batch['x_ot'].to(self.device)  # [B, 12, 1]
            context_text = batch['context_text']   # List[str]

            # ===== 1. TS-TS 对比损失（主任务，可选） =====
            if not getattr(self.args, 'ts_text_only', False) and self.lambda_ts_ts > 0:
                z1, z2 = self.forward_ts_ts(x_ot)
                loss_ts_ts = self.ntxent_loss(z1, z2, exclude_neighbors=self.neg_neighbor_exclude)
            else:
                # 跳过 TS-TS 分支时给零损失，保持下游加权计算安全
                loss_ts_ts = torch.tensor(0.0, device=self.device)

            # ===== 2. TS-Text 对比损失（辅助任务） =====
            use_ts_text = getattr(self.args, 'ts_text_only', False) or self.lambda_ts_text > 0
            if use_ts_text:
                ts_proj, text_proj = self.forward_ts_text(x_ot, context_text)
                loss_ts_text = self.clip_loss(ts_proj, text_proj, exclude_neighbors=self.neg_neighbor_exclude)
            else:
                loss_ts_text = torch.tensor(0.0, device=self.device)

            # ===== 3. 总损失（加权组合） =====
            loss = (self.lambda_ts_ts * loss_ts_ts +
                   self.lambda_ts_text * loss_ts_text)

            # ===== 4. 反向传播 =====
            self.ts_encoder_optimizer.zero_grad()
            self.ts_proj_optimizer.zero_grad()
            if self.text_proj_optimizer is not None:
                self.text_proj_optimizer.zero_grad()
            if self.text_encoder_optimizer is not None:
                self.text_encoder_optimizer.zero_grad()

            loss.backward()

            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.ts_encoder.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(self.ts_projector.parameters(), max_norm=1.0)
            if self.text_proj_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.text_projector.parameters(), max_norm=1.0)
            if self.text_encoder_optimizer is not None:
                torch.nn.utils.clip_grad_norm_(self.text_encoder.parameters(), max_norm=1.0)

            self.ts_encoder_optimizer.step()
            self.ts_proj_optimizer.step()
            if self.text_proj_optimizer is not None:
                self.text_proj_optimizer.step()
            if self.text_encoder_optimizer is not None:
                self.text_encoder_optimizer.step()

            # ===== 5. 记录 =====
            total_loss += loss.item()
            total_loss_ts_ts += loss_ts_ts.item()
            total_loss_ts_text += loss_ts_text.item()

            # 打印训练进度
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch} [{batch_idx}/{len(train_loader)}] '
                      f'Loss: {loss.item():.4f} '
                      f'(TS-TS: {loss_ts_ts.item():.4f}, '
                      f'TS-Text: {loss_ts_text.item():.4f})')

        # 计算平均损失
        avg_loss = total_loss / len(train_loader)
        avg_loss_ts_ts = total_loss_ts_ts / len(train_loader)
        avg_loss_ts_text = total_loss_ts_text / len(train_loader)

        return avg_loss, avg_loss_ts_ts, avg_loss_ts_text


    def validate(self, val_loader):
        """
        Validate contrastive objectives (TS-TS and/or TS-Text).

        Returns:
            dict: overall_acc/overall_loss plus per-branch metrics.
        """
        self.ts_encoder.eval()
        self.ts_projector.eval()
        self.text_projector.eval()
        self.text_encoder.eval()

        use_ts_ts = (not getattr(self.args, 'ts_text_only', False)) and self.lambda_ts_ts > 0
        use_ts_text = getattr(self.args, 'ts_text_only', False) or self.lambda_ts_text > 0

        ts_ts_z1, ts_ts_z2 = [], []
        ts_ts_total_loss = 0.0

        ts_text_ts_proj, ts_text_text_proj = [], []
        ts_text_total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_ot = batch['x_ot'].to(self.device)
                context_text = batch['context_text'] if use_ts_text else None

                if use_ts_ts:
                    z1, z2 = self.forward_ts_ts(x_ot)
                    loss_ts_ts = self.ntxent_loss(z1, z2, exclude_neighbors=self.neg_neighbor_exclude)
                    ts_ts_total_loss += loss_ts_ts.item()
                    ts_ts_z1.append(z1.cpu())
                    ts_ts_z2.append(z2.cpu())

                if use_ts_text:
                    ts_proj, text_proj = self.forward_ts_text(x_ot, context_text)
                    loss_ts_text = self.clip_loss(ts_proj, text_proj, exclude_neighbors=self.neg_neighbor_exclude)
                    ts_text_total_loss += loss_ts_text.item()
                    ts_text_ts_proj.append(ts_proj.cpu())
                    ts_text_text_proj.append(text_proj.cpu())

        metrics = {}
        overall_acc_numer = 0.0
        overall_loss_numer = 0.0
        weight_sum = 0.0
        branch_acc = []
        branch_loss = []

        if use_ts_ts and len(ts_ts_z1) > 0:
            all_z1 = torch.cat(ts_ts_z1, dim=0)
            all_z2 = torch.cat(ts_ts_z2, dim=0)
            similarity = all_z1 @ all_z2.T

            z1_to_z2 = similarity.argmax(dim=1)
            z1_to_z2_acc = (z1_to_z2 == torch.arange(len(all_z1))).float().mean()

            z2_to_z1 = similarity.argmax(dim=0)
            z2_to_z1_acc = (z2_to_z1 == torch.arange(len(all_z2))).float().mean()

            k = min(5, similarity.size(0))
            z1_topk = similarity.topk(k, dim=1).indices
            z1_targets = torch.arange(len(all_z1)).unsqueeze(1)
            z1_to_z2_topk_acc = (z1_topk == z1_targets).any(dim=1).float().mean()

            z2_topk = similarity.topk(k, dim=0).indices
            z2_targets = torch.arange(len(all_z2)).unsqueeze(0)
            z2_to_z1_topk_acc = (z2_topk == z2_targets).any(dim=0).float().mean()

            avg_acc_top1 = (z1_to_z2_acc + z2_to_z1_acc) / 2
            avg_acc_top5 = (z1_to_z2_topk_acc + z2_to_z1_topk_acc) / 2
            avg_loss = ts_ts_total_loss / len(val_loader)

            metrics['ts_ts'] = {
                'acc_top1': avg_acc_top1.item(),
                'acc_top5': avg_acc_top5.item(),
                'loss': avg_loss
            }

            weight_sum += self.lambda_ts_ts
            overall_acc_numer += self.lambda_ts_ts * metrics['ts_ts']['acc_top1']
            overall_loss_numer += self.lambda_ts_ts * metrics['ts_ts']['loss']
            branch_acc.append(metrics['ts_ts']['acc_top1'])
            branch_loss.append(metrics['ts_ts']['loss'])

            print(f"Validation (TS-TS) - z1->z2 Top1: {z1_to_z2_acc:.4f}, Top5: {z1_to_z2_topk_acc:.4f}; "
                  f"z2->z1 Top1: {z2_to_z1_acc:.4f}, Top5: {z2_to_z1_topk_acc:.4f}; "
                  f"Avg Top1: {avg_acc_top1:.4f}, Avg Top5: {avg_acc_top5:.4f}, "
                  f"Loss: {avg_loss:.4f}")

        if use_ts_text and len(ts_text_ts_proj) > 0:
            all_ts_proj = torch.cat(ts_text_ts_proj, dim=0)
            all_text_proj = torch.cat(ts_text_text_proj, dim=0)
            similarity = all_ts_proj @ all_text_proj.T

            ts2text_pred = similarity.argmax(dim=1)
            ts2text_acc = (ts2text_pred == torch.arange(len(all_ts_proj))).float().mean()

            k = min(5, similarity.size(0))
            ts2text_topk = similarity.topk(k, dim=1).indices
            ts_indices = torch.arange(len(all_ts_proj)).unsqueeze(1)
            ts2text_topk_acc = (ts2text_topk == ts_indices).any(dim=1).float().mean()

            text2ts_pred = similarity.argmax(dim=0)
            text2ts_acc = (text2ts_pred == torch.arange(len(all_text_proj))).float().mean()

            text2ts_topk = similarity.topk(k, dim=0).indices
            text_indices = torch.arange(len(all_text_proj)).unsqueeze(0)
            text2ts_topk_acc = (text2ts_topk == text_indices).any(dim=0).float().mean()

            avg_acc_top1 = (ts2text_acc + text2ts_acc) / 2
            avg_acc_top5 = (ts2text_topk_acc + text2ts_topk_acc) / 2
            avg_loss = ts_text_total_loss / len(val_loader)

            metrics['ts_text'] = {
                'acc_top1': avg_acc_top1.item(),
                'acc_top5': avg_acc_top5.item(),
                'loss': avg_loss
            }

            weight_sum += self.lambda_ts_text
            overall_acc_numer += self.lambda_ts_text * metrics['ts_text']['acc_top1']
            overall_loss_numer += self.lambda_ts_text * metrics['ts_text']['loss']
            branch_acc.append(metrics['ts_text']['acc_top1'])
            branch_loss.append(metrics['ts_text']['loss'])

            print(f"Validation (TS-Text) - TS->Text Top1: {ts2text_acc:.4f}, Top5: {ts2text_topk_acc:.4f}; "
                  f"Text->TS Top1: {text2ts_acc:.4f}, Top5: {text2ts_topk_acc:.4f}; "
                  f"Avg Top1: {avg_acc_top1:.4f}, Avg Top5: {avg_acc_top5:.4f}, "
                  f"Loss: {avg_loss:.4f}")

        if weight_sum > 0:
            metrics['overall_acc'] = overall_acc_numer / weight_sum
            metrics['overall_loss'] = overall_loss_numer / weight_sum
        elif branch_acc:
            metrics['overall_acc'] = float(np.mean(branch_acc))
            metrics['overall_loss'] = float(np.mean(branch_loss))
        else:
            metrics['overall_acc'] = 0.0
            metrics['overall_loss'] = 0.0

        if 'ts_ts' in metrics and 'ts_text' in metrics:
            print(f"Validation (Combined) - Overall Top1: {metrics['overall_acc']:.4f}, "
                  f"Loss: {metrics['overall_loss']:.4f}")
            print(f"  TS-TS   - Top1: {metrics['ts_ts']['acc_top1']:.4f}, "
                  f"Top5: {metrics['ts_ts']['acc_top5']:.4f}, "
                  f"Loss: {metrics['ts_ts']['loss']:.4f}")
            print(f"  TS-Text - Top1: {metrics['ts_text']['acc_top1']:.4f}, "
                  f"Top5: {metrics['ts_text']['acc_top5']:.4f}, "
                  f"Loss: {metrics['ts_text']['loss']:.4f}")
        return metrics

    def save_checkpoint(self, path):
        """
        保存检查点

        Args:
            path: 保存路径
        """
        checkpoint = {
            'ts_encoder': self.ts_encoder.state_dict(),
            'ts_projector': self.ts_projector.state_dict(),
            'text_encoder': self.text_encoder.state_dict(),
            'text_projector': self.text_projector.state_dict(),
            'ts_encoder_optimizer': self.ts_encoder_optimizer.state_dict(),
            'ts_proj_optimizer': self.ts_proj_optimizer.state_dict(),
            'text_proj_optimizer': self.text_proj_optimizer.state_dict() if self.text_proj_optimizer is not None else None,
            'text_encoder_optimizer': self.text_encoder_optimizer.state_dict() if self.text_encoder_optimizer is not None else None,
            'args': self.args
        }

        torch.save(checkpoint, path)
        print(f'✓ Checkpoint saved to {path}')


    def load_checkpoint(self, path):
        """
        加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.ts_encoder.load_state_dict(checkpoint['ts_encoder'])
        self.ts_projector.load_state_dict(checkpoint['ts_projector'])
        if 'text_encoder' in checkpoint:
            self.text_encoder.load_state_dict(checkpoint['text_encoder'])
        self.text_projector.load_state_dict(checkpoint['text_projector'])

        if 'ts_encoder_optimizer' in checkpoint:
            self.ts_encoder_optimizer.load_state_dict(checkpoint['ts_encoder_optimizer'])
            self.ts_proj_optimizer.load_state_dict(checkpoint['ts_proj_optimizer'])
            if self.text_proj_optimizer is not None and checkpoint.get('text_proj_optimizer') is not None:
                self.text_proj_optimizer.load_state_dict(checkpoint['text_proj_optimizer'])
            if self.text_encoder_optimizer is not None and checkpoint.get('text_encoder_optimizer') is not None:
                self.text_encoder_optimizer.load_state_dict(checkpoint['text_encoder_optimizer'])

        print(f'✓ Checkpoint loaded from {path}')


    def extract_features(self, dataloader, modality='ts'):
        """
        提取特征（用于下游任务或可视化）

        Args:
            dataloader: 数据加载器
            modality: 'ts' 或 'text' 或 'both'

        Returns:
            features: numpy数组或字典
        """
        self.ts_encoder.eval()
        self.ts_projector.eval()
        self.text_projector.eval()

        ts_features = []
        text_features = []

        with torch.no_grad():
            for batch in dataloader:
                x_ot = batch['x_ot'].to(self.device)
                context_text = batch['context_text']

                if modality in ['ts', 'both']:
                    ts_hidden = self._encode_timeseries(x_ot)
                    ts_proj = self.ts_projector(ts_hidden)
                    ts_proj = F.normalize(ts_proj, dim=-1)
                    ts_features.append(ts_proj.cpu().numpy())

                if modality in ['text', 'both']:
                    text_hidden = self.encode_text(context_text)
                    text_proj = self.text_projector(text_hidden)
                    text_proj = F.normalize(text_proj, dim=-1)
                    text_features.append(text_proj.cpu().numpy())

        if modality == 'ts':
            return np.concatenate(ts_features, axis=0)
        elif modality == 'text':
            return np.concatenate(text_features, axis=0)
        else:
            return {
                'ts': np.concatenate(ts_features, axis=0),
                'text': np.concatenate(text_features, axis=0)
            }
