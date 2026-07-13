"""
简化版多模态融合 - 针对小数据集优化

策略:
1. 减少参数量 - 避免过拟合
2. 简化架构 - 更容易优化
3. 保守融合 - 确保不比TS-only差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightweightFusionModule(nn.Module):
    """
    轻量级融合模块 - 适用于小数据集

    设计原则:
    1. 参数量小 - 避免过拟合
    2. 简单有效 - 易于优化
    3. 保守融合 - 加权和而非复杂注意力

    核心思想:
    - 学习一个标量权重 λ ∈ [0, 1]
    - fused = (1-λ) * ts_features + λ * text_proj
    - 如果文本无用，λ → 0，退化为TS-only
    - 如果文本有用，λ → 1，充分利用文本
    """

    def __init__(self, d_model, d_text, use_text, use_rag, output_dim=256):
        super().__init__()
        self.d_model = d_model
        self.use_text = use_text
        self.use_rag = use_rag

        if use_text:
            # 方案1: 标量权重融合 (最简单，参数最少)
            # 只学习一个标量，决定接受多少文本信息
            self.text_proj = nn.Linear(d_text, d_model)
            self.fusion_weight = nn.Parameter(torch.tensor(0.1))  # 初始值0.1，保守融合

            # 可选：学习每个维度的权重
            # self.fusion_weight = nn.Parameter(torch.ones(d_model) * 0.1)

        if use_rag:
            self.rag_proj = nn.Linear(d_model, d_model)
            self.rag_weight = nn.Parameter(torch.tensor(0.1))

        # 最终投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, ts_features, text_features=None, rag_features=None):
        """
        加权融合: fused = α*TS + β*Text + γ*RAG
        其中 α + β + γ = 1 (通过softmax保证)
        """
        fused = ts_features

        if self.use_text and text_features is not None:
            # 投影文本到TS空间
            text_proj = self.text_proj(text_features)

            # 标量权重融合 (自动在[0,1]之间)
            lambda_text = torch.sigmoid(self.fusion_weight)
            fused = (1 - lambda_text) * ts_features + lambda_text * text_proj

        if self.use_rag and rag_features is not None:
            rag_proj = self.rag_proj(rag_features)
            lambda_rag = torch.sigmoid(self.rag_weight)
            fused = (1 - lambda_rag) * fused + lambda_rag * rag_proj

        return self.output_proj(fused)


class AdaptiveGatedFusionModule(nn.Module):
    """
    自适应门控融合 - 样本自适应但参数少

    核心改进:
    1. 门控向量而非矩阵 - 减少参数量
    2. 初始化偏向TS - 保守融合
    3. 简单投影 - 避免过拟合
    """

    def __init__(self, d_model, d_text, use_text, use_rag, output_dim=256):
        super().__init__()
        self.d_model = d_model
        self.use_text = use_text
        self.use_rag = use_rag

        if use_text:
            # 文本投影
            self.text_proj = nn.Linear(d_text, d_model, bias=False)

            # 门控网络 (参数少)
            self.gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),  # 降维减少参数
                nn.ReLU(),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid()
            )
            # 初始化使gate初始值偏向0 (保守融合)
            with torch.no_grad():
                self.gate_net[-2].bias.data.fill_(-2.0)  # Sigmoid(-2) ≈ 0.12

        if use_rag:
            self.rag_proj = nn.Linear(d_model, d_model, bias=False)
            self.rag_gate_net = nn.Sequential(
                nn.Linear(d_model * 2, d_model // 4),
                nn.ReLU(),
                nn.Linear(d_model // 4, d_model),
                nn.Sigmoid()
            )
            with torch.no_grad():
                self.rag_gate_net[-2].bias.data.fill_(-2.0)

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, ts_features, text_features=None, rag_features=None):
        """
        样本自适应门控: gate = f(TS, Text)
        fused = TS + gate * Text
        """
        fused = ts_features

        if self.use_text and text_features is not None:
            text_proj = self.text_proj(text_features)

            # 计算门控 (每个样本不同)
            gate_input = torch.cat([ts_features, text_proj], dim=1)
            gate = self.gate_net(gate_input)  # [B, d_model]

            # 门控融合
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

            # 简化的注意力 (2个头)
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=2,  # 减少头数
                dropout=0.3,  # 增加dropout
                batch_first=True
            )

            self.norm = nn.LayerNorm(d_model)

            # 可学习的残差权重
            self.residual_weight = nn.Parameter(torch.tensor(0.9))  # 初始偏向TS

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
            nn.Dropout(0.3)  # 增加正则化
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

            # 可学习的残差权重
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
