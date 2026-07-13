import torch
import torch.nn as nn


class ProjectionHead(nn.Module):
    """
    MLP投影头（用于对比学习）

    将编码器的输出映射到低维对比学习空间

    Args:
        input_dim: 输入维度（d_model或d_llm）
        hidden_dims: 隐藏层维度列表，如 [512, 256]
        output_dim: 输出维度（proj_dim，如128）
        use_bn: 是否使用BatchNorm
        dropout: dropout率（默认0.1）

    Example:
        >>> proj = ProjectionHead(input_dim=768, hidden_dims=[512, 256], output_dim=128)
        >>> x = torch.randn(32, 768)
        >>> out = proj(x)  # [32, 128]
    """
    def __init__(self, input_dim, hidden_dims, output_dim, use_bn=True, dropout=0.1):
        super().__init__()

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]

        for i in range(len(dims) - 1):
            # 线性层
            layers.append(nn.Linear(dims[i], dims[i+1]))

            # 中间层添加激活函数、BN和Dropout
            if i < len(dims) - 2:
                if use_bn:
                    layers.append(nn.BatchNorm1d(dims[i+1]))
                layers.append(nn.ReLU(inplace=True))
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.projection = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: [batch, input_dim] 编码器输出
        Returns:
            out: [batch, output_dim] 投影后的特征
        """
        return self.projection(x)


class DualProjectionHead(nn.Module):
    """
    双投影头（用于需要两个独立投影的场景）

    Args:
        ts_input_dim: 时序编码器输出维度
        text_input_dim: 文本编码器输出维度
        hidden_dims: 隐藏层维度列表
        output_dim: 共享的输出维度
    """
    def __init__(self, ts_input_dim, text_input_dim, hidden_dims, output_dim, use_bn=True):
        super().__init__()

        self.ts_projector = ProjectionHead(
            input_dim=ts_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_bn=use_bn
        )

        self.text_projector = ProjectionHead(
            input_dim=text_input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            use_bn=use_bn
        )

    def forward_ts(self, ts_features):
        """时序特征投影"""
        return self.ts_projector(ts_features)

    def forward_text(self, text_features):
        """文本特征投影"""
        return self.text_projector(text_features)
