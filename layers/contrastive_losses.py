import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    NT-Xent损失（Normalized Temperature-scaled Cross Entropy Loss）
    也称为InfoNCE损失，用于自监督对比学习

    用于时序-时序对比学习，拉近同一样本不同增强视图的距离，
    推远不同样本的距离。

    Args:
        temperature: 温度参数，控制softmax的平滑程度（default: 0.07）
                    较小的温度使模型对困难负样本更敏感

    Reference:
        Chen et al. "A Simple Framework for Contrastive Learning of Visual Representations" (SimCLR)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, exclude_neighbors: int = 0):
        """
        Args:
            z1: [batch, proj_dim] 第一个视图的投影
            z2: [batch, proj_dim] 第二个视图的投影
            exclude_neighbors: int，可选，屏蔽|i-j|<=k的近邻负样本（默认0，不屏蔽）
        Returns:
            loss: scalar 对比损失
        """
        batch_size = z1.shape[0]

        # 拼接两个视图
        z = torch.cat([z1, z2], dim=0)  # [2B, proj_dim]

        # 计算余弦相似度矩阵
        # 归一化（z已在外部归一化，这里确保）
        z = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z, z.T) / self.temperature  # [2B, 2B]

        # 构造标签（对角线偏移B的位置为正样本）
        # 对于z1[i]，其正样本是z2[i]，即sim_matrix[i, i+B]
        # 对于z2[i]，其正样本是z1[i]，即sim_matrix[i+B, i]
        labels = torch.arange(batch_size, dtype=torch.long).to(z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # [2B]

        # 移除自身相似度（对角线元素设为极小值）
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # 屏蔽近邻负样本
        if exclude_neighbors > 0:
            idx = torch.arange(batch_size, device=z.device)
            orig_idx = torch.cat([idx, idx])  # [2B]
            diff = (orig_idx.unsqueeze(0) - orig_idx.unsqueeze(1)).abs()
            near_mask = (diff <= exclude_neighbors) & (orig_idx.unsqueeze(0) != orig_idx.unsqueeze(1))
            sim_matrix = sim_matrix.masked_fill(near_mask, -9e15)

        # 计算交叉熵损失
        loss = F.cross_entropy(sim_matrix, labels)

        return loss


class CLIPLoss(nn.Module):
    """
    CLIP风格的对比损失
    用于时序-文本跨模态对比学习

    通过双向对比损失对齐时序和文本的语义空间：
    - TS → Text: 给定时序，找到对应的文本
    - Text → TS: 给定文本，找到对应的时序

    Args:
        temperature: 温度参数（default: 0.07）

    Reference:
        Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, ts_proj, text_proj, exclude_neighbors: int = 0):
        """
        Args:
            ts_proj: [batch, proj_dim] 时序投影特征
            text_proj: [batch, proj_dim] 文本投影特征
            exclude_neighbors: int，可选，屏蔽|i-j|<=k的近邻负样本（默认0，不屏蔽）
        Returns:
            loss: scalar 跨模态对比损失
        """
        batch_size = ts_proj.shape[0]

        # 归一化（确保在单位球面上）
        ts_proj = F.normalize(ts_proj, dim=1)
        text_proj = F.normalize(text_proj, dim=1)

        # 计算相似度矩阵
        logits_per_ts = torch.mm(ts_proj, text_proj.T) / self.temperature    # [B, B]
        logits_per_text = torch.mm(text_proj, ts_proj.T) / self.temperature  # [B, B]

        if exclude_neighbors > 0:
            idx = torch.arange(batch_size, device=ts_proj.device)
            diff = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()
            near_mask = (diff <= exclude_neighbors) & (~torch.eye(batch_size, dtype=torch.bool, device=ts_proj.device))
            logits_per_ts = logits_per_ts.masked_fill(near_mask, -9e15)
            logits_per_text = logits_per_text.masked_fill(near_mask, -9e15)

        # 标签（对角线为正样本）
        labels = torch.arange(batch_size, dtype=torch.long).to(ts_proj.device)

        # 双向损失
        loss_ts2text = F.cross_entropy(logits_per_ts, labels)
        loss_text2ts = F.cross_entropy(logits_per_text, labels)

        # 平均
        loss = (loss_ts2text + loss_text2ts) / 2

        return loss


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss（监督对比损失）
    如果有标签信息，可以使用此损失

    Args:
        temperature: 温度参数
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch, proj_dim] 特征向量
            labels: [batch] 类别标签
        Returns:
            loss: scalar
        """
        batch_size = features.shape[0]

        # 归一化
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        sim_matrix = torch.mm(features, features.T) / self.temperature  # [B, B]

        # 构建标签mask（同类为正样本）
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)  # [B, B]

        # 移除对角线
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        # 计算exp
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # 计算损失
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        loss = -mean_log_prob_pos.mean()

        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss（三元组损失）
    用于对比学习的另一种形式

    Args:
        margin: 边界值（default: 1.0）
    """
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Args:
            anchor: [batch, dim] 锚点
            positive: [batch, dim] 正样本
            negative: [batch, dim] 负样本
        Returns:
            loss: scalar
        """
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)

        loss = F.relu(pos_dist - neg_dist + self.margin)

        return loss.mean()
