import torch
import torch.nn.functional as F
from torch import nn

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    """
    def __init__(self, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, features, labels):
        """
        features: [batch_size, feature_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        labels = labels.contiguous().view(-1, 1)

        valid_mask = (labels != -1).float()  # 只保留闭集样本
        mask = (labels == labels.T).float() * (valid_mask @ valid_mask.T)

        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        exp_sim = torch.exp(similarity_matrix) * logits_mask

        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        loss = -mean_log_prob_pos.mean()

        return loss

    # def forward(self, features, labels):
    #     """
    #     features: [batch_size, feature_dim]
    #     labels: [batch_size]
    #     """
    #     device = features.device
    #     batch_size = features.size(0)
    #
    #     # 特征归一化
    #     features = F.normalize(features, dim=1)
    #
    #     # 相似度矩阵计算
    #     similarity_matrix = torch.matmul(features, features.T) / self.temperature
    #
    #     # 创建正负样本mask
    #     labels = labels.contiguous().view(-1, 1)
    #     mask = torch.eq(labels, labels.T).float().to(device)
    #
    #     # 排除自身的相似度
    #     logits_mask = torch.ones_like(mask).fill_diagonal_(0)
    #
    #     # exp(sim)
    #     exp_sim = torch.exp(similarity_matrix) * logits_mask
    #
    #     # 对比损失计算
    #     log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)
    #
    #     mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
    #
    #     loss = -mean_log_prob_pos.mean()
    #
    #     return loss


class ContrastiveLossWithCE(nn.Module):
    """
    Combining SupCon Loss and CrossEntropy Loss
    """
    def __init__(self, temperature=0.07, weight_ce=2):
        super(ContrastiveLossWithCE, self).__init__()
        self.supcon_loss = SupConLoss(temperature)
        self.ce_loss = nn.CrossEntropyLoss()
        self.weight_ce = weight_ce

    def forward(self, features, labels=None, logits=None):
        # 计算对比损失
        supcon_loss = self.supcon_loss(features, labels)

        # 如果提供了logits（分类输出），则计算CE损失
        if logits is not None:
            ce_loss = self.ce_loss(logits, labels)
            total_loss = supcon_loss + self.weight_ce * ce_loss
        else:
            total_loss = supcon_loss

        # print(f"supcon_loss:{supcon_loss}. ce_loss:{ce_loss}.total_loss:{total_loss}")

        return total_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal.mean()
