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

        # 只保留列方向上为有效样本的
        # valid_mask_vec = valid_mask.view(-1)  # [B]
        # exp_sim = exp_sim * valid_mask_vec.unsqueeze(0)  # 屏蔽对无效样本的关注
        #
        # log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)

        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        loss = -mean_log_prob_pos.mean()

        return loss

class SupConLoss_SoftNegative(nn.Module):
    def __init__(self, temperature=0.07):
        super(SupConLoss_SoftNegative, self).__init__()
        self.temperature = temperature
        self.eps = 1e-8

    def forward(self, features, labels):
        """
        features: [B, D]
        labels: [B] (开集标签为 -1)
        """
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        labels = labels.contiguous().view(-1, 1)  # [B, 1]
        valid_mask = (labels != -1).float()  # [B, 1]

        # 正样本掩码，仅用于 known-closed
        mask = (labels == labels.T).float() * (valid_mask @ valid_mask.T)

        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        exp_sim = torch.exp(similarity_matrix) * logits_mask

        # ----------------------------
        # soft penalty on open-set negatives
        # ----------------------------
        open_mask_row = (labels == -1).float()  # [B, 1]
        open_mask_col = (labels.T == -1).float()  # [1, B]
        open_mask = open_mask_row @ torch.ones_like(open_mask_col) + torch.ones_like(open_mask_row) @ open_mask_col
        open_mask = torch.clamp(open_mask, 0, 1) * logits_mask  # remove diagonal

        # similarity 越高，w 越小（惩罚越轻）
        weights = 1.0 - similarity_matrix.detach()
        exp_sim = exp_sim * (1 - open_mask + open_mask * weights)

        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        loss = -mean_log_prob_pos.mean()

        return loss



class SupConLoss_DynamicMargin(nn.Module):
    def __init__(self, temperature=0.07, base_margin=0.3, beta=0.4):
        super(SupConLoss_DynamicMargin, self).__init__()
        self.temperature = temperature
        self.base_margin = base_margin
        self.beta = beta
        self.eps = 1e-8

    def forward(self, features, labels):
        device = features.device
        batch_size = features.size(0)

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature  # [B, B]

        labels = labels.contiguous().view(-1, 1)  # [B, 1]
        valid_mask = (labels != -1).float()

        mask = (labels == labels.T).float() * (valid_mask @ valid_mask.T)
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        exp_sim = torch.exp(similarity_matrix) * logits_mask

        # ----------------------------
        # Dynamic Margin for open-set negatives
        # ----------------------------
        open_mask_row = (labels == -1).float()
        open_mask_col = (labels.T == -1).float()
        open_mask = open_mask_row @ torch.ones_like(open_mask_col) + torch.ones_like(open_mask_row) @ open_mask_col
        open_mask = torch.clamp(open_mask, 0, 1) * logits_mask

        dynamic_margin = self.base_margin + self.beta * (1 - similarity_matrix.detach())
        similarity_matrix = similarity_matrix - open_mask * dynamic_margin

        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True) + self.eps)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)
        loss = -mean_log_prob_pos.mean()

        return loss



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
