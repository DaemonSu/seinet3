import torch
import numpy as np
import torch.nn.functional as F


class PrototypeMemory:
    def __init__(self, num_classes, embed_dim, momentum=0.9, device='cuda'):
        self.num_classes = num_classes
        self.momentum = momentum
        self.device = device
        self.prototypes = torch.zeros(num_classes, embed_dim).to(device)

    def update(self, features, labels):
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                proto_mean = features[mask].mean(0)
                self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * proto_mean
    # 获取某个标签对应的原型
    def getPrototy(self, labels):
        for i in range(self.num_classes):
            mask = (labels == i)
            if mask.sum() > 0:
                proto_mean = features[mask].mean(0)
                self.prototypes[i] = self.momentum * self.prototypes[i] + (1 - self.momentum) * proto_mean

    def classify(self, features,threshold=None,all_logit=None):
        dist = torch.cdist(features, self.prototypes)
        # 获得最小的距离以及该距离所述分类
        min_dist,preds = dist.min(dim=1)

        # 取出最相似的原型
        matched_prototypes = self.prototypes[preds]  # [B, feature_dim]

        diff = features - matched_prototypes
        cos = F.cosine_similarity(features, matched_prototypes, dim=1).unsqueeze(1)
        distance = torch.norm(diff, dim=1, keepdim=True)
        x = torch.cat([diff, cos, distance],dim=1)



        # 计算与原型的差值向量
        # diff_vector = features - matched_prototypes  # [B, feature_dim]
        # diff_vector = torch.cat((features, matched_prototypes), dim=-1)  # or dim=2

        return preds, min_dist, x


    def classify_topN_diff(self, features, N=3):
        """
        输入:
            features: [B, D]
            N:        Top-N 原型数量

        返回:
            diff_vectors: [B, N, D]  每个样本与 Top-N 原型的差值
            topN_indices: [B, N]     每个样本匹配的 Top-N 原型的索引位置
        """
        # 1. 计算特征与所有原型之间的距离 => [B, P]
        dist = torch.cdist(features, self.prototypes)

        # 2. 找到 Top-N 距离最小的原型索引 => [B, N]
        topN_dists, topN_indices = torch.topk(dist, N, dim=1, largest=False)

        # 3. 提取对应的 Top-N 原型向量 => [B, N, D]
        matched_prototypes = self.prototypes[topN_indices]  # [B, N, D]

        # 4. 扩展特征维度 => [B, N, D]
        features_expanded = features.unsqueeze(1).expand_as(matched_prototypes)

        diff = features_expanded - matched_prototypes
        cos = F.cosine_similarity(features_expanded, matched_prototypes, dim=1).unsqueeze(1)
        distance = torch.norm(diff, dim=1, keepdim=True)
        x = torch.cat([diff, cos, distance], dim=-1)

        # 沿最后一个维度拼接 → [B, N, 2F]
        # cat_vectors = torch.cat((features_expanded, matched_prototypes), dim=-1)  # or dim=2


        return topN_dists,x,topN_indices



