import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import KnownDataset, UnknownDataset
from model_mix import FeatureExtractor, ClassifierHead
from loss import SupConLoss_DynamicMargin
from PrototypeMemory import PrototypeMemory
from util.utils import set_seed, adjust_lr, topaccuracy, save_object
import os
from config import parse_args
import torch.nn.functional as F

def train_open_contrastive(config):
    set_seed(config.seed)

    # ============ 数据加载 ============
    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, config.batch_size, True)

    unknown_trainset = UnknownDataset(config.train_data_open)
    unknown_loader = DataLoader(unknown_trainset, batch_size=config.batch_size, shuffle=True)

    # ============ 模型初始化 ============
    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)

    # ============ 损失函数 ============
    supcon_loss_fn = SupConLoss_DynamicMargin()
    ce_loss_fn = nn.CrossEntropyLoss()

    # ============ 优化器 ============
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr)

    prototype = PrototypeMemory(config.num_classes, config.embedding_dim, config.device,config.prototype_momentum)


    # ============ 训练开始 ============
    for epoch in range(config.epochs):
        encoder.train()
        classifier.train()
        total_loss, total_acc = 0, 0

        known_iter = iter(known_loader)
        unknown_iter = iter(unknown_loader)

        for i in range(len(known_loader)):
            try:
                x_known, y_known = next(known_iter)
            except StopIteration:
                known_iter = iter(known_loader)
                x_known, y_known = next(known_iter)

            try:
                x_unknown, _ = next(unknown_iter)
            except StopIteration:
                unknown_iter = iter(unknown_loader)
                x_unknown, _ = next(unknown_iter)

            x_known, y_known = x_known.to(config.device), y_known.to(config.device)
            x_unknown = x_unknown.to(config.device)

            # 提取特征
            feat_known = encoder(x_known)
            feat_unknown = encoder(x_unknown)

            # 闭集分类损失
            logits = classifier(feat_known)
            ce_loss = ce_loss_fn(logits, y_known)

            logits_unknown = classifier(feat_unknown)
            probs_unknown = F.softmax(logits_unknown, dim=1)
            max_probs, _ = probs_unknown.max(dim=1)

            # 惩罚：鼓励开集样本的最大概率越低越好
            # 比如超过阈值的部分才被惩罚（soft margin）
            penalty = torch.clamp(max_probs -  config.open_threshold, min=0)
            penalty_loss = penalty.mean()


            # 对比损失：已知类之间 + 已知类 vs 未知类
            feat_all = torch.cat([feat_known, feat_unknown], dim=0)
            labels_all = torch.cat([y_known, torch.full((x_unknown.size(0),), -1, device=config.device)], dim=0)
            con_loss = supcon_loss_fn(feat_all, labels_all)

            con_weight = 1.0 + max(0.0, 1.0 - ce_loss.item()) * 1.5
            loss = ce_loss + con_weight * con_loss + penalty_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 更新原型
            prototype.update(feat_known.detach(), y_known.detach())


            acc = topaccuracy(logits, y_known)
            total_loss += loss.item()
            total_acc += acc

        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(known_loader):.4f} | Acc: {total_acc/len(known_loader):.2f}")

        adjust_lr(optimizer, epoch, config)

    # 模型保存
    os.makedirs(config.save_dir, exist_ok=True)
    torch.save({'encoder': encoder.state_dict()},
               os.path.join(config.save_dir, 'encoder.pth'))

    torch.save({'classifier': classifier.state_dict()},
               os.path.join(config.save_dir, 'classifier.pth'))

    # 将原型以文件的形式保存到文件夹中
    save_object(prototype, 'model/prototype2.pkl')

if __name__ == "__main__":

    config = parse_args()
    train_open_contrastive(config)
