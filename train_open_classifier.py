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

def train_open_classifier(config):
    set_seed(config.seed)

    # ============ 数据加载 ============
    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, config.batch_size, True)

    unknown_trainset = UnknownDataset(config.train_data_open)
    unknown_loader = DataLoader(unknown_trainset, batch_size=config.batch_size, shuffle=True)

    # ============ 加载 encoder 并冻结 ============
    encoder = FeatureExtractor(1024).to(config.device)
    ckpt1 = torch.load(os.path.join(config.save_dir, 'encoder.pth'), map_location=config.device)
    encoder.load_state_dict(ckpt1['encoder'])
    encoder.eval()

    for param in encoder.parameters():
        param.requires_grad = False

    # ============ 初始化 classifier ============
    classifier = ClassifierHead(1024, config.num_classes).to(config.device)

    # ============ 优化器 & 损失函数 ============
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=config.lr)

    ce_loss_fn = nn.CrossEntropyLoss()

    # ============ 开始训练 ============
    for epoch in range(config.epochs2):
        classifier.train()
        total_loss, total_acc = 0, 0

        known_iter = iter(known_loader)
        unknown_iter = iter(unknown_loader)

        for x, y in known_loader:
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

            with torch.no_grad():
                # 提取特征
                feat_known = encoder(x_known)
                feat_unknown = encoder(x_unknown)
            # feat_all = torch.cat([feat_known, feat_unknown], dim=0)
            # labels_all = torch.cat([y_known, torch.full((x_unknown.size(0),), -1, device=config.device)], dim=0)

            logits = classifier(feat_known)

            ce_loss = ce_loss_fn(logits, y_known)

            # 惩罚：鼓励开集样本的最大概率越低越好
            # 比如超过阈值的部分才被惩罚（soft margin）
            logits_unknown = classifier(feat_unknown)
            probs_unknown = F.softmax(logits_unknown, dim=1)
            max_probs, _ = probs_unknown.max(dim=1)
            penalty = torch.clamp(max_probs - config.open_threshold, min=0)
            penalty_loss = penalty.mean()

            uniform = torch.full_like(probs_unknown, 1.0 / probs_unknown.size(1))
            kl_loss = F.kl_div(probs_unknown.log(), uniform, reduction='batchmean')


            loss= ce_loss + penalty_loss + 2*kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = logits.argmax(dim=1)
            acc = (pred == y_known).float().mean()

            total_loss += loss.item()
            total_acc += acc.item()
            print(
            f" CE_Loss: {ce_loss:.4f} | penalty_loss: {penalty_loss:.4f}| kl_loss: {kl_loss:.4f}")

        print(
            f"[Epoch {epoch + 1}] Loss: {total_loss / len(known_loader):.4f} | Acc: {total_acc / len(known_loader):.4f}")
    os.makedirs(config.save_dir, exist_ok=True)
    torch.save({'classifier': classifier.state_dict()},
               os.path.join(config.save_dir, 'classifier.pth'))

if __name__ == "__main__":

    config = parse_args()
    train_open_classifier(config)
