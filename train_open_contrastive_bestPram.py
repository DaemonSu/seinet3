import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import KnownDataset, UnknownDataset, MixedDataset
from model_open import FeatureExtractor, ClassifierHead
from loss import SupConLoss_DynamicMargin
from util.utils import set_seed, adjust_lr, topaccuracy, save_object
from config import parse_args
import itertools


def train_open_contrastive(config, criterion_params):
    set_seed(config.seed)

    known_trainset = KnownDataset(config.train_data_close)
    known_loader = DataLoader(known_trainset, config.batch_size, True)

    unknown_trainset = UnknownDataset(config.train_data_open)
    unknown_loader = DataLoader(unknown_trainset, config.batch_size, shuffle=True)

    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)

    supcon_loss_fn = SupConLoss_DynamicMargin(
        temperature=criterion_params["temperature"],
        base_margin=criterion_params["base_margin"],
        beta=criterion_params["beta"]
    )
    ce_loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=config.lr)

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

            feat_known = encoder(x_known)
            feat_unknown = encoder(x_unknown)

            logits = classifier(feat_known)
            ce_loss = ce_loss_fn(logits, y_known)

            feat_all = torch.cat([feat_known, feat_unknown], dim=0)
            labels_all = torch.cat([y_known, torch.full((x_unknown.size(0),), -1, device=config.device)], dim=0)
            con_loss = supcon_loss_fn(feat_all, labels_all)

            con_weight = 1.0 + max(0.0, 1.0 - ce_loss.item()) * 1.5
            loss = ce_loss + con_weight * con_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = topaccuracy(logits, y_known)
            total_loss += loss.item()
            total_acc += acc
        if epoch % 10 == 0:
            print(f"[Epoch {epoch + 1}] Loss: {total_loss / len(known_loader):.4f} | Acc: {total_acc / len(known_loader):.2f}")

        adjust_lr(optimizer, epoch, config)

    return encoder, classifier




import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
from tqdm import tqdm

def validate_mixed(encoder, classifier, val_loader, config, open_threshold):
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_logits = []
    all_feats = []

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validation"):
            x, y = x.to(config.device), y.to(config.device)
            feat = encoder(x)
            logits = classifier(feat)

            prob = F.softmax(logits, dim=1)
            max_probs, preds = prob.max(dim=1)

            # Threshold-based prediction
            pred_labels = []
            for p, pred in zip(max_probs, preds):
                if p < open_threshold:
                    pred_labels.append(-1)  # Open-set
                else:
                    pred_labels.append(pred.item())  # Closed-set prediction

            all_preds.extend(pred_labels)
            all_labels.extend(y.cpu().tolist())
            all_logits.append(max_probs.cpu().numpy())
            all_feats.extend(feat.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    closed_mask = y_true != -1
    open_mask = y_true == -1

    if closed_mask.sum() > 0:
        closed_acc = accuracy_score(y_true[closed_mask], y_pred[closed_mask])
    else:
        closed_acc = float('nan')

    if open_mask.sum() > 0:
        open_recognition_rate = np.mean(y_pred[open_mask] == -1)
    else:
        open_recognition_rate = float('nan')

    overall_acc = np.mean(y_pred == y_true)
    f1_open = f1_score(y_true == -1, y_pred == -1)

    print(f"closed_acc: {closed_acc:.4f}")
    print(f"open_recognition_rate: {open_recognition_rate:.4f}")
    print(f"overall_acc: {overall_acc:.4f}")
    print(f"f1_open: {f1_open:.4f}")

    return overall_acc


import itertools

def run_grid_search(config):
    param_grid = {
        "temperature": [  0.07, 0.09, 0.11, 0.13,0.15],
        "base_margin": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
        "beta": [0.0,  0.2,  0.4,  0.6, 0.8 , 1.0],
    }

    best_score = -1
    best_params = None
    best_threshold = None
    results = []

    for combo in itertools.product(*param_grid.values()):
        params = dict(zip(param_grid.keys(), combo))
        print(f"\n[GridSearch] Params: {params}")

        # 训练模型
        encoder, classifier = train_open_contrastive(config, params)

        # 验证集
        val_dataset = MixedDataset(config.val_data)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        # 遍历不同的 open_threshold，找到最佳的阈值
        best_threshold_score = -1
        best_threshold_for_combo = None
        for open_threshold in [ 0.75,0.80, 0.85,0.90,0.95]:
            score = validate_mixed(encoder, classifier, val_loader, config, open_threshold)
            # 更新最佳阈值
            if score > best_threshold_score:
                best_threshold_score = score
                best_threshold_for_combo = open_threshold

        # 记录当前组合及其最佳 open_threshold 和得分
        results.append((params, best_threshold_for_combo, best_threshold_score))
        print(f"===========================")
        # 打印该参数组合下最佳的 open_threshold 结果
        print(f"Best open_threshold for this combination: {best_threshold_for_combo}")
        print(f"Score: {best_threshold_score:.4f}")

        # 更新最佳结果
        if best_threshold_score > best_score:
            best_score = best_threshold_score
            best_params = params
            best_threshold = best_threshold_for_combo

    print("\n========== Grid Search Complete ==========")
    print(f"Best Params: {best_params}")
    print(f"Best Threshold: {best_threshold}")
    print(f"Best Score: {best_score:.4f}")

    return best_params, best_threshold, best_score, results




if __name__ == "__main__":
    config = parse_args()
    run_grid_search(config)
