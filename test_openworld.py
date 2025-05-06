import os

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from config import parse_args
from dataset import MixedDataset
from model_open import ClassifierHead, FeatureExtractor, OpenDetector


@torch.no_grad()
def test_openworld(encoder, classifier, open_detector, test_loader, config):
    encoder.eval()
    classifier.eval()
    open_detector.eval()

    all_preds = []
    all_labels = []

    for x, y in tqdm(test_loader, desc="Testing OpenWorld"):
        x, y = x.to(config.device), y.to(config.device)

        # 提取特征和分类器输出
        features = encoder(x)
        logits = classifier(features)
        probs = F.softmax(logits, dim=1)
        # 拼接 features 和 logits 作为 open_detector 输入
        open_input = torch.cat([features, logits], dim=1)
        # 将features + logits送入open detector，得到开集概率
        open_logits = open_detector(open_input).squeeze()
        open_probs = torch.sigmoid(open_logits)

        pred_labels = []
        max_probs, preds = probs.max(dim=1)
        for p_open,  pred in zip(open_probs,  preds):
            if p_open > config.open_detector_threshold:
                pred_labels.append(-1)  # 判定为 open-set (未知设备)
            else:
                pred_labels.append(pred.item())  # 判定为闭集设备

        all_preds.extend(pred_labels)
        all_labels.extend(y.cpu().tolist())

    # 转成numpy数组
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # 评估指标
    closed_mask = (y_true != -1)
    open_mask = (y_true == -1)

    closed_correct = (y_pred[closed_mask] == y_true[closed_mask]).sum()
    closed_total = closed_mask.sum()

    open_correct = (y_pred[open_mask] == -1).sum()
    open_total = open_mask.sum()

    closed_acc = 100.0 * closed_correct / closed_total if closed_total > 0 else 0
    open_recognition = 100.0 * open_correct / open_total if open_total > 0 else 0
    overall_acc = 100.0 * (closed_correct + open_correct) / len(y_true)

    open_f1 = f1_score((y_true == -1).astype(int), (y_pred == -1).astype(int))

    print(f"Closed-set Accuracy      : {closed_acc:.2f}%")
    print(f"Open-set Recognition Rate: {open_recognition:.2f}%")
    print(f"Overall Accuracy         : {overall_acc:.2f}%")
    print(f"Open-set F1 Score        : {open_f1:.4f}")

    return closed_acc, open_recognition, overall_acc, open_f1


if __name__ == "__main__":
    config = parse_args()
    # ============ 数据加载 ============

    mixed_testset = MixedDataset(config.test_mixed)
    mixed_loader = DataLoader(mixed_testset, batch_size=config.batch_size, shuffle=False)

    # ============ 加载模型 ============
    encoder = FeatureExtractor(1024).to(config.device)
    classifier = ClassifierHead(1024, 10).to(config.device)


    ckpt = torch.load(os.path.join(config.save_dir, 'model_opencon2.pth'), map_location=config.device)
    encoder.load_state_dict(ckpt['encoder'])
    classifier.load_state_dict(ckpt['classifier'])

    ckpt2 = torch.load(os.path.join(config.save_dir, 'open_detector.pth'), map_location=config.device)
    open_detector = OpenDetector(feature_dim=config.embedding_dim, num_classes=config.num_classes).to(config.device)

    open_detector.load_state_dict(ckpt2)

    # 推理测试
    test_openworld(encoder, classifier, open_detector, mixed_loader, config)
