import os

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from config import parse_args
from dataset import MixedDataset
from model_open import FeatureExtractor, ClassifierHead


def test_mixed(encoder, classifier, test_loader, config):
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing"):
            x, y = x.to(config.device), y.to(config.device)
            feat = encoder(x)
            logits = classifier(feat)
            prob = F.softmax(logits, dim=1)
            max_probs, preds = prob.max(dim=1)

            # Threshold-based prediction
            pred_labels = []
            for p, pred in zip(max_probs, preds):
                if p < config.open_threshold:
                    pred_labels.append(-1)  # Open-set
                else:
                    pred_labels.append(pred.item())  # Closed-set prediction

            all_preds.extend(pred_labels)
            all_labels.extend(y.cpu().tolist())
            all_logits.append(max_probs.cpu().numpy())

    all_logits = np.concatenate(all_logits)

    # Convert to numpy arrays
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    # Masking
    closed_mask = y_true != -1
    open_mask = y_true == -1

    # Closed-set Accuracy
    if closed_mask.sum() > 0:
        closed_acc = accuracy_score(y_true[closed_mask], y_pred[closed_mask])
    else:
        closed_acc = float('nan')

    # Open-set Recognition Rate
    if open_mask.sum() > 0:
        open_recognition_rate = np.mean(y_pred[open_mask] == -1)
    else:
        open_recognition_rate = float('nan')

    # Overall Accuracy
    overall_acc = np.mean(y_pred == y_true)

    # Open-set F1 Score (binary classification: -1 vs not -1)
    f1_open = f1_score(y_true == -1, y_pred == -1)

    print("\n===== Mixed Test Results =====")
    print(f"Closed-set Accuracy     : {closed_acc * 100:.2f}%")
    print(f"Open-set Recognition Rate: {open_recognition_rate * 100:.2f}%")
    print(f"Overall Accuracy        : {overall_acc * 100:.2f}%")
    print(f"Open-set F1 Score       : {f1_open:.4f}")

    return closed_acc, open_recognition_rate, overall_acc, f1_open


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


    # 假设已有：encoder, classifier, config, test_loader
    test_mixed(encoder, classifier, mixed_loader, config)


