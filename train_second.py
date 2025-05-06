import os
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from OpenSetDetector import OpenSetDetector, FeatureMatchNet
from dataset import SEIDataset
from config import parse_args
from evaluate import evaluate_openset_predictions
from utils import load_object, accuracy, save_checkpoint, testAccuracy
from loss import FocalLoss

torch.set_printoptions(profile="full")

def load_model_and_prototype(model_path, prototype_path):
    """ 加载模型和原型，并处理异常 """
    try:
        model_one = torch.load(model_path)
        prototype = load_object(prototype_path)
        return model_one, prototype
    except FileNotFoundError as e:
        print(f"文件加载失败: {e}")
        raise

def train_openset_detector(
    model, dataloader, val_dataloader, num_epochs=50, lr=5e-4, device='cuda', N=1
):
    """ 训练二阶段检测器 """
    # 加载第一阶段模型和原型
    model_one, prototype = load_model_and_prototype("model/seinet_49.pth", 'model/prototype.pkl')
    model_one.eval()
    for param in model_one.parameters():
        param.requires_grad = False

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            x1, labels = batch
            x1, labels = x1.to(device), labels.to(device)

            with torch.no_grad():
                logits, feature = model_one(x1, return_feature=True)

            # topN_dists, cat_vectors, topN_indices = prototype.classify_topN_diff(feature, N)
            preds, min_dist, x = prototype.classify(feature)


            isopen = model(x)
            binary_labels = (labels < 10).float().unsqueeze(1)
            loss = criterion(isopen, binary_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"第二阶段训练 [Epoch {epoch+1}/{num_epochs}] Loss: {avg_loss:.4f}")

        if epoch % 10 == 0:
            validate_model(model, model_one, prototype, val_dataloader, device, N)

def validate_model(model, model_one, prototype, dataloader, device, N):
    """ 验证模型性能 """
    model.eval()
    all_predict = torch.empty(0, dtype=torch.float32).to(device)
    all_binary_labels = torch.empty(0, dtype=torch.float32).to(device)

    with torch.no_grad():
        for x2, labelVal in dataloader:
            x2, labelVal = x2.to(device), labelVal.to(device)
            logits, feature = model_one(x2, return_feature=True)
            preds, min_dists, diff_vectors = prototype.classify(feature, N)
            binary_labels = (labelVal < 10).float().unsqueeze(1)

            # min_dists = min_dists.unsqueeze(1)
            # x3_combined = torch.cat((min_dists, diff_vectors), dim=1)
            # x3_final = x3_combined.view(-1, x3_combined.shape[-1])

            isopen = model(diff_vectors)
            probs = torch.sigmoid(isopen)
            predictions = (probs >= 0.63).float()



            all_predict = torch.cat((all_predict, predictions), dim=0)
            all_binary_labels = torch.cat((all_binary_labels, binary_labels), dim=0)

        correct = (all_predict == all_binary_labels).float().sum()
        total = all_binary_labels.numel()
        accuracyVal = correct / total
        print(f'验证集 情况，数据总量为 {total}，Acc: {accuracyVal:.4f}')

        # 调用评估函数
        best_threshold, best_metrics = evaluate_openset_predictions(all_predict, all_binary_labels)

        print(f"阈值: {best_threshold:.4f}, 准确率: {best_metrics['accuracy']:.4f}, 精确率: {best_metrics['precision']:.4f}, 召回率: {best_metrics['recall']:.4f}, F1: {best_metrics['f1']:.4f}")

def main():
    args = parse_args()
    device = args.device

    train_second = SEIDataset(args.train_second)
    val_second = SEIDataset(args.val_second)

    train_second_loader = DataLoader(train_second, batch_size=args.batch_size, shuffle=True)
    val_second_loader = DataLoader(val_second, batch_size=args.batch_size, shuffle=True)

    model = FeatureMatchNet(1026)
    train_openset_detector(model, train_second_loader, val_second_loader, num_epochs=150, device=device, N=1)

    save_path = os.path.join(args.save_dir, "train_second4.pth")
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_checkpoint(model, save_path)

if __name__ == "__main__":
    main()
