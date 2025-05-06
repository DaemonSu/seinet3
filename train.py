import os

import torch
from torch.utils.data import DataLoader
from model import SEIModel
from loss import ContrastiveLossWithCE
from prototype import PrototypeMemory
from dataset import SEIDataset
from config import parse_args
from utils import accuracy,save_checkpoint,save_object
from torch.optim.lr_scheduler import StepLR

args = parse_args()
device = args.device
# 加载数据
train_dataset = SEIDataset(args.train_data)
val_dataset = SEIDataset(args.val_data)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)


model = SEIModel(num_classes=10,input_dim=args.input_dim, seq_len=args.seq_len, embed_dim=args.embedding_dim).to(device)
criterion = ContrastiveLossWithCE(args.margin)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # 每10个epoch，学习率减少为原来的0.1
# prototype = PrototypeMemory()
prototype = PrototypeMemory(args.num_classes, args.embedding_dim, args.prototype_momentum, device)
for epoch in range(args.epochs):
    model.train()
    total_loss=0

    for batch in train_loader:
        x1,  labels = batch
        x1,  labels = x1.to(device),  labels.to(device)
        logits,features = model(x1, return_feature=True)

        # 打印logits和features的形状以及数值范围
        # print(f"logits: {logits.mean()}, {logits.std()}")  # 查看logits的均值和标准差
        # print(f"features: {features.mean()}, {features.std()}")  # 查看features的均值和标准差
        # 计算损失
        loss = criterion(features, labels, logits)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        prototype.update(features.detach(), labels.detach())

    scheduler.step()

    print(f"Epoch {epoch + 1}, train_loss: {total_loss:.4f}")
    # val
    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for x, label in val_loader:
            x, label = x.to(device), label.to(device)
            logit,feature = model(x,return_feature=True)
            all_features.append(feature)
            all_labels.append(label)
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    preds,min_dist,_ = prototype.classify(all_features)
    acc = accuracy(preds, all_labels)
    print(f'Epoch {epoch+1}, Acc: {acc:.4f}')

    # 将原型以文件的形式保存到文件夹中
    save_object(prototype, 'model/prototype.pkl')
    # 将深度学习模型保存到文件中
    # save_path: str = 'model_weight/CNN_FixMatch_SSML_Norm_Rotate2_n_classes_16_5label_95unlabel_rand30.pth',
    save_path = os.path.join(args.save_dir, f"seinet_{epoch}.pth")
    save_checkpoint(model,  save_path)
