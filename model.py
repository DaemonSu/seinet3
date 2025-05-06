import torch
import torch.nn as nn
import torch.nn.functional as F
from complexcnn import ComplexConv
from torch.nn import MaxPool1d, Flatten, BatchNorm1d, LazyLinear, Dropout


class CNNMLPBackbone(nn.Module):
    """
    CNN + MLP 特征提取器
    """
    def __init__(self, input_dim=2, seq_len=14000, embed_dim=256):
        super(CNNMLPBackbone, self).__init__()
        # self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, stride=1)
        # self.bn1 = nn.BatchNorm1d(64)
        # self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1)
        # self.bn2 = nn.BatchNorm1d(128)
        # self.fc = nn.Linear(128 * (seq_len - 8), embed_dim)  # 调整为特定序列长度的输出
        self.conv1 = ComplexConv(in_channels=1, out_channels=64, kernel_size=3)
        self.batchnorm1 = BatchNorm1d(num_features=192)
        self.maxpool1 = MaxPool1d(kernel_size=2)

        self.conv2 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm2 = BatchNorm1d(num_features=192)
        self.maxpool2 = MaxPool1d(kernel_size=2)

        self.conv3 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm3 = BatchNorm1d(num_features=192)
        self.maxpool3 = MaxPool1d(kernel_size=2)

        self.conv4 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm4 = BatchNorm1d(num_features=192)
        self.maxpool4 = MaxPool1d(kernel_size=2)

        self.conv5 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm5 = BatchNorm1d(num_features=192)
        self.maxpool5 = MaxPool1d(kernel_size=2)

        self.conv6 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm6 = BatchNorm1d(num_features=192)
        self.maxpool6 = MaxPool1d(kernel_size=2)

        self.conv7 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm7 = BatchNorm1d(num_features=192)
        self.maxpool7 = MaxPool1d(kernel_size=2)

        self.conv8 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm8 = BatchNorm1d(num_features=192)
        self.maxpool8 = MaxPool1d(kernel_size=2)

        self.conv9 = ComplexConv(in_channels=64, out_channels=64, kernel_size=3)
        self.batchnorm9 = BatchNorm1d(num_features=192)
        self.maxpool9 = MaxPool1d(kernel_size=2)

        self.flatten = Flatten()
        self.linear1 = LazyLinear(1024)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # 转换为 [B, C, L]，适配 Conv1d 输入
        # x = F.relu(self.bn1(self.conv1(x)))
        # x = F.relu(self.bn2(self.conv2(x)))
        # x = x.view(x.size(0), -1)  # Flatten
        # x = self.fc(x)
        # x = F.normalize(x, p=2, dim=-1)  # 归一化特征
        x = self.conv1(x)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.batchnorm3(x)
        x = self.maxpool3(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.batchnorm4(x)
        x = self.maxpool4(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = self.batchnorm5(x)
        x = self.maxpool5(x)

        x = self.conv6(x)
        x = F.relu(x)
        x = self.batchnorm6(x)
        x = self.maxpool6(x)

        x = self.conv7(x)
        x = F.relu(x)
        x = self.batchnorm7(x)
        x = self.maxpool7(x)

        x = self.conv8(x)
        x = F.relu(x)
        x = self.batchnorm8(x)
        x = self.maxpool8(x)

        x = self.conv9(x)
        x = F.relu(x)
        x = self.batchnorm9(x)
        x = self.maxpool9(x)

        x = self.flatten(x)

        x = self.linear1(x)
        embedding = F.relu(x)

        # output = self.linear2(embedding)

        return embedding


class SEIModel(nn.Module):
    """
    SEI模型（CNN + MLP + 分类器）
    """
    def __init__(self, num_classes=10, input_dim=2, seq_len=14000, embed_dim=1024):
        super(SEIModel, self).__init__()
        self.backbone = CNNMLPBackbone(input_dim, seq_len, embed_dim)
        self.classifier = LazyLinear( num_classes)  # 分类头

    def forward(self, x, return_feature=False):
        features = self.backbone(x)  # 提取特征
        logits = self.classifier(features)  # 分类输出 logits
        if return_feature:
            return logits, features
        else:
            return logits
