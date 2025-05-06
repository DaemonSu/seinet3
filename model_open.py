# 文件：model.py

import torch


import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)

class FeatureExtractor(nn.Module):
    def __init__(self, feature_dim=128):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=7, stride=2, padding=3),  # [B, 64, 3500]
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.layer1 = ResidualBlock(64, 128, stride=2)    # -> [B, 128, 1750]
        self.layer2 = ResidualBlock(128, 128, stride=2)   # -> [B, 128, 875]
        self.layer3 = ResidualBlock(128, 256, stride=2)   # -> [B, 256, 438]
        self.layer4 = ResidualBlock(256, 256, stride=2)   # -> [B, 256, 219]

        self.pool = nn.AdaptiveAvgPool1d(1)               # -> [B, 256, 1]
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = x.transpose(1, 2)  # [B, 3, 7000]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)  # [B, 256]
        feat = F.normalize(self.fc(x), dim=-1)  # [B, feature_dim]
        return feat


class ClassifierHead(nn.Module):
    def __init__(self, in_dim=128, num_classes=10):
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)

    # open_detector.py



class OpenDetector(nn.Module):
    def __init__(self, feature_dim, num_classes, hidden_dim=128):
        super(OpenDetector, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim + num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)  # 输出1个logit，代表开/闭
        )

    def forward(self, x):
        """
        feat: [batch_size, feature_dim]
        logits: [batch_size, num_classes]
        """
        out = self.fc(x)
        return out.squeeze(-1)

