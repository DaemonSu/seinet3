import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenSetDetector(nn.Module):
    def __init__(self, input_dim=1025):
        super(OpenSetDetector, self).__init__()

        self.feature_conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Conv1d(1024, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # 输出 [B, 64, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),           # [B, 64, 1] → [B, 64]
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)        # 输出 [B, 1]
        )

    def forward(self, x):
        # 输入 x: [B, 1025] → 需要 reshape 成 [B, 1, 1025]
        x = x.unsqueeze(1)
        x = self.feature_conv(x)
        x = self.classifier(x)
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMatchNet(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block4 = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block5 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.block6 = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.Dropout(0.3)
        )

        self.block7 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

        self.block8 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # nn.Dropout(0.1)
        )

        # 残差连接，输入 feature_dim → 128 维以便与最后加法对齐
        self.residual = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.BatchNorm1d(32)
        )

        self.out = nn.Linear(32, 1)

    def forward(self, f):
        x = self.block1(f)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        shortcut = self.residual(f)
        x = x + shortcut  # 残差连接
        x = F.relu(x)

        return self.out(x)  # 直接输出 logits，配合 BCEWithLogitsLoss 使用




