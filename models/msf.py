import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleFusion(nn.Module):

    def __init__(
            self,
            low_channels=128,
            high_channels=960,
            out_channels=256
    ):
        super().__init__()

        # 浅层特征压缩
        self.low_proj = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 深层特征压缩
        self.high_proj = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, low_feat, high_feat):

        # 通道对齐
        low_feat = self.low_proj(low_feat)
        high_feat = self.high_proj(high_feat)

        # 尺寸对齐
        high_feat = F.interpolate(
            high_feat,
            size=low_feat.shape[2:],
            mode='bilinear',
            align_corners=False
        )

        # 特征拼接
        x = torch.cat([low_feat, high_feat], dim=1)

        # 融合
        x = self.fusion(x)

        return x