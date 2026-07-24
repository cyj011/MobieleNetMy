import torch
import torch.nn as nn
import torch.nn.functional as F

class LiteChannelAttention(nn.Module):
    """
    Lite Channel Attention (LCA)
    轻量通道注意力：GAP + 自适应1D卷积 + Sigmoid
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        # 自适应核大小
        kernel_size = int(abs((torch.log2(torch.tensor(channels)) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.gap(x)          # [B, C, 1, 1]
        out = out.squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        out = self.conv(out)       # [B, 1, C]
        out = out.transpose(-1, -2).unsqueeze(-1) # [B, C, 1, 1]
        out = self.sigmoid(out)
        return x * out