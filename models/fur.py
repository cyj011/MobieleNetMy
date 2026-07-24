import torch
import torch.nn as nn


class FrequencyUncertaintyReweighting(nn.Module):

    def __init__(self, channels, reduction=4):
        super().__init__()

        hidden = max(channels // reduction, 32)

        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        B, C, H, W = x.shape

        fft = torch.fft.fft2(x)

        magnitude = torch.abs(fft)

        freq_feature = magnitude.mean(dim=(2, 3))

        weight = self.fc(freq_feature)

        weight = weight.view(B, C, 1, 1)

        return x * weight