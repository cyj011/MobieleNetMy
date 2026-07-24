
import torch
import torch.nn as nn


class DynamicTokenPruning(nn.Module):
    """
    Dynamic Token Pruning for CNN Feature Maps
    """

    def __init__(
        self,
        channels,
        keep_ratio=0.7
    ):
        super().__init__()

        self.keep_ratio = keep_ratio

        self.score = nn.Conv2d(
            channels,
            1,
            kernel_size=1
        )

    def forward(self, x):

        B, C, H, W = x.shape

        importance = self.score(x)

        importance = importance.view(B, -1)

        k = max(
            1,
            int(H * W * self.keep_ratio)
        )

        topk_idx = torch.topk(
            importance,
            k,
            dim=1
        )[1]

        mask = torch.zeros_like(importance)

        mask.scatter_(
            1,
            topk_idx,
            1.0
        )

        mask = mask.view(B, 1, H, W)

        return x * mask