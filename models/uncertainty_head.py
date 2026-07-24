# uncertainty_head.py

import torch
import torch.nn as nn

class UAHClassifier(nn.Module):

    def __init__(
        self,
        in_features,
        num_classes
    ):
        super().__init__()

        self.mean_head = nn.Linear(
            in_features,
            num_classes
        )

        self.var_head = nn.Sequential(
            nn.Linear(
                in_features,
                num_classes
            ),
            nn.Softplus()
        )

    def forward(self, x):

        mean = self.mean_head(x)

        variance = self.var_head(x)

        return mean, variance