"""Detection head over [z_v; z_a; C_i; A; TIL] (2D + 3 inputs)."""

import torch.nn as nn


class DetectionHead(nn.Module):
    def __init__(self, embed_dim=512, n_scores=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2 + n_scores, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)                           # logits [B]
