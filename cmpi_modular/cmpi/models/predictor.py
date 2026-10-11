"""Cross-modal predictors T_av (audio -> visual) and T_va (visual -> audio)."""

import torch.nn as nn


class CrossModalPredictor(nn.Module):
    """Three-layer MLP: in_dim -> hidden -> hidden -> out_dim."""

    def __init__(self, in_dim=512, hidden_dim=1024, out_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
