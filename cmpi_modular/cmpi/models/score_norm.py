"""EMA running standardisation of the scalar forensic scores (C_i, A, TIL)."""

import torch
import torch.nn as nn


class ScoreNormalizer(nn.Module):
    def __init__(self, names=('ci', 'asym', 'til'), momentum=0.99, warmup_updates=10):
        super().__init__()
        self.names = tuple(names)
        self.momentum = momentum
        self.warmup_updates = warmup_updates
        for n in self.names:
            self.register_buffer(f'{n}_mean', torch.tensor(1.0))
            self.register_buffer(f'{n}_std', torch.tensor(1.0))
        self.register_buffer('n_updates', torch.tensor(0))

    @torch.no_grad()
    def update(self, **scores):
        m = self.momentum if self.n_updates > self.warmup_updates else 0.5
        for n, v in scores.items():
            mu, std = v.mean(), v.std().clamp(min=1e-6)
            setattr(self, f'{n}_mean', m * getattr(self, f'{n}_mean') + (1 - m) * mu)
            setattr(self, f'{n}_std', m * getattr(self, f'{n}_std') + (1 - m) * std)
        self.n_updates += 1

    def forward(self, value, name):
        mean = getattr(self, f'{name}_mean')
        std = getattr(self, f'{name}_std').clamp(min=1e-6)
        return ((value - mean) / std).unsqueeze(1)               # [B, 1]
