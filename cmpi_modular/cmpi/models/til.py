"""Temporal Inconsistency Localization (TIL).

Per-frame error  e_t = ||T_av(z_a) - h_t||  against the frame features h_t,
scored with additive attention  alpha_t = softmax(w^T tanh(W e_t)),
and pooled into  TIL = sum_t alpha_t * e_t.
"""

import torch
import torch.nn as nn


class TemporalAttentionScorer(nn.Module):
    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                nn.Linear(hidden, 1))

    def forward(self, err_seq):
        # err_seq: [B, T]
        alpha = torch.softmax(self.fc(err_seq.unsqueeze(-1)).squeeze(-1), dim=1)
        return (alpha * err_seq).sum(dim=1), alpha               # [B], [B, T]


class TemporalInconsistencyLocalization(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.scorer = TemporalAttentionScorer(in_dim=1, hidden=hidden)

    def forward(self, pred_v, frame_features):
        frame_err = torch.norm(pred_v.unsqueeze(1) - frame_features, dim=2)  # [B, T]
        til, alpha = self.scorer(frame_err)
        return {'til': til, 'til_alpha': alpha, 'frame_err': frame_err}
