"""Directional Inconsistency Asymmetry Modeling (DIAM).

Given clip embeddings and the two cross-modal predictions:
    eps_f = ||T_av(z_a) - z_v||      forward (audio -> visual) error
    eps_b = ||T_va(z_v) - z_a||      backward (visual -> audio) error
    C_i   = eps_f + eps_b            inconsistency magnitude
    A     = eps_f - eps_b            signed asymmetry; A > 0 suggests the visual
                                     stream is inconsistent, A < 0 the audio stream
"""

import torch
import torch.nn as nn


class DIAM(nn.Module):
    def forward(self, z_v, z_a, pred_v, pred_a):
        eps_f = torch.norm(pred_v - z_v, dim=1)
        eps_b = torch.norm(pred_a - z_a, dim=1)
        return {'eps_f': eps_f, 'eps_b': eps_b,
                'C_i': eps_f + eps_b, 'A': eps_f - eps_b}
