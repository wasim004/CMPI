"""Training objective:  L = L_focal(det) + lambda * (L_av + L_va),
with the predictive losses computed on REAL samples only."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def focal_loss(logits, labels, alpha=0.75, gamma=2.0):
    labels = labels.float()
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt = torch.exp(-bce)
    w = torch.where(labels == 1, alpha, 1 - alpha)
    return (w * (1 - pt) ** gamma * bce).mean()


def predictive_loss(out, labels):
    real = labels == 0
    if real.sum() == 0:
        z = torch.tensor(0.0, device=labels.device)
        return z, z
    L_av = F.mse_loss(out['pred_v'][real], out['z_v'][real].detach())
    L_va = F.mse_loss(out['pred_a'][real], out['z_a'][real].detach())
    return L_av, L_va


class CMPILoss(nn.Module):
    def __init__(self, lambda_pred=0.1, focal_alpha=0.75, focal_gamma=2.0):
        super().__init__()
        self.lambda_pred = lambda_pred
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(self, out, labels):
        L_det = focal_loss(out['det_logits'], labels, self.focal_alpha, self.focal_gamma)
        L_av, L_va = predictive_loss(out, labels)
        L_pred = L_av + L_va
        total = L_det + self.lambda_pred * L_pred
        return total, {'total': total.item(), 'det': L_det.item(), 'L_av': L_av.item(),
                       'L_va': L_va.item(), 'L_pred': L_pred.item()}
