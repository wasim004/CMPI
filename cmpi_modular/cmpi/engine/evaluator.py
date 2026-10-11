"""Inference over a loader and metric computation."""

import numpy as np
import torch

from ..utils.metrics import detection_metrics, attribution_metrics


@torch.no_grad()
def collect_outputs(model, loader, device, criterion=None):
    model.eval()
    keys = ('det_probs', 'C_i', 'A', 'til')
    buf = {k: [] for k in keys}
    labels, mod_v, mod_a = [], [], []
    loss_sums, n = {}, 0
    for batch in loader:
        y = batch['label'].to(device)
        out = model(batch['frames'].to(device), batch['audio'].to(device), update_stats=False)
        if criterion is not None:
            _, ld = criterion(out, y)
            for k, v in ld.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v
            n += 1
        for k in keys:
            buf[k].append(out[k].float().cpu().numpy())
        labels.append(y.cpu().numpy())
        if 'modify_video' in batch:
            mod_v.append(np.asarray(batch['modify_video'], dtype=int))
            mod_a.append(np.asarray(batch['modify_audio'], dtype=int))
    res = {k: np.concatenate(v) for k, v in buf.items()}
    res['labels'] = np.concatenate(labels)
    if mod_v:
        res['modify_video'], res['modify_audio'] = np.concatenate(mod_v), np.concatenate(mod_a)
    res['losses'] = {k: v / max(n, 1) for k, v in loss_sums.items()}
    return res


def evaluate_loader(model, loader, device, criterion=None):
    r = collect_outputs(model, loader, device, criterion)
    m = detection_metrics(r['labels'], r['det_probs'], r['C_i'], r['A'], r['til'])
    if 'modify_video' in r:
        m.update(attribution_metrics(r['labels'], r['det_probs'], r['A'],
                                     r['modify_video'], r['modify_audio']))
    if r['losses']:
        m['loss'] = r['losses']['total']
        m['det_loss'] = r['losses']['det']
        m['pred_loss'] = r['losses']['L_pred']
    return m
