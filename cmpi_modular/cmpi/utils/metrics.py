import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


def safe_auroc(y, s):
    return float('nan') if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def detection_metrics(labels, probs, ci, asym, til, threshold=0.5):
    real, fake = labels == 0, labels == 1
    return {
        'n_real': int(real.sum()), 'n_fake': int(fake.sum()),
        'auroc': safe_auroc(labels, probs),
        'bal_acc': float(balanced_accuracy_score(labels, (probs >= threshold).astype(int))),
        'ci_auroc': safe_auroc(labels, ci),
        'til_auroc': safe_auroc(labels, til),
        'ci_mean_real': float(ci[real].mean()), 'ci_mean_fake': float(ci[fake].mean()),
        'a_mean_real': float(asym[real].mean()), 'a_mean_fake': float(asym[fake].mean()),
    }


def attribution_metrics(labels, probs, asym, modify_video, modify_audio):
    """Per-manipulation-type detection AUROC and mean asymmetry (LAV-DF)."""
    res = {}
    fake = labels == 1
    groups = {'video_fake': fake & (modify_video == 1) & (modify_audio == 0),
              'audio_fake': fake & (modify_video == 0) & (modify_audio == 1),
              'both_fake': fake & (modify_video == 1) & (modify_audio == 1)}
    for name, mask in groups.items():
        if mask.sum() == 0:
            continue
        idx = np.where((labels == 0) | mask)[0]
        res[f'auroc_det_{name}'] = safe_auroc(labels[idx], probs[idx])
        res[f'a_mean_{name}'] = float(asym[mask].mean())
    return res
