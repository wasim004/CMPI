"""
CMPI+DIAM — Further Experiments for CVIU Revision
====================================================
Wasim Ahmad, BIT — August 2026

Implements the two pending experiments identified in the TIFS rejection
progress report that are runnable with existing checkpoints/data:

  Exp 2 — Test-time score calibration. Re-estimate the C_i / A / TIL
          normalisation statistics from a small *unlabeled* held-out
          slice of the target domain (LAV-DF, DFDM), then evaluate the
          (frozen) detection head on the remaining held-out samples.
          Compared against the FakeAVCeleb-calibrated baseline on the
          identical held-out slice.

  Exp 4 — Modality-attribution accuracy. Using only the *sign* of the
          raw (unnormalised) asymmetry A = eps_f - eps_b, classify each
          single-modality LAV-DF fake as video-fake (A>0 expected) or
          audio-fake (A<0 expected), and report accuracy + AUROC of A
          as an attribution score.

Both experiments reuse the CMPIDIAMModel / CrossDataset classes and
encoder forward pass from eval_crossdataset_diam.py — a single encoder
forward pass per dataset is cached, then both experiments are computed
from the cached raw signals (z_v, z_a, C_i, A, til, labels, mod flags)
without re-running the encoders.

Usage
-----
python exp_further_experiments.py \\
    --checkpoint  results/diam/best_model.pt \\
    --data_root   data/crossdataset \\
    --output_dir  results/further_experiments
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crossdataset_diam import CMPIDIAMModel, CrossDataset


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


@torch.no_grad()
def cache_raw_signals(model, loader, device):
    """Single encoder forward pass; caches everything needed for both experiments."""
    zv_l, za_l, ci_l, a_l, til_l, label_l, mv_l, ma_l = [], [], [], [], [], [], [], []

    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)

        vis_out = model.visual_encoder(frames)
        z_v     = vis_out['video_embedding']
        h_t     = vis_out['frame_features']
        z_a     = model.audio_encoder(audio)['audio_embedding']
        pred_v  = model.T_av(z_a)
        pred_a  = model.T_va(z_v)
        eps_f   = torch.norm(pred_v - z_v, dim=1)
        eps_b   = torch.norm(pred_a - z_a, dim=1)
        C_i     = eps_f + eps_b
        A       = eps_f - eps_b
        frame_err = torch.norm(pred_v.unsqueeze(1) - h_t, dim=2)
        til, _  = model.til_scorer(frame_err)

        zv_l.append(z_v.cpu().numpy())
        za_l.append(z_a.cpu().numpy())
        ci_l.append(C_i.cpu().numpy())
        a_l.append(A.cpu().numpy())
        til_l.append(til.cpu().numpy())
        label_l.append(batch['label'].numpy())
        if 'modify_video' in batch:
            mv_l.append(np.array([int(x) for x in batch['modify_video']]))
            ma_l.append(np.array([int(x) for x in batch['modify_audio']]))

    out = dict(
        z_v=np.concatenate(zv_l), z_a=np.concatenate(za_l),
        ci=np.concatenate(ci_l), a=np.concatenate(a_l), til=np.concatenate(til_l),
        labels=np.concatenate(label_l),
    )
    if mv_l:
        out['modify_video'] = np.concatenate(mv_l)
        out['modify_audio'] = np.concatenate(ma_l)
    return out


def detection_head_forward(model, z_v, z_a, ci, a, til, stats, device):
    """Recompute det_probs given arbitrary (mean, std) normalisation stats."""
    ci_m, ci_s   = stats['ci']
    a_m,  a_s    = stats['a']
    til_m, til_s = stats['til']

    ci_n  = (torch.tensor(ci,  dtype=torch.float32) - ci_m)  / max(ci_s,  1e-6)
    a_n   = (torch.tensor(a,   dtype=torch.float32) - a_m)   / max(a_s,   1e-6)
    til_n = (torch.tensor(til, dtype=torch.float32) - til_m) / max(til_s, 1e-6)

    det_in = torch.cat([
        torch.tensor(z_v, dtype=torch.float32),
        torch.tensor(z_a, dtype=torch.float32),
        ci_n.unsqueeze(1), a_n.unsqueeze(1), til_n.unsqueeze(1),
    ], dim=1).to(device)

    with torch.no_grad():
        logits = model.detection_head(det_in).squeeze(-1)
    return torch.sigmoid(logits).cpu().numpy()


def exp2_test_time_calibration(model, cache, device, calib_frac, seed=42):
    """Re-estimate C_i/A/TIL normalisation stats from an unlabeled calib slice."""
    n = len(cache['labels'])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_calib = max(20, int(round(n * calib_frac)))
    calib_idx, eval_idx = idx[:n_calib], idx[n_calib:]

    # Baseline (FakeAVCeleb-fitted) EMA stats, as stored in the checkpoint buffers.
    baseline_stats = {
        'ci':  (model.ci_mean.item(),   model.ci_std.item()),
        'a':   (model.asym_mean.item(), model.asym_std.item()),
        'til': (model.til_mean.item(),  model.til_std.item()),
    }

    # Recalibrated stats — computed ONLY from the unlabeled calib slice of the
    # target domain (raw C_i / A / TIL values; labels are never used here).
    recal_stats = {
        'ci':  (float(cache['ci'][calib_idx].mean()),  float(cache['ci'][calib_idx].std())),
        'a':   (float(cache['a'][calib_idx].mean()),   float(cache['a'][calib_idx].std())),
        'til': (float(cache['til'][calib_idx].mean()), float(cache['til'][calib_idx].std())),
    }

    def eval_on(stats):
        probs = detection_head_forward(
            model, cache['z_v'][eval_idx], cache['z_a'][eval_idx],
            cache['ci'][eval_idx], cache['a'][eval_idx], cache['til'][eval_idx],
            stats, device)
        y = cache['labels'][eval_idx]
        return {
            'auroc_det': safe_auroc(y, probs),
            'bal_acc':   float(balanced_accuracy_score(y, (probs >= 0.5).astype(int))),
        }

    return {
        'n_calib': int(len(calib_idx)), 'n_eval': int(len(eval_idx)),
        'baseline_stats': baseline_stats, 'recalibrated_stats': recal_stats,
        'baseline':     eval_on(baseline_stats),
        'recalibrated': eval_on(recal_stats),
        # raw C_i / A AUROC on the eval slice is unaffected by any affine
        # renormalisation — reported once for reference.
        'auroc_ci_raw_eval':  safe_auroc(cache['labels'][eval_idx], cache['ci'][eval_idx]),
        'auroc_a_raw_eval':   safe_auroc(cache['labels'][eval_idx], cache['a'][eval_idx]),
    }


def exp4_modality_attribution(cache):
    """Does sign(raw A) identify which modality was manipulated?"""
    if 'modify_video' not in cache:
        return None
    mv, ma = cache['modify_video'].astype(bool), cache['modify_audio'].astype(bool)
    fake = cache['labels'] == 1
    a = cache['a']

    video_only = fake & mv & (~ma)   # expect A > 0 (eps_f >> eps_b)
    audio_only = fake & (~mv) & ma   # expect A < 0 (eps_b >> eps_f)

    mask = video_only | audio_only
    if mask.sum() == 0:
        return {'note': 'no single-modality fake samples found'}

    y_true = video_only[mask].astype(int)     # 1 = video-fake, 0 = audio-fake
    a_sub  = a[mask]
    y_pred = (a_sub > 0).astype(int)

    return {
        'n_video_only': int(video_only.sum()),
        'n_audio_only': int(audio_only.sum()),
        'attribution_accuracy': float(accuracy_score(y_true, y_pred)),
        'attribution_auroc':    safe_auroc(y_true, a_sub),
        'a_mean_video_fake': float(a[video_only].mean()),
        'a_mean_audio_fake': float(a[audio_only].mean()),
        'a_std_video_fake':  float(a[video_only].std()),
        'a_std_audio_fake':  float(a[audio_only].std()),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root',  required=True)
    p.add_argument('--output_dir', default='results/further_experiments')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--datasets', nargs='+', default=['lavdf', 'dfdm'])
    p.add_argument('--calib_frac', type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n  CMPI+DIAM Further Experiments (Exp 2: test-time calibration, "
          f"Exp 4: modality attribution)\n  Device: {device}\n{'='*70}\n")

    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"best val AUROC={ckpt.get('best_val_auroc','?')})\n")

    all_results = {}
    data_root = Path(args.data_root)

    for ds_name in args.datasets:
        ds_dir = data_root / ds_name
        meta, prep = ds_dir / 'metadata.csv', ds_dir / 'preprocessed'
        if not meta.exists() or not prep.exists():
            print(f"  Skipping {ds_name} — data missing")
            continue

        print(f"  --- {ds_name.upper()} ---")
        ds = CrossDataset(str(meta), str(prep))
        loader = DataLoader(ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        cache = cache_raw_signals(model, loader, device)

        exp2 = exp2_test_time_calibration(model, cache, device, args.calib_frac)
        exp4 = exp4_modality_attribution(cache)

        print(f"  [Exp2] baseline AUROC={exp2['baseline']['auroc_det']:.4f} "
              f"-> recalibrated AUROC={exp2['recalibrated']['auroc_det']:.4f} "
              f"(n_calib={exp2['n_calib']}, n_eval={exp2['n_eval']})")
        if exp4 and 'attribution_accuracy' in exp4:
            print(f"  [Exp4] modality-attribution accuracy={exp4['attribution_accuracy']:.4f}, "
                  f"AUROC={exp4['attribution_auroc']:.4f} "
                  f"(video-only n={exp4['n_video_only']}, audio-only n={exp4['n_audio_only']})")

        all_results[ds_name] = {'exp2_test_time_calibration': exp2,
                                 'exp4_modality_attribution': exp4}
        print()

    out_file = out_dir / 'further_experiments_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {out_file}\n")


if __name__ == '__main__':
    main()
