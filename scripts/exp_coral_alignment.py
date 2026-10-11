"""
CMPI+DIAM — CORAL Domain Alignment for Cross-Dataset Generalisation
======================================================================
Wasim Ahmad, BIT — August 2026

Motivated by the Exp2 finding (test-time renormalisation of C_i/A/TIL
does NOT recover LAV-DF performance): the bottleneck is the encoder
embeddings (z_v, z_a) themselves, not the scalar forensic signals.

This script tests CORrelation ALignment (CORAL, Sun & Saenko 2016):
align the second-order statistics (covariance) of the target-domain
embeddings to the source (FakeAVCeleb) domain, using ONLY unlabeled
target-domain data, then re-run T_av/T_va/TIL/detection head on the
aligned embeddings.

Procedure per target dataset:
  1. Cache raw z_v, z_a, per-frame h_t on an unlabeled calib split.
  2. Fit a CORAL whitening+recoloring affine map (per modality) from
     FakeAVCeleb train-set embeddings (source) and the calib split
     (target), using shrinkage-regularised covariance estimates.
  3. Apply the map to the held-out eval split's z_v, z_a, h_t.
  4. Recompute eps_f, eps_b, C_i, A, TIL, det_probs from the ALIGNED
     features via the frozen T_av/T_va/til_scorer/detection_head.
  5. Compare against the unaligned baseline on the identical eval
     split.

Usage
-----
python exp_coral_alignment.py \\
    --checkpoint       results/diam/best_model.pt \\
    --metadata_csv     ../data/metadata.csv \\
    --preprocessed_dir ../data/preprocessed \\
    --data_root        data/crossdataset \\
    --output_dir       results/further_experiments
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crossdataset_diam import CMPIDIAMModel, CrossDataset

ROOT = Path(__file__).resolve().parent.parent
AMDD_DATA = ROOT / "data"
if str(AMDD_DATA) not in sys.path:
    sys.path.insert(0, str(AMDD_DATA))
from dataset import DeepfakeDataset


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def sym_matrix_power(C, power, eps_floor=1e-6):
    """Matrix power of a symmetric PSD matrix via eigendecomposition."""
    w, V = np.linalg.eigh(C)
    w = np.clip(w, eps_floor, None)
    return (V * (w ** power)) @ V.T


def fit_coral(source_feats, target_feats, shrinkage=0.3, eps=1e-2):
    """
    Returns an affine map (A, b) such that:
        aligned = target_feats @ A + b
    moves target covariance/mean toward source covariance/mean.
    `shrinkage` interpolates each covariance toward a scaled identity
    (Ledoit-Wolf-style) for numerical stability with few target samples.
    """
    mu_s, mu_t = source_feats.mean(0), target_feats.mean(0)
    Xs, Xt = source_feats - mu_s, target_feats - mu_t
    D = Xs.shape[1]

    def shrunk_cov(X):
        n = max(len(X) - 1, 1)
        C = (X.T @ X) / n
        avg_var = np.trace(C) / D
        return (1 - shrinkage) * C + shrinkage * avg_var * np.eye(D) + eps * np.eye(D)

    Cs = shrunk_cov(Xs)
    Ct = shrunk_cov(Xt)

    Ct_inv_sqrt = sym_matrix_power(Ct, -0.5)
    Cs_sqrt = sym_matrix_power(Cs, 0.5)

    A = Ct_inv_sqrt @ Cs_sqrt              # [D, D], applied as X @ A
    b = mu_s - mu_t @ A                    # so that (Xt) @ A + b centers on mu_s
    return A.astype(np.float32), b.astype(np.float32)


def apply_affine(X, A, b):
    return X @ A + b


@torch.no_grad()
def cache_source_embeddings(model, loader, device, max_samples=None):
    zv_l, za_l = [], []
    n = 0
    for batch in loader:
        frames = batch['frames'].to(device)
        audio = batch['audio'].to(device)
        z_v = model.visual_encoder(frames)['video_embedding']
        z_a = model.audio_encoder(audio)['audio_embedding']
        zv_l.append(z_v.cpu().numpy())
        za_l.append(z_a.cpu().numpy())
        n += len(z_v)
        if max_samples and n >= max_samples:
            break
    return np.concatenate(zv_l)[:max_samples], np.concatenate(za_l)[:max_samples]


@torch.no_grad()
def cache_target_full(model, loader, device):
    """Cache everything needed to recompute forward pass post-alignment."""
    zv_l, za_l, ht_l, label_l, mv_l, ma_l = [], [], [], [], [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        audio = batch['audio'].to(device)
        vis_out = model.visual_encoder(frames)
        z_v, h_t = vis_out['video_embedding'], vis_out['frame_features']
        z_a = model.audio_encoder(audio)['audio_embedding']
        zv_l.append(z_v.cpu().numpy())
        za_l.append(z_a.cpu().numpy())
        ht_l.append(h_t.cpu().numpy())
        label_l.append(batch['label'].numpy())
        if 'modify_video' in batch:
            mv_l.append(np.array([int(x) for x in batch['modify_video']]))
            ma_l.append(np.array([int(x) for x in batch['modify_audio']]))
    out = dict(z_v=np.concatenate(zv_l), z_a=np.concatenate(za_l),
               h_t=np.concatenate(ht_l), labels=np.concatenate(label_l))
    if mv_l:
        out['modify_video'] = np.concatenate(mv_l)
        out['modify_audio'] = np.concatenate(ma_l)
    return out


def forward_from_embeddings(model, z_v, z_a, h_t, stats, device):
    """Recompute C_i/A/TIL/det_probs from (possibly aligned) embeddings."""
    z_v_t = torch.tensor(z_v, dtype=torch.float32, device=device)
    z_a_t = torch.tensor(z_a, dtype=torch.float32, device=device)
    h_t_t = torch.tensor(h_t, dtype=torch.float32, device=device)

    with torch.no_grad():
        pred_v = model.T_av(z_a_t)
        pred_a = model.T_va(z_v_t)
        eps_f = torch.norm(pred_v - z_v_t, dim=1)
        eps_b = torch.norm(pred_a - z_a_t, dim=1)
        C_i = eps_f + eps_b
        A = eps_f - eps_b
        frame_err = torch.norm(pred_v.unsqueeze(1) - h_t_t, dim=2)
        til, _ = model.til_scorer(frame_err)

        ci_m, ci_s = stats['ci']
        a_m, a_s = stats['a']
        til_m, til_s = stats['til']
        ci_n = ((C_i - ci_m) / max(ci_s, 1e-6)).unsqueeze(1)
        a_n = ((A - a_m) / max(a_s, 1e-6)).unsqueeze(1)
        til_n = ((til - til_m) / max(til_s, 1e-6)).unsqueeze(1)

        det_in = torch.cat([z_v_t, z_a_t, ci_n, a_n, til_n], dim=1)
        probs = torch.sigmoid(model.detection_head(det_in).squeeze(-1))

    return C_i.cpu().numpy(), A.cpu().numpy(), til.cpu().numpy(), probs.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--output_dir', default='results/further_experiments')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--datasets', nargs='+', default=['lavdf', 'dfdm'])
    p.add_argument('--calib_frac', type=float, default=0.2)
    p.add_argument('--shrinkage', type=float, default=0.3)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n  CORAL Domain Alignment Experiment\n  Device: {device}\n{'='*70}\n")

    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    baseline_stats = {
        'ci':  (model.ci_mean.item(),   model.ci_std.item()),
        'a':   (model.asym_mean.item(), model.asym_std.item()),
        'til': (model.til_mean.item(),  model.til_std.item()),
    }

    print("  Caching FakeAVCeleb (source) embeddings...")
    src_ds = DeepfakeDataset(args.metadata_csv, 'train', args.preprocessed_dir)
    src_loader = DataLoader(src_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    src_zv, src_za = cache_source_embeddings(model, src_loader, device)
    print(f"  Source: {src_zv.shape[0]} samples\n")

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
        cache = cache_target_full(model, loader, device)

        n = len(cache['labels'])
        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        n_calib = max(20, int(round(n * args.calib_frac)))
        calib_idx, eval_idx = idx[:n_calib], idx[n_calib:]

        # Fit CORAL maps on UNLABELED calib slice only
        A_v, b_v = fit_coral(src_zv, cache['z_v'][calib_idx], shrinkage=args.shrinkage)
        A_a, b_a = fit_coral(src_za, cache['z_a'][calib_idx], shrinkage=args.shrinkage)

        # Apply to eval slice (z_v, z_a, and per-frame h_t via the same z_v map)
        zv_eval = cache['z_v'][eval_idx]
        za_eval = cache['z_a'][eval_idx]
        ht_eval = cache['h_t'][eval_idx]
        y_eval = cache['labels'][eval_idx]

        zv_aligned = apply_affine(zv_eval, A_v, b_v)
        za_aligned = apply_affine(za_eval, A_a, b_a)
        T = ht_eval.shape[1]
        ht_aligned = apply_affine(ht_eval.reshape(-1, ht_eval.shape[-1]), A_v, b_v).reshape(ht_eval.shape)

        # Baseline (no alignment) on the SAME eval slice
        _, _, _, probs_base = forward_from_embeddings(
            model, zv_eval, za_eval, ht_eval, baseline_stats, device)
        # CORAL-aligned
        ci_al, a_al, til_al, probs_al = forward_from_embeddings(
            model, zv_aligned, za_aligned, ht_aligned, baseline_stats, device)

        result = {
            'n_calib': int(len(calib_idx)), 'n_eval': int(len(eval_idx)),
            'baseline_auroc': safe_auroc(y_eval, probs_base),
            'baseline_bal_acc': float(balanced_accuracy_score(y_eval, (probs_base >= 0.5).astype(int))),
            'coral_auroc': safe_auroc(y_eval, probs_al),
            'coral_bal_acc': float(balanced_accuracy_score(y_eval, (probs_al >= 0.5).astype(int))),
            'coral_auroc_ci': safe_auroc(y_eval, ci_al),
            'coral_auroc_a':  safe_auroc(y_eval, a_al),
        }
        print(f"  Baseline AUROC={result['baseline_auroc']:.4f} (BalAcc={result['baseline_bal_acc']:.4f})")
        print(f"  CORAL    AUROC={result['coral_auroc']:.4f} (BalAcc={result['coral_bal_acc']:.4f})")
        print(f"  CORAL C_i-AUROC={result['coral_auroc_ci']:.4f}  A-AUROC={result['coral_auroc_a']:.4f}")

        all_results[ds_name] = result
        print()

    out_file = out_dir / 'coral_alignment_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {out_file}\n")


if __name__ == '__main__':
    main()
