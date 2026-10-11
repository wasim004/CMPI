"""
CMPI Cross-Dataset Evaluation
================================
Wasim Ahmad, BIT

Loads the trained CMPI checkpoint (trained on FakeAVCeleb) and evaluates
it on DF-TIMIT, LAV-DF, and DFDM without any retraining.

This directly validates the causal invariance claim: if the predictive
coupling mechanism generalises across datasets, AUROC should remain high
even on unseen manipulation types and recording conditions.

Usage
-----
python eval_crossdataset.py \
    --checkpoint  results/training_v2/best_model.pt \
    --data_root   data/crossdataset \
    --output_dir  results/crossdataset

# Evaluate one dataset only:
python eval_crossdataset.py --checkpoint ... --data_root ... --dataset dftimit
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_auc_score, roc_curve,
                             balanced_accuracy_score,
                             confusion_matrix)
from scipy.stats import mannwhitneyu

# ── Resolve AMDD / CMPI paths ────────────────────────────────────────────
AMDD_ROOT  = Path(__file__).resolve().parent.parent
CMPI_ROOT = Path(__file__).resolve().parent.parent

for _p in [AMDD_ROOT, AMDD_ROOT / "models", AMDD_ROOT / "data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from encoders import VisualEncoder, AudioEncoder
except ModuleNotFoundError:
    from models.encoders import VisualEncoder, AudioEncoder

from dataset import DeepfakeDataset


# ══════════════════════════════════════════════════════════════════════════
# Model (copy from train_ccmpm.py — must be identical)
# ══════════════════════════════════════════════════════════════════════════

class CrossModalPredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
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


class DetectionHead(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),    nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class CMPIModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim, dropout=dropout)
        self.audio_encoder  = AudioEncoder(embed_dim=embed_dim,  dropout=dropout)
        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.register_buffer('ci_running_mean', torch.tensor(1.0))
        self.register_buffer('ci_running_std',  torch.tensor(1.0))
        self.register_buffer('ci_n_updates',    torch.tensor(0))
        self.detection_head = DetectionHead(embed_dim * 2 + 1, dropout)

    def forward(self, frames, audio):
        z_v    = self.visual_encoder(frames)['video_embedding']
        z_a    = self.audio_encoder(audio)['audio_embedding']
        pred_v = self.T_av(z_a)
        pred_a = self.T_va(z_v)
        err_av = torch.norm(pred_v - z_v, dim=1)
        err_va = torch.norm(pred_a - z_a, dim=1)
        C_i    = err_av + err_va
        C_i_norm = ((C_i - self.ci_running_mean) /
                    self.ci_running_std.clamp(min=1e-6)).unsqueeze(1)
        det_input  = torch.cat([z_v, z_a, C_i_norm], dim=1)
        det_logits = self.detection_head(det_input)
        det_probs  = torch.sigmoid(det_logits)
        return {
            'z_v': z_v, 'z_a': z_a,
            'pred_v': pred_v, 'pred_a': pred_a,
            'C_i': C_i, 'err_av': err_av, 'err_va': err_va,
            'det_logits': det_logits, 'det_probs': det_probs,
        }


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def calibrate_ci_stats(model, loader, device, n_real_samples=100):
    """
    Update ci_running_mean/std from real samples of the target domain.
    Corrects for absolute C_i scale differences across datasets.
    Uses real samples only — no label leakage.
    """
    model.eval()
    ci_values = []
    collected = 0
    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label']
        real_mask = (labels == 0)
        if real_mask.sum() == 0:
            continue
        z_v    = model.visual_encoder(frames[real_mask])['video_embedding']
        z_a    = model.audio_encoder(audio[real_mask])['audio_embedding']
        pred_v = model.T_av(z_a)
        pred_a = model.T_va(z_v)
        ci     = torch.norm(pred_v - z_v, dim=1) + torch.norm(pred_a - z_a, dim=1)
        ci_values.append(ci.cpu())
        collected += real_mask.sum().item()
        if collected >= n_real_samples:
            break
    if not ci_values:
        print("  Warning: no real samples found for calibration — stats unchanged")
        return
    ci_all   = torch.cat(ci_values)
    new_mean = ci_all.mean()
    new_std  = ci_all.std().clamp(min=1e-6)
    model.ci_running_mean.copy_(new_mean)
    model.ci_running_std.copy_(new_std)
    print(f"  C_i calibrated: mean={new_mean:.2f}  std={new_std:.2f} "
          f"(from {len(ci_all)} real samples)")


@torch.no_grad()
def evaluate_dataset(model, loader, device, dataset_name):
    model.eval()
    all_probs, all_ci, all_err_av, all_err_va, all_labels = [], [], [], [], []

    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label']

        out = model(frames, audio)
        all_probs.append(out['det_probs'].cpu().numpy())
        all_ci.append(out['C_i'].cpu().numpy())
        all_err_av.append(out['err_av'].cpu().numpy())
        all_err_va.append(out['err_va'].cpu().numpy())
        all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    ci     = np.concatenate(all_ci)
    err_av = np.concatenate(all_err_av)
    err_va = np.concatenate(all_err_va)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    n_real = (labels == 0).sum()
    n_fake = (labels == 1).sum()

    auroc    = roc_auc_score(labels, probs)   if len(np.unique(labels)) > 1 else float('nan')
    ci_auroc = roc_auc_score(labels, ci)      if len(np.unique(labels)) > 1 else float('nan')
    bal_acc  = balanced_accuracy_score(labels, preds)

    # Per-arm stats
    ci_real = ci[labels == 0]
    ci_fake = ci[labels == 1]
    av_real = err_av[labels == 0]
    av_fake = err_av[labels == 1]
    va_real = err_va[labels == 0]
    va_fake = err_va[labels == 1]

    mw_stat, mw_p = (mannwhitneyu(ci_fake, ci_real, alternative='greater')
                     if n_real > 0 and n_fake > 0
                     else (float('nan'), float('nan')))

    results = {
        'dataset':        dataset_name,
        'n_real':         int(n_real),
        'n_fake':         int(n_fake),
        'auroc':          float(auroc),
        'ci_auroc':       float(ci_auroc),
        'bal_acc':        float(bal_acc),
        'ci_mean_real':   float(ci_real.mean()) if n_real > 0 else float('nan'),
        'ci_mean_fake':   float(ci_fake.mean()) if n_fake > 0 else float('nan'),
        'ci_std_real':    float(ci_real.std())  if n_real > 0 else float('nan'),
        'ci_std_fake':    float(ci_fake.std())  if n_fake > 0 else float('nan'),
        'separation':     float(ci_fake.mean() / (ci_real.mean() + 1e-8))
                          if n_real > 0 and n_fake > 0 else float('nan'),
        # Per-arm breakdown
        'err_av_real':    float(av_real.mean()) if n_real > 0 else float('nan'),
        'err_av_fake':    float(av_fake.mean()) if n_fake > 0 else float('nan'),
        'err_va_real':    float(va_real.mean()) if n_real > 0 else float('nan'),
        'err_va_fake':    float(va_fake.mean()) if n_fake > 0 else float('nan'),
        'mannwhitney_p':  float(mw_p),
    }

    print(f"\n{'─'*60}")
    print(f"  {dataset_name.upper()}")
    print(f"{'─'*60}")
    print(f"  Samples        : {n_real} real / {n_fake} fake")
    print(f"  AUROC          : {auroc:.4f}")
    print(f"  C_i AUROC      : {ci_auroc:.4f}")
    print(f"  Balanced Acc   : {bal_acc:.4f}")
    print(f"  C_i real       : {ci_real.mean():.2f} ± {ci_real.std():.2f}")
    print(f"  C_i fake       : {ci_fake.mean():.2f} ± {ci_fake.std():.2f}")
    print(f"  Separation     : {results['separation']:.2f}×")
    print(f"  err_av (a→v)   : real={av_real.mean():.2f}  fake={av_fake.mean():.2f}")
    print(f"  err_va (v→a)   : real={va_real.mean():.2f}  fake={va_fake.mean():.2f}")
    print(f"  Mann-Whitney p : {mw_p:.2e}")

    return results, probs, ci, labels


def plot_results(all_results, all_probs, all_ci, all_labels,
                 output_dir, in_dist_auroc=None):
    """Generate cross-dataset summary plots."""
    n_ds = len(all_results)
    colors = ['steelblue', 'darkorange', 'seagreen', 'crimson']

    # ── Figure 1: ROC curves ──────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, (res, probs, ci, labels) in enumerate(
            zip(all_results, all_probs, all_ci, all_labels)):
        if np.isnan(res['auroc']):
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{res['dataset'].upper()} (AUC={res['auroc']:.4f})")

    if in_dist_auroc is not None:
        ax.axhline(in_dist_auroc, color='gray', ls=':', lw=1.5,
                   label=f"FakeAVCeleb in-dist ({in_dist_auroc:.4f})")

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Cross-Dataset ROC — CMPI (trained on FakeAVCeleb)')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'crossdataset_roc.png'), dpi=150)
    plt.close(fig)

    # ── Figure 2: C_i distributions per dataset ───────────────────────────
    fig, axes = plt.subplots(1, n_ds, figsize=(6 * n_ds, 5))
    if n_ds == 1:
        axes = [axes]

    for ax, res, ci, labels in zip(axes, all_results, all_ci, all_labels):
        real_c = ci[labels == 0]
        fake_c = ci[labels == 1]
        bins   = np.linspace(ci.min(), ci.max(), 50)
        if len(real_c):
            ax.hist(real_c, bins=bins, alpha=0.65, color='royalblue',
                    density=True, label=f'Real μ={real_c.mean():.1f}')
        if len(fake_c):
            ax.hist(fake_c, bins=bins, alpha=0.65, color='tomato',
                    density=True, label=f'Fake μ={fake_c.mean():.1f}')
        auroc_str = f"AUC={res['ci_auroc']:.3f}" if not np.isnan(res['ci_auroc']) else ""
        ax.set_title(f"{res['dataset'].upper()} — C_i {auroc_str}")
        ax.set_xlabel('Predictive Inconsistency C_i')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle('C_i Distributions — Cross-Dataset (CMPI)', fontsize=13)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'crossdataset_ci_dist.png'), dpi=150)
    plt.close(fig)

    # ── Figure 3: Per-arm bar chart ───────────────────────────────────────
    # Shows err_av and err_va separately for real vs fake per dataset
    # This is the key figure for the causal theory analysis in the paper
    ds_names = [r['dataset'].upper() for r in all_results]
    x        = np.arange(len(ds_names))
    width    = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*width, [r['err_av_real'] for r in all_results],
           width, label='err_av real (a→v)', color='royalblue', alpha=0.8)
    ax.bar(x - 0.5*width, [r['err_av_fake'] for r in all_results],
           width, label='err_av fake (a→v)', color='royalblue', alpha=0.4,
           hatch='//')
    ax.bar(x + 0.5*width, [r['err_va_real'] for r in all_results],
           width, label='err_va real (v→a)', color='darkorange', alpha=0.8)
    ax.bar(x + 1.5*width, [r['err_va_fake'] for r in all_results],
           width, label='err_va fake (v→a)', color='darkorange', alpha=0.4,
           hatch='//')

    ax.set_xticks(x)
    ax.set_xticklabels(ds_names)
    ax.set_ylabel('Mean Prediction Error')
    ax.set_title('Per-Arm Prediction Error: Real vs Fake Across Datasets\n'
                 '(DF-TIMIT: visual-only fake; LAV-DF: audio+video fake)')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_dir / 'crossdataset_perarm.png'), dpi=150)
    plt.close(fig)

    print(f"\n  Saved: crossdataset_roc.png")
    print(f"  Saved: crossdataset_ci_dist.png")
    print(f"  Saved: crossdataset_perarm.png")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='CMPI Cross-Dataset Evaluation')
    p.add_argument('--checkpoint', required=True,
                   help='Path to trained CMPI best_model.pt')
    p.add_argument('--data_root',
                   default='data/crossdataset',
                   help='Root dir containing dftimit/ lavdf/ dfdm/ subdirs')
    p.add_argument('--output_dir',
                   default='results/crossdataset')
    p.add_argument('--dataset', default='all',
                   choices=['all', 'dftimit', 'lavdf', 'dfdm'])
    p.add_argument('--batch_size',  type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--embed_dim',   type=int, default=512)
    p.add_argument('--hidden_dim',  type=int, default=1024)
    p.add_argument('--in_dist_auroc', type=float, default=0.9982,
                   help='FakeAVCeleb test AUROC for reference line in plots')
    p.add_argument('--device', default='auto')
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  CMPI Cross-Dataset Evaluation")
    print(f"  Device    : {device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'═'*60}")

    # ── Load model ────────────────────────────────────────────────────────
    model = CMPIModel(embed_dim=args.embed_dim,
                       hidden_dim=args.hidden_dim).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device,
                       weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"\n  Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
          f"val AUROC={ckpt.get('best_val_auroc', '?'):.4f})")

    # ── Identify datasets to evaluate ─────────────────────────────────────
    data_root = Path(args.data_root)
    datasets  = (['dftimit', 'lavdf', 'dfdm'] if args.dataset == 'all'
                 else [args.dataset])

    all_results, all_probs_list, all_ci_list, all_labels_list = [], [], [], []

    for ds_name in datasets:
        ds_dir = data_root / ds_name
        meta_csv = ds_dir / 'metadata.csv'
        preproc_dir = ds_dir / 'preprocessed'

        if not meta_csv.exists():
            print(f"\n[{ds_name.upper()}] metadata.csv not found at {meta_csv}")
            print(f"  → Run preprocess_crossdataset.py --dataset {ds_name} first")
            continue

        print(f"\n[{ds_name.upper()}] Loading dataset from {ds_dir}")
        try:
            dataset = DeepfakeDataset(
                metadata_csv=str(meta_csv),
                split='test',
                preprocessed_dir=str(preproc_dir),
            )
        except Exception as e:
            print(f"  Failed to load dataset: {e}")
            continue

        if len(dataset) == 0:
            print(f"  Dataset is empty, skipping.")
            continue

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        )

        print(f"\n[{ds_name.upper()}] Calibrating C_i stats from real samples...")
        calibrate_ci_stats(model, loader, device, n_real_samples=100)

        results, probs, ci, labels = evaluate_dataset(
            model, loader, device, ds_name
        )
        all_results.append(results)
        all_probs_list.append(probs)
        all_ci_list.append(ci)
        all_labels_list.append(labels)

        # Save per-dataset arrays
        np.save(str(output_dir / f'{ds_name}_probs.npy'),  probs)
        np.save(str(output_dir / f'{ds_name}_ci.npy'),     ci)
        np.save(str(output_dir / f'{ds_name}_labels.npy'), labels)

    if not all_results:
        print("\nNo datasets evaluated. Run preprocessor first.")
        return

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n\n{'═'*60}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Dataset':<12} {'AUROC':>8} {'C_i AUC':>9} "
          f"{'BalAcc':>8} {'Sep':>7} {'p-val':>12}")
    print(f"  {'-'*60}")
    for r in all_results:
        print(f"  {r['dataset'].upper():<12} "
              f"{r['auroc']:>8.4f} "
              f"{r['ci_auroc']:>9.4f} "
              f"{r['bal_acc']:>8.4f} "
              f"{r['separation']:>7.2f}× "
              f"{r['mannwhitney_p']:>12.2e}")
    print(f"  {'FakeAVCeleb':<12} {args.in_dist_auroc:>8.4f}  "
          f"{'(in-dist reference)':}")
    print(f"{'═'*60}")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[Plots]")
    plot_results(all_results, all_probs_list, all_ci_list, all_labels_list,
                 output_dir, in_dist_auroc=args.in_dist_auroc)

    # ── Save JSON ─────────────────────────────────────────────────────────
    summary = {
        'checkpoint':     args.checkpoint,
        'in_dist_auroc':  args.in_dist_auroc,
        'results':        all_results,
    }
    with open(str(output_dir / 'crossdataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Done] All results in: {output_dir}")


if __name__ == '__main__':
    main()
