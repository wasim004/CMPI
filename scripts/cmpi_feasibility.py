"""
CMPI Feasibility Experiment
=============================
Causal Cross-Modal Predictive Modeling — Wasim Ahmad, BIT

Goal
----
Validate the core hypothesis: real audio-visual pairs have lower bidirectional
predictive inconsistency (C_i) than fake pairs, using frozen AMDD encoders
and lightweight MLP predictors trained on real samples only.

Setup
-----
- Encoders  : loaded from AMDD checkpoint (frozen)
- Predictors: T_{a→v} and T_{v→a}, trained from scratch on REAL samples only
- Signal    : C_i = ||ẑ_v - z̃_v||_2 + ||ẑ_a - z̃_a||_2
- Success   : AUROC(C_i) > 0.70 on FakeAVCeleb test split

Usage
-----
# With AMDD checkpoint (recommended):
python ccmpm_feasibility.py \
    --metadata_csv     AMDD/data/metadata.csv \
    --preprocessed_dir AMDD/data/preprocessed \
    --checkpoint       AMDD/checkpoints/best_model.pt \
    --output_dir       results/feasibility

# Without checkpoint (random encoder weights — baseline only):
python ccmpm_feasibility.py \
    --metadata_csv     AMDD/data/metadata.csv \
    --preprocessed_dir AMDD/data/preprocessed \
    --output_dir       results/feasibility

# Fast rerun (skip embedding extraction if already cached):
python ccmpm_feasibility.py \
    --metadata_csv     AMDD/data/metadata.csv \
    --preprocessed_dir AMDD/data/preprocessed \
    --checkpoint       AMDD/checkpoints/best_model.pt \
    --output_dir       results/feasibility \
    --skip_embedding_extraction
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')  # no display needed
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import mannwhitneyu

# ── Resolve AMDD source root ──────────────────────────────────────────────
AMDD_ROOT     = Path(__file__).resolve().parent.parent / "AMDD"
CMPI_ROOT    = Path(__file__).resolve().parent.parent

# Add AMDD root so `from models.encoders import ...` resolves correctly.
# Also add AMDD/training in case dataset.py lives there, and AMDD/models
# for a flat `from encoders import ...` fallback.
for _p in [
    AMDD_ROOT,
    AMDD_ROOT / "models",
    AMDD_ROOT / "training",
    AMDD_ROOT / "data",
]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Import AMDD components.
# Priority order reflects your confirmed folder layout:
#   AMDD/models/encoders.py   → VisualEncoder, AudioEncoder
#   AMDD/training/dataset.py  → DeepfakeDataset   (try here first)
#   AMDD/dataset.py           → DeepfakeDataset   (fallback)
try:
    from encoders import VisualEncoder, AudioEncoder   # AMDD/models/ on path
    print("✓ Imported VisualEncoder, AudioEncoder from AMDD/models/encoders.py")
except ModuleNotFoundError:
    try:
        from models.encoders import VisualEncoder, AudioEncoder
        print("✓ Imported VisualEncoder, AudioEncoder via models.encoders")
    except ModuleNotFoundError as e:
        print(f"✗ Could not import encoder modules: {e}")
        print("  Expected: AMDD/models/encoders.py")
        sys.exit(1)

try:
    from dataset import DeepfakeDataset                # AMDD/training/ on path
    print("✓ Imported DeepfakeDataset from AMDD/training/dataset.py")
except ModuleNotFoundError:
    try:
        sys.path.insert(0, str(AMDD_ROOT))
        from dataset import DeepfakeDataset            # AMDD root fallback
        print("✓ Imported DeepfakeDataset from AMDD/dataset.py")
    except ModuleNotFoundError as e:
        print(f"✗ Could not import DeepfakeDataset: {e}")
        print("  Expected: AMDD/data/dataset.py")
        print("        or: AMDD/dataset.py")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════
# 1.  Cross-modal predictor MLPs
# ══════════════════════════════════════════════════════════════════════════

class CrossModalPredictor(nn.Module):
    """
    Lightweight MLP that predicts one modality's representation from the other.
    Architecture: Linear → LayerNorm → ReLU → Linear → LayerNorm → ReLU → Linear
    Both predictors share the same class; only input/output dims differ.
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ══════════════════════════════════════════════════════════════════════════
# 2.  Embedding extraction (frozen encoders)
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_embeddings(visual_enc, audio_enc, dataloader, device,
                       split_name=""):
    """
    Forward-pass all samples through frozen encoders.
    Returns numpy arrays for downstream predictor training / evaluation.
    """
    visual_enc.eval()
    audio_enc.eval()

    z_v_list, z_a_list, label_list, uid_list = [], [], [], []
    t0 = time.time()

    for batch_idx, batch in enumerate(dataloader):
        frames = batch['frames'].to(device)   # [B, T, C, H, W]
        audio  = batch['audio'].to(device)    # [B, 128, 128]
        labels = batch['label']               # [B]  (cpu)
        uids   = batch['uid']

        vis_out = visual_enc(frames)
        aud_out = audio_enc(audio)

        z_v_list.append(vis_out['video_embedding'].cpu().numpy())
        z_a_list.append(aud_out['audio_embedding'].cpu().numpy())
        label_list.append(labels.numpy())
        uid_list.extend(uids)

        if (batch_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{split_name}] {batch_idx+1}/{len(dataloader)} "
                  f"batches — {elapsed:.1f}s")

    z_v    = np.concatenate(z_v_list,    axis=0)   # [N, embed_dim]
    z_a    = np.concatenate(z_a_list,    axis=0)   # [N, embed_dim]
    labels = np.concatenate(label_list,  axis=0)   # [N]

    real_n = (labels == 0).sum()
    fake_n = (labels == 1).sum()
    print(f"  [{split_name}] {len(labels)} samples | "
          f"real={real_n}  fake={fake_n}")

    return z_v, z_a, labels, uid_list


# ══════════════════════════════════════════════════════════════════════════
# 3.  Predictor training (real samples only)
# ══════════════════════════════════════════════════════════════════════════

def train_predictors(z_v_train, z_a_train, labels_train,
                     embed_dim, device, args):
    """
    Train T_{a→v} and T_{v→a} on REAL training samples only.
    The predictors learn what 'normal coupling' looks like.
    Fake samples will show higher reconstruction error at test time.
    """
    real_mask = labels_train == 0
    z_v_real  = torch.tensor(z_v_train[real_mask], dtype=torch.float32)
    z_a_real  = torch.tensor(z_a_train[real_mask], dtype=torch.float32)

    n_real = z_v_real.shape[0]
    print(f"\n[Predictor Training] Using {n_real} real samples")
    print(f"  embed_dim={embed_dim}, hidden_dim={args.hidden_dim}, "
          f"epochs={args.pred_epochs}, lr={args.pred_lr}")

    # Audio → Visual predictor
    T_av = CrossModalPredictor(embed_dim, args.hidden_dim, embed_dim).to(device)
    # Visual → Audio predictor
    T_va = CrossModalPredictor(embed_dim, args.hidden_dim, embed_dim).to(device)

    opt_av = torch.optim.AdamW(T_av.parameters(), lr=args.pred_lr,
                                weight_decay=1e-4)
    opt_va = torch.optim.AdamW(T_va.parameters(), lr=args.pred_lr,
                                weight_decay=1e-4)

    sched_av = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_av, T_max=args.pred_epochs)
    sched_va = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt_va, T_max=args.pred_epochs)

    history = {'av_loss': [], 'va_loss': []}

    for epoch in range(1, args.pred_epochs + 1):
        T_av.train(); T_va.train()

        # Mini-batch loop
        perm     = torch.randperm(n_real)
        ep_av, ep_va, n_batches = 0.0, 0.0, 0

        for start in range(0, n_real, args.pred_batch):
            idx   = perm[start: start + args.pred_batch]
            bz_v  = z_v_real[idx].to(device)
            bz_a  = z_a_real[idx].to(device)

            # T_{a→v}
            opt_av.zero_grad()
            pred_v = T_av(bz_a)
            loss_av = F.mse_loss(pred_v, bz_v)
            loss_av.backward()
            nn.utils.clip_grad_norm_(T_av.parameters(), 1.0)
            opt_av.step()

            # T_{v→a}
            opt_va.zero_grad()
            pred_a = T_va(bz_v)
            loss_va = F.mse_loss(pred_a, bz_a)
            loss_va.backward()
            nn.utils.clip_grad_norm_(T_va.parameters(), 1.0)
            opt_va.step()

            ep_av += loss_av.item()
            ep_va += loss_va.item()
            n_batches += 1

        sched_av.step(); sched_va.step()

        avg_av = ep_av / n_batches
        avg_va = ep_va / n_batches
        history['av_loss'].append(avg_av)
        history['va_loss'].append(avg_va)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.pred_epochs} | "
                  f"L_av={avg_av:.5f}  L_va={avg_va:.5f}")

    return T_av, T_va, history


# ══════════════════════════════════════════════════════════════════════════
# 4.  Compute C_i scores
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_consistency_scores(T_av, T_va, z_v, z_a, device,
                                batch_size=256):
    """
    C_i = ||T_{a→v}(z̃_a) - z̃_v||_2  +  ||T_{v→a}(z̃_v) - z̃_a||_2
    Returns numpy array [N].
    """
    T_av.eval(); T_va.eval()
    N = z_v.shape[0]
    scores = []

    for start in range(0, N, batch_size):
        bz_v = torch.tensor(z_v[start:start+batch_size],
                             dtype=torch.float32, device=device)
        bz_a = torch.tensor(z_a[start:start+batch_size],
                             dtype=torch.float32, device=device)

        pred_v = T_av(bz_a)
        pred_a = T_va(bz_v)

        err_av = torch.norm(pred_v - bz_v, dim=1)   # [B]
        err_va = torch.norm(pred_a - bz_a, dim=1)   # [B]
        ci     = (err_av + err_va).cpu().numpy()
        scores.append(ci)

    return np.concatenate(scores)


# ══════════════════════════════════════════════════════════════════════════
# 5.  Evaluation and reporting
# ══════════════════════════════════════════════════════════════════════════

def evaluate(scores, labels, split_name, output_dir):
    """
    Compute AUROC, Mann-Whitney U test, mean/std per class.
    Saves a result dict and returns it.
    """
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]

    # AUROC — higher C_i → more likely fake, so use scores directly
    auroc = roc_auc_score(labels, scores)

    # Mann-Whitney U test: are fake C_i stochastically larger than real C_i?
    stat, pval = mannwhitneyu(fake_scores, real_scores, alternative='greater')

    results = {
        'split':            split_name,
        'n_real':           int((labels == 0).sum()),
        'n_fake':           int((labels == 1).sum()),
        'auroc':            float(auroc),
        'mannwhitney_stat': float(stat),
        'mannwhitney_pval': float(pval),
        'real_ci_mean':     float(real_scores.mean()),
        'real_ci_std':      float(real_scores.std()),
        'fake_ci_mean':     float(fake_scores.mean()),
        'fake_ci_std':      float(fake_scores.std()),
        'separation_ratio': float(fake_scores.mean() / (real_scores.mean() + 1e-8)),
    }

    print(f"\n{'─'*55}")
    print(f"  Results — {split_name}")
    print(f"{'─'*55}")
    print(f"  Samples        : {results['n_real']} real / {results['n_fake']} fake")
    print(f"  AUROC (C_i)    : {auroc:.4f}  {'✓ ABOVE 0.70' if auroc > 0.70 else '✗ below 0.70'}")
    print(f"  C_i real       : {results['real_ci_mean']:.4f} ± {results['real_ci_std']:.4f}")
    print(f"  C_i fake       : {results['fake_ci_mean']:.4f} ± {results['fake_ci_std']:.4f}")
    print(f"  Separation     : fake/real mean ratio = {results['separation_ratio']:.3f}")
    print(f"  Mann-Whitney U : stat={stat:.0f}  p={pval:.2e}  "
          f"{'(significant)' if pval < 0.05 else '(not significant)'}")
    print(f"{'─'*55}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# 6.  Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_all(scores_dict, labels_dict, train_history, output_dir):
    """
    Generate all figures:
    1. Training loss curves (predictor)
    2. C_i histograms — val and test
    3. ROC curves — val and test
    4. Combined summary figure
    """
    output_dir = Path(output_dir)
    fig_paths  = []

    # ── Figure 1: Training loss ──────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(train_history['av_loss']) + 1)

    axes[0].plot(epochs, train_history['av_loss'], color='steelblue', lw=2)
    axes[0].set_title('T_{a→v} MSE Loss', fontsize=12)
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE Loss')
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_history['va_loss'], color='darkorange', lw=2)
    axes[1].set_title('T_{v→a} MSE Loss', fontsize=12)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE Loss')
    axes[1].grid(alpha=0.3)

    fig.suptitle('Cross-Modal Predictor Training (Real Samples Only)', fontsize=13)
    fig.tight_layout()
    p = output_dir / 'fig1_training_loss.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(p)
    print(f"  Saved: {p.name}")

    # ── Figure 2: C_i histograms ─────────────────────────────────────────
    n_splits = len(scores_dict)
    fig, axes = plt.subplots(1, n_splits, figsize=(6 * n_splits, 5))
    if n_splits == 1:
        axes = [axes]

    for ax, (split, scores) in zip(axes, scores_dict.items()):
        labels = labels_dict[split]
        real_s = scores[labels == 0]
        fake_s = scores[labels == 1]

        bins = np.linspace(scores.min(), scores.max(), 50)
        ax.hist(real_s, bins=bins, alpha=0.65, color='royalblue',
                label=f'Real (n={len(real_s)})', density=True)
        ax.hist(fake_s, bins=bins, alpha=0.65, color='tomato',
                label=f'Fake (n={len(fake_s)})', density=True)

        ax.axvline(real_s.mean(), color='royalblue', ls='--', lw=1.5,
                   label=f'Real μ={real_s.mean():.3f}')
        ax.axvline(fake_s.mean(), color='tomato', ls='--', lw=1.5,
                   label=f'Fake μ={fake_s.mean():.3f}')

        ax.set_title(f'C_i Distribution — {split.upper()}', fontsize=12)
        ax.set_xlabel('Predictive Inconsistency Score C_i')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'CMPI Hypothesis: Real samples exhibit lower C_i than fakes',
        fontsize=13
    )
    fig.tight_layout()
    p = output_dir / 'fig2_ci_histograms.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(p)
    print(f"  Saved: {p.name}")

    # ── Figure 3: ROC curves ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'val': 'darkorange', 'test': 'steelblue', 'train': 'seagreen'}

    for split, scores in scores_dict.items():
        labels = labels_dict[split]
        auroc  = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)
        c = colors.get(split, 'gray')
        ax.plot(fpr, tpr, color=c, lw=2,
                label=f'{split.upper()} (AUC={auroc:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5000)')
    ax.axhline(0.90, color='gray', ls=':', lw=1, alpha=0.5)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve — Unsupervised C_i Score (CMPI)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])

    fig.tight_layout()
    p = output_dir / 'fig3_roc_curves.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(p)
    print(f"  Saved: {p.name}")

    # ── Figure 4: Summary panel ──────────────────────────────────────────
    # Box-and-whisker of C_i for real vs fake, per split
    fig, axes = plt.subplots(1, n_splits, figsize=(5 * n_splits, 5))
    if n_splits == 1:
        axes = [axes]

    for ax, (split, scores) in zip(axes, scores_dict.items()):
        labels = labels_dict[split]
        real_s = scores[labels == 0]
        fake_s = scores[labels == 1]

        bp = ax.boxplot(
            [real_s, fake_s],
            labels=['Real', 'Fake'],
            patch_artist=True,
            notch=True,
            medianprops=dict(color='black', lw=2),
        )
        bp['boxes'][0].set_facecolor('royalblue')
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][1].set_facecolor('tomato')
        bp['boxes'][1].set_alpha(0.7)

        auroc = roc_auc_score(labels, scores)
        ax.set_title(f'{split.upper()} | AUROC={auroc:.4f}', fontsize=12)
        ax.set_ylabel('C_i (Predictive Inconsistency Score)')
        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('C_i Distribution: Real vs Fake (Box plots)', fontsize=13)
    fig.tight_layout()
    p = output_dir / 'fig4_boxplots.png'
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    fig_paths.append(p)
    print(f"  Saved: {p.name}")

    return fig_paths


# ══════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description='CMPI Feasibility Experiment'
    )
    # Paths
    p.add_argument('--metadata_csv', required=True,
                   help='Path to AMDD metadata CSV '
                        '(e.g. <path>/AMDD/data/metadata.csv)')
    p.add_argument('--preprocessed_dir', required=True,
                   help='Path to AMDD preprocessed data directory '
                        '(e.g. <path>/AMDD/data/preprocessed)')
    p.add_argument('--checkpoint', default=None,
                   help='Path to AMDD model checkpoint '
                        '(e.g. <path>/AMDD/checkpoints/best_model.pt)')
    p.add_argument('--output_dir',
                   default='results/feasibility',
                   help='Directory for saving results and figures '
                        '(default: CMPI/results/feasibility)')

    # Encoder settings (must match AMDD training)
    p.add_argument('--embed_dim', type=int, default=512,
                   help='Encoder embedding dimension (default: 512)')
    p.add_argument('--enc_batch_size', type=int, default=8,
                   help='Batch size for embedding extraction')
    p.add_argument('--num_workers', type=int, default=4)

    # Predictor settings
    p.add_argument('--hidden_dim', type=int, default=1024,
                   help='Hidden dim for predictor MLPs (default: 1024)')
    p.add_argument('--pred_epochs', type=int, default=50,
                   help='Predictor training epochs (default: 50)')
    p.add_argument('--pred_lr', type=float, default=1e-3,
                   help='Predictor learning rate (default: 1e-3)')
    p.add_argument('--pred_batch', type=int, default=64,
                   help='Predictor training batch size (default: 64)')
    p.add_argument('--dropout', type=float, default=0.1)

    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='auto',
                   help='"auto", "cuda", "cpu", or "cuda:N"')
    p.add_argument('--skip_embedding_extraction', action='store_true',
                   help='Load cached embeddings if already extracted')

    return p.parse_args()


def main():
    args = parse_args()

    # ── Seeding ───────────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\n{'═'*55}")
    print(f"  CMPI Feasibility Experiment")
    print(f"  Device: {device}")
    print(f"{'═'*55}")

    # ── Output directory ──────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    emb_cache  = output_dir / 'embeddings_cache'
    emb_cache.mkdir(exist_ok=True)

    # ── Build encoders ────────────────────────────────────────────────────
    visual_enc = VisualEncoder(embed_dim=args.embed_dim).to(device)
    audio_enc  = AudioEncoder(embed_dim=args.embed_dim).to(device)

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
        if ckpt_path.exists():
            print(f"\n[Checkpoint] Loading from: {ckpt_path}")
            ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

            # Handle various checkpoint formats from AMDD training scripts
            state = ckpt.get('model_state', ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)))

            # Extract encoder sub-keys if saved as full model
            def _load_submodule(enc, prefix):
                sub_state = {
                    k[len(prefix):]: v
                    for k, v in state.items()
                    if k.startswith(prefix)
                }
                if sub_state:
                    missing, unexpected = enc.load_state_dict(
                        sub_state, strict=False
                    )
                    print(f"  {prefix}: loaded {len(sub_state)} keys "
                          f"(missing={len(missing)}, "
                          f"unexpected={len(unexpected)})")
                else:
                    print(f"  {prefix}: no matching keys found — "
                          f"using ImageNet init")

            _load_submodule(visual_enc, 'visual_encoder.')
            _load_submodule(audio_enc,  'audio_encoder.')
        else:
            print(f"[Warning] Checkpoint not found at {ckpt_path} — "
                  f"using ImageNet init weights")
    else:
        print("\n[Encoders] No checkpoint supplied — "
              "using ImageNet init weights (baseline mode)")

    # Freeze encoders — CMPI predictors are trained independently
    for param in visual_enc.parameters():
        param.requires_grad = False
    for param in audio_enc.parameters():
        param.requires_grad = False

    visual_enc.eval()
    audio_enc.eval()

    # ── Datasets and dataloaders ──────────────────────────────────────────
    print("\n[Data] Building datasets …")
    splits_to_load = ['train', 'val', 'test']
    datasets = {}
    for split in splits_to_load:
        try:
            datasets[split] = DeepfakeDataset(
                metadata_csv=args.metadata_csv,
                split=split,
                preprocessed_dir=args.preprocessed_dir,
            )
        except Exception as e:
            print(f"  Warning: could not load split '{split}': {e}")

    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.enc_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == 'cuda'),
        )
        for split, ds in datasets.items()
    }

    # ── Extract (or load cached) embeddings ───────────────────────────────
    embeddings = {}
    for split in datasets:
        cache_v = emb_cache / f'{split}_z_v.npy'
        cache_a = emb_cache / f'{split}_z_a.npy'
        cache_l = emb_cache / f'{split}_labels.npy'

        if args.skip_embedding_extraction and cache_v.exists():
            print(f"\n[Embeddings] Loading cached {split} embeddings …")
            embeddings[split] = (
                np.load(str(cache_v)),
                np.load(str(cache_a)),
                np.load(str(cache_l)),
            )
        else:
            print(f"\n[Embeddings] Extracting {split} …")
            z_v, z_a, labels, _ = extract_embeddings(
                visual_enc, audio_enc, loaders[split], device, split
            )
            np.save(str(cache_v), z_v)
            np.save(str(cache_a), z_a)
            np.save(str(cache_l), labels)
            embeddings[split] = (z_v, z_a, labels)
            print(f"  Cached to {emb_cache}/")

    z_v_train, z_a_train, labels_train = embeddings['train']

    # ── Train predictors on real samples only ─────────────────────────────
    print("\n[Predictors] Training T_{a→v} and T_{v→a} on real samples …")
    T_av, T_va, train_history = train_predictors(
        z_v_train, z_a_train, labels_train,
        embed_dim=args.embed_dim,
        device=device,
        args=args,
    )

    # Save predictor weights
    torch.save(T_av.state_dict(), str(output_dir / 'T_av.pt'))
    torch.save(T_va.state_dict(), str(output_dir / 'T_va.pt'))
    print("  Saved predictor weights to output_dir/")

    # ── Compute C_i on all splits ─────────────────────────────────────────
    print("\n[Scoring] Computing C_i scores …")
    scores_dict = {}
    labels_dict = {}
    all_results = []

    for split in ['val', 'test']:
        if split not in embeddings:
            continue
        z_v, z_a, labels = embeddings[split]
        scores = compute_consistency_scores(T_av, T_va, z_v, z_a, device)
        scores_dict[split] = scores
        labels_dict[split] = labels

        # Also save raw scores for later analysis
        np.save(str(output_dir / f'{split}_ci_scores.npy'), scores)
        np.save(str(output_dir / f'{split}_labels.npy'),   labels)

    # Optionally also score training set (should be biased toward low C_i for real)
    z_v, z_a, labels = embeddings['train']
    scores = compute_consistency_scores(T_av, T_va, z_v, z_a, device)
    scores_dict['train'] = scores
    labels_dict['train'] = labels

    # ── Evaluate ──────────────────────────────────────────────────────────
    print("\n[Evaluation]")
    for split in ['train', 'val', 'test']:
        if split not in scores_dict:
            continue
        res = evaluate(scores_dict[split], labels_dict[split],
                       split, output_dir)
        all_results.append(res)

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n[Plots] Generating figures …")
    # Show val and test in plots (exclude train to avoid clutter)
    plot_scores  = {k: v for k, v in scores_dict.items() if k != 'train'}
    plot_labels  = {k: v for k, v in labels_dict.items() if k != 'train'}
    plot_all(plot_scores, plot_labels, train_history, output_dir)

    # ── Save JSON summary ─────────────────────────────────────────────────
    summary = {
        'experiment': 'CMPI_feasibility',
        'hypothesis': ('Real samples have lower predictive inconsistency C_i '
                       'than fake samples under frozen AMDD encoders '
                       'with predictors trained on real only.'),
        'success_criterion': 'AUROC(C_i, test) > 0.70',
        'args': vars(args),
        'results': all_results,
    }

    test_results = [r for r in all_results if r['split'] == 'test']
    if test_results:
        test_auroc = test_results[0]['auroc']
        summary['verdict'] = (
            f"HYPOTHESIS VALIDATED — test AUROC={test_auroc:.4f} > 0.70"
            if test_auroc > 0.70
            else f"HYPOTHESIS NOT YET MET — test AUROC={test_auroc:.4f} < 0.70 "
                 f"(check encoder quality, try more predictor epochs)"
        )
        print(f"\n{'═'*55}")
        print(f"  VERDICT: {summary['verdict']}")
        print(f"{'═'*55}")

    summary_path = output_dir / 'feasibility_summary.json'
    with open(str(summary_path), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Done] Full summary saved to: {summary_path}")
    print(f"  Figures and scores in: {output_dir}/")


if __name__ == '__main__':
    main()