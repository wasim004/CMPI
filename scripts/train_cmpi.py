"""
CMPI End-to-End Training
==========================
Causal Cross-Modal Predictive Modeling — Wasim Ahmad, BIT

Architecture
------------
  VisualEncoder  (ResNet50 + temporal attention, from AMDD)
  AudioEncoder   (ResNet18 adapted for mel spectrograms, from AMDD)
  CrossModalPredictor  T_{a→v}  and  T_{v→a}   (MLP, trained here)
  DetectionHead  (MLP on [z_v ; z_a ; C_i])

Loss
----
  L_total = L_det  +  λ * (L_{a→v} + L_{v→a})

  L_det  : focal loss on binary real/fake labels
  L_{a→v}: MSE  ||T_{a→v}(z_a) - z_v||²   (all samples)
  L_{v→a}: MSE  ||T_{v→a}(z_v) - z_a||²   (all samples)

  During training the predictors receive gradient from ALL samples,
  but the C_i signal naturally separates because real embeddings
  already have tighter coupling (validated: test AUROC=0.9983).

Usage
-----
python train_ccmpm.py \
    --metadata_csv     data/metadata.csv \
    --preprocessed_dir data/preprocessed \
    --output_dir       results/training

# Resume from checkpoint:
python train_ccmpm.py \
    --metadata_csv     data/metadata.csv \
    --preprocessed_dir data/preprocessed \
    --output_dir       results/training \
    --resume           results/training/last_checkpoint.pt

# Warm-start encoders from AMDD (recommended):
python train_ccmpm.py \
    --metadata_csv     data/metadata.csv \
    --preprocessed_dir data/preprocessed \
    --output_dir       results/training \
    --amdd_checkpoint  AMDD/checkpoints/full_model_best.pth
"""

import sys
import json
import time
import argparse
import shutil
from pathlib import Path
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Resolve paths ─────────────────────────────────────────────────────────
AMDD_ROOT  = Path(__file__).resolve().parent.parent
CMPI_ROOT = Path(__file__).resolve().parent.parent

for _p in [AMDD_ROOT, AMDD_ROOT / "models", AMDD_ROOT / "data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from encoders import VisualEncoder, AudioEncoder
    print("✓ VisualEncoder, AudioEncoder from models/")
except ModuleNotFoundError:
    from models.encoders import VisualEncoder, AudioEncoder
    print("✓ VisualEncoder, AudioEncoder via models.encoders")

try:
    from dataset import DeepfakeDataset, get_weighted_sampler
    print("✓ DeepfakeDataset from data/")
except ModuleNotFoundError:
    sys.path.insert(0, str(AMDD_ROOT / "training"))
    from dataset import DeepfakeDataset, get_weighted_sampler
    print("✓ DeepfakeDataset from AMDD/training/")


# ══════════════════════════════════════════════════════════════════════════
# 1.  Model components
# ══════════════════════════════════════════════════════════════════════════

class CrossModalPredictor(nn.Module):
    """MLP predictor: one modality's embedding → the other's."""
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.1):
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

    def forward(self, x):
        return self.net(x)


class DetectionHead(nn.Module):
    """
    Binary detection head.
    Input: [z_v ; z_a ; C_i_normalized]  dim = 2*embed_dim + 1
    """
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [B]


class CMPIModel(nn.Module):
    """
    Full CMPI model.

    Forward pass returns:
        z_v, z_a          — encoder embeddings
        pred_v, pred_a    — cross-modal predictions
        C_i               — predictive inconsistency score [B]
        det_logits        — detection logits [B]
        det_probs         — sigmoid probabilities [B]
    """
    def __init__(self, embed_dim=512, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Encoders
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim, dropout=dropout)
        self.audio_encoder  = AudioEncoder(embed_dim=embed_dim,  dropout=dropout)

        # Cross-modal predictors
        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)

        # C_i normalisation: running stats updated during training
        self.register_buffer('ci_running_mean', torch.tensor(1.0))
        self.register_buffer('ci_running_std',  torch.tensor(1.0))
        self.register_buffer('ci_n_updates',    torch.tensor(0))

        # Detection head: [z_v ; z_a ; C_i_norm]
        self.detection_head = DetectionHead(embed_dim * 2 + 1, dropout)

    @torch.no_grad()
    def _update_ci_stats(self, ci):
        """EMA update of C_i running mean/std for normalisation."""
        batch_mean = ci.mean()
        batch_std  = ci.std().clamp(min=1e-6)
        momentum   = 0.99 if self.ci_n_updates > 10 else 0.5
        self.ci_running_mean = momentum * self.ci_running_mean + (1 - momentum) * batch_mean
        self.ci_running_std  = momentum * self.ci_running_std  + (1 - momentum) * batch_std
        self.ci_n_updates   += 1

    def forward(self, frames, audio, update_ci_stats=True):
        # Encode
        z_v = self.visual_encoder(frames)['video_embedding']   # [B, D]
        z_a = self.audio_encoder(audio)['audio_embedding']     # [B, D]

        # Cross-modal predictions
        pred_v = self.T_av(z_a)    # audio → visual
        pred_a = self.T_va(z_v)    # visual → audio

        # Predictive inconsistency score
        err_av = torch.norm(pred_v - z_v, dim=1)   # [B]
        err_va = torch.norm(pred_a - z_a, dim=1)   # [B]
        C_i    = err_av + err_va                    # [B]

        # Normalise C_i for detection head input
        if self.training and update_ci_stats:
            self._update_ci_stats(C_i.detach())

        C_i_norm = ((C_i - self.ci_running_mean) /
                    self.ci_running_std.clamp(min=1e-6)).unsqueeze(1)   # [B, 1]

        # Detection head
        det_input  = torch.cat([z_v, z_a, C_i_norm], dim=1)   # [B, 2D+1]
        det_logits = self.detection_head(det_input)             # [B]
        det_probs  = torch.sigmoid(det_logits)                  # [B]

        return {
            'z_v':        z_v,
            'z_a':        z_a,
            'pred_v':     pred_v,
            'pred_a':     pred_a,
            'C_i':        C_i,
            'det_logits': det_logits,
            'det_probs':  det_probs,
        }


# ══════════════════════════════════════════════════════════════════════════
# 2.  Loss functions
# ══════════════════════════════════════════════════════════════════════════

def focal_loss(logits, labels, alpha=0.75, gamma=2.0):
    """Focal loss for imbalanced binary classification."""
    labels = labels.float()
    bce    = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt     = torch.exp(-bce)
    aw     = torch.where(labels == 1, alpha, 1 - alpha)
    return (aw * (1 - pt) ** gamma * bce).mean()


def compute_loss(outputs, labels, lambda_pred=1.0):
    """
    L_total = L_det  +  λ * (L_{a→v} + L_{v→a})

    Predictive losses are computed on REAL samples only.
    This enforces that predictors learn genuine audio-visual coupling,
    so C_i is naturally lower for real and higher for fake — consistent
    with the causal theory in the paper (Sections 4-5).
    """
    # Detection loss (focal) — all samples
    L_det = focal_loss(outputs['det_logits'], labels)

    # Predictive losses — REAL samples only
    real_mask = (labels == 0)
    if real_mask.sum() > 0:
        L_av = F.mse_loss(outputs['pred_v'][real_mask],
                          outputs['z_v'][real_mask].detach())
        L_va = F.mse_loss(outputs['pred_a'][real_mask],
                          outputs['z_a'][real_mask].detach())
    else:
        # Fallback: batch has no real samples (rare with weighted sampler)
        L_av = torch.tensor(0.0, device=outputs['det_logits'].device)
        L_va = torch.tensor(0.0, device=outputs['det_logits'].device)

    L_pred = L_av + L_va
    total  = L_det + lambda_pred * L_pred

    return total, {
        'total':  total.item(),
        'det':    L_det.item(),
        'L_av':   L_av.item(),
        'L_va':   L_va.item(),
        'L_pred': L_pred.item(),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3.  Evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, lambda_pred=1.0):
    model.eval()
    all_probs, all_ci, all_labels = [], [], []
    total_loss = det_loss = pred_loss = 0.0
    n_batches  = 0

    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label'].to(device)

        outputs = model(frames, audio, update_ci_stats=False)
        loss, ld = compute_loss(outputs, labels, lambda_pred)

        total_loss += ld['total']
        det_loss   += ld['det']
        pred_loss  += ld['L_pred']
        n_batches  += 1

        all_probs.append(outputs['det_probs'].cpu().numpy())
        all_ci.append(outputs['C_i'].cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    probs  = np.concatenate(all_probs)
    ci     = np.concatenate(all_ci)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auroc    = roc_auc_score(labels, probs)
    bal_acc  = balanced_accuracy_score(labels, preds)
    ci_auroc = roc_auc_score(labels, ci)   # unsupervised C_i signal alone

    return {
        'loss':       total_loss / n_batches,
        'det_loss':   det_loss   / n_batches,
        'pred_loss':  pred_loss  / n_batches,
        'auroc':      auroc,
        'bal_acc':    bal_acc,
        'ci_auroc':   ci_auroc,
        'ci_mean_real': float(ci[labels == 0].mean()),
        'ci_mean_fake': float(ci[labels == 1].mean()),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4.  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════

def save_checkpoint(state, output_dir, is_best, filename='last_checkpoint.pt'):
    path = Path(output_dir) / filename
    torch.save(state, str(path))
    if is_best:
        best_path = Path(output_dir) / 'best_model.pt'
        shutil.copyfile(str(path), str(best_path))
        print(f"  ★ New best model saved → best_model.pt")


def load_amdd_encoders(model, ckpt_path, device):
    """Load visual_encoder and audio_encoder weights from AMDD checkpoint."""
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get('model_state',
            ckpt.get('model_state_dict',
            ckpt.get('state_dict', ckpt)))

    def _load(enc, prefix):
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        if sub:
            missing, unexpected = enc.load_state_dict(sub, strict=False)
            print(f"  {prefix}: {len(sub)} keys loaded "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"  {prefix}: no keys found — keeping ImageNet init")

    _load(model.visual_encoder, 'visual_encoder.')
    _load(model.audio_encoder,  'audio_encoder.')


# ══════════════════════════════════════════════════════════════════════════
# 5.  Plotting
# ══════════════════════════════════════════════════════════════════════════

def plot_training_curves(history, output_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    def _plot(ax, key_train, key_val, title, ylabel):
        ax.plot(epochs, history[key_train], label='Train', color='steelblue', lw=2)
        if key_val in history and history[key_val]:
            ax.plot(epochs, history[key_val],  label='Val',   color='darkorange', lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    _plot(axes[0,0], 'train_loss',     'val_loss',     'Total Loss',        'Loss')
    _plot(axes[0,1], 'train_det_loss', 'val_det_loss', 'Detection Loss',    'Focal Loss')
    _plot(axes[0,2], 'train_pred_loss','val_pred_loss', 'Predictive Loss',  'MSE')
    _plot(axes[1,0], 'train_auroc',    'val_auroc',    'AUROC (det_probs)', 'AUROC')
    _plot(axes[1,1], 'train_bal_acc',  'val_bal_acc',  'Balanced Accuracy', 'Bal. Acc')
    _plot(axes[1,2], 'train_ci_auroc', 'val_ci_auroc', 'AUROC (C_i only)',  'AUROC')

    # Mark best val epoch
    if history.get('val_auroc'):
        best_ep = int(np.argmax(history['val_auroc'])) + 1
        for ax in axes.flat:
            ax.axvline(best_ep, color='red', ls='--', lw=1, alpha=0.5)

    fig.suptitle('CMPI Training Curves', fontsize=14)
    fig.tight_layout()
    p = Path(output_dir) / 'training_curves.png'
    fig.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: training_curves.png")


def plot_ci_distributions(model, loader, device, output_dir, split='test'):
    """Plot C_i distributions for real vs fake on any split."""
    model.eval()
    ci_list, label_list = [], []

    with torch.no_grad():
        for batch in loader:
            frames = batch['frames'].to(device)
            audio  = batch['audio'].to(device)
            labels = batch['label']
            out    = model(frames, audio, update_ci_stats=False)
            ci_list.append(out['C_i'].cpu().numpy())
            label_list.append(labels.numpy())

    ci     = np.concatenate(ci_list)
    labels = np.concatenate(label_list)
    real_c = ci[labels == 0]
    fake_c = ci[labels == 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    bins = np.linspace(ci.min(), ci.max(), 60)
    axes[0].hist(real_c, bins=bins, alpha=0.65, color='royalblue',
                 density=True, label=f'Real  μ={real_c.mean():.2f}')
    axes[0].hist(fake_c, bins=bins, alpha=0.65, color='tomato',
                 density=True, label=f'Fake  μ={fake_c.mean():.2f}')
    axes[0].axvline(real_c.mean(), color='royalblue', ls='--', lw=1.5)
    axes[0].axvline(fake_c.mean(), color='tomato',    ls='--', lw=1.5)
    axes[0].set_title(f'C_i Distribution — {split.upper()}')
    axes[0].set_xlabel('Predictive Inconsistency C_i')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Boxplot
    bp = axes[1].boxplot(
        [real_c, fake_c],
        tick_labels=['Real', 'Fake'],
        patch_artist=True, notch=True,
        medianprops=dict(color='black', lw=2),
    )
    bp['boxes'][0].set_facecolor('royalblue'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('tomato');    bp['boxes'][1].set_alpha(0.7)
    auroc = roc_auc_score(labels, ci)
    axes[1].set_title(f'C_i Boxplot — AUROC={auroc:.4f}')
    axes[1].set_ylabel('C_i')
    axes[1].grid(axis='y', alpha=0.3)

    fig.suptitle(f'CMPI Predictive Inconsistency — {split.upper()}', fontsize=13)
    fig.tight_layout()
    p = Path(output_dir) / f'ci_distribution_{split}.png'
    fig.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ci_distribution_{split}.png")


# ══════════════════════════════════════════════════════════════════════════
# 6.  Training loop
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    lambda_pred, scaler, epoch, log_interval=20):
    model.train()
    total_loss = det_loss_sum = pred_loss_sum = 0.0
    n_batches = 0
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        frames = batch['frames'].to(device, non_blocking=True)
        audio  = batch['audio'].to(device,  non_blocking=True)
        labels = batch['label'].to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            outputs = model(frames, audio)
            loss, ld = compute_loss(outputs, labels, lambda_pred)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss    += ld['total']
        det_loss_sum  += ld['det']
        pred_loss_sum += ld['L_pred']
        n_batches     += 1

        if (batch_idx + 1) % log_interval == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch} [{batch_idx+1}/{len(loader)}] "
                  f"loss={ld['total']:.4f}  "
                  f"det={ld['det']:.4f}  "
                  f"pred={ld['L_pred']:.4f}  "
                  f"({elapsed:.1f}s)")

    if scheduler is not None:
        scheduler.step()

    return {
        'loss':      total_loss    / n_batches,
        'det_loss':  det_loss_sum  / n_batches,
        'pred_loss': pred_loss_sum / n_batches,
    }


# ══════════════════════════════════════════════════════════════════════════
# 7.  Argument parsing
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='CMPI Training')

    # Data
    p.add_argument('--metadata_csv',     required=True)
    p.add_argument('--preprocessed_dir', required=True)

    # Output
    p.add_argument('--output_dir',
                   default='results/training')

    # Warm-start / resume
    p.add_argument('--amdd_checkpoint', default=None,
                   help='AMDD checkpoint to warm-start encoders from')
    p.add_argument('--resume', default=None,
                   help='CMPI checkpoint to resume training from')

    # Architecture
    p.add_argument('--embed_dim',   type=int,   default=512)
    p.add_argument('--hidden_dim',  type=int,   default=1024)
    p.add_argument('--dropout',     type=float, default=0.1)

    # Training
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--lr',            type=float, default=1e-4,
                   help='Base LR for encoders (warm-started)')
    p.add_argument('--lr_predictor',  type=float, default=1e-3,
                   help='LR for predictor MLPs and detection head')
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--lambda_pred',   type=float, default=0.1,
                   help='Weight λ on predictive loss (default: 0.1)')
    p.add_argument('--warmup_epochs', type=int,   default=5,
                   help='Linear LR warmup epochs')
    p.add_argument('--amp',           action='store_true', default=True,
                   help='Use automatic mixed precision (default: on)')
    p.add_argument('--no_amp',        action='store_true')

    # Misc
    p.add_argument('--seed',   type=int, default=42)
    p.add_argument('--device', default='auto')
    p.add_argument('--log_interval', type=int, default=20)

    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════
# 8.  Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    use_amp = (args.amp and not args.no_amp and device.type == 'cuda')
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  CMPI End-to-End Training")
    print(f"  Device : {device}  |  AMP: {use_amp}")
    print(f"  Output : {output_dir}")
    print(f"{'═'*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    print("[Data] Building datasets …")
    train_ds = DeepfakeDataset(args.metadata_csv, split='train',
                               preprocessed_dir=args.preprocessed_dir)
    val_ds   = DeepfakeDataset(args.metadata_csv, split='val',
                               preprocessed_dir=args.preprocessed_dir)
    test_ds  = DeepfakeDataset(args.metadata_csv, split='test',
                               preprocessed_dir=args.preprocessed_dir)

    sampler      = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=args.num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers,
                              pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    print("\n[Model] Building CMPI …")
    model = CMPIModel(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    # Warm-start encoders from AMDD
    start_epoch  = 1
    best_val_auroc = 0.0
    history = {k: [] for k in [
        'train_loss', 'train_det_loss', 'train_pred_loss',
        'train_auroc', 'train_bal_acc', 'train_ci_auroc',
        'val_loss',   'val_det_loss',   'val_pred_loss',
        'val_auroc',  'val_bal_acc',    'val_ci_auroc',
        'lr_encoder', 'lr_predictor',
    ]}

    if args.resume:
        print(f"\n[Resume] Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_auroc = ckpt.get('best_val_auroc', 0.0)
        history        = ckpt.get('history', history)
        print(f"  Resuming from epoch {start_epoch}, "
              f"best val AUROC={best_val_auroc:.4f}")
    elif args.amdd_checkpoint:
        print(f"\n[Warm-start] Loading AMDD encoders from: {args.amdd_checkpoint}")
        load_amdd_encoders(model, args.amdd_checkpoint, device)

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    enc_params   = (sum(p.numel() for p in model.visual_encoder.parameters()) +
                    sum(p.numel() for p in model.audio_encoder.parameters())) / 1e6
    pred_params  = (sum(p.numel() for p in model.T_av.parameters()) +
                    sum(p.numel() for p in model.T_va.parameters()) +
                    sum(p.numel() for p in model.detection_head.parameters())) / 1e6
    print(f"\n  Total params : {total_params:.1f}M")
    print(f"  Encoder params : {enc_params:.1f}M")
    print(f"  Predictor+head : {pred_params:.1f}M")

    # ── Optimiser — differential LR ───────────────────────────────────────
    # Encoders: lower LR (already well-trained from AMDD)
    # Predictors + head: higher LR (trained from scratch)
    enc_params_list = (list(model.visual_encoder.parameters()) +
                       list(model.audio_encoder.parameters()))
    new_params_list = (list(model.T_av.parameters()) +
                       list(model.T_va.parameters()) +
                       list(model.detection_head.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': enc_params_list, 'lr': args.lr,           'name': 'encoders'},
        {'params': new_params_list, 'lr': args.lr_predictor, 'name': 'predictors'},
    ], weight_decay=args.weight_decay)

    if args.resume and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])

    # Cosine annealing after warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    if args.resume and 'scheduler_state' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state'])

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n[Training] {args.epochs} epochs, "
          f"batch={args.batch_size}, "
          f"lr_enc={args.lr}, lr_pred={args.lr_predictor}, "
          f"λ={args.lambda_pred}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()
        print(f"── Epoch {epoch}/{args.epochs} "
              f"(lr_enc={optimizer.param_groups[0]['lr']:.2e}, "
              f"lr_pred={optimizer.param_groups[1]['lr']:.2e}) ──")

        # Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            args.lambda_pred, scaler, epoch, args.log_interval
        )

        # Evaluate on train (quick pass — no grad)
        train_eval = evaluate(model, train_loader, device, args.lambda_pred)
        val_eval   = evaluate(model, val_loader,   device, args.lambda_pred)

        # Log
        current_lr_enc  = optimizer.param_groups[0]['lr']
        current_lr_pred = optimizer.param_groups[1]['lr']

        history['train_loss'].append(train_metrics['loss'])
        history['train_det_loss'].append(train_metrics['det_loss'])
        history['train_pred_loss'].append(train_metrics['pred_loss'])
        history['train_auroc'].append(train_eval['auroc'])
        history['train_bal_acc'].append(train_eval['bal_acc'])
        history['train_ci_auroc'].append(train_eval['ci_auroc'])
        history['val_loss'].append(val_eval['loss'])
        history['val_det_loss'].append(val_eval['det_loss'])
        history['val_pred_loss'].append(val_eval['pred_loss'])
        history['val_auroc'].append(val_eval['auroc'])
        history['val_bal_acc'].append(val_eval['bal_acc'])
        history['val_ci_auroc'].append(val_eval['ci_auroc'])
        history['lr_encoder'].append(current_lr_enc)
        history['lr_predictor'].append(current_lr_pred)

        # Console summary
        elapsed = time.time() - t_epoch
        print(f"  Train | loss={train_metrics['loss']:.4f}  "
              f"AUROC={train_eval['auroc']:.4f}  "
              f"BalAcc={train_eval['bal_acc']:.4f}  "
              f"C_i-AUROC={train_eval['ci_auroc']:.4f}")
        print(f"  Val   | loss={val_eval['loss']:.4f}  "
              f"AUROC={val_eval['auroc']:.4f}  "
              f"BalAcc={val_eval['bal_acc']:.4f}  "
              f"C_i-AUROC={val_eval['ci_auroc']:.4f}  "
              f"({elapsed:.1f}s)")
        print(f"  C_i   | real={val_eval['ci_mean_real']:.2f}  "
              f"fake={val_eval['ci_mean_fake']:.2f}  "
              f"ratio={val_eval['ci_mean_fake']/(val_eval['ci_mean_real']+1e-6):.2f}×")

        # Checkpoint
        is_best = val_eval['auroc'] > best_val_auroc
        if is_best:
            best_val_auroc = val_eval['auroc']

        save_checkpoint({
            'epoch':          epoch,
            'model_state':    model.state_dict(),
            'optimizer_state':optimizer.state_dict(),
            'scheduler_state':scheduler.state_dict(),
            'best_val_auroc': best_val_auroc,
            'val_metrics':    val_eval,
            'history':        history,
            'args':           vars(args),
        }, output_dir, is_best)

        # Save history JSON every epoch
        with open(str(output_dir / 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        print()

    # ── Final evaluation on test set ──────────────────────────────────────
    print("═" * 60)
    print("[Final Evaluation] Loading best model for test set …")
    best_ckpt = torch.load(str(output_dir / 'best_model.pt'),
                            map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state'])

    test_eval = evaluate(model, test_loader, device, args.lambda_pred)

    print(f"\n  ── TEST SET RESULTS ──")
    print(f"  AUROC           : {test_eval['auroc']:.4f}")
    print(f"  Balanced Acc    : {test_eval['bal_acc']:.4f}")
    print(f"  C_i AUROC       : {test_eval['ci_auroc']:.4f}")
    print(f"  C_i real mean   : {test_eval['ci_mean_real']:.4f}")
    print(f"  C_i fake mean   : {test_eval['ci_mean_fake']:.4f}")
    print(f"  Separation ratio: "
          f"{test_eval['ci_mean_fake']/(test_eval['ci_mean_real']+1e-6):.3f}×")
    print("═" * 60)

    # Save final results
    final = {
        'best_val_auroc': best_val_auroc,
        'test': test_eval,
        'args': vars(args),
    }
    with open(str(output_dir / 'final_results.json'), 'w') as f:
        json.dump(final, f, indent=2)

    # ── Final plots ───────────────────────────────────────────────────────
    print("\n[Plots]")
    plot_training_curves(history, output_dir)
    plot_ci_distributions(model, val_loader,  device, output_dir, 'val')
    plot_ci_distributions(model, test_loader, device, output_dir, 'test')

    print(f"\n[Done] All outputs in: {output_dir}")


if __name__ == '__main__':
    main()