"""
CMPI Ablation Study
=====================
Wasim Ahmad, BIT

Trains five model variants on FakeAVCeleb and reports test metrics.
All variants use identical hyperparameters — only architecture differs.

Variants
--------
  full        : Complete CMPI (both predictors, C_i in detection head)
  no_T_av     : Remove T_{a→v} predictor (audio→visual arm disabled)
  no_T_va     : Remove T_{v→a} predictor (visual→audio arm disabled)
  no_ci_input : Remove C_i from detection head input (head sees only z_v, z_a)
  no_pred_loss: λ=0 — encoders + head trained but no predictive loss at all

Run all variants sequentially (recommended — ~6 hours total):
  python ablation_ccmpm.py --output_dir results/ablation \
      --metadata_csv data/metadata.csv \
      --preprocessed_dir data/preprocessed \
      --amdd_checkpoint AMDD/checkpoints/full_model_best.pth

Run a single variant:
  python ablation_ccmpm.py --variant no_T_av --output_dir results/ablation ...
"""

import sys
import json
import time
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── AMDD imports ──────────────────────────────────────────────────────────
AMDD_ROOT = Path(__file__).resolve().parent.parent
for _p in [AMDD_ROOT, AMDD_ROOT / "models", AMDD_ROOT / "data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from encoders import VisualEncoder, AudioEncoder
except ModuleNotFoundError:
    from models.encoders import VisualEncoder, AudioEncoder

try:
    from dataset import DeepfakeDataset, get_weighted_sampler
except ModuleNotFoundError:
    sys.path.insert(0, str(AMDD_ROOT / "training"))
    from dataset import DeepfakeDataset, get_weighted_sampler


# ══════════════════════════════════════════════════════════════════════════
# Ablation model variants
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


class CMPIAblation(nn.Module):
    """
    CMPI with configurable ablations.

    variant options:
      'full'         — complete model (T_av + T_va + C_i in head)
      'no_T_av'      — T_{a→v} disabled; C_i uses only err_va
      'no_T_va'      — T_{v→a} disabled; C_i uses only err_av
      'no_ci_input'  — C_i not fed to detection head; head sees [z_v; z_a] only
      'no_pred_loss' — full architecture but λ=0 (handled in loss, not here)
    """
    def __init__(self, variant='full', embed_dim=512,
                 hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.variant    = variant
        self.embed_dim  = embed_dim

        self.visual_encoder = VisualEncoder(embed_dim=embed_dim, dropout=dropout)
        self.audio_encoder  = AudioEncoder(embed_dim=embed_dim,  dropout=dropout)

        # Predictors — instantiate both, disable in forward as needed
        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)

        # Running stats for C_i normalisation
        self.register_buffer('ci_running_mean', torch.tensor(1.0))
        self.register_buffer('ci_running_std',  torch.tensor(1.0))
        self.register_buffer('ci_n_updates',    torch.tensor(0))

        # Detection head — input dim depends on variant
        if variant == 'no_ci_input':
            head_in = embed_dim * 2          # [z_v ; z_a]
        else:
            head_in = embed_dim * 2 + 1      # [z_v ; z_a ; C_i_norm]
        self.detection_head = DetectionHead(head_in, dropout)

    @torch.no_grad()
    def _update_ci_stats(self, ci):
        m  = 0.99 if self.ci_n_updates > 10 else 0.5
        self.ci_running_mean = m * self.ci_running_mean + (1-m) * ci.mean()
        self.ci_running_std  = m * self.ci_running_std  + (1-m) * ci.std().clamp(min=1e-6)
        self.ci_n_updates   += 1

    def forward(self, frames, audio, update_ci_stats=True):
        z_v = self.visual_encoder(frames)['video_embedding']
        z_a = self.audio_encoder(audio)['audio_embedding']

        # Compute predictions according to variant
        if self.variant == 'no_T_av':
            # Only T_{v→a} active; err_av = 0
            pred_a = self.T_va(z_v)
            err_av = torch.zeros(z_v.shape[0], device=z_v.device)
            err_va = torch.norm(pred_a - z_a, dim=1)
            pred_v = z_v   # unused but kept for loss fn compatibility
        elif self.variant == 'no_T_va':
            # Only T_{a→v} active; err_va = 0
            pred_v = self.T_av(z_a)
            err_av = torch.norm(pred_v - z_v, dim=1)
            err_va = torch.zeros(z_v.shape[0], device=z_v.device)
            pred_a = z_a
        else:
            # Both active (full, no_ci_input, no_pred_loss)
            pred_v = self.T_av(z_a)
            pred_a = self.T_va(z_v)
            err_av = torch.norm(pred_v - z_v, dim=1)
            err_va = torch.norm(pred_a - z_a, dim=1)

        C_i = err_av + err_va

        if self.training and update_ci_stats:
            self._update_ci_stats(C_i.detach())

        # Build detection head input
        if self.variant == 'no_ci_input':
            det_input = torch.cat([z_v, z_a], dim=1)
        else:
            C_i_norm  = ((C_i - self.ci_running_mean) /
                         self.ci_running_std.clamp(min=1e-6)).unsqueeze(1)
            det_input = torch.cat([z_v, z_a, C_i_norm], dim=1)

        det_logits = self.detection_head(det_input)
        det_probs  = torch.sigmoid(det_logits)

        return {
            'z_v': z_v, 'z_a': z_a,
            'pred_v': pred_v, 'pred_a': pred_a,
            'C_i': C_i, 'err_av': err_av, 'err_va': err_va,
            'det_logits': det_logits, 'det_probs': det_probs,
        }


# ══════════════════════════════════════════════════════════════════════════
# Loss, evaluation, training
# ══════════════════════════════════════════════════════════════════════════

def focal_loss(logits, labels, alpha=0.75, gamma=2.0):
    labels = labels.float()
    bce    = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt     = torch.exp(-bce)
    aw     = torch.where(labels == 1, alpha, 1 - alpha)
    return (aw * (1 - pt) ** gamma * bce).mean()


def compute_loss(outputs, labels, variant, lambda_pred=0.1):
    L_det = focal_loss(outputs['det_logits'], labels)

    if variant == 'no_pred_loss' or lambda_pred == 0.0:
        return L_det, {'total': L_det.item(), 'det': L_det.item(),
                       'L_av': 0.0, 'L_va': 0.0}

    real_mask = (labels == 0)
    L_av = L_va = torch.tensor(0.0, device=labels.device)

    if real_mask.sum() > 0:
        if variant != 'no_T_av':
            L_av = F.mse_loss(outputs['pred_v'][real_mask],
                              outputs['z_v'][real_mask].detach())
        if variant != 'no_T_va':
            L_va = F.mse_loss(outputs['pred_a'][real_mask],
                              outputs['z_a'][real_mask].detach())

    total = L_det + lambda_pred * (L_av + L_va)
    return total, {
        'total': total.item(), 'det': L_det.item(),
        'L_av': L_av.item(),   'L_va': L_va.item(),
    }


@torch.no_grad()
def evaluate(model, loader, device, variant):
    model.eval()
    all_probs, all_ci, all_labels = [], [], []

    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label']
        out    = model(frames, audio, update_ci_stats=False)
        all_probs.append(out['det_probs'].cpu().numpy())
        all_ci.append(out['C_i'].cpu().numpy())
        all_labels.append(labels.numpy())

    probs  = np.concatenate(all_probs)
    ci     = np.concatenate(all_ci)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    auroc   = roc_auc_score(labels, probs)   if len(np.unique(labels)) > 1 else float('nan')
    ci_auc  = roc_auc_score(labels, ci)      if len(np.unique(labels)) > 1 else float('nan')
    bal_acc = balanced_accuracy_score(labels, preds)

    ci_real = ci[labels == 0]
    ci_fake = ci[labels == 1]

    return {
        'auroc':   auroc,
        'ci_auroc': ci_auc,
        'bal_acc': bal_acc,
        'ci_mean_real': float(ci_real.mean()) if len(ci_real) else float('nan'),
        'ci_mean_fake': float(ci_fake.mean()) if len(ci_fake) else float('nan'),
        'separation':   float(ci_fake.mean() / (ci_real.mean() + 1e-8))
                        if len(ci_real) and len(ci_fake) else float('nan'),
    }


def load_amdd_encoders(model, ckpt_path, device):
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model_state_dict',
            ckpt.get('state_dict', ckpt)))

    def _load(enc, prefix):
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        if sub:
            missing, unexpected = enc.load_state_dict(sub, strict=False)
            print(f"    {prefix}: {len(sub)} keys "
                  f"(missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            print(f"    {prefix}: no keys — ImageNet init")

    _load(model.visual_encoder, 'visual_encoder.')
    _load(model.audio_encoder,  'audio_encoder.')


def train_variant(variant, args, device, train_loader, val_loader,
                  test_loader, output_dir):
    print(f"\n{'═'*60}")
    print(f"  Variant: {variant}")
    print(f"{'═'*60}")

    model = CMPIAblation(
        variant=variant,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    if args.amdd_checkpoint:
        print("  Loading AMDD encoder weights...")
        load_amdd_encoders(model, args.amdd_checkpoint, device)

    # Differential LR: encoders lower, new components higher
    enc_params  = (list(model.visual_encoder.parameters()) +
                   list(model.audio_encoder.parameters()))
    new_params  = (list(model.T_av.parameters()) +
                   list(model.T_va.parameters()) +
                   list(model.detection_head.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': enc_params, 'lr': args.lr,           'name': 'encoders'},
        {'params': new_params, 'lr': args.lr_predictor, 'name': 'predictors'},
    ], weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        p = (ep - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = (torch.amp.GradScaler('cuda')
                 if device.type == 'cuda' else None)

    best_val_auroc = 0.0
    best_test      = None
    var_dir        = output_dir / variant
    var_dir.mkdir(parents=True, exist_ok=True)

    lambda_pred = 0.0 if variant == 'no_pred_loss' else args.lambda_pred

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            frames = batch['frames'].to(device, non_blocking=True)
            audio  = batch['audio'].to(device,  non_blocking=True)
            labels = batch['label'].to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                out  = model(frames, audio)
                loss, _ = compute_loss(out, labels, variant, lambda_pred)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        scheduler.step()

        val_metrics = evaluate(model, val_loader, device, variant)

        if val_metrics['auroc'] > best_val_auroc:
            best_val_auroc = val_metrics['auroc']
            torch.save({'model_state': model.state_dict(),
                        'epoch': epoch,
                        'val_auroc': best_val_auroc,
                        'variant': variant},
                       str(var_dir / 'best.pt'))

        if epoch % 10 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"val AUROC={val_metrics['auroc']:.4f}  "
                  f"C_i-AUC={val_metrics['ci_auroc']:.4f}  "
                  f"BalAcc={val_metrics['bal_acc']:.4f}  "
                  f"sep={val_metrics['separation']:.2f}×")

    # Load best and evaluate on test
    best_ckpt = torch.load(str(var_dir / 'best.pt'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state'])
    test_metrics = evaluate(model, test_loader, device, variant)

    print(f"\n  ── TEST ({variant}) ──")
    print(f"  AUROC      : {test_metrics['auroc']:.4f}")
    print(f"  C_i AUROC  : {test_metrics['ci_auroc']:.4f}")
    print(f"  Balanced   : {test_metrics['bal_acc']:.4f}")
    print(f"  C_i sep    : {test_metrics['separation']:.2f}×")
    print(f"  C_i real   : {test_metrics['ci_mean_real']:.2f}")
    print(f"  C_i fake   : {test_metrics['ci_mean_fake']:.2f}")

    return {
        'variant':      variant,
        'best_val_auroc': best_val_auroc,
        'test':         test_metrics,
    }


# ══════════════════════════════════════════════════════════════════════════
# Summary and plotting
# ══════════════════════════════════════════════════════════════════════════

VARIANT_LABELS = {
    'full':         'Full CMPI',
    'no_T_av':      'w/o T_{a→v}',
    'no_T_va':      'w/o T_{v→a}',
    'no_ci_input':  'w/o C_i input',
    'no_pred_loss': 'w/o pred. loss (λ=0)',
}


def print_ablation_table(results):
    print(f"\n\n{'═'*72}")
    print(f"  ABLATION TABLE (FakeAVCeleb test set)")
    print(f"{'═'*72}")
    print(f"  {'Variant':<26} {'AUROC':>8} {'C_i AUC':>9} "
          f"{'BalAcc':>8} {'Sep':>7}")
    print(f"  {'─'*62}")
    for r in results:
        t = r['test']
        print(f"  {VARIANT_LABELS[r['variant']]:<26} "
              f"{t['auroc']:>8.4f} "
              f"{t['ci_auroc']:>9.4f} "
              f"{t['bal_acc']:>8.4f} "
              f"{t['separation']:>7.2f}×")
    print(f"{'═'*72}")


def plot_ablation(results, output_dir):
    variants = [r['variant'] for r in results]
    labels   = [VARIANT_LABELS[v] for v in variants]
    aurocs   = [r['test']['auroc']   for r in results]
    ci_aucs  = [r['test']['ci_auroc'] for r in results]
    bal_accs = [r['test']['bal_acc'] for r in results]
    seps     = [r['test']['separation'] for r in results]

    x     = np.arange(len(variants))
    width = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUROC and C_i AUROC bars
    ax = axes[0]
    ax.bar(x - width/2, aurocs,  width, label='AUROC (det_probs)',
           color='steelblue', alpha=0.85)
    ax.bar(x + width/2, ci_aucs, width, label='C_i AUROC (unsup.)',
           color='darkorange', alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('AUROC')
    ax.set_ylim(0.5, 1.02)
    ax.set_title('Detection AUROC per Ablation Variant')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    # Annotate bars
    for i, (a, c) in enumerate(zip(aurocs, ci_aucs)):
        ax.text(i - width/2, a + 0.005, f'{a:.3f}', ha='center',
                va='bottom', fontsize=7)
        ax.text(i + width/2, c + 0.005, f'{c:.3f}', ha='center',
                va='bottom', fontsize=7)

    # Right: C_i separation ratio
    ax2 = axes[1]
    colors = ['steelblue' if v == 'full' else 'gray' for v in variants]
    bars   = ax2.bar(x, seps, color=colors, alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
    ax2.set_ylabel('C_i Separation Ratio (fake/real mean)')
    ax2.set_title('C_i Separation per Ablation Variant')
    ax2.axhline(1.0, color='red', ls='--', lw=1, label='No separation (ratio=1)')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    for bar, v in zip(bars, seps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f'{v:.2f}×', ha='center', va='bottom', fontsize=8)

    fig.suptitle('CMPI Ablation Study — FakeAVCeleb Test Set', fontsize=13)
    fig.tight_layout()
    p = output_dir / 'ablation_results.png'
    fig.savefig(str(p), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: ablation_results.png")


# ══════════════════════════════════════════════════════════════════════════
# Argument parsing and main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description='CMPI Ablation Study')
    p.add_argument('--metadata_csv',     required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--amdd_checkpoint',  default=None)
    p.add_argument('--output_dir',
                   default='results/ablation')
    p.add_argument('--variant', default='all',
                   choices=['all', 'full', 'no_T_av', 'no_T_va',
                            'no_ci_input', 'no_pred_loss'])
    # Training (match train_ccmpm.py v2 defaults)
    p.add_argument('--epochs',        type=int,   default=50)
    p.add_argument('--batch_size',    type=int,   default=8)
    p.add_argument('--num_workers',   type=int,   default=4)
    p.add_argument('--embed_dim',     type=int,   default=512)
    p.add_argument('--hidden_dim',    type=int,   default=1024)
    p.add_argument('--dropout',       type=float, default=0.1)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--lr_predictor',  type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-4)
    p.add_argument('--lambda_pred',   type=float, default=0.1)
    p.add_argument('--warmup_epochs', type=int,   default=5)
    p.add_argument('--seed',          type=int,   default=42)
    p.add_argument('--device',        default='auto')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  CMPI Ablation Study")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}  Batch: {args.batch_size}")
    print(f"  λ_pred: {args.lambda_pred}")
    print(f"{'═'*60}")

    # Data
    print("\n[Data]")
    train_ds = DeepfakeDataset(args.metadata_csv, 'train', args.preprocessed_dir)
    val_ds   = DeepfakeDataset(args.metadata_csv, 'val',   args.preprocessed_dir)
    test_ds  = DeepfakeDataset(args.metadata_csv, 'test',  args.preprocessed_dir)

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

    # Variants to run
    all_variants = ['full', 'no_T_av', 'no_T_va', 'no_ci_input', 'no_pred_loss']
    variants     = all_variants if args.variant == 'all' else [args.variant]

    all_results = []
    t0 = time.time()

    for variant in variants:
        result = train_variant(
            variant, args, device,
            train_loader, val_loader, test_loader,
            output_dir
        )
        all_results.append(result)

        # Save running summary after each variant
        with open(str(output_dir / 'ablation_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

    elapsed = (time.time() - t0) / 3600
    print(f"\n[Timing] Total: {elapsed:.1f}h")

    if len(all_results) > 1:
        print_ablation_table(all_results)
        print("\n[Plots]")
        plot_ablation(all_results, output_dir)

    print(f"\n[Done] Results in: {output_dir}")


if __name__ == '__main__':
    main()
