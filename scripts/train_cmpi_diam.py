"""
CMPI + DIAM + TIL  Training Script
====================================
Wasim Ahmad, BIT  —  August 2026

Changes from original CMPI
---------------------------
1. Directional Inconsistency Asymmetry Modeling (DIAM)
   - Computes signed asymmetry  A = ε_f − ε_b
   - Magnitude C_i = ε_f + ε_b  (original)
   - Both fed to detection head: qualitatively new forensic signal
   - Sign of A encodes which modality is inconsistent (novelty argument)

2. Temporal Inconsistency Localization (TIL) head
   - Uses per-frame visual features h_t from VisualEncoder
   - Frame-level error:  err_t = ||T_av(z_a) − h_t||
   - Temporal attention: α_t = softmax(w · err_t)
   - TIL score = Σ_t α_t * err_t   (attention-weighted per-frame error)
   - Directly addresses LAV-DF collapse (partial fakes diluted by clip pooling)

Detection head input: [z_v, z_a, C_i_norm, A_norm, TIL_norm]  dim = 2D+3

Usage
-----
python train_cmpi_diam.py \\
    --metadata_csv     /path/to/AMDD/data/metadata.csv \\
    --preprocessed_dir /path/to/AMDD/data/preprocessed \\
    --amdd_checkpoint  /path/to/AMDD/checkpoints/full_model_best.pth \\
    --output_dir       results/diam
"""

import sys, json, time, argparse, shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Resolve encoder / dataset imports ────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
AMDD_MODELS = ROOT / "AMDD" / "models"
AMDD_DATA   = ROOT / "AMDD" / "data"

for _p in [str(AMDD_MODELS), str(AMDD_DATA)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoders import VisualEncoder, AudioEncoder
from dataset  import DeepfakeDataset, get_weighted_sampler


# ══════════════════════════════════════════════════════════════════════════
# 1.  Model
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


class TemporalAttentionScorer(nn.Module):
    """Scalar attention over per-frame error sequence → TIL score."""
    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, err_seq):
        # err_seq: [B, T]  per-frame error magnitudes
        x = err_seq.unsqueeze(-1)          # [B, T, 1]
        logits = self.fc(x).squeeze(-1)    # [B, T]
        alpha  = torch.softmax(logits, dim=1)   # [B, T]
        til    = (alpha * err_seq).sum(dim=1)   # [B]
        return til, alpha


class CMPIDIAMModel(nn.Module):
    """
    CMPI + DIAM + TIL model.

    Outputs:
      z_v, z_a          — clip-level encoder embeddings
      pred_v, pred_a    — cross-modal predictions (clip level)
      C_i               — magnitude: ε_f + ε_b           [B]
      A                 — asymmetry: ε_f − ε_b           [B]
      til               — TIL attention-weighted score    [B]
      til_alpha         — per-frame attention weights     [B, T]
      det_logits        — detection logits                [B]
      det_probs         — sigmoid detection probabilities [B]
    """
    def __init__(self, embed_dim=512, hidden_dim=1024, dropout=0.1, n_frames=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_frames  = n_frames

        self.visual_encoder = VisualEncoder(embed_dim=embed_dim, dropout=dropout)
        self.audio_encoder  = AudioEncoder(embed_dim=embed_dim,  dropout=dropout)

        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)

        # TIL head — attends over per-frame error sequence
        self.til_scorer = TemporalAttentionScorer(in_dim=1, hidden=32)

        # Running stats for normalisation (EMA)
        for name in ('ci', 'asym', 'til'):
            self.register_buffer(f'{name}_mean', torch.tensor(1.0))
            self.register_buffer(f'{name}_std',  torch.tensor(1.0))
        self.register_buffer('n_updates', torch.tensor(0))

        # Detection head: [z_v, z_a, C_i_norm, A_norm, TIL_norm]
        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),               nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @torch.no_grad()
    def _update_stats(self, ci, asym, til):
        m = 0.99 if self.n_updates > 10 else 0.5
        for val, name in [(ci, 'ci'), (asym, 'asym'), (til, 'til')]:
            mu  = val.mean()
            std = val.std().clamp(min=1e-6)
            setattr(self, f'{name}_mean', m * getattr(self, f'{name}_mean') + (1-m)*mu)
            setattr(self, f'{name}_std',  m * getattr(self, f'{name}_std')  + (1-m)*std)
        self.n_updates += 1

    def _norm(self, val, name):
        return ((val - getattr(self, f'{name}_mean')) /
                getattr(self, f'{name}_std').clamp(min=1e-6)).unsqueeze(1)

    def forward(self, frames, audio, update_stats=True):
        # ── Encode ────────────────────────────────────────────────────────
        vis_out = self.visual_encoder(frames)
        z_v     = vis_out['video_embedding']   # [B, D]
        h_t     = vis_out['frame_features']    # [B, T, D]
        z_a     = self.audio_encoder(audio)['audio_embedding']  # [B, D]

        # ── Clip-level predictions ─────────────────────────────────────────
        pred_v = self.T_av(z_a)   # [B, D]  audio → visual
        pred_a = self.T_va(z_v)   # [B, D]  visual → audio

        # ── DIAM: directional asymmetry ────────────────────────────────────
        eps_f = torch.norm(pred_v - z_v, dim=1)   # [B]  forward error
        eps_b = torch.norm(pred_a - z_a, dim=1)   # [B]  backward error
        C_i   = eps_f + eps_b                     # [B]  magnitude
        A     = eps_f - eps_b                     # [B]  signed asymmetry

        # ── TIL: per-frame error against T_av prediction ───────────────────
        # pred_v [B, D] broadcast over T frames
        frame_err = torch.norm(
            pred_v.unsqueeze(1) - h_t, dim=2
        )                                         # [B, T]
        til, til_alpha = self.til_scorer(frame_err)   # [B], [B, T]

        # ── Normalise for detection head ───────────────────────────────────
        if self.training and update_stats:
            self._update_stats(C_i.detach(), A.detach(), til.detach())

        ci_n   = self._norm(C_i, 'ci')    # [B, 1]
        asym_n = self._norm(A,   'asym')  # [B, 1]
        til_n  = self._norm(til, 'til')   # [B, 1]

        # ── Detection head ─────────────────────────────────────────────────
        det_in     = torch.cat([z_v, z_a, ci_n, asym_n, til_n], dim=1)  # [B, 2D+3]
        det_logits = self.detection_head(det_in).squeeze(-1)              # [B]
        det_probs  = torch.sigmoid(det_logits)

        return {
            'z_v':       z_v,
            'z_a':       z_a,
            'pred_v':    pred_v,
            'pred_a':    pred_a,
            'C_i':       C_i,
            'A':         A,
            'til':       til,
            'til_alpha': til_alpha,
            'frame_err': frame_err,
            'det_logits': det_logits,
            'det_probs':  det_probs,
        }


# ══════════════════════════════════════════════════════════════════════════
# 2.  Loss
# ══════════════════════════════════════════════════════════════════════════

def focal_loss(logits, labels, alpha=0.75, gamma=2.0):
    labels = labels.float()
    bce    = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt     = torch.exp(-bce)
    aw     = torch.where(labels == 1, alpha, 1 - alpha)
    return (aw * (1 - pt)**gamma * bce).mean()


def compute_loss(outputs, labels, lambda_pred=0.1):
    """
    L = L_det  +  λ * (L_av + L_va)

    Predictive losses on REAL samples only — same as original CMPI.
    DIAM and TIL are differentiable and receive gradients through det_logits.
    """
    L_det = focal_loss(outputs['det_logits'], labels)

    real_mask = labels == 0
    if real_mask.sum() > 0:
        L_av = F.mse_loss(outputs['pred_v'][real_mask],
                          outputs['z_v'][real_mask].detach())
        L_va = F.mse_loss(outputs['pred_a'][real_mask],
                          outputs['z_a'][real_mask].detach())
    else:
        L_av = L_va = torch.tensor(0.0, device=labels.device)

    L_pred = L_av + L_va
    total  = L_det + lambda_pred * L_pred
    return total, {
        'total': total.item(), 'det': L_det.item(),
        'L_av': L_av.item(),  'L_va': L_va.item(),
        'L_pred': L_pred.item(),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3.  Evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, lambda_pred=0.1):
    model.eval()
    probs_l, ci_l, asym_l, til_l, label_l = [], [], [], [], []
    tot_loss = det_loss = pred_loss = 0.0
    n = 0

    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label'].to(device)
        out    = model(frames, audio, update_stats=False)
        loss, ld = compute_loss(out, labels, lambda_pred)

        tot_loss  += ld['total']
        det_loss  += ld['det']
        pred_loss += ld['L_pred']
        n += 1

        probs_l.append(out['det_probs'].cpu().numpy())
        ci_l.append(out['C_i'].cpu().numpy())
        asym_l.append(out['A'].cpu().numpy())
        til_l.append(out['til'].cpu().numpy())
        label_l.append(labels.cpu().numpy())

    probs  = np.concatenate(probs_l)
    ci     = np.concatenate(ci_l)
    asym   = np.concatenate(asym_l)
    til    = np.concatenate(til_l)
    labels = np.concatenate(label_l)
    preds  = (probs >= 0.5).astype(int)

    auroc     = roc_auc_score(labels, probs)
    bal_acc   = balanced_accuracy_score(labels, preds)
    ci_auroc  = roc_auc_score(labels, ci)
    til_auroc = roc_auc_score(labels, til)

    return {
        'loss':         tot_loss  / n,
        'det_loss':     det_loss  / n,
        'pred_loss':    pred_loss / n,
        'auroc':        auroc,
        'bal_acc':      bal_acc,
        'ci_auroc':     ci_auroc,
        'til_auroc':    til_auroc,
        'ci_mean_real': float(ci[labels == 0].mean()),
        'ci_mean_fake': float(ci[labels == 1].mean()),
        'a_mean_real':  float(asym[labels == 0].mean()),
        'a_mean_fake':  float(asym[labels == 1].mean()),
    }


# ══════════════════════════════════════════════════════════════════════════
# 4.  Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════

def save_checkpoint(state, out_dir, is_best):
    p = Path(out_dir) / 'last_checkpoint.pt'
    torch.save(state, str(p))
    if is_best:
        shutil.copyfile(str(p), str(Path(out_dir) / 'best_model.pt'))
        print("  ★ New best model saved")


def load_amdd_encoders(model, ckpt_path, device):
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model_state_dict',
                     ckpt.get('state_dict', ckpt)))
    for enc, prefix in [(model.visual_encoder, 'visual_encoder.'),
                        (model.audio_encoder,  'audio_encoder.')]:
        sub = {k[len(prefix):]: v for k, v in state.items()
               if k.startswith(prefix)}
        if sub:
            enc.load_state_dict(sub, strict=False)
            print(f"  {prefix}: {len(sub)} keys loaded")
        else:
            print(f"  {prefix}: not found — keeping ImageNet init")


# ══════════════════════════════════════════════════════════════════════════
# 5.  Training loop
# ══════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, optimizer, scheduler, device,
                    lambda_pred, scaler, epoch, log_interval=20):
    model.train()
    tot = det = pred = 0.0
    n = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        frames = batch['frames'].to(device, non_blocking=True)
        audio  = batch['audio'].to(device,  non_blocking=True)
        labels = batch['label'].to(device,  non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            out  = model(frames, audio)
            loss, ld = compute_loss(out, labels, lambda_pred)

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

        tot  += ld['total']
        det  += ld['det']
        pred += ld['L_pred']
        n    += 1

        if (i + 1) % log_interval == 0:
            print(f"    Ep{epoch} [{i+1}/{len(loader)}] "
                  f"loss={ld['total']:.4f} det={ld['det']:.4f} "
                  f"pred={ld['L_pred']:.4f}  ({time.time()-t0:.1f}s)")

    if scheduler:
        scheduler.step()
    return {'loss': tot/n, 'det_loss': det/n, 'pred_loss': pred/n}


# ══════════════════════════════════════════════════════════════════════════
# 6.  Plots
# ══════════════════════════════════════════════════════════════════════════

def plot_curves(history, out_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    pairs = [
        ('train_loss',     'val_loss',     'Total Loss',       'Loss'),
        ('train_det_loss', 'val_det_loss', 'Detection Loss',   'Focal Loss'),
        ('train_pred_loss','val_pred_loss','Predictive Loss',  'MSE'),
        ('train_auroc',    'val_auroc',    'AUROC (det)',      'AUROC'),
        ('train_ci_auroc', 'val_ci_auroc', 'AUROC (C_i)',      'AUROC'),
        ('train_til_auroc','val_til_auroc','AUROC (TIL)',      'AUROC'),
    ]
    for ax, (tr, va, title, ylabel) in zip(axes.flat, pairs):
        ax.plot(epochs, history[tr], label='Train', color='steelblue', lw=2)
        if history.get(va):
            ax.plot(epochs, history[va], label='Val', color='darkorange', lw=2)
        ax.set_title(title, fontsize=11); ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel); ax.legend(fontsize=9); ax.grid(alpha=0.3)

    if history.get('val_auroc'):
        best = int(np.argmax(history['val_auroc'])) + 1
        for ax in axes.flat:
            ax.axvline(best, color='red', ls='--', lw=1, alpha=0.5)

    fig.suptitle('CMPI+DIAM+TIL Training Curves', fontsize=14)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / 'training_curves.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata_csv',     required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--output_dir',       default='results/diam')
    p.add_argument('--amdd_checkpoint',  default=None)
    p.add_argument('--resume',           default=None)
    p.add_argument('--embed_dim',        type=int,   default=512)
    p.add_argument('--hidden_dim',       type=int,   default=1024)
    p.add_argument('--dropout',          type=float, default=0.1)
    p.add_argument('--epochs',           type=int,   default=50)
    p.add_argument('--batch_size',       type=int,   default=8)
    p.add_argument('--num_workers',      type=int,   default=4)
    p.add_argument('--lr',               type=float, default=1e-4)
    p.add_argument('--lr_predictor',     type=float, default=1e-3)
    p.add_argument('--weight_decay',     type=float, default=1e-4)
    p.add_argument('--lambda_pred',      type=float, default=0.1)
    p.add_argument('--warmup_epochs',    type=int,   default=5)
    p.add_argument('--seed',             type=int,   default=42)
    p.add_argument('--log_interval',     type=int,   default=20)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  CMPI + DIAM + TIL  |  Device: {device}  |  AMP: {use_amp}")
    print(f"  Output: {out_dir}")
    print(f"{'═'*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────
    train_ds = DeepfakeDataset(args.metadata_csv, 'train', args.preprocessed_dir)
    val_ds   = DeepfakeDataset(args.metadata_csv, 'val',   args.preprocessed_dir)
    test_ds  = DeepfakeDataset(args.metadata_csv, 'test',  args.preprocessed_dir)

    sampler = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = CMPIDIAMModel(
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)

    # History
    hist_keys = [
        'train_loss','train_det_loss','train_pred_loss',
        'train_auroc','train_bal_acc','train_ci_auroc','train_til_auroc',
        'val_loss',  'val_det_loss',  'val_pred_loss',
        'val_auroc', 'val_bal_acc',   'val_ci_auroc',  'val_til_auroc',
    ]
    history = {k: [] for k in hist_keys}
    start_epoch = 1
    best_auroc  = 0.0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        start_epoch = ckpt['epoch'] + 1
        best_auroc  = ckpt.get('best_val_auroc', 0.0)
        history     = ckpt.get('history', history)
        print(f"Resumed from epoch {start_epoch}, best AUROC={best_auroc:.4f}")
    elif args.amdd_checkpoint:
        print(f"Warm-starting encoders from: {args.amdd_checkpoint}")
        load_amdd_encoders(model, args.amdd_checkpoint, device)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Total params: {total_params:.1f}M")

    # ── Optimiser (differential LR) ───────────────────────────────────────
    enc_params  = (list(model.visual_encoder.parameters()) +
                   list(model.audio_encoder.parameters()))
    head_params = (list(model.T_av.parameters()) +
                   list(model.T_va.parameters()) +
                   list(model.til_scorer.parameters()) +
                   list(model.detection_head.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': enc_params,  'lr': args.lr,           'name': 'encoders'},
        {'params': head_params, 'lr': args.lr_predictor, 'name': 'predictors'},
    ], weight_decay=args.weight_decay)

    def lr_lambda(ep):
        if ep < args.warmup_epochs:
            return (ep + 1) / args.warmup_epochs
        prog = (ep - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * prog))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──────────────────────────────────────────────────────────
    print(f"\n[Training] {args.epochs} epochs  |  batch={args.batch_size}  "
          f"|  lr_enc={args.lr}  |  lr_pred={args.lr_predictor}  "
          f"|  λ={args.lambda_pred}\n")

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        print(f"── Epoch {epoch}/{args.epochs} "
              f"(lr_enc={optimizer.param_groups[0]['lr']:.2e}, "
              f"lr_pred={optimizer.param_groups[1]['lr']:.2e}) ──")

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            device, args.lambda_pred, scaler, epoch, args.log_interval)

        train_eval = evaluate(model, train_loader, device, args.lambda_pred)
        val_eval   = evaluate(model, val_loader,   device, args.lambda_pred)

        for k in ['loss','det_loss','pred_loss']:
            history[f'train_{k}'].append(train_metrics[k])
        for k in ['auroc','bal_acc','ci_auroc','til_auroc']:
            history[f'train_{k}'].append(train_eval[k])
        for k in ['loss','det_loss','pred_loss','auroc','bal_acc','ci_auroc','til_auroc']:
            history[f'val_{k}'].append(val_eval[k])

        elapsed = time.time() - t0
        print(f"  Train | AUROC={train_eval['auroc']:.4f}  "
              f"C_i-AUROC={train_eval['ci_auroc']:.4f}  "
              f"TIL-AUROC={train_eval['til_auroc']:.4f}")
        print(f"  Val   | AUROC={val_eval['auroc']:.4f}  "
              f"C_i-AUROC={val_eval['ci_auroc']:.4f}  "
              f"TIL-AUROC={val_eval['til_auroc']:.4f}  "
              f"BalAcc={val_eval['bal_acc']:.4f}  ({elapsed:.1f}s)")
        print(f"  DIAM  | C_i real={val_eval['ci_mean_real']:.2f}  "
              f"fake={val_eval['ci_mean_fake']:.2f}  "
              f"A real={val_eval['a_mean_real']:.2f}  "
              f"fake={val_eval['a_mean_fake']:.2f}")

        is_best = val_eval['auroc'] > best_auroc
        if is_best:
            best_auroc = val_eval['auroc']

        save_checkpoint({
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_val_auroc': best_auroc,
            'val_metrics': val_eval,
            'history': history,
            'args': vars(args),
        }, out_dir, is_best)

        with open(str(out_dir / 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)
        print()

    # ── Final test evaluation ─────────────────────────────────────────────
    print("═" * 60)
    ckpt = torch.load(str(out_dir / 'best_model.pt'),
                      map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    test_eval = evaluate(model, test_loader, device, args.lambda_pred)

    print(f"\n  ── TEST SET RESULTS (CMPI+DIAM+TIL) ──")
    print(f"  AUROC (det):   {test_eval['auroc']:.4f}")
    print(f"  Balanced Acc:  {test_eval['bal_acc']:.4f}")
    print(f"  C_i AUROC:     {test_eval['ci_auroc']:.4f}")
    print(f"  TIL AUROC:     {test_eval['til_auroc']:.4f}")
    print(f"  C_i real/fake: {test_eval['ci_mean_real']:.2f} / "
          f"{test_eval['ci_mean_fake']:.2f}")
    print("═" * 60)

    final = {'best_val_auroc': best_auroc, 'test': test_eval, 'args': vars(args)}
    with open(str(out_dir / 'final_results.json'), 'w') as f:
        json.dump(final, f, indent=2)

    plot_curves(history, out_dir)
    print(f"\n[Done] All outputs in: {out_dir}")


if __name__ == '__main__':
    main()
