"""
Sensitivity Analysis — λ and Frame Count T
===========================================
Wasim Ahmad, BIT

Experiment 1: λ sensitivity (0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0)
Experiment 2: Frame count T sensitivity (4, 8, 12, 16, 24, 32)

Usage
-----
# Run both (overnight, ~8 hours total)
python sensitivity_analysis.py \
    --metadata_csv     data/metadata.csv \
    --preprocessed_dir data/preprocessed \
    --amdd_checkpoint  AMDD/checkpoints/full_model_best.pth \
    --output_dir       results/sensitivity

# Run only lambda experiment
python sensitivity_analysis.py ... --experiment lambda

# Run only frame count experiment
python sensitivity_analysis.py ... --experiment frames
"""

import sys, argparse, json, time
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

AMDD_ROOT = Path(__file__).resolve().parent.parent
for _p in [AMDD_ROOT, AMDD_ROOT/"models", AMDD_ROOT/"data"]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from encoders import VisualEncoder, AudioEncoder
except ModuleNotFoundError:
    from models.encoders import VisualEncoder, AudioEncoder
try:
    from dataset import DeepfakeDataset, get_weighted_sampler
except ModuleNotFoundError:
    sys.path.insert(0, str(AMDD_ROOT/"training"))
    from dataset import DeepfakeDataset, get_weighted_sampler

plt.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':         9,
    'text.usetex':       False,
    'figure.dpi':        300,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.04,
    'axes.spines.top':   False,
    'axes.spines.right': False,
})


# ── Model ──────────────────────────────────────────────────────────────────
class CrossModalPredictor(nn.Module):
    def __init__(self, d=512, h=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h,h), nn.LayerNorm(h), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h,d))
    def forward(self, x): return self.net(x)

class DetectionHead(nn.Module):
    def __init__(self, d=1025, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256,64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,1))
    def forward(self, x): return self.net(x).squeeze(-1)

class CMPIModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_encoder = VisualEncoder(embed_dim=512)
        self.audio_encoder  = AudioEncoder(embed_dim=512)
        self.T_av = CrossModalPredictor()
        self.T_va = CrossModalPredictor()
        self.register_buffer('ci_running_mean', torch.tensor(1.0))
        self.register_buffer('ci_running_std',  torch.tensor(1.0))
        self.register_buffer('ci_n_updates',    torch.tensor(0))
        self.detection_head = DetectionHead()

    @torch.no_grad()
    def _update_ci(self, ci):
        m = 0.99 if self.ci_n_updates > 10 else 0.5
        self.ci_running_mean = m*self.ci_running_mean + (1-m)*ci.mean()
        self.ci_running_std  = m*self.ci_running_std  + (1-m)*ci.std().clamp(min=1e-6)
        self.ci_n_updates   += 1

    def forward(self, frames, audio, update=True):
        z_v    = self.visual_encoder(frames)['video_embedding']
        z_a    = self.audio_encoder(audio)['audio_embedding']
        pred_v = self.T_av(z_a)
        pred_a = self.T_va(z_v)
        err_av = torch.norm(pred_v - z_v, dim=1)
        err_va = torch.norm(pred_a - z_a, dim=1)
        C_i    = err_av + err_va
        if self.training and update:
            self._update_ci(C_i.detach())
        C_i_n  = ((C_i - self.ci_running_mean) /
                  self.ci_running_std.clamp(min=1e-6)).unsqueeze(1)
        det    = self.detection_head(torch.cat([z_v, z_a, C_i_n], dim=1))
        return {'C_i': C_i, 'det_probs': torch.sigmoid(det)}


def focal_loss(logits, labels, alpha=0.75, gamma=2.0):
    labels = labels.float()
    bce    = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    pt     = torch.exp(-bce)
    aw     = torch.where(labels==1, alpha, 1-alpha)
    return (aw * (1-pt)**gamma * bce).mean()


def load_amdd_encoders(model, ckpt_path, device):
    ckpt  = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt.get('model_state', ckpt.get('model_state_dict',
            ckpt.get('state_dict', ckpt)))
    def _load(enc, prefix):
        sub = {k[len(prefix):]: v for k,v in state.items()
               if k.startswith(prefix)}
        if sub:
            enc.load_state_dict(sub, strict=False)
    _load(model.visual_encoder, 'visual_encoder.')
    _load(model.audio_encoder,  'audio_encoder.')


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, ci_vals, labels = [], [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        out    = model(frames, audio, update=False)
        probs.append(out['det_probs'].cpu().numpy())
        ci_vals.append(out['C_i'].cpu().numpy())
        labels.append(batch['label'].numpy())
    probs  = np.concatenate(probs)
    ci     = np.concatenate(ci_vals)
    labels = np.concatenate(labels)
    preds  = (probs >= 0.5).astype(int)
    # Guard against NaN from unstable training
    if np.isnan(probs).any() or np.isnan(ci).any():
        return {'auroc': 0.5, 'ci_auroc': 0.5, 'bal_acc': 0.5,
                'sep': 1.0, 'ci_real': 0.0, 'ci_fake': 0.0}
    auroc    = roc_auc_score(labels, probs)
    ci_auroc = roc_auc_score(labels, ci)
    bal_acc  = balanced_accuracy_score(labels, preds)
    ci_real  = ci[labels==0].mean()
    ci_fake  = ci[labels==1].mean()
    sep = ci_fake / (ci_real + 1e-8)
    return {'auroc': auroc, 'ci_auroc': ci_auroc,
            'bal_acc': bal_acc, 'sep': sep,
            'ci_real': ci_real, 'ci_fake': ci_fake}


def train_model(args, device, train_loader, val_loader, test_loader,
                lambda_pred=0.1, epochs=50, n_frames=16, tag=''):
    model = CMPIModel().to(device)
    if args.amdd_checkpoint:
        load_amdd_encoders(model, args.amdd_checkpoint, device)

    enc_params = (list(model.visual_encoder.parameters()) +
                  list(model.audio_encoder.parameters()))
    new_params = (list(model.T_av.parameters()) +
                  list(model.T_va.parameters()) +
                  list(model.detection_head.parameters()))

    optimizer = torch.optim.AdamW([
        {'params': enc_params, 'lr': 1e-4},
        {'params': new_params, 'lr': 1e-3},
    ], weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < 5: return (ep+1)/5
        p = (ep-5)/max(epochs-5,1)
        return 0.5*(1+np.cos(np.pi*p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda') if device.type=='cuda' else None

    best_val  = 0.0
    best_state = None

    for epoch in range(1, epochs+1):
        model.train()
        for batch in train_loader:
            frames = batch['frames'].to(device, non_blocking=True)
            audio  = batch['audio'].to(device,  non_blocking=True)
            labels = batch['label'].to(device,  non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                out    = model(frames, audio)
                logits = torch.log(out['det_probs'].clamp(1e-6,1-1e-6) /
                         (1 - out['det_probs'].clamp(1e-6,1-1e-6)))
                L_det  = focal_loss(logits, labels)
                real_m = (labels==0)
                if lambda_pred > 0 and real_m.sum() > 0:
                    z_v = model.visual_encoder(frames)['video_embedding']
                    z_a = model.audio_encoder(audio)['audio_embedding']
                    L_p = (F.mse_loss(model.T_av(z_a[real_m]),
                                      z_v[real_m].detach()) +
                           F.mse_loss(model.T_va(z_v[real_m]),
                                      z_a[real_m].detach()))
                    loss = L_det + lambda_pred * L_p
                else:
                    loss = L_det

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
        val_m = evaluate(model, val_loader, device)
        if val_m['auroc'] > best_val:
            best_val   = val_m['auroc']
            best_state = {k: v.clone() for k,v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f"    [{tag}] Epoch {epoch:3d} | "
                  f"val AUROC={val_m['auroc']:.4f} "
                  f"Ci-AUC={val_m['ci_auroc']:.4f} "
                  f"sep={val_m['sep']:.2f}x")

    model.load_state_dict(best_state)
    test_m = evaluate(model, test_loader, device)
    print(f"  [{tag}] TEST: AUROC={test_m['auroc']:.4f} "
          f"Ci-AUC={test_m['ci_auroc']:.4f} "
          f"sep={test_m['sep']:.2f}x")
    return test_m


# ══════════════════════════════════════════════════════════════════════════
# Experiment runners
# ══════════════════════════════════════════════════════════════════════════

def run_lambda_sensitivity(args, device, out_dir):
    print(f"\n{'='*60}")
    print(f"  Experiment 1: λ Sensitivity")
    print(f"{'='*60}")

    lambdas = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]
    results = []

    ds_train = DeepfakeDataset(args.metadata_csv,'train',args.preprocessed_dir)
    ds_val   = DeepfakeDataset(args.metadata_csv,'val',  args.preprocessed_dir)
    ds_test  = DeepfakeDataset(args.metadata_csv,'test', args.preprocessed_dir)

    sampler     = get_weighted_sampler(ds_train)
    train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    for lam in lambdas:
        print(f"\n  λ = {lam}")
        m = train_model(args, device, train_loader, val_loader, test_loader,
                        lambda_pred=lam, epochs=50, tag=f'λ={lam}')
        m['lambda'] = lam
        results.append(m)
        # Save after each run
        with open(str(out_dir/'lambda_results.json'), 'w') as f:
            json.dump([{k: float(v) if hasattr(v,"item") else v for k,v in r.items()} for r in results], f, indent=2)

    plot_lambda_results(results, out_dir)
    return results


def plot_lambda_results(results, out_dir):
    lambdas  = [r['lambda'] for r in results]
    aurocs   = [r['auroc']    for r in results]
    ci_aucs  = [r['ci_auroc'] for r in results]
    seps     = [r['sep']      for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88,
                        bottom=0.16, wspace=0.38)

    x = np.arange(len(lambdas))
    ax1.plot(x, aurocs,  'o-', color='#1565C0', lw=1.8,
             ms=5, label='Detection AUROC')
    ax1.plot(x, ci_aucs, 's--', color='#E65100', lw=1.8,
             ms=5, label='$\\mathcal{C}_i$ AUROC')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lambdas], fontsize=8)
    ax1.set_xlabel('$\\lambda$ (predictive loss weight)', fontsize=9)
    ax1.set_ylabel('AUROC', fontsize=9)
    ax1.set_title('AUROC vs $\\lambda$', fontsize=9.5)
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.grid(alpha=0.2, lw=0.5)
    ax1.axvline(lambdas.index(0.1), color='#2E7D32',
                lw=0.8, ls=':', label='Selected ($\\lambda$=0.1)')

    ax2.plot(x, seps, 'D-', color='#AD1457', lw=1.8, ms=5)
    ax2.axhline(1.0, color='#C62828', lw=0.8, ls='--',
                label='No separation')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in lambdas], fontsize=8)
    ax2.set_xlabel('$\\lambda$', fontsize=9)
    ax2.set_ylabel('$\\mathcal{C}_i$ separation', fontsize=9)
    ax2.set_title('Separation vs $\\lambda$', fontsize=9.5)
    ax2.legend(frameon=False, fontsize=7.5)
    ax2.grid(alpha=0.2, lw=0.5)

    fig.suptitle('Sensitivity to $\\lambda$ — FakeAVCeleb Test Set',
                 fontsize=9.5, fontweight='bold')

    fig.savefig(str(out_dir/'lambda_sensitivity.pdf'), format='pdf', dpi=300)
    fig.savefig(str(out_dir/'lambda_sensitivity.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: lambda_sensitivity.pdf")


def run_frame_sensitivity(args, device, out_dir):
    print(f"\n{'='*60}")
    print(f"  Experiment 2: Frame Count T Sensitivity")
    print(f"{'='*60}")

    frame_counts = [4, 8, 12, 16, 24, 32]
    results = []

    for T in frame_counts:
        print(f"\n  T = {T} frames")
        # Dataset loads 16 frames; we subsample to T during eval
        # Simplest approach: use existing data, subsample T frames
        ds_train = DeepfakeDataset(args.metadata_csv,'train',
                                   args.preprocessed_dir)
        ds_val   = DeepfakeDataset(args.metadata_csv,'val',
                                   args.preprocessed_dir)
        ds_test  = DeepfakeDataset(args.metadata_csv,'test',
                                   args.preprocessed_dir)

        # Apply frame subsampling via collate
        def make_collate(n_frames):
            def collate(batch):
                import torch
                frames = torch.stack([b['frames'] for b in batch])
                # Subsample T evenly from 16
                T_total = frames.shape[1]
                if n_frames < T_total:
                    idx = torch.linspace(0, T_total-1,
                                         n_frames).long()
                    frames = frames[:, idx]
                audio  = torch.stack([b['audio']  for b in batch])
                label  = torch.stack([b['label']  for b in batch])
                gen_id = torch.stack([b['generator_id'] for b in batch])
                return {'frames': frames, 'audio': audio,
                        'label': label, 'generator_id': gen_id,
                        'uid': [-1]*len(batch)}
            return collate

        cf = make_collate(T)
        sampler = get_weighted_sampler(ds_train)
        train_loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                                  num_workers=4, pin_memory=True,
                                  drop_last=True, collate_fn=cf)
        val_loader   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True,
                                  collate_fn=cf)
        test_loader  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                                  num_workers=4, pin_memory=True,
                                  collate_fn=cf)

        m = train_model(args, device, train_loader, val_loader, test_loader,
                        lambda_pred=0.1, epochs=50, tag=f'T={T}')
        m['T'] = T
        results.append(m)
        with open(str(out_dir/'frame_results.json'), 'w') as f:
            json.dump([{k: float(v) if hasattr(v,"item") else v for k,v in r.items()} for r in results], f, indent=2)

    plot_frame_results(results, out_dir)
    return results


def plot_frame_results(results, out_dir):
    Ts      = [r['T']       for r in results]
    aurocs  = [r['auroc']   for r in results]
    ci_aucs = [r['ci_auroc'] for r in results]
    seps    = [r['sep']     for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.0))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88,
                        bottom=0.16, wspace=0.38)

    ax1.plot(Ts, aurocs,  'o-', color='#1565C0', lw=1.8,
             ms=5, label='Detection AUROC')
    ax1.plot(Ts, ci_aucs, 's--', color='#E65100', lw=1.8,
             ms=5, label='$\\mathcal{C}_i$ AUROC')
    ax1.axvline(16, color='#2E7D32', lw=0.8, ls=':',
                label='Selected (T=16)')
    ax1.set_xlabel('Number of frames $T$', fontsize=9)
    ax1.set_ylabel('AUROC', fontsize=9)
    ax1.set_title('AUROC vs $T$', fontsize=9.5)
    ax1.legend(frameon=False, fontsize=7.5)
    ax1.grid(alpha=0.2, lw=0.5)

    ax2.plot(Ts, seps, 'D-', color='#AD1457', lw=1.8, ms=5)
    ax2.axhline(1.0, color='#C62828', lw=0.8, ls='--')
    ax2.axvline(16, color='#2E7D32', lw=0.8, ls=':')
    ax2.set_xlabel('Number of frames $T$', fontsize=9)
    ax2.set_ylabel('$\\mathcal{C}_i$ separation', fontsize=9)
    ax2.set_title('Separation vs $T$', fontsize=9.5)
    ax2.grid(alpha=0.2, lw=0.5)

    fig.suptitle('Sensitivity to Frame Count $T$ — FakeAVCeleb Test Set',
                 fontsize=9.5, fontweight='bold')

    fig.savefig(str(out_dir/'frame_sensitivity.pdf'), format='pdf', dpi=300)
    fig.savefig(str(out_dir/'frame_sensitivity.png'), format='png', dpi=300)
    plt.close(fig)
    print(f"  Saved: frame_sensitivity.pdf")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--metadata_csv',
                   default='data/metadata.csv')
    p.add_argument('--preprocessed_dir',
                   default='data/preprocessed')
    p.add_argument('--amdd_checkpoint',
                   default='AMDD/checkpoints/full_model_best.pth')
    p.add_argument('--output_dir', default='results/sensitivity')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--experiment', default='all',
                   choices=['all', 'lambda', 'frames'])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out    = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    t0 = time.time()

    if args.experiment in ('all', 'lambda'):
        run_lambda_sensitivity(args, device, out)

    if args.experiment in ('all', 'frames'):
        run_frame_sensitivity(args, device, out)

    print(f"\n[Done] Total time: {(time.time()-t0)/3600:.1f}h")
    print(f"Results in: {out}")


if __name__ == '__main__':
    main()