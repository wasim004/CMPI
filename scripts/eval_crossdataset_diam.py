"""
CMPI+DIAM Cross-Dataset Evaluation — LAV-DF & DFDM
=====================================================
Wasim Ahmad, BIT — August 2026

Loads the trained CMPI+DIAM checkpoint and evaluates zero-shot on LAV-DF
and DFDM. Reports AUROC for det_probs, C_i, A (asymmetry), and TIL score.

On LAV-DF the metadata has modify_video / modify_audio columns — we report
results per manipulation type so we can show DIAM's asymmetry correctly
identifies which modality was manipulated.

Usage
-----
python eval_crossdataset_diam.py \\
    --checkpoint  results/diam/best_model.pt \\
    --data_root   data/crossdataset \\
    --output_dir  results/crossdataset_diam
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ── Imports ───────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
AMDD_MODELS = ROOT / "AMDD" / "models"
AMDD_DATA   = ROOT / "AMDD" / "data"

for _p in [str(AMDD_MODELS), str(AMDD_DATA)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoders import VisualEncoder, AudioEncoder
import torchvision.transforms as T

# ── Paste the model classes (must be identical to training) ───────────────

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
    def forward(self, x): return self.net(x)


class TemporalAttentionScorer(nn.Module):
    def __init__(self, in_dim=1, hidden=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 1),
        )
    def forward(self, err_seq):
        x     = err_seq.unsqueeze(-1)
        alpha = torch.softmax(self.fc(x).squeeze(-1), dim=1)
        return (alpha * err_seq).sum(dim=1), alpha


class CMPIDIAMModel(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=1024, dropout=0.1, n_frames=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.visual_encoder = VisualEncoder(embed_dim=embed_dim, dropout=dropout)
        self.audio_encoder  = AudioEncoder(embed_dim=embed_dim,  dropout=dropout)
        self.T_av = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.T_va = CrossModalPredictor(embed_dim, hidden_dim, embed_dim, dropout)
        self.til_scorer = TemporalAttentionScorer()
        for name in ('ci', 'asym', 'til'):
            self.register_buffer(f'{name}_mean', torch.tensor(1.0))
            self.register_buffer(f'{name}_std',  torch.tensor(1.0))
        self.register_buffer('n_updates', torch.tensor(0))
        self.detection_head = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 64),               nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def _norm(self, val, name):
        return ((val - getattr(self, f'{name}_mean')) /
                getattr(self, f'{name}_std').clamp(min=1e-6)).unsqueeze(1)

    def forward(self, frames, audio):
        vis_out = self.visual_encoder(frames)
        z_v     = vis_out['video_embedding']
        h_t     = vis_out['frame_features']
        z_a     = self.audio_encoder(audio)['audio_embedding']
        pred_v  = self.T_av(z_a)
        pred_a  = self.T_va(z_v)
        eps_f   = torch.norm(pred_v - z_v, dim=1)
        eps_b   = torch.norm(pred_a - z_a, dim=1)
        C_i     = eps_f + eps_b
        A       = eps_f - eps_b
        frame_err = torch.norm(pred_v.unsqueeze(1) - h_t, dim=2)
        til, til_alpha = self.til_scorer(frame_err)
        ci_n   = self._norm(C_i, 'ci')
        asym_n = self._norm(A,   'asym')
        til_n  = self._norm(til, 'til')
        det_in     = torch.cat([z_v, z_a, ci_n, asym_n, til_n], dim=1)
        det_logits = self.detection_head(det_in).squeeze(-1)
        return {
            'C_i': C_i, 'A': A, 'til': til,
            'frame_err': frame_err, 'til_alpha': til_alpha,
            'det_probs': torch.sigmoid(det_logits),
        }


# ══════════════════════════════════════════════════════════════════════════
# Cross-dataset dataset class
# ══════════════════════════════════════════════════════════════════════════

VAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CrossDataset(Dataset):
    """Loads preprocessed cross-dataset samples."""
    def __init__(self, metadata_csv, preprocessed_dir):
        self.df  = pd.read_csv(metadata_csv)
        self.df  = self.df[self.df['success'] == True].reset_index(drop=True)
        self.frames_dir = Path(preprocessed_dir) / 'frames'
        self.audio_dir  = Path(preprocessed_dir) / 'audio'
        print(f"  CrossDataset: {len(self.df)} samples "
              f"(real={int((self.df.label==0).sum())}, "
              f"fake={int((self.df.label==1).sum())})")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = str(row['uid']).zfill(6)

        frames = np.load(str(self.frames_dir / f"{uid}.npy"))
        ft = []
        for i in range(frames.shape[0]):
            ft.append(VAL_TRANSFORM(frames[i]))
        frames_t = torch.stack(ft, dim=0)

        mel_path = self.audio_dir / f"{uid}.npy"
        if mel_path.exists():
            mel = np.load(str(mel_path)).astype(np.float32)
            mn, mx = mel.min(), mel.max()
            if mx > mn:
                mel = 2.0 * (mel - mn) / (mx - mn) - 1.0
        else:
            mel = np.zeros((128, 128), dtype=np.float32)

        item = {
            'frames': frames_t,
            'audio':  torch.tensor(mel, dtype=torch.float32),
            'label':  torch.tensor(int(row['label']), dtype=torch.long),
            'uid':    uid,
        }
        for col in ('modify_video', 'modify_audio'):
            if col in row:
                item[col] = bool(row[col])
        return item


# ══════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    probs_l, ci_l, asym_l, til_l, label_l = [], [], [], [], []
    mod_v_l, mod_a_l = [], []

    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        labels = batch['label']
        out    = model(frames, audio)

        probs_l.append(out['det_probs'].cpu().numpy())
        ci_l.append(out['C_i'].cpu().numpy())
        asym_l.append(out['A'].cpu().numpy())
        til_l.append(out['til'].cpu().numpy())
        label_l.append(labels.numpy())
        if 'modify_video' in batch:
            mod_v_l.append(np.array([int(x) for x in batch['modify_video']]))
            mod_a_l.append(np.array([int(x) for x in batch['modify_audio']]))

    probs  = np.concatenate(probs_l)
    ci     = np.concatenate(ci_l)
    asym   = np.concatenate(asym_l)
    til    = np.concatenate(til_l)
    labels = np.concatenate(label_l)

    def safe_auroc(y, s):
        if len(np.unique(y)) < 2:
            return float('nan')
        return float(roc_auc_score(y, s))

    results = {
        'n_real': int((labels==0).sum()),
        'n_fake': int((labels==1).sum()),
        'auroc_det':  safe_auroc(labels, probs),
        'auroc_ci':   safe_auroc(labels, ci),
        'auroc_til':  safe_auroc(labels, til),
        'bal_acc':    float(balanced_accuracy_score(labels, (probs>=0.5).astype(int))),
        'ci_real':    float(ci[labels==0].mean()),
        'ci_fake':    float(ci[labels==1].mean()),
        'a_real':     float(asym[labels==0].mean()),
        'a_fake':     float(asym[labels==1].mean()),
    }

    if mod_v_l:
        mv = np.concatenate(mod_v_l)
        ma = np.concatenate(mod_a_l)
        fakes = labels == 1

        # Video-only fake: mv=1, ma=0
        mask_vonly = fakes & (mv==1) & (ma==0)
        if mask_vonly.sum() > 0:
            y2 = np.where(mask_vonly, 1, 0)
            real_mask = labels == 0
            idx = np.where(real_mask | mask_vonly)[0]
            results['auroc_det_video_fake'] = safe_auroc(
                labels[idx], probs[idx])
            # For video fakes, A should be positive (ε_f > ε_b)
            results['a_mean_video_fake'] = float(asym[mask_vonly].mean())

        # Audio-only fake: mv=0, ma=1
        mask_aonly = fakes & (mv==0) & (ma==1)
        if mask_aonly.sum() > 0:
            idx = np.where((labels==0) | mask_aonly)[0]
            results['auroc_det_audio_fake'] = safe_auroc(
                labels[idx], probs[idx])
            # For audio fakes, A should be negative (ε_b > ε_f)
            results['a_mean_audio_fake'] = float(asym[mask_aonly].mean())

        # Both fake: mv=1, ma=1
        mask_both = fakes & (mv==1) & (ma==1)
        if mask_both.sum() > 0:
            idx = np.where((labels==0) | mask_both)[0]
            results['auroc_det_both_fake'] = safe_auroc(
                labels[idx], probs[idx])
            results['a_mean_both_fake'] = float(asym[mask_both].mean())

    return results


def print_results(name, r):
    print(f"\n  ── {name} ──")
    print(f"  Samples: real={r['n_real']}, fake={r['n_fake']}")
    print(f"  AUROC (det):    {r['auroc_det']:.4f}")
    print(f"  AUROC (C_i):    {r['auroc_ci']:.4f}")
    print(f"  AUROC (TIL):    {r['auroc_til']:.4f}")
    print(f"  Balanced Acc:   {r['bal_acc']:.4f}")
    print(f"  C_i real/fake:  {r['ci_real']:.2f} / {r['ci_fake']:.2f}")
    print(f"  A   real/fake:  {r['a_real']:.2f} / {r['a_fake']:.2f}")
    for k in ['auroc_det_video_fake', 'auroc_det_audio_fake', 'auroc_det_both_fake']:
        if k in r:
            print(f"  {k}: {r[k]:.4f}")
    for k in ['a_mean_video_fake', 'a_mean_audio_fake', 'a_mean_both_fake']:
        if k in r:
            print(f"  {k}: {r[k]:.4f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root',  required=True,
                   help='Root dir with lavdf/, dfdm/, dftimit/ subfolders')
    p.add_argument('--output_dir', default='results/crossdataset_diam')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers',type=int, default=4)
    p.add_argument('--datasets',   nargs='+', default=['lavdf', 'dfdm'])
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*60}")
    print(f"  CMPI+DIAM Cross-Dataset Evaluation")
    print(f"  Device: {device}  |  Checkpoint: {args.checkpoint}")
    print(f"{'═'*60}\n")

    # ── Load model ────────────────────────────────────────────────────────
    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"  Loaded checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"best val AUROC={ckpt.get('best_val_auroc', '?')})\n")

    all_results = {}
    data_root   = Path(args.data_root)

    for ds_name in args.datasets:
        ds_dir = data_root / ds_name
        meta   = ds_dir / 'metadata.csv'
        prep   = ds_dir / 'preprocessed'
        if not meta.exists() or not prep.exists():
            print(f"  Skipping {ds_name} — metadata or preprocessed dir missing")
            continue

        print(f"  Evaluating on: {ds_name.upper()}")
        ds     = CrossDataset(str(meta), str(prep))
        loader = DataLoader(ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        r      = run_eval(model, loader, device)
        print_results(ds_name.upper(), r)
        all_results[ds_name] = r

    # ── Save ──────────────────────────────────────────────────────────────
    out_file = out_dir / 'crossdataset_results.json'
    with open(str(out_file), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {out_file}")

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    print(f"  {'Dataset':<12} {'AUROC(det)':>10} {'AUROC(TIL)':>10} {'BalAcc':>8}")
    for name, r in all_results.items():
        print(f"  {name:<12} {r['auroc_det']:>10.4f} {r['auroc_til']:>10.4f} "
              f"{r['bal_acc']:>8.4f}")
    print(f"{'═'*60}\n")


if __name__ == '__main__':
    main()
