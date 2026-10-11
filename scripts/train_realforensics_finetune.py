"""
RealForensics — supervised fine-tuning on FakeAVCeleb
========================================================
Wasim Ahmad, BIT — September 2026

ISWA revision (ISWA-D-26-01461), Reviewer #1 pt.2: complete the Table 1
comparison by fine-tuning RealForensics on FakeAVCeleb, matching the same
"reproduced by us using the official implementation" ($\S$) convention
already used for the FAME and Hashmi et al. baselines in that table —
rather than reporting the zero-shot FF++ checkpoint number, which isn't
methodologically comparable to the other (in-domain-trained) rows.

Architecture (CSN-101 backbone + cosine-similarity linear head) is
reconstructed standalone in eval_realforensics_baseline.py / here, since the
official repo's full training scaffolding (Hydra + PyTorch Lightning 1.3.8 +
PyTorch 1.9/CUDA 11.1) cannot run on this machine's GPU at all (see that
script's docstring). This fine-tunes from their released FF++ checkpoint —
a reasonable and defensible initialization, analogous to how CMPI itself
warm-starts from the AMDD checkpoint rather than from scratch.

Simplifications vs. the official stage-2 training recipe (documented for
the rebuttal): plain BCE-with-logits supervised fine-tuning, no concurrent
self-supervised LRW loss term, no logit-adjustment prior, standard Adam
instead of AdamP. This is a supervised-fine-tuning baseline, not a
byte-for-byte reproduction of their training procedure — same spirit as
how FAME/Hashmi were already "reproduced using the official implementation"
rather than trained pixel-identically to their original papers.

Usage
-----
python train_realforensics_finetune.py \\
    --init_checkpoint  realforensics_ff.pth \\
    --metadata_csv     ../data/metadata.csv \\
    --preprocessed_dir ../data/preprocessed \\
    --output_dir       results/realforensics_finetuned
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_realforensics_baseline import create_csn, MeanLinear, IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parent.parent
AMDD_DATA = ROOT / "data"
if str(AMDD_DATA) not in sys.path:
    sys.path.insert(0, str(AMDD_DATA))
from dataset import DeepfakeDataset


def frames_to_csn_clip(frames_bt_c_h_w, out_size=112):
    """frames: [B, T, C, H, W] already ImageNet-normalized at 224x224 (AMDD's
    own VAL/TRAIN_TRANSFORMS). Un-normalize, resize to 112, re-normalize
    (identical stats, so this is really just a resize) -> [B, C, T, 112, 112]."""
    B, T, C, H, W = frames_bt_c_h_w.shape
    x = frames_bt_c_h_w.view(B * T, C, H, W)
    x = F.interpolate(x, size=(out_size, out_size), mode='bilinear', align_corners=False)
    x = x.view(B, T, C, out_size, out_size).permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]
    return x


def load_pretrained(backbone, df_head, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    wb = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
    wh = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
    mb, ub = backbone.load_state_dict(wb, strict=False)
    mh, uh = df_head.load_state_dict(wh, strict=False)
    print(f"  Loaded init: backbone missing={len(mb)} unexpected={len(ub)}  "
          f"df_head missing={len(mh)} unexpected={len(uh)}")


@torch.no_grad()
def evaluate(backbone, df_head, loader, device):
    backbone.eval(); df_head.eval()
    probs_l, labels_l = [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        clip = frames_to_csn_clip(frames)
        feat = backbone(clip)
        logit = df_head(feat).squeeze(-1)
        probs_l.append(torch.sigmoid(logit).cpu().numpy())
        labels_l.append(batch['label'].numpy())
    probs = np.concatenate(probs_l)
    labels = np.concatenate(labels_l)
    auroc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float('nan')
    bal_acc = float(balanced_accuracy_score(labels, (probs >= 0.5).astype(int)))
    return auroc, bal_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init_checkpoint', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--output_dir', default='results/realforensics_finetuned')
    p.add_argument('--epochs', type=int, default=15)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--lr_backbone', type=float, default=1e-5)
    p.add_argument('--lr_head', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}\n  RealForensics fine-tuning on FakeAVCeleb\n  Device: {device}\n{'='*70}")

    backbone = create_csn(model_depth=101)
    df_head = MeanLinear(in_dim=2048, out_dim=1, norm_linear=True, scale=64)
    load_pretrained(backbone, df_head, args.init_checkpoint)
    backbone, df_head = backbone.to(device), df_head.to(device)

    train_ds = DeepfakeDataset(args.metadata_csv, 'train', args.preprocessed_dir)
    val_ds = DeepfakeDataset(args.metadata_csv, 'val', args.preprocessed_dir)
    test_ds = DeepfakeDataset(args.metadata_csv, 'test', args.preprocessed_dir)
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    opt = torch.optim.Adam([
        {'params': backbone.parameters(), 'lr': args.lr_backbone},
        {'params': df_head.parameters(), 'lr': args.lr_head},
    ], weight_decay=1e-4)

    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        backbone.train(); df_head.train()
        losses = []
        for batch in train_loader:
            frames = batch['frames'].to(device)
            labels = batch['label'].float().to(device)
            clip = frames_to_csn_clip(frames)

            opt.zero_grad()
            feat = backbone(clip)
            logit = df_head(feat).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, labels)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        val_auroc, val_bal_acc = evaluate(backbone, df_head, val_loader, device)
        train_loss = float(np.mean(losses))
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_AUROC={val_auroc:.4f}  val_BalAcc={val_bal_acc:.4f}")
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_auroc': val_auroc, 'val_bal_acc': val_bal_acc})

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {
                'backbone': {k: v.cpu().clone() for k, v in backbone.state_dict().items()},
                'df_head': {k: v.cpu().clone() for k, v in df_head.state_dict().items()},
                'epoch': epoch, 'val_auroc': val_auroc,
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"  Early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
                break

    torch.save(best_state, out_dir / 'best_model.pt')
    backbone.load_state_dict(best_state['backbone'])
    df_head.load_state_dict(best_state['df_head'])

    test_auroc, test_bal_acc = evaluate(backbone, df_head, test_loader, device)
    print(f"\nBest val epoch: {best_state['epoch']} (val AUROC={best_state['val_auroc']:.4f})")
    print(f"Test set (FakeAVCeleb, n={len(test_ds)}): AUROC={test_auroc:.4f}  BalAcc={test_bal_acc:.4f}")

    result = {
        'best_val_epoch': best_state['epoch'], 'best_val_auroc': best_state['val_auroc'],
        'test_auroc': test_auroc, 'test_bal_acc': test_bal_acc,
        'history': history,
        'note': ('RealForensics (CSN-101 backbone + cosine-linear head), initialized from the '
                 'officially released FF++-trained checkpoint (realforensics_ff.pth), fine-tuned '
                 'supervised (plain BCE) on FakeAVCeleb train split, matching the same '
                 '"reproduced by us using the official implementation" convention as the '
                 'FAME/Hashmi et al. rows in Table 1. Architecture reconstructed standalone since '
                 'the original repo\'s training scaffolding requires PyTorch 1.9/CUDA 11.1, which '
                 'has no kernels for this machine\'s GPU.'),
    }
    with open(out_dir / 'realforensics_finetuned_results.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_dir / 'realforensics_finetuned_results.json'}")


if __name__ == '__main__':
    main()
