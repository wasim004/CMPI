"""
RealForensics baseline — zero-shot eval on FakeAVCeleb (CMPI's own in-domain test set)
=========================================================================================
Wasim Ahmad, BIT — September 2026

ISWA revision (ISWA-D-26-01461), Reviewer #1 pt.2 / Reviewer #2 (implicit):
add a RealForensics comparison to the baseline table.

Full reproduction of RealForensics (Haliassos et al., CVPR 2022) is not
feasible here: stage 1 is a DDP self-supervised pretraining run on LRW
across 8 GPUs the authors say they "have not tested on a single GPU," and
the full repo pins PyTorch 1.9/CUDA 11.1 (pytorch-lightning 1.3.8, hydra
1.1.0) which has no compiled kernels for this machine's GPU (RTX 5060 Ti,
Blackwell) — the published environment literally cannot run on this
hardware.

Instead: their officially released "cross-dataset generalisation" checkpoint
(realforensics_ff.pth, trained on FaceForensics++) is loaded directly. The
stage-2 detection model architecture (CSN-101 video backbone -> temporal
mean-pool -> cosine-similarity linear head, see stage2/models/backbones/
csn.py + stage2/models/linear.py in the official repo) is pure torch.nn
code with no version-specific API, so it's reconstructed standalone here
and the checkpoint's state_dict is loaded directly (split by "backbone."/
"df_head." prefix, exactly as the official stage2/eval.py does) — bypassing
the old Hydra/Lightning/CUDA-11.1 scaffolding entirely.

Evaluation protocol simplification (documented explicitly for the rebuttal):
this reuses CMPI's own already-preprocessed FakeAVCeleb test frames
(AMDD/data/preprocessed, the same 220 test videos / 16 evenly-sampled
face-cropped frames used for every other row in CMPI's comparison table),
resized 224->112, rather than replicating RealForensics' own bespoke
RetinaFace+FAN crop pipeline and multi-clip video-level aggregation. CSN's
architecture is clip-length-agnostic (temporal global average pooling), so
feeding 16 frames instead of their native 25 is architecturally valid. This
gives every baseline in the table the identical input pixels, which is
arguably a fairer uniform protocol than reproducing each method's own
distinct eval pipeline — but it is a real deviation from RealForensics'
reported numbers and must be described as such, not presented as a
faithful reproduction.

Usage
-----
python eval_realforensics_baseline.py \\
    --checkpoint /tmp/realforensics_ff.pth \\
    --metadata_csv "../AMDD/data/metadata.csv" \\
    --preprocessed_dir "../AMDD/data/preprocessed" \\
    --output_dir results/realforensics_baseline
"""

import sys, json, math, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ── Model code copied verbatim from the official repo (pure nn.Module, no
#    version-specific Hydra/Lightning dependency) ───────────────────────────
try:
    from pytorchvideo.models.resnet import Net, create_bottleneck_block, create_res_stage
    from pytorchvideo.models.stem import create_res_basic_stem
except ImportError as e:
    raise SystemExit(
        "pytorchvideo is required (pip install pytorchvideo). "
        f"Original error: {e}")


class LambdaModule(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def create_csn(
    *, input_channel=3, model_depth=101, norm=nn.BatchNorm3d, activation=nn.ReLU,
    stem_dim_out=64, stem_conv_kernel_size=(3, 7, 7), stem_conv_stride=(1, 2, 2),
    stem_pool=None, stem_pool_kernel_size=(1, 3, 3), stem_pool_stride=(1, 2, 2),
    stage_conv_a_kernel_size=(1, 1, 1), stage_conv_b_kernel_size=(3, 3, 3),
    stage_conv_b_width_per_group=1, stage_spatial_stride=(1, 2, 2, 2),
    stage_temporal_stride=(1, 2, 2, 2), bottleneck=create_bottleneck_block,
    bottleneck_ratio=4,
):
    _MODEL_STAGE_DEPTH = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}
    stage_depths = _MODEL_STAGE_DEPTH[model_depth]

    blocks = []
    stem = create_res_basic_stem(
        in_channels=input_channel, out_channels=stem_dim_out,
        conv_kernel_size=stem_conv_kernel_size, conv_stride=stem_conv_stride,
        conv_padding=[s // 2 for s in stem_conv_kernel_size],
        pool=stem_pool, pool_kernel_size=stem_pool_kernel_size,
        pool_stride=stem_pool_stride,
        pool_padding=[s // 2 for s in stem_pool_kernel_size],
        norm=norm, activation=activation,
    )
    blocks.append(stem)

    stage_dim_in = stem_dim_out
    stage_dim_out = stage_dim_in * 4
    for idx in range(len(stage_depths)):
        stage_dim_inner = stage_dim_out // bottleneck_ratio
        depth = stage_depths[idx]
        stage_conv_b_stride = (stage_temporal_stride[idx], stage_spatial_stride[idx], stage_spatial_stride[idx])
        stage = create_res_stage(
            depth=depth, dim_in=stage_dim_in, dim_inner=stage_dim_inner, dim_out=stage_dim_out,
            bottleneck=bottleneck, conv_a_kernel_size=stage_conv_a_kernel_size, conv_a_stride=(1, 1, 1),
            conv_a_padding=[s // 2 for s in stage_conv_a_kernel_size],
            conv_b_kernel_size=stage_conv_b_kernel_size, conv_b_stride=stage_conv_b_stride,
            conv_b_padding=[s // 2 for s in stage_conv_b_kernel_size],
            conv_b_num_groups=(stage_dim_inner // stage_conv_b_width_per_group),
            conv_b_dilation=(1, 1, 1), norm=norm, activation=activation,
        )
        blocks.append(stage)
        stage_dim_in = stage_dim_out
        stage_dim_out = stage_dim_out * 2

    blocks.append(LambdaModule(lambda x: x.mean(dim=(3, 4))))
    return Net(blocks=nn.ModuleList(blocks))


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


class MeanLinear(nn.Module):
    def __init__(self, in_dim, out_dim=1, norm_linear=True, scale=64):
        super().__init__()
        if norm_linear:
            self.weight = nn.Parameter(torch.Tensor(out_dim, in_dim))
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.scale = scale
            self.linear = Lambda(lambda x: F.linear(F.normalize(x), F.normalize(self.weight)) * self.scale)
        else:
            self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = x.mean(-1)
        return self.linear(x)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)


def preprocess_clip(frames_np, out_size=112):
    """frames_np: [T,H,W,C] uint8 (already face-cropped by CMPI's own pipeline).
    Returns [1, 3, T, out_size, out_size] float tensor, ImageNet-normalized."""
    t = torch.from_numpy(frames_np).permute(0, 3, 1, 2).float() / 255.0  # [T,C,H,W]
    t = F.interpolate(t, size=(out_size, out_size), mode='bilinear', align_corners=False)
    t = t.permute(1, 0, 2, 3).unsqueeze(0)  # [1, C, T, H, W]
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--split_col', default='split')
    p.add_argument('--split_val', default='test')
    p.add_argument('--source_prefix', default='favceleb')
    p.add_argument('--output_dir', default='results/realforensics_baseline')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    backbone = create_csn(model_depth=101)
    df_head = MeanLinear(in_dim=2048, out_dim=1, norm_linear=True, scale=64)

    state_dict = torch.load(args.checkpoint, map_location='cpu')
    if 'state_dict' in state_dict:  # in case it's a full Lightning checkpoint, not a bare state_dict
        state_dict = state_dict['state_dict']
    weights_backbone = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("backbone")}
    weights_df_head  = {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("df_head")}
    print(f"  backbone keys: {len(weights_backbone)}  df_head keys: {len(weights_df_head)}")
    missing_b, unexpected_b = backbone.load_state_dict(weights_backbone, strict=False)
    missing_h, unexpected_h = df_head.load_state_dict(weights_df_head, strict=False)
    print(f"  backbone missing={len(missing_b)} unexpected={len(unexpected_b)}")
    print(f"  df_head  missing={len(missing_h)} unexpected={len(unexpected_h)}")
    if missing_b or unexpected_b:
        print(f"    backbone missing keys (first 5): {missing_b[:5]}")
        print(f"    backbone unexpected keys (first 5): {unexpected_b[:5]}")

    backbone = backbone.to(device).eval()
    df_head = df_head.to(device).eval()

    df = pd.read_csv(args.metadata_csv)
    if 'success' in df.columns:
        df = df[df['success'] == True]
    df = df[df[args.split_col] == args.split_val]
    if args.source_prefix:
        df = df[df['source'].str.startswith(args.source_prefix)]
    df = df.reset_index(drop=True)
    print(f"  Eval set: {len(df)} samples (real={int((df.label==0).sum())}, fake={int((df.label==1).sum())})")

    frames_dir = Path(args.preprocessed_dir) / 'frames'

    logits_l, labels_l, uids_l = [], [], []
    with torch.no_grad():
        for i, row in df.iterrows():
            uid = str(row['uid']).zfill(6)
            fpath = frames_dir / f"{uid}.npy"
            if not fpath.exists():
                continue
            frames_np = np.load(str(fpath))
            clip = preprocess_clip(frames_np).to(device)
            feat = backbone(clip)             # [1, 2048, T']
            logit = df_head(feat).squeeze()   # scalar
            logits_l.append(float(logit.item()))
            labels_l.append(int(row['label']))
            uids_l.append(uid)
            if (i + 1) % 50 == 0:
                print(f"    {i+1}/{len(df)} done")

    logits = np.array(logits_l)
    labels = np.array(labels_l)
    probs = 1 / (1 + np.exp(-logits))

    auroc = float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else float('nan')
    bal_acc = float(balanced_accuracy_score(labels, (probs >= 0.5).astype(int)))

    print(f"\nRealForensics (FF++-pretrained checkpoint) zero-shot on FakeAVCeleb test set:")
    print(f"  n={len(labels)}  AUROC={auroc:.4f}  BalAcc={bal_acc:.4f}")
    print(f"  logit range: [{logits.min():.3f}, {logits.max():.3f}]  mean={logits.mean():.3f}")

    result = {
        'n': len(labels), 'auroc': auroc, 'bal_acc': bal_acc,
        'logit_mean': float(logits.mean()), 'logit_std': float(logits.std()),
        'note': ('Officially released realforensics_ff.pth (trained on FaceForensics++) '
                 'evaluated zero-shot on CMPI\'s FakeAVCeleb test set, using CMPI\'s own '
                 'preprocessed 16-frame face crops (resized 224->112) rather than '
                 'RealForensics\' own crop+multi-clip-aggregation pipeline — see script '
                 'docstring for full justification.'),
    }
    with open(str(out_dir / 'realforensics_favceleb_results.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_dir / 'realforensics_favceleb_results.json'}")


if __name__ == '__main__':
    main()
