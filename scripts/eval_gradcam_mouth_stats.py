"""
CMPI+DIAM — GradCAM++ quantitative attention statistics
=========================================================
Wasim Ahmad, BIT — September 2026

ISWA revision (ISWA-D-26-01461):
  Reviewer #1 pt.8: convert the qualitative GradCAM++ examples into a
    quantitative statistic — % of attention mass falling in the mouth region,
    real vs. fake.
  Reviewer #2 pt.9: add GradCAM++ visualizations on LAV-DF (cross-domain),
    not just FakeAVCeleb (in-domain) — to see whether attention shifts.

Approach
--------
GradCAM++ (Chattopadhay et al. 2018) on the last conv block of the visual
encoder (ResNet50 layer4, i.e. `visual_encoder.backbone[7]`, output
[T, 2048, 7, 7] since the encoder processes all T frames of one clip as a
T-batch). Target scalar = the detection logit (pre-sigmoid), computed via a
manual forward pass that mirrors CMPIDIAMModel.forward() but stops before
the sigmoid so gradients are well-behaved regardless of how saturated the
probability is.

Mouth region per frame comes from face_alignment 2D landmarks (68-point
iBug scheme, indices 48-67) run on the same 224x224 face crop the model
sees — a convex hull over those points, dilated by a small margin. Frames
where landmark detection fails are skipped (NaN), not zero-filled.

Requires the `torch` conda env (has face-alignment installed; the project's
default `torch2` env does not):
    python eval_gradcam_mouth_stats.py ...

Usage
-----
python eval_gradcam_mouth_stats.py \\
    --checkpoint    results/diam/best_model.pt \\
    --metadata_csv  /path/to/metadata.csv \\
    --preprocessed_dir /path/to/preprocessed \\
    --dataset_name  favceleb \\
    --split_col     split --split_val test \\
    --source_prefix favceleb \\
    --n_per_class   999999 \\
    --output_dir    results/gradcam_mouth/favceleb
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torchvision.transforms as T

ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT / "models"), str(ROOT / "data"), str(Path(__file__).resolve().parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoders import VisualEncoder, AudioEncoder
from eval_crossdataset_diam import CMPIDIAMModel

VAL_TRANSFORM = T.Compose([
    T.ToPILImage(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MOUTH_IDX = list(range(48, 68))  # iBug 68-point mouth landmarks


def gradcam_pp(activations, gradients):
    """activations, gradients: [T, C, H, W] for one clip. Returns [T, H, W] cams."""
    grads_2 = gradients ** 2
    grads_3 = gradients ** 3
    sum_act = activations.sum(dim=(2, 3), keepdim=True)
    eps = 1e-8
    alpha = grads_2 / (2 * grads_2 + sum_act * grads_3 + eps)
    weights = (alpha * F.relu(gradients)).sum(dim=(2, 3))  # [T, C]
    cams = F.relu((weights.unsqueeze(-1).unsqueeze(-1) * activations).sum(dim=1))  # [T, H, W]
    return cams


def get_mouth_mask(fa_model, frame_uint8, size=224, dilate_px=8):
    preds = fa_model.get_landmarks(frame_uint8)
    if preds is None or len(preds) == 0:
        return None
    lm = preds[0][MOUTH_IDX].astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    hull = cv2.convexHull(lm)
    cv2.fillConvexPoly(mask, hull, 1)
    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask.astype(bool)


class ManualForward:
    """Mirrors CMPIDIAMModel.forward() but exposes pre-sigmoid logits and lets
    us hook layer4 activations/gradients per clip."""
    def __init__(self, model):
        self.model = model

    def __call__(self, frames, audio):
        m = self.model
        vis_out = m.visual_encoder(frames)
        z_v, h_t = vis_out['video_embedding'], vis_out['frame_features']
        z_a = m.audio_encoder(audio)['audio_embedding']
        pred_v, pred_a = m.T_av(z_a), m.T_va(z_v)
        eps_f = torch.norm(pred_v - z_v, dim=1)
        eps_b = torch.norm(pred_a - z_a, dim=1)
        C_i, A = eps_f + eps_b, eps_f - eps_b
        frame_err = torch.norm(pred_v.unsqueeze(1) - h_t, dim=2)
        til, _ = m.til_scorer(frame_err)
        ci_n, asym_n, til_n = m._norm(C_i, 'ci'), m._norm(A, 'asym'), m._norm(til, 'til')
        det_in = torch.cat([z_v, z_a, ci_n, asym_n, til_n], dim=1)
        logits = m.detection_head(det_in).squeeze(-1)
        return logits


def process_sample(model, manual_fwd, fa_model, frames_np, mel_np, device, act_holder):
    """frames_np: [T,H,W,C] uint8. Returns (mean_mass_in_mouth, n_valid_frames)."""
    T_ = frames_np.shape[0]
    ft = torch.stack([VAL_TRANSFORM(frames_np[i]) for i in range(T_)], dim=0).unsqueeze(0).to(device)
    ft.requires_grad_(False)
    mel = torch.tensor(mel_np, dtype=torch.float32).unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)
    logits = manual_fwd(ft, mel)
    logits.sum().backward()

    act = act_holder['act']          # [T, 2048, 7, 7]
    grad = act.grad                  # [T, 2048, 7, 7]
    cams = gradcam_pp(act.detach(), grad.detach())  # [T, 7, 7]
    cams = F.interpolate(cams.unsqueeze(1), size=(224, 224), mode='bilinear', align_corners=False).squeeze(1)
    cams = cams.cpu().numpy()

    masses, areas = [], []
    for t in range(T_):
        cam = cams[t]
        s = cam.sum()
        if s <= 1e-8:
            continue
        cam_norm = cam / s
        mask = get_mouth_mask(fa_model, frames_np[t])
        if mask is None:
            continue
        masses.append(float(cam_norm[mask].sum()))
        areas.append(float(mask.mean()))
    if not masses:
        return None, None, 0
    return float(np.mean(masses)), float(np.mean(areas)), len(masses)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--split_col', default=None)
    p.add_argument('--split_val', default=None)
    p.add_argument('--source_prefix', default=None)
    p.add_argument('--n_per_class', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', required=True)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import face_alignment
    # CPU only: this env's torch build hits an nvrtc/JIT-fusion mismatch on
    # GPU for face_alignment's network specifically (unrelated to the main
    # CMPIDIAMModel GradCAM computation below, which stays on `device`).
    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cpu', flip_input=False)

    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(True)  # need grads w.r.t. activations, not weights, but keep graph intact
    manual_fwd = ManualForward(model)

    act_holder = {}
    def fwd_hook(module, inp, out):
        out.retain_grad()
        act_holder['act'] = out
    handle = model.visual_encoder.backbone[7].register_forward_hook(fwd_hook)

    df = pd.read_csv(args.metadata_csv)
    df = df[df['success'] == True] if 'success' in df.columns else df
    if args.split_col:
        df = df[df[args.split_col] == args.split_val]
    if args.source_prefix and 'source' in df.columns:
        df = df[df['source'].str.startswith(args.source_prefix)]
    df = df.reset_index(drop=True)

    frames_dir = Path(args.preprocessed_dir) / 'frames'
    audio_dir = Path(args.preprocessed_dir) / 'audio'

    rng = np.random.RandomState(args.seed)
    results = {}
    for label, name in [(0, 'real'), (1, 'fake')]:
        sub = df[df['label'] == label]
        if len(sub) > args.n_per_class:
            sub = sub.sample(n=args.n_per_class, random_state=args.seed)
        print(f"[{args.dataset_name}] {name}: evaluating {len(sub)} samples")

        sample_means, sample_areas = [], []
        n_skipped = 0
        for _, row in sub.iterrows():
            uid = str(row['uid']).zfill(6)
            fpath = frames_dir / f"{uid}.npy"
            apath = audio_dir / f"{uid}.npy"
            if not fpath.exists() or not apath.exists():
                n_skipped += 1
                continue
            frames_np = np.load(str(fpath))
            mel = np.load(str(apath)).astype(np.float32)
            mn, mx = mel.min(), mel.max()
            if mx > mn:
                mel = 2.0 * (mel - mn) / (mx - mn) - 1.0
            mass, area, n_valid = process_sample(model, manual_fwd, fa_model, frames_np, mel, device, act_holder)
            if mass is None:
                n_skipped += 1
                continue
            sample_means.append(mass)
            sample_areas.append(area)

        results[name] = {
            'n_samples': len(sample_means),
            'n_skipped': n_skipped,
            'mouth_attention_mass_mean': float(np.mean(sample_means)) if sample_means else float('nan'),
            'mouth_attention_mass_std': float(np.std(sample_means)) if sample_means else float('nan'),
            'mouth_attention_mass_median': float(np.median(sample_means)) if sample_means else float('nan'),
            'mouth_area_fraction_mean': float(np.mean(sample_areas)) if sample_areas else float('nan'),
        }
        print(f"  -> mean mouth-attention mass = {results[name]['mouth_attention_mass_mean']:.4f} "
              f"(mouth area fraction = {results[name]['mouth_area_fraction_mean']:.4f}, "
              f"n={results[name]['n_samples']}, skipped={n_skipped})")

    handle.remove()

    out_file = out_dir / f'gradcam_mouth_stats_{args.dataset_name}.json'
    with open(str(out_file), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
