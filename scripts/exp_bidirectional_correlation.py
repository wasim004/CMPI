"""
CMPI+DIAM — Forward/Backward Error Correlation Analysis
==========================================================
Wasim Ahmad, BIT — August 2026

Directly addresses the TIFS reviewer concern that forward (eps_f) and
backward (eps_b) prediction errors might be "two correlated versions of
the same signal" rather than a qualitatively new bidirectional signal.

We compute per-sample eps_f, eps_b on the FakeAVCeleb held-out test set
and report the Pearson correlation r(eps_f, eps_b) separately for real
and fake samples. The DIAM hypothesis predicts:
  - High correlation for REAL samples (both errors driven by the same
    residual encoder/predictor noise around the shared causal coupling)
  - Substantially lower correlation for FAKE samples (the coupling break
    is directional, so eps_f and eps_b diverge independently depending on
    which modality was manipulated)
A large real-vs-fake correlation gap is direct quantitative evidence that
eps_f and eps_b are NOT redundant, and that their signed difference (A)
carries information the sum (C_i) discards.

Usage
-----
python exp_bidirectional_correlation.py \\
    --checkpoint  results/diam/best_model.pt \\
    --metadata_csv     ../data/metadata.csv \\
    --preprocessed_dir ../data/preprocessed \\
    --output_dir  results/further_experiments
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crossdataset_diam import CMPIDIAMModel

ROOT = Path(__file__).resolve().parent.parent
AMDD_DATA = ROOT / "data"
if str(AMDD_DATA) not in sys.path:
    sys.path.insert(0, str(AMDD_DATA))
from dataset import DeepfakeDataset


@torch.no_grad()
def collect_eps(model, loader, device):
    eps_f_l, eps_b_l, label_l, gen_l = [], [], [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        vis_out = model.visual_encoder(frames)
        z_v = vis_out['video_embedding']
        z_a = model.audio_encoder(audio)['audio_embedding']
        pred_v = model.T_av(z_a)
        pred_a = model.T_va(z_v)
        eps_f = torch.norm(pred_v - z_v, dim=1)
        eps_b = torch.norm(pred_a - z_a, dim=1)
        eps_f_l.append(eps_f.cpu().numpy())
        eps_b_l.append(eps_b.cpu().numpy())
        label_l.append(batch['label'].numpy())
        if 'generator_id' in batch:
            gen_l.append(batch['generator_id'].numpy())
    return (np.concatenate(eps_f_l), np.concatenate(eps_b_l),
            np.concatenate(label_l),
            np.concatenate(gen_l) if gen_l else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--output_dir', default='results/further_experiments')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    test_ds = DeepfakeDataset(args.metadata_csv, 'test', args.preprocessed_dir)
    loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True)

    eps_f, eps_b, labels, gen_id = collect_eps(model, loader, device)

    results = {}
    for name, mask in [('real', labels == 0), ('fake', labels == 1), ('all', np.ones_like(labels, dtype=bool))]:
        if mask.sum() < 2:
            continue
        r, pval = sstats.pearsonr(eps_f[mask], eps_b[mask])
        results[name] = {
            'n': int(mask.sum()),
            'pearson_r': float(r),
            'p_value': float(pval),
            'eps_f_mean': float(eps_f[mask].mean()),
            'eps_b_mean': float(eps_b[mask].mean()),
        }
        print(f"  {name:5s} (n={mask.sum():3d}): r(eps_f, eps_b) = {r:.4f}  (p={pval:.2e})  "
              f"eps_f={eps_f[mask].mean():.2f}  eps_b={eps_b[mask].mean():.2f}")

    # ── Per-generator breakdown (in-domain modality-attribution check) ──────
    gen_names = {0: 'real', 1: 'faceswap', 2: 'wav2lip', 3: 'fsgan'}
    if gen_id is not None:
        print("\n  Per-generator breakdown (FakeAVCeleb test set):")
        gen_results = {}
        A = eps_f - eps_b
        for gid in sorted(np.unique(gen_id)):
            m = gen_id == gid
            name = gen_names.get(int(gid), str(gid))
            gen_results[name] = {
                'n': int(m.sum()),
                'eps_f_mean': float(eps_f[m].mean()),
                'eps_b_mean': float(eps_b[m].mean()),
                'A_mean': float(A[m].mean()),
                'A_std':  float(A[m].std()),
            }
            print(f"    {name:10s} (n={m.sum():3d}): eps_f={eps_f[m].mean():6.2f}  "
                  f"eps_b={eps_b[m].mean():6.2f}  A={A[m].mean():7.2f} +/- {A[m].std():.2f}")
        results['per_generator'] = gen_results

    out_file = out_dir / 'bidirectional_correlation.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    np.save(out_dir / 'eps_f.npy', eps_f)
    np.save(out_dir / 'eps_b.npy', eps_b)
    np.save(out_dir / 'labels.npy', labels)
    if gen_id is not None:
        np.save(out_dir / 'generator_id.npy', gen_id)

    print(f"\n  Saved: {out_file}")


if __name__ == '__main__':
    main()
