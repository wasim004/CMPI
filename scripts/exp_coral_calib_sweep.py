"""
CMPI+DIAM — CORAL calibration-set-size sweep
================================================
Wasim Ahmad, BIT — September 2026

ISWA revision (ISWA-D-26-01461), Reviewer #2 pt.6: CORAL alignment WORSENS
DFDM performance in the original experiment (results/further_experiments/
coral_alignment_results.json, calib_frac=0.2) — analyze whether small
calibration-set sizes are the cause, by sweeping calib size.

Unlike calling exp_coral_alignment.py repeatedly with different
--calib_frac (which reuses the same train/test split so calib and eval set
sizes are coupled — larger calib always means smaller eval, confounding the
comparison), this script fixes ONE held-out eval set per dataset (last
`--fixed_eval_frac` of the permuted samples) and draws calibration sets of
increasing size from the *remaining* pool only. That isolates the effect of
calibration-set size from eval-set-size noise. Source (FakeAVCeleb) and
target embeddings are cached once and reused across all sweep points to
avoid re-running the encoders per calib size.

Usage
-----
python exp_coral_calib_sweep.py \\
    --checkpoint       results/diam/best_model.pt \\
    --metadata_csv     ../data/metadata.csv \\
    --preprocessed_dir ../data/preprocessed \\
    --data_root        data/crossdataset \\
    --output_dir       results/coral_calib_sweep
"""

import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crossdataset_diam import CMPIDIAMModel, CrossDataset
from exp_coral_alignment import (
    fit_coral, apply_affine, forward_from_embeddings,
    cache_source_embeddings, cache_target_full, safe_auroc,
)

ROOT = Path(__file__).resolve().parent.parent
AMDD_DATA = ROOT / "data"
if str(AMDD_DATA) not in sys.path:
    sys.path.insert(0, str(AMDD_DATA))
from dataset import DeepfakeDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--metadata_csv', required=True)
    p.add_argument('--preprocessed_dir', required=True)
    p.add_argument('--data_root', required=True)
    p.add_argument('--output_dir', default='results/coral_calib_sweep')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--datasets', nargs='+', default=['lavdf', 'dfdm'])
    p.add_argument('--fixed_eval_frac', type=float, default=0.5,
                    help='Fraction of each dataset permanently held out as eval, never used for calibration.')
    p.add_argument('--calib_sizes', nargs='+', type=int,
                    default=[10, 20, 50, 100, 200, 400])
    p.add_argument('--shrinkage', type=float, default=0.3)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n  CORAL Calibration-Set-Size Sweep\n  Device: {device}\n{'='*70}\n")

    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    baseline_stats = {
        'ci':  (model.ci_mean.item(),   model.ci_std.item()),
        'a':   (model.asym_mean.item(), model.asym_std.item()),
        'til': (model.til_mean.item(),  model.til_std.item()),
    }

    print("  Caching FakeAVCeleb (source) embeddings...")
    src_ds = DeepfakeDataset(args.metadata_csv, 'train', args.preprocessed_dir)
    src_loader = DataLoader(src_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    src_zv, src_za = cache_source_embeddings(model, src_loader, device)
    print(f"  Source: {src_zv.shape[0]} samples\n")

    all_results = {}
    data_root = Path(args.data_root)

    for ds_name in args.datasets:
        ds_dir = data_root / ds_name
        meta, prep = ds_dir / 'metadata.csv', ds_dir / 'preprocessed'
        if not meta.exists() or not prep.exists():
            print(f"  Skipping {ds_name} — data missing")
            continue

        print(f"  --- {ds_name.upper()} ---")
        ds = CrossDataset(str(meta), str(prep))
        loader = DataLoader(ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
        cache = cache_target_full(model, loader, device)
        n = len(cache['labels'])

        rng = np.random.default_rng(42)
        idx = rng.permutation(n)
        n_eval = max(20, int(round(n * args.fixed_eval_frac)))
        eval_idx = idx[:n_eval]
        pool_idx = idx[n_eval:]  # calibration candidates only, fixed eval never touched
        print(f"  n={n}  fixed eval={n_eval}  calib pool={len(pool_idx)}")

        zv_eval, za_eval, ht_eval = cache['z_v'][eval_idx], cache['z_a'][eval_idx], cache['h_t'][eval_idx]
        y_eval = cache['labels'][eval_idx]

        _, _, _, probs_base = forward_from_embeddings(
            model, zv_eval, za_eval, ht_eval, baseline_stats, device)
        baseline_auroc = safe_auroc(y_eval, probs_base)
        print(f"  Baseline (no CORAL) AUROC={baseline_auroc:.4f}")

        sweep = []
        for n_calib in args.calib_sizes:
            n_calib_actual = min(n_calib, len(pool_idx))
            calib_idx = pool_idx[:n_calib_actual]

            A_v, b_v = fit_coral(src_zv, cache['z_v'][calib_idx], shrinkage=args.shrinkage)
            A_a, b_a = fit_coral(src_za, cache['z_a'][calib_idx], shrinkage=args.shrinkage)

            zv_al = apply_affine(zv_eval, A_v, b_v)
            za_al = apply_affine(za_eval, A_a, b_a)
            T = ht_eval.shape[1]
            ht_al = apply_affine(ht_eval.reshape(-1, ht_eval.shape[-1]), A_v, b_v).reshape(ht_eval.shape)

            _, _, _, probs_al = forward_from_embeddings(model, zv_al, za_al, ht_al, baseline_stats, device)
            auroc = safe_auroc(y_eval, probs_al)
            bal_acc = float(balanced_accuracy_score(y_eval, (probs_al >= 0.5).astype(int)))
            print(f"    n_calib={n_calib_actual:>4}  CORAL AUROC={auroc:.4f}  "
                  f"BalAcc={bal_acc:.4f}  (delta vs baseline={auroc - baseline_auroc:+.4f})")
            sweep.append({'n_calib': int(n_calib_actual), 'auroc': auroc, 'bal_acc': bal_acc,
                          'delta_vs_baseline': float(auroc - baseline_auroc)})

        all_results[ds_name] = {
            'n_total': int(n), 'n_eval_fixed': int(n_eval),
            'baseline_auroc': baseline_auroc, 'sweep': sweep,
        }
        print()

    out_file = out_dir / 'coral_calib_sweep_results.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved: {out_file}\n")


if __name__ == '__main__':
    main()
