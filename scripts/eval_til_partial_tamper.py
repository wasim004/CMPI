"""
CMPI+DIAM — TIL gain vs. tamper-fraction on LAV-DF
====================================================
Wasim Ahmad, BIT — September 2026

ISWA revision (ISWA-D-26-01461), Reviewer #1 pt.4 / Reviewer #2 pt.2: quantify
the improvement TIL brings specifically on partially-tampered clips.

LAV-DF's own test-set fakes are essentially ALL partial by construction: the
manipulated segment covers a median of only 8.9% of clip duration (1999/2000
test fakes fall under a 30% tamper fraction, checked separately) — there is
no natural "fully tampered" subset to contrast against. So instead of a
partial-vs-full binary split (which the data doesn't support), this bins
fakes by tamper fraction (manipulated seconds / clip duration, reconstructed
from the original LAV-DF metadata.json — see uid-reconstruction note below)
and reports AUROC for C_i alone vs. TIL alone vs. the full detector in each
bin, to show whether TIL's edge over the un-localized C_i signal grows as
the fake segment shrinks.

uid -> original LAV-DF entry reconstruction
--------------------------------------------
preprocess_crossdataset.py's build_lavdf() assigns uid sequentially over
[real_entries[:2000] then fake_entries[:2000]] from metadata.json's test
split, and every one of those 4000 entries was confirmed to have succeeded
(final CSV has exactly 2000/2000, the cap, with zero gaps) — so uid i for
i<2000 is real_entries[i], and uid i for i>=2000 is fake_entries[i-2000], in
metadata.json order. Verified by joining back on modify_video/modify_audio/
label against the existing metadata.csv: 0 mismatches across all 4000 rows.

Usage
-----
python eval_til_partial_tamper.py \\
    --checkpoint  results/diam/best_model.pt \\
    --data_root   data/crossdataset \\
    --output_dir  results/til_partial_tamper
"""

import json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

import sys
ROOT = Path(__file__).resolve().parent.parent
for _p in [str(ROOT / "AMDD" / "models"), str(ROOT / "AMDD" / "data")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the exact model + dataset classes from the cross-dataset eval script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_crossdataset_diam import CMPIDIAMModel, CrossDataset

LAVDF_ROOT = Path("datasets/LAV-DF/LAV-DF")


def reconstruct_tamper_fractions():
    """Rebuild uid -> tamper_fraction for the 4000-row LAV-DF cross-dataset
    test set, using the same selection order as preprocess_crossdataset.py's
    build_lavdf(). Returns dict uid(str, zero-padded) -> float in [0, 1]."""
    with open(str(LAVDF_ROOT / 'metadata.json')) as f:
        meta = json.load(f)
    entries      = [e for e in meta if e['split'] == 'test']
    real_entries = [e for e in entries if e['n_fakes'] == 0][:2000]
    fake_entries = [e for e in entries if e['n_fakes'] >  0][:2000]

    frac = {}
    uid_counter = 0
    for e in real_entries:
        frac[str(uid_counter).zfill(6)] = 0.0
        uid_counter += 1
    for e in fake_entries:
        dur = e['duration']
        tampered = sum(end - start for start, end in e['fake_periods'])
        frac[str(uid_counter).zfill(6)] = float(tampered / dur) if dur > 0 else float('nan')
        uid_counter += 1
    return frac


@torch.no_grad()
def collect_per_sample(model, loader, device):
    model.eval()
    uids, probs_l, ci_l, til_l, label_l = [], [], [], [], []
    for batch in loader:
        frames = batch['frames'].to(device)
        audio  = batch['audio'].to(device)
        out    = model(frames, audio)
        uids.extend(batch['uid'])
        probs_l.append(out['det_probs'].cpu().numpy())
        ci_l.append(out['C_i'].cpu().numpy())
        til_l.append(out['til'].cpu().numpy())
        label_l.append(batch['label'].numpy())
    return (uids, np.concatenate(probs_l), np.concatenate(ci_l),
            np.concatenate(til_l), np.concatenate(label_l))


def safe_auroc(y, s):
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, s))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data_root',  required=True)
    p.add_argument('--output_dir', default='results/til_partial_tamper')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=4)
    args = p.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    model = CMPIDIAMModel(embed_dim=512, hidden_dim=1024).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')})")

    meta_csv = Path(args.data_root) / 'lavdf' / 'metadata.csv'
    prep_dir = Path(args.data_root) / 'lavdf' / 'preprocessed'
    ds       = CrossDataset(str(meta_csv), str(prep_dir))
    loader   = DataLoader(ds, args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True)

    uids, probs, ci, til, labels = collect_per_sample(model, loader, device)

    frac_map = reconstruct_tamper_fractions()
    tamper_frac = np.array([frac_map.get(u, float('nan')) for u in uids])
    n_missing = int(np.isnan(tamper_frac).sum())
    if n_missing:
        print(f"WARNING: {n_missing} uids missing from reconstructed tamper-fraction map")

    real_mask = labels == 0
    fake_mask = labels == 1
    fake_frac = tamper_frac[fake_mask]

    print(f"\nFake tamper-fraction distribution (n={fake_mask.sum()}):")
    print(f"  mean={np.nanmean(fake_frac):.4f}  median={np.nanmedian(fake_frac):.4f}")
    for q in (0.1, 0.25, 0.5, 0.75, 0.9):
        print(f"  {int(q*100)}th pct: {np.nanquantile(fake_frac, q):.4f}")

    # Quartile bins over the fake tamper-fraction distribution
    edges = np.nanquantile(fake_frac, [0.0, 0.25, 0.5, 0.75, 1.0])
    edges[0] = -np.inf
    edges[-1] = np.inf
    bin_labels = ['Q1 (most partial)', 'Q2', 'Q3', 'Q4 (least partial)']

    results = {
        'n_real': int(real_mask.sum()),
        'n_fake': int(fake_mask.sum()),
        'fake_tamper_fraction': {
            'mean': float(np.nanmean(fake_frac)),
            'median': float(np.nanmedian(fake_frac)),
        },
        'overall': {
            'auroc_ci':  safe_auroc(labels, ci),
            'auroc_til': safe_auroc(labels, til),
            'auroc_det': safe_auroc(labels, probs),
        },
        'bins': [],
    }

    print(f"\n{'Bin':<20}{'n':>6}{'frac range':>18}{'AUROC(C_i)':>13}{'AUROC(TIL)':>13}{'AUROC(det)':>13}{'TIL-C_i gain':>14}")
    fake_idx = np.where(fake_mask)[0]
    real_idx = np.where(real_mask)[0]
    for i in range(4):
        lo, hi = edges[i], edges[i + 1]
        sel_fake = fake_idx[(fake_frac >= lo) & (fake_frac < hi)]
        idx = np.concatenate([real_idx, sel_fake])
        y   = labels[idx]
        a_ci  = safe_auroc(y, ci[idx])
        a_til = safe_auroc(y, til[idx])
        a_det = safe_auroc(y, probs[idx])
        rng = f"[{lo:.3f}, {hi:.3f})" if np.isfinite(lo) and np.isfinite(hi) else \
              (f"[0, {hi:.3f})" if not np.isfinite(lo) else f"[{lo:.3f}, max]")
        print(f"{bin_labels[i]:<20}{len(sel_fake):>6}{rng:>18}{a_ci:>13.4f}{a_til:>13.4f}{a_det:>13.4f}{(a_til-a_ci):>14.4f}")
        results['bins'].append({
            'bin': bin_labels[i], 'n_fake': int(len(sel_fake)),
            'frac_low': float(lo) if np.isfinite(lo) else 0.0,
            'frac_high': float(hi) if np.isfinite(hi) else float(fake_frac.max()),
            'auroc_ci': a_ci, 'auroc_til': a_til, 'auroc_det': a_det,
            'til_minus_ci': float(a_til - a_ci) if not (np.isnan(a_til) or np.isnan(a_ci)) else float('nan'),
        })

    out_file = out_dir / 'til_partial_tamper_results.json'
    with open(str(out_file), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == '__main__':
    main()
