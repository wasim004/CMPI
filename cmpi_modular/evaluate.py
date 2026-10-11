"""Evaluate a trained checkpoint in-domain (FakeAVCeleb test split) and/or
zero-shot on cross-domain benchmarks (LAV-DF, DFDM, Celeb-DF).

    python evaluate.py --checkpoint runs/cmpi_diam_til/best_model.pt --crossdomain
    python evaluate.py --checkpoint best_model.pt --indomain --crossdomain eval.datasets=[lavdf]

Checkpoints from the original scripts/train_cmpi_diam.py load as well.
"""

import argparse
import json
from pathlib import Path

import torch

from cmpi.data import FakeAVCelebDataset, build_crossdomain_loader
from cmpi.engine import evaluate_loader
from cmpi.models import build_model
from cmpi.utils import load_config, apply_overrides, load_checkpoint
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/cmpi_diam_til.yaml')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--indomain', action='store_true', help='evaluate FakeAVCeleb test split')
    p.add_argument('--crossdomain', action='store_true', help='evaluate eval.datasets')
    p.add_argument('--output', default=None, help='JSON output path')
    p.add_argument('overrides', nargs='*')
    return p.parse_args()


def show(name, r):
    line = (f"  {name:<10} AUROC={r['auroc']:.4f}  BalAcc={r['bal_acc']:.4f}  "
            f"C_i={r['ci_auroc']:.4f}  TIL={r['til_auroc']:.4f}  "
            f"A real/fake={r['a_mean_real']:.2f}/{r['a_mean_fake']:.2f}")
    print(line)
    for k in ('video_fake', 'audio_fake', 'both_fake'):
        if f'a_mean_{k}' in r:
            print(f"             {k}: AUROC={r[f'auroc_det_{k}']:.4f}  mean A={r[f'a_mean_{k}']:.2f}")


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    if not (args.indomain or args.crossdomain):
        args.indomain = args.crossdomain = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg['model']['pretrained_backbones'] = False      # weights come from the checkpoint
    model = build_model(cfg).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, device)
    print(f"Loaded {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")

    e, results = cfg['eval'], {}
    if args.indomain:
        ds = FakeAVCelebDataset(cfg['data']['metadata_csv'], 'test', cfg['data']['preprocessed_dir'])
        loader = DataLoader(ds, batch_size=e['batch_size'], shuffle=False,
                            num_workers=cfg['data']['num_workers'], pin_memory=True)
        results['fakeavceleb'] = evaluate_loader(model, loader, device)
        show('FakeAVCeleb', results['fakeavceleb'])
    if args.crossdomain:
        for name in e['datasets']:
            print(f"{name}:")
            loader = build_crossdomain_loader(e['crossdomain_root'], name, e['batch_size'],
                                              cfg['data']['num_workers'])
            if loader is None:
                print(f"  skipped: {e['crossdomain_root']}/{name} not found")
                continue
            results[name] = evaluate_loader(model, loader, device)
            show(name, results[name])

    out = Path(args.output or Path(args.checkpoint).with_name('eval_results.json'))
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
