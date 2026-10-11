"""Train CMPI + DIAM + TIL on FakeAVCeleb and report test-set results.

    python train.py --config configs/cmpi_diam_til.yaml
    python train.py --config configs/cmpi_diam_til.yaml train.epochs=10 data.num_workers=8
"""

import argparse
import json
from pathlib import Path

import torch

from cmpi.data import build_fakeavceleb_loaders
from cmpi.engine import Trainer, evaluate_loader
from cmpi.losses import CMPILoss
from cmpi.models import build_model
from cmpi.utils import (load_config, apply_overrides, set_seed,
                        load_amdd_encoders, load_checkpoint)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config', default='configs/cmpi_diam_til.yaml')
    p.add_argument('--resume', default=None, help='last_checkpoint.pt to resume from')
    p.add_argument('overrides', nargs='*', help='dotted key=value overrides')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cfg['output_dir'])
    print(f"Device: {device} | output: {out_dir}")

    loaders = build_fakeavceleb_loaders(cfg)
    model = build_model(cfg).to(device)
    criterion = CMPILoss(**cfg['loss'])
    trainer = Trainer(model, criterion, loaders, cfg, device, out_dir)

    if args.resume:
        trainer.resume(args.resume)
    elif cfg['model'].get('amdd_checkpoint'):
        print(f"Warm-starting encoders from {cfg['model']['amdd_checkpoint']}")
        load_amdd_encoders(model, cfg['model']['amdd_checkpoint'], device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    best = trainer.fit()

    load_checkpoint(model, out_dir / 'best_model.pt', device)
    test = evaluate_loader(model, loaders['test'], device, criterion)
    print(f"\nTEST  AUROC={test['auroc']:.4f}  BalAcc={test['bal_acc']:.4f}  "
          f"C_i-AUROC={test['ci_auroc']:.4f}  TIL-AUROC={test['til_auroc']:.4f}")
    with open(out_dir / 'final_results.json', 'w') as f:
        json.dump({'best_val_auroc': best, 'test': test, 'config': cfg}, f, indent=2)


if __name__ == '__main__':
    main()
