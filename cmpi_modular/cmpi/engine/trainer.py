"""Training loop: differential learning rates, warmup + cosine schedule,
AMP, gradient clipping, best-by-val-AUROC checkpointing."""

import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

from .evaluator import evaluate_loader
from ..utils.checkpoint import save_checkpoint
from ..utils.plotting import plot_training_curves


def build_optimizer(model, cfg):
    t = cfg['train']
    return torch.optim.AdamW([
        {'params': model.encoder_parameters(), 'lr': t['lr_encoder'], 'name': 'encoders'},
        {'params': model.head_parameters(), 'lr': t['lr_head'], 'name': 'heads'},
    ], weight_decay=t['weight_decay'])


def build_scheduler(optimizer, cfg):
    t = cfg['train']
    warm, total = t['warmup_epochs'], t['epochs']

    def lr_lambda(ep):
        if ep < warm:
            return (ep + 1) / warm
        prog = (ep - warm) / max(total - warm, 1)
        return 0.5 * (1.0 + math.cos(math.pi * prog))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


HIST_KEYS = ['loss', 'det_loss', 'pred_loss', 'auroc', 'bal_acc', 'ci_auroc', 'til_auroc']


class Trainer:
    def __init__(self, model, criterion, loaders, cfg, device, out_dir):
        self.model, self.criterion, self.loaders = model, criterion, loaders
        self.cfg, self.device = cfg, device
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.optimizer = build_optimizer(model, cfg)
        self.scheduler = build_scheduler(self.optimizer, cfg)
        self.use_amp = device.type == 'cuda' and cfg['train'].get('amp', True)
        self.scaler = torch.amp.GradScaler('cuda') if self.use_amp else None
        self.grad_clip = cfg['train'].get('grad_clip', 1.0)
        self.log_interval = cfg['train'].get('log_interval', 20)
        self.history = {f'{s}_{k}': [] for s in ('train', 'val') for k in HIST_KEYS}
        self.start_epoch, self.best_auroc = 1, 0.0

    def resume(self, path):
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_checkpoint_state(ckpt['model_state'])
        if 'optimizer_state' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state'])
        if 'scheduler_state' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state'])
        self.start_epoch = ckpt['epoch'] + 1
        self.best_auroc = ckpt.get('best_val_auroc', 0.0)
        self.history = ckpt.get('history', self.history)
        print(f"Resumed at epoch {self.start_epoch} (best val AUROC {self.best_auroc:.4f})")

    def train_one_epoch(self, epoch):
        self.model.train()
        sums, n, t0 = {'total': 0.0, 'det': 0.0, 'L_pred': 0.0}, 0, time.time()
        loader = self.loaders['train']
        for i, batch in enumerate(loader):
            frames = batch['frames'].to(self.device, non_blocking=True)
            audio = batch['audio'].to(self.device, non_blocking=True)
            labels = batch['label'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model(frames, audio)
                loss, ld = self.criterion(out, labels)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            for k in sums:
                sums[k] += ld[k]
            n += 1
            if (i + 1) % self.log_interval == 0:
                print(f"    ep{epoch} [{i + 1}/{len(loader)}] loss={ld['total']:.4f} "
                      f"det={ld['det']:.4f} pred={ld['L_pred']:.4f} ({time.time() - t0:.0f}s)")
        self.scheduler.step()
        return {'loss': sums['total'] / n, 'det_loss': sums['det'] / n,
                'pred_loss': sums['L_pred'] / n}

    def fit(self):
        epochs = self.cfg['train']['epochs']
        for epoch in range(self.start_epoch, epochs + 1):
            t0 = time.time()
            lrs = [g['lr'] for g in self.optimizer.param_groups]
            print(f"── Epoch {epoch}/{epochs}  lr_enc={lrs[0]:.2e} lr_head={lrs[1]:.2e}")

            tr = self.train_one_epoch(epoch)
            tr_eval = evaluate_loader(self.model, self.loaders['train'], self.device, self.criterion)
            va = evaluate_loader(self.model, self.loaders['val'], self.device, self.criterion)

            for k in ('loss', 'det_loss', 'pred_loss'):
                self.history[f'train_{k}'].append(tr[k])
            for k in ('auroc', 'bal_acc', 'ci_auroc', 'til_auroc'):
                self.history[f'train_{k}'].append(tr_eval[k])
            for k in HIST_KEYS:
                self.history[f'val_{k}'].append(va[k])

            print(f"  val  AUROC={va['auroc']:.4f}  C_i={va['ci_auroc']:.4f}  "
                  f"TIL={va['til_auroc']:.4f}  BalAcc={va['bal_acc']:.4f}  ({time.time() - t0:.0f}s)")
            print(f"  DIAM C_i real/fake={va['ci_mean_real']:.2f}/{va['ci_mean_fake']:.2f}  "
                  f"A real/fake={va['a_mean_real']:.2f}/{va['a_mean_fake']:.2f}")

            is_best = va['auroc'] > self.best_auroc
            if is_best:
                self.best_auroc = va['auroc']
                print("  * new best")
            save_checkpoint({
                'epoch': epoch, 'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'best_val_auroc': self.best_auroc, 'val_metrics': va,
                'history': self.history, 'config': self.cfg,
            }, self.out_dir, is_best)
            with open(self.out_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)

        plot_training_curves(self.history, self.out_dir)
        return self.best_auroc
