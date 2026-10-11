from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PANELS = [('loss', 'Total loss'), ('det_loss', 'Detection loss (focal)'),
          ('pred_loss', 'Predictive loss (MSE)'), ('auroc', 'AUROC (detector)'),
          ('ci_auroc', 'AUROC (C_i)'), ('til_auroc', 'AUROC (TIL)')]


def plot_training_curves(history, out_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (key, title) in zip(axes.flat, PANELS):
        ax.plot(epochs, history[f'train_{key}'], label='Train', lw=2)
        if history.get(f'val_{key}'):
            ax.plot(epochs, history[f'val_{key}'], label='Val', lw=2)
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.grid(alpha=0.3)
        ax.legend()
    if history.get('val_auroc'):
        best = int(np.argmax(history['val_auroc'])) + 1
        for ax in axes.flat:
            ax.axvline(best, color='red', ls='--', lw=1, alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(Path(out_dir) / 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
