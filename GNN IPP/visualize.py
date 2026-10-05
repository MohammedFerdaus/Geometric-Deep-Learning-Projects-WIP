import numpy as np
import matplotlib.pyplot as plt

from model.mpnn import batched_mpnn_forward
from train import train_loop


def get_predictions(params, graphs, target_mean, target_std):
    preds = np.array(batched_mpnn_forward(params, graphs)).reshape(-1) * target_std + target_mean
    targets = np.array([g['target'] for g in graphs])
    return preds, targets


def plot_loss_curve(history, save_path=None):
    epochs = range(len(history['train_loss']))

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, history['train_loss'], color='tab:blue', label='train loss (normalized MSE)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train loss (normalized MSE)', color='tab:blue')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(epochs, history['val_mae'], color='tab:orange', label='val MAE (eV)')
    ax2.set_ylabel('Val MAE (eV)', color='tab:orange')
    ax2.set_yscale('log')

    plt.title('Training loss and validation MAE')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_parity(params, test_graphs, target_mean, target_std, save_path=None):
    preds, targets = get_predictions(params, test_graphs, target_mean, target_std)

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.5, s=12)
    lims = [min(targets.min(), preds.min()), max(targets.max(), preds.max())]
    plt.plot(lims, lims, 'r--', label='y = x (perfect prediction)')
    plt.xlabel('True target (eV)')
    plt.ylabel('Predicted target (eV)')
    plt.title('Parity plot: predicted vs. true')
    plt.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_error_histogram(params, test_graphs, target_mean, target_std, save_path=None):
    preds, targets = get_predictions(params, test_graphs, target_mean, target_std)
    errors = preds - targets

    plt.figure(figsize=(7, 5))
    plt.hist(errors, bins=40)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Prediction error (eV)  [pred - true]')
    plt.ylabel('Count')
    plt.title(f'Error distribution (mean {errors.mean():.1f} eV, std {errors.std():.1f} eV)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_error_vs_molecule_size(params, test_graphs, target_mean, target_std, save_path=None):
    preds, targets = get_predictions(params, test_graphs, target_mean, target_std)
    abs_errors = np.abs(preds - targets)
    sizes = np.array([g['num_atoms'] for g in test_graphs])

    plt.figure(figsize=(7, 5))
    plt.scatter(sizes, abs_errors, alpha=0.5, s=12)
    plt.xlabel('Number of atoms')
    plt.ylabel('Absolute error (eV)')
    plt.title('Error vs. molecule size')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    params, history, test_graphs, test_metrics, target_mean, target_std = train_loop(
        num_epochs=40, n_molecules=200, batch_size=16
    )
    print(f"target_mean={target_mean:.1f} eV, target_std={target_std:.1f} eV")

    plot_loss_curve(history, save_path='loss_curve.png')
    plot_parity(params, test_graphs, target_mean, target_std, save_path='parity.png')
    plot_error_histogram(params, test_graphs, target_mean, target_std, save_path='error_hist.png')
    plot_error_vs_molecule_size(params, test_graphs, target_mean, target_std, save_path='error_vs_size.png')
