import sys
import os

from data.load_qm9 import load_qm9_subset, train_val_test_split
from data.graph_utils import build_dataset_graphs
from train import load_checkpoint, evaluate, CHECKPOINT_DIR
from visualize import (plot_loss_curve, plot_parity, plot_error_histogram,
                       plot_error_vs_molecule_size)


def main(run_name):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{run_name}.pkl')
    state = load_checkpoint(ckpt_path)
    cfg = state['run_config']
    print(f"Loaded {ckpt_path}: {state['epoch']} epochs completed, config={cfg}")

    molecules = load_qm9_subset(n_molecules=cfg['n_molecules'], seed=cfg['seed'],
                                target_index=cfg['target_index'])
    graphs = build_dataset_graphs(molecules)
    _, _, test_graphs = train_val_test_split(graphs, seed=cfg['seed'])

    params = state['params']
    target_mean, target_std = state['target_mean'], state['target_std']

    print("test metrics:", evaluate(params, test_graphs, target_mean, target_std))
    print(f"target_std={target_std:.2f}  (predict-the-mean MAE is roughly {0.8 * target_std:.2f})")

    plot_loss_curve(state['history'], save_path=f'{run_name}_loss.png')
    plot_parity(params, test_graphs, target_mean, target_std, save_path=f'{run_name}_parity.png')
    plot_error_histogram(params, test_graphs, target_mean, target_std, save_path=f'{run_name}_hist.png')
    plot_error_vs_molecule_size(params, test_graphs, target_mean, target_std,
                                save_path=f'{run_name}_error_vs_size.png')


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else 'ae_1000')
