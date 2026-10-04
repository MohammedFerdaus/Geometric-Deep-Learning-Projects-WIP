import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax

from data.load_qm9 import load_qm9_subset, train_val_test_split
from data.graph_utils import build_dataset_graphs
from model.init import init_mpnn_params
from model.mpnn import mpnn_forward, batched_mpnn_forward

CHECKPOINT_DIR = 'checkpoints'

def compute_loss(params, graph, target_mean, target_std):
    pred = mpnn_forward(params, graph['node_features'], graph['edge_index'], graph['edge_attr'])
    target = jnp.array([(graph['target'] - target_mean) / target_std])
    
    return jnp.mean((pred - target) ** 2)

def compute_batch_loss(params, graphs, target_mean, target_std):
    losses = jnp.array([compute_loss(params, g, target_mean, target_std) for g in graphs])
    
    return jnp.mean(losses)

def train_step(params, opt_state, graphs, optimizer, target_mean, target_std):
    loss, grads = jax.value_and_grad(compute_batch_loss)(params, graphs, target_mean, target_std)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

def evaluate(params, graphs, target_mean, target_std):
    preds = batched_mpnn_forward(params, graphs).reshape(-1) * target_std + target_mean
    targets = jnp.array([g['target'] for g in graphs])

    mse = jnp.mean((preds - targets) ** 2)
    mae = jnp.mean(jnp.abs(preds - targets))
    max_err = jnp.max(jnp.abs(preds - targets))

    return {'mse': float(mse), 'mae': float(mae), 'max_err': float(max_err)}

def save_checkpoint(path, params, opt_state, history, epoch, target_mean, target_std, run_config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        'params': jax.tree_util.tree_map(np.asarray, params),
        'opt_state': jax.tree_util.tree_map(np.asarray, opt_state),
        'history': history,
        'epoch': epoch, 
        'target_mean': target_mean,
        'target_std': target_std,
        'run_config': run_config,
    }
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump(state, f)
    os.replace(tmp_path, path)

def load_checkpoint(path):
    with open(path, 'rb') as f:
        state = pickle.load(f)
    state['params'] = jax.tree_util.tree_map(jnp.asarray, state['params'])
    state['opt_state'] = jax.tree_util.tree_map(jnp.asarray, state['opt_state'])
    
    return state

def train_loop(num_epochs=50, n_molecules=2000, batch_size=32, learning_rate=1e-3, seed=0,
               target_index=7, save_every=3, run_name='run', resume=True):
    ckpt_path = os.path.join(CHECKPOINT_DIR, f'{run_name}.pkl')
    run_config = {'n_molecules': n_molecules, 'batch_size': batch_size, 'learning_rate': learning_rate,
                  'seed': seed, 'target_index': target_index}

    key = jax.random.PRNGKey(seed)
    molecules = load_qm9_subset(n_molecules=n_molecules, seed=seed, target_index=target_index)
    graphs = build_dataset_graphs(molecules)
    train_graphs, val_graphs, test_graphs = train_val_test_split(graphs, seed=seed)

    optimizer = optax.adam(learning_rate)
    start_epoch = 0

    if resume and os.path.exists(ckpt_path):
        state = load_checkpoint(ckpt_path)
        if state['run_config'] != run_config:
            raise ValueError(
                f"Checkpoint {ckpt_path} was made with a different config {state['run_config']} "
                f"than this run {run_config}. Use a different run_name or delete the checkpoint."
            )
        params = state['params']
        opt_state = state['opt_state']
        history = state['history']
        target_mean = state['target_mean']
        target_std = state['target_std']
        start_epoch = state['epoch']
        print(f"Resuming from {ckpt_path} at epoch {start_epoch}")
    else:
        train_targets = np.array([g['target'] for g in train_graphs])
        target_mean = float(train_targets.mean())
        target_std = float(train_targets.std())

        params = init_mpnn_params(key)
        opt_state = optimizer.init(params)
        history = {'train_loss': [], 'val_mse': [], 'val_mae': []}

    print(f"target_mean={target_mean:.1f}, target_std={target_std:.1f}, "
          f"train/val/test = {len(train_graphs)}/{len(val_graphs)}/{len(test_graphs)}", flush=True)

    for epoch in range(start_epoch, num_epochs):
        rng = np.random.default_rng([seed, epoch])
        perm = rng.permutation(len(train_graphs))
        shuffled = [train_graphs[i] for i in perm]

        epoch_losses = []
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start:start + batch_size]
            params, opt_state, loss = train_step(
                params, opt_state, batch, optimizer, target_mean, target_std
            )
            epoch_losses.append(float(loss))

        val_metrics = evaluate(params, val_graphs, target_mean, target_std)

        history['train_loss'].append(float(np.mean(epoch_losses)))
        history['val_mse'].append(val_metrics['mse'])
        history['val_mae'].append(val_metrics['mae'])

        print(f"epoch {epoch}: train_loss={np.mean(epoch_losses):.4f}  "
              f"val_mse={val_metrics['mse']:.4f}  val_mae={val_metrics['mae']:.4f}", flush=True)

        completed = epoch + 1
        if completed % save_every == 0 or completed == num_epochs:
            save_checkpoint(ckpt_path, params, opt_state, history, completed,
                            target_mean, target_std, run_config)
            print(f"  [checkpoint saved: {ckpt_path} at epoch {completed}]", flush=True)

    test_metrics = evaluate(params, test_graphs, target_mean, target_std)
    print(f"FINAL test metrics: {test_metrics}")

    return params, history, test_graphs, test_metrics, target_mean, target_std

if __name__ == "__main__":
    train_loop(num_epochs=50, n_molecules=1000, batch_size=16, target_index=12,
               save_every=3, run_name='ae_1000')
