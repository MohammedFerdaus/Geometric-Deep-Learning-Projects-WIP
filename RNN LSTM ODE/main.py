import numpy as np
import jax
import optax

from physics import default_params, simulate
from data import (
    generate_trajectory_dataset,
    train_val_test_split,
    compute_normalization_stats,
    normalize,
    make_windows,
    to_jax_arrays,)

from lstm import init_lstm_params
from train import train_loop
from rollout import autoregressive_rollout, compare_to_ground_truth
from evaluate import (
    plot_trajectory_comparison,
    plot_phase_space,
    plot_error_vs_time,
    analyze_failure_regions,)

CONFIG = {
    'n_trajectories': 150,
    't_span': (0.0, 20.0),
    'dt': 0.01,
    'seq_len': 20,
    'split_ratios': (0.7, 0.15, 0.15),
    'hidden_dim': 32,
    'input_dim': 3,
    'output_dim': 2,
    'learning_rate': 1e-3,
    'n_epochs': 100,
    'batch_size': 25,
    'rollout_n_steps': 50,
    'seed': 0,}

def main():
    rng = np.random.default_rng(CONFIG['seed'])
    jax_key = jax.random.PRNGKey(CONFIG['seed'])
    
    params_physics = default_params()
    
    trajectories = generate_trajectory_dataset(
        CONFIG['n_trajectories'], CONFIG['t_span'], CONFIG['dt'], rng, params_physics)
    
    train_traj, val_traj, test_traj = train_val_test_split(
        trajectories, CONFIG['split_ratios'], rng)
    
    stats = compute_normalization_stats(train_traj)
    train_traj_norm = [normalize(traj, stats) for traj in train_traj]
    val_traj_norm = [normalize(traj, stats) for traj in val_traj]
    test_traj_norm = [normalize(traj, stats) for traj in test_traj]
    
    def build_xy(traj_list):
        Xs, Ys = [], []
        for traj in traj_list:
            X, Y = make_windows(traj, CONFIG['seq_len'])
            Xs.append(X)
            Ys.append(Y)
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)
    
    X_train, Y_train = build_xy(train_traj_norm)
    X_val, Y_val = build_xy(val_traj_norm)
    
    X_train, Y_train = to_jax_arrays(X_train, Y_train)
    X_val, Y_val = to_jax_arrays(X_val, Y_val)
    
    lstm_params = init_lstm_params(
        jax_key, CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['output_dim'])
    
    optimizer = optax.adam(CONFIG['learning_rate'])
    optimizer_state = optimizer.init(lstm_params)
    
    trained_params, history = train_loop(
        lstm_params, optimizer_state, optimizer,
        (X_train, Y_train), (X_val, Y_val),
        CONFIG['n_epochs'], CONFIG['batch_size'], rng)
    
    test_traj_example = test_traj_norm[0]
    seq_len = CONFIG['seq_len']
    
    rollout_results = []
    for i in range(min(10, len(test_traj_norm))):
        test_example = test_traj_norm[i]
        test_example_raw = test_traj[i]
        
        initial_window = np.stack([
            test_example['n'][:seq_len],
            test_example['T'][:seq_len],
            test_example['P_heat'][:seq_len],],
            axis=1)

        pred_traj = autoregressive_rollout(
            trained_params, initial_window, test_example['P_heat'],
            CONFIG['rollout_n_steps'], stats)

        true_traj_physical = {
            'n': test_example_raw['n'][seq_len : seq_len + CONFIG['rollout_n_steps']],
            'T': test_example_raw['T'][seq_len : seq_len + CONFIG['rollout_n_steps']],}
    
        t_rollout = test_example_raw['t'][seq_len : seq_len + CONFIG['rollout_n_steps']]

        error_metrics = compare_to_ground_truth(pred_traj, true_traj_physical)
        rollout_results.append({
            't': t_rollout,
            'true_traj': true_traj_physical,
            'error_metrics': error_metrics,
            'pred_traj': pred_traj,})

    failure_analysis = analyze_failure_regions(rollout_results)

    example = rollout_results[0]
    fig1 = plot_trajectory_comparison(example['t'], example['true_traj'], example['pred_traj'])
    fig2 = plot_phase_space(example['true_traj'], example['pred_traj'])
    fig3 = plot_error_vs_time(example['t'], example['error_metrics'])

    fig1.savefig('trajectory_comparison.png')
    fig2.savefig('phase_space.png')
    fig3.savefig('error_vs_time.png')

    mse_total_avg = np.mean([r['error_metrics']['mse_total'] for r in rollout_results])
    print(f"rollout total MSE (averaged over {len(rollout_results)} test trajectories): {mse_total_avg}")
    print(f"error vs |dT/dt| correlation (pooled): {failure_analysis['correlation']}")

if __name__ == '__main__':
    main()