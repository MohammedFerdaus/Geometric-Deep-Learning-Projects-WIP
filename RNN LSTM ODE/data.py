import numpy as np
import jax.numpy as jnp
from physics import simulate
from forcing import random_forcing_generator

def generate_trajectory_dataset(n_trajectories, t_span, dt, rng, params):
    trajectories = []
    for _ in range(n_trajectories):
        n0 = rng.uniform(0.5, 1.5)
        T0 = rng.uniform(0.3, 1.5)

        P_heat_fn = random_forcing_generator(rng, t_span=t_span)
        t_arr, n_arr, T_arr = simulate((n0, T0), t_span, dt, P_heat_fn, params)
        P_heat_arr = np.array([P_heat_fn(t) for t in t_arr])

        trajectories.append({'t': t_arr, 'n': n_arr, 'T': T_arr, 'P_heat': P_heat_arr})

    return trajectories

def train_val_test_split(trajectories, ratios=(0.7, 0.15, 0.15), rng=None):
    n = len(trajectories)
    idx = np.arange(n)
    if rng is not None:
        rng.shuffle(idx)

    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    
    train = [trajectories[i] for i in train_idx]
    val = [trajectories[i] for i in val_idx]
    test = [trajectories[i] for i in test_idx]
    return train, val, test

def compute_normalization_stats(trajectories):
    all_n = np.concatenate([traj['n'] for traj in trajectories])
    all_T = np.concatenate([traj['T'] for traj in trajectories])
    all_P = np.concatenate([traj['P_heat'] for traj in trajectories])
    
    stats = {
        'n': {'mean': all_n.mean(), 'std': all_n.std()},
        'T': {'mean': all_T.mean(), 'std': all_T.std()},
        'P_heat': {'mean': all_P.mean(), 'std': all_P.std()}}
    
    return stats

def normalize(trajectory, stats):
    def z(x, s):
        return(x - s['mean']) / (s['std'] + 1e-8)
    
    return {
        't': trajectory['t'],
        'n': z(trajectory['n'], stats['n']),
        'T': z(trajectory['T'], stats['T']),
        'P_heat': z(trajectory['P_heat'], stats['P_heat'])}

def denormalize(trajectory, stats):
    def unz(x, s):
        return x * s['std'] + s['mean']
    
    return {
        't': trajectory['t'],
        'n': unz(trajectory['n'], stats['n']),
        'T': unz(trajectory['T'], stats['T']),
        'P_heat': unz(trajectory['P_heat'], stats['P_heat'])}

def make_windows(trajectory, seq_len):
    n, T, P = trajectory['n'], trajectory['T'], trajectory['P_heat']
    total_steps = len(n)
    n_windows = total_steps - seq_len
    
    X = np.zeros((n_windows, seq_len, 3))
    Y = np.zeros((n_windows, 2))
    
    for i in range(n_windows):
        X[i, :, 0] = n[i : i + seq_len]
        X[i, :, 1] = T[i : i + seq_len]
        X[i, :, 2] = P[i : i + seq_len]
        Y[i] = [n[i + seq_len], T[i + seq_len]]
        
    return X, Y

def to_jax_arrays(X, Y):
    return jnp.array(X), jnp.array(Y)