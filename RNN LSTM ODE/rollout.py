import numpy as np
import jax.numpy as jnp
from lstm import predict
from data import normalize, denormalize


def autoregressive_rollout(params, initial_window, P_heat_sequence, n_steps, stats):
    seq_len = initial_window.shape[0]
    window = jnp.array(initial_window) 

    predicted_n = []
    predicted_T = []

    for step in range(n_steps):
        Y_pred = predict(params, window)
        n_pred_norm, T_pred_norm = Y_pred[0], Y_pred[1]

        next_P_heat_norm = P_heat_sequence[seq_len + step] 

        new_row = jnp.array([n_pred_norm, T_pred_norm, next_P_heat_norm])
        window = jnp.concatenate([window[1:], new_row[None, :]], axis=0)

        predicted_n.append(float(n_pred_norm))
        predicted_T.append(float(T_pred_norm))

    pred_traj_norm = {
        'n': np.array(predicted_n),
        'T': np.array(predicted_T),
        'P_heat': np.zeros(n_steps),  
        't': np.zeros(n_steps),     
    }
    pred_traj_physical = denormalize(pred_traj_norm, stats)

    return {'n': pred_traj_physical['n'], 'T': pred_traj_physical['T']}


def compare_to_ground_truth(predicted_traj, true_traj):
    n_pred, T_pred = predicted_traj['n'], predicted_traj['T']
    n_true, T_true = true_traj['n'], true_traj['T']

    n_error = np.abs(n_pred - n_true)
    T_error = np.abs(T_pred - T_true)

    mse_per_step = (n_pred - n_true) ** 2 + (T_pred - T_true) ** 2
    mse_total = mse_per_step.mean()

    return {
        'mse_per_step': mse_per_step,
        'n_error': n_error,
        'T_error': T_error,
        'mse_total': mse_total,}