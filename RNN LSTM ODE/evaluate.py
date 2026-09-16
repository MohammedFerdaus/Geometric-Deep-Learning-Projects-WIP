import numpy as np
import matplotlib.pyplot as plt

def plot_trajectory_comparison(t, true_traj, pred_traj):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

    axes[0].plot(t, true_traj['n'], label='RK4 (true)', color='black')
    axes[0].plot(t, pred_traj['n'], label='LSTM', color='tab:red', linestyle='--')
    axes[0].set_ylabel('n (density)')
    axes[0].legend()

    axes[1].plot(t, true_traj['T'], label='RK4 (true)', color='black')
    axes[1].plot(t, pred_traj['T'], label='LSTM', color='tab:red', linestyle='--')
    axes[1].set_ylabel('T (temperature)')
    axes[1].set_xlabel('time')
    axes[1].legend()

    fig.tight_layout()
    return fig

def plot_phase_space(true_traj, pred_traj):
    fig, ax = plt.subplots(figsize=(6,6))

    ax.plot(true_traj['n'], true_traj['T'], label='RK4 (true)', color='black')
    ax.plot(pred_traj['n'], pred_traj['T'], label='LSTM', color='tab:red', linestyle='--')

    ax.scatter(true_traj['n'][0], true_traj['T'][0], color='green', zorder=5, label='start')

    ax.set_xlabel('n (density)')
    ax.set_ylabel('T (temperature)')
    ax.legend()

    fig.tight_layout()
    return fig

def plot_error_vs_time(t, error_metrics):
    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(t, error_metrics['mse_per_step'], label='combined MSE')
    ax.set_yscale('log')
    ax.set_xlabel('time (rollout step)')
    ax.set_ylabel('error (log scale)')
    ax.legend()

    fig.tight_layout()
    return fig

def analyze_failure_regions(rollout_results, threshold_percentile=90):
    all_dT_dt = []
    all_error = []
    
    for result in rollout_results:
        dT_dt = np.gradient(result['true_traj']['T'], result['t'])
        error = result['error_metrics']['mse_per_step']
        all_dT_dt.append(dT_dt)
        all_error.append(error)
    
    dT_dt_pooled = np.concatenate(all_dT_dt)
    error_pooled = np.concatenate(all_error)
    
    threshold = np.percentile(error_pooled, threshold_percentile)
    high_error_mask = error_pooled >= threshold
    
    correlation = np.corrcoef(np.abs(dT_dt_pooled), error_pooled)[0, 1]
    
    return {
        'dT_dt_pooled': dT_dt_pooled,
        'error_pooled': error_pooled,
        'high_error_mask': high_error_mask,
        'correlation': correlation,
    }