import os
import pickle as pkl

import numpy as np
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import jax.random as jrand

jax.config.update("jax_enable_x64", True)

from lj import lj_forces_analytic, initial_cluster
from data import generate_trajectory
from md_rollout import load_model, make_force_fn

COLORS = {"true": "k", "autodiff": "C0", "direct": "C1"}
FLOOR = 1e-16 

def load_rollout(path="rollout.npz"):
    data = np.load(path)
    return {k: data[k] for k in data.files}

def plot_training(path="egnn_autodiff.pkl", out="training.png"):
    if not os.path.exists(path):
        print(f"{path} not found, skipping training plot")
        return
    with open(path, "rb") as fh:
        saved = pkl.load(fh)
    if "history" not in saved:
        print("no history stored in pickle, skipping training plot")
        return

    train_loss = np.array([float(v) for v in saved["history"]["train_loss"]])
    val_mae = np.array([float(v) for v in saved["history"]["val_force_mae"]])
    epochs = np.arange(len(train_loss))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(epochs, train_loss)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("train loss (normalized units)")
    axes[0].set_title("training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(epochs, val_mae)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("val force MAE (normalized units)")
    axes[1].set_title("validation force MAE")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_energy(r, out="energy.png"):
    t = np.asarray(r["t"])

    series = [
        ("E_true", "true LJ", COLORS["true"], "-"),
        ("E_model_autodiff", "autodiff: model's own energy", COLORS["autodiff"], "--"),
        ("E_modelpath_autodiff", "autodiff: true energy on model path", COLORS["autodiff"], "-"),
        ("E_modelpath_direct", "direct: true energy on model path", COLORS["direct"], "-"),
    ]
    series = [s for s in series if s[0] in r]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for key, label, color, ls in series:
        E = np.asarray(r[key])
        dev = E - E[0]
        axes[0].semilogy(t, np.abs(dev) + FLOOR, color=color, linestyle=ls, label=label)
        axes[1].plot(t, dev, color=color, linestyle=ls, label=label)

    axes[0].set_xlabel("time (reduced units)")
    axes[0].set_ylabel("|E(t) - E(0)|")
    axes[0].set_title("energy deviation (log scale)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("time (reduced units)")
    axes[1].set_ylabel("E(t) - E(0)")
    axes[1].set_title("energy deviation (linear scale)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_rollout_quality(r, out="rollout_quality.png"):
    t = np.asarray(r["t"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for mode in ["autodiff", "direct"]:
        if f"div_{mode}" not in r:
            continue
        c = COLORS[mode]
        axes[0].semilogy(t, np.asarray(r[f"div_{mode}"]) + FLOOR, color=c, label=mode)
        axes[1].semilogy(t, np.asarray(r[f"ferr_{mode}"]) + FLOOR, color=c, label=mode)
        axes[2].plot(t, np.asarray(r[f"rmax_{mode}"]), color=c, label=mode)

    titles = ["divergence from true trajectory",
              "force RMS error along rollout",
              "max atom distance from origin"]
    ylabels = ["mean atom displacement", "force RMS error (real units)", "max radius"]
    for ax, title, ylabel in zip(axes, titles, ylabels):
        ax.set_xlabel("time (reduced units)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_parity(path="egnn_autodiff.pkl", out="parity.png", seed=999):
    if not os.path.exists(path):
        print(f"{path} not found, skipping parity plot")
        return

    params, stats = load_model(path)
    force_fn = make_force_fn(params, stats, "autodiff")

    x0 = initial_cluster()
    xs = generate_trajectory(jrand.PRNGKey(seed), x0, n_steps=3000, dt=0.005, kT=0.3)
    xs = xs[500::20]

    f_true = np.asarray(jax.vmap(lj_forces_analytic)(xs)).ravel()
    f_pred = np.asarray(jax.vmap(force_fn)(xs)).ravel()

    rmse = float(np.sqrt(np.mean((f_pred - f_true) ** 2)))
    f_std = float(np.std(f_true))

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(f_true, f_pred, s=2, alpha=0.3)
    lo, hi = float(f_true.min()), float(f_true.max())
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set_xlabel("true force component")
    ax.set_ylabel("predicted force component")
    ax.set_title(f"force parity (RMSE {rmse:.3f}, true std {f_std:.3f})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    r = load_rollout()
    plot_training()
    plot_energy(r)
    plot_rollout_quality(r)
    plot_parity()
    print("saved figures")