import os
import sys
import pickle as pkl

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand

jax.config.update("jax_enable_x64", True)

from lj import (initial_cluster,
                check_forces_agree,
                lj_energy)
from data import (build_dataset,
                  normalize,
                  split)

from egnn import init_egnn
from train import train
from equivariance_test import run_all_tests
import md_rollout
import plots

config = {
    "seed": 0,
    "stages": ["check", "train", "equivariance", "rollout", "plots"],

    # which force heads to train: "autodiff", "direct", or both
    "modes": ["autodiff", "direct"],
 
    # dataset (changing any of these automatically builds a new cached file)
    "n_traj": 20,
    "n_steps_data": 20000,
    "dt": 0.005,
    "kT": 0.3,
    "burn_in": 1000,
    "stride": 20,
 
    # model
    "d": 32,
    "n_layers": 3,
 
    # training
    "epochs": 100,
    "batch_size": 64,
    "lr": 3e-4,
 
    # md rollout
    "rollout_steps": 4000,
    "rollout_seed": 123
}

def model_path(mode):
    return f"egnn_{mode}.pkl"

def get_data(cfg, key):
    name = (f"dataset_{cfg['n_traj']}_{cfg['n_steps_data']}_{cfg['dt']}_"
            f"{cfg['kT']}_{cfg['burn_in']}_{cfg['stride']}.npz")
    
    if os.path.exists(name):
        print(f"loading cached dataset {name}")
        d = np.load(name)
        
        return jnp.asarray(d["X"]), jnp.asarray(d["E"]), jnp.asarray(d["F"])
 
    x0 = initial_cluster()
    X, E, F = build_dataset(key, x0, cfg["n_traj"], cfg["n_steps_data"], cfg["dt"],
                            cfg["kT"], cfg["burn_in"], cfg["stride"])
    
    np.savez(name, X=np.asarray(X), E=np.asarray(E), F=np.asarray(F))
    print(f"cached dataset to {name}")
    
    return X, E, F

def stage_check(cfg):
    print("\n check: LJ forces and relaxed energy ")
    
    x = initial_cluster()
    check_forces_agree(x)
    
    print(f"relaxed LJ13 energy: {float(lj_energy(x)):.4f}")

def stage_train(cfg, key):
    print("\n train ")
    
    k_data, k_split, k_train = jrand.split(key, 3)
    X, E, F = get_data(cfg, k_data)
    E, F, stats = normalize(E, F)

    splits = split(k_split, X, E, F)

    for mode in cfg["modes"]:
        print(f"\n training mode: {mode} ")
        params, history = train(k_train, splits, mode, d=cfg["d"],
                                n_layers=cfg["n_layers"], epochs=cfg["epochs"],
                                batch_size=cfg["batch_size"], lr=cfg["lr"])
        
        with open(model_path(mode), "wb") as fh:
            pkl.dump({"params": params, "stats": stats,
                         "history": history, "config": cfg}, fh)
        print(f"saved {model_path(mode)}")

def stage_equivariance(cfg, key):
    print("\n equivariance ")
    k_init, k_noise, k_t1, k_t2 = jrand.split(key, 4)

    x = initial_cluster()
    x = x + 0.1 * jrand.normal(k_noise, x.shape)
 
    print(" untrained model ")
    params = init_egnn(k_init, d=cfg["d"], n_layers=cfg["n_layers"])
    run_all_tests(params, x, k_t1)

    for mode in cfg["modes"]:
        if os.path.exists(model_path(mode)):
            print(f" trained model ({mode}) ")
            
            with open(model_path(mode), "rb") as fh:
                saved = pkl.load(fh)
            run_all_tests(saved["params"], x, k_t2)

def stage_rollout(cfg):
    print("\n md rollout ")
    md_rollout.run_rollout(dt=cfg["dt"], n_steps=cfg["rollout_steps"],
                           kT=cfg["kT"], seed=cfg["rollout_seed"])
  
def stage_plots(cfg):
    print("\n plots ")
    r = plots.load_rollout()
    plots.plot_training()
    plots.plot_energy(r)
    plots.plot_rollout_quality(r)
    plots.plot_parity()
    print("saved figures")
 
if __name__ == "__main__":
    cfg = config
    stages = sys.argv[1:] if len(sys.argv) > 1 else cfg["stages"]
    key = jrand.PRNGKey(cfg["seed"])
 
    if "check" in stages:
        stage_check(cfg)
    if "train" in stages:
        stage_train(cfg, key)
    if "equivariance" in stages:
        stage_equivariance(cfg, jrand.fold_in(key, 1))
    if "rollout" in stages:
        stage_rollout(cfg)
    if "plots" in stages:
        stage_plots(cfg)
 