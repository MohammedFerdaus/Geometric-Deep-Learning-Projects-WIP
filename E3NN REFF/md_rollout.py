import os
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrand
import pickle as pkl

from egnn import (predict_forces_autodiff,
                  predict_forces_direct,
                  predict_energy)

from lj import (kinetic_energy,
                lj_forces_analytic,
                initial_cluster,
                run_md)

from data import sample_velocities

jax.config.update("jax_enable_x64", True)

def load_model(path):
    with open(path, "rb") as f:
        d = pkl.load(f)
    return d["params"], d["stats"]

def make_force_fn(params, stats, mode):
    scale = stats["scale"]
    if mode == "autodiff":
        return lambda x: scale * predict_forces_autodiff(params, x)
    else:
        return lambda x: scale * predict_forces_direct(params, x)

def make_energy_fn(params, stats):
    scale = stats["scale"]          
    e_mean = stats["e_mean"]

    return lambda x: scale * predict_energy(params, x) + e_mean

def model_total_energy(xs, vs, energy_fn):
    pe = jax.vmap(energy_fn)(xs)
    ke = jax.vmap(kinetic_energy)(vs)

    return pe + ke

def drift_stats(t, energies):
    dev = energies - energies[0]
    max_dev = jnp.max(jnp.abs(dev))

    slope = jnp.polyfit(t, energies, deg=1)[0]

    rel = max_dev / jnp.abs(energies[0])

    return {"max_dev": max_dev, "slope": slope, "rel": rel}

def divergence(xs_a, xs_b):
    per_atom = jnp.linalg.norm(xs_a - xs_b, axis=-1)    
    return jnp.mean(per_atom, axis=1)                 

def force_error_along(xs, force_fn):
    f_true = jax.vmap(lj_forces_analytic)(xs)
    f_pred = jax.vmap(force_fn)(xs)

    return jnp.sqrt(jnp.mean((f_pred - f_true) ** 2, axis=(1, 2)))

def print_drift(label, stats):
    print(f"{label:28s} max|dE| {float(stats['max_dev']):.3e}   "
          f"slope {float(stats['slope']):+.3e}   rel {float(stats['rel']):.3e}")

def run_rollout(dt=0.005, n_steps=4000, kT=0.3, seed=123):
    key = jrand.PRNGKey(seed)

    x0 = initial_cluster()
    n_atoms = x0.shape[0]
    v0 = sample_velocities(key, n_atoms, kT)
    t = jnp.linspace(0.0, dt * n_steps, n_steps + 1)

    xs_t, vs_t, E_t = run_md(x0, v0, lj_forces_analytic, dt, n_steps)

    results = {"t": t, "E_true": E_t}
    results["drift_true"] = drift_stats(t, E_t)
    print_drift("true LJ", results["drift_true"])

    for mode, path in [("autodiff", "egnn_autodiff.pkl"), ("direct", "egnn_direct.pkl")]:
        if not os.path.exists(path):
            continue

        params, stats = load_model(path)
        force_fn = make_force_fn(params, stats, mode)

        xs_m, vs_m, E_m_true = run_md(x0, v0, force_fn, dt, n_steps)

        results[f"E_modelpath_{mode}"] = E_m_true
        results[f"div_{mode}"] = divergence(xs_m, xs_t)
        results[f"ferr_{mode}"] = force_error_along(xs_m, force_fn)
        results[f"rmax_{mode}"] = jnp.max(jnp.linalg.norm(xs_m, axis=-1), axis=1)

        results[f"drift_modelpath_{mode}"] = drift_stats(t, E_m_true)
        print_drift(f"{mode}: true E on model path", results[f"drift_modelpath_{mode}"])

        if mode == "autodiff":
            energy_fn = make_energy_fn(params, stats)
            E_model = model_total_energy(xs_m, vs_m, energy_fn)     
            results["E_model_autodiff"] = E_model
            results["drift_model_autodiff"] = drift_stats(t, E_model)
            print_drift("autodiff: model's own E", results["drift_model_autodiff"])

        print(f"{mode}: finite rollout = {bool(jnp.all(jnp.isfinite(xs_m)))}")

    np.savez("rollout.npz",
             **{k: v for k, v in results.items() if not isinstance(v, dict)})
    print("saved rollout.npz")

if __name__ == "__main__":
    run_rollout()