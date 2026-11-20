import jax
import jax.numpy as jnp

from lj import run_md, lj_forces_analytic, lj_energy

def sample_velocities(key, n_atoms, kT, mass=1.0):
    std = jnp.sqrt(kT / mass)

    v = jax.random.normal(key, shape=(n_atoms, 3)) * std
    v = v - jnp.mean(v, axis=0, keepdims=True)

    return v

def generate_trajectory(key, x0, n_steps, dt, kT):
    key, subkey = jax.random.split(key)
    v0 = sample_velocities(subkey, n_atoms=x0.shape[0], kT=kT)
    xs, vs, energies = run_md(x0, v0, lj_forces_analytic, dt, n_steps)
    print(f"energy drift: {jnp.max(jnp.abs(energies - energies[0]))}")
    
    return xs

def build_dataset(key, x0, n_traj, n_steps, dt, kT, burn_in, stride):
    keys = jax.random.split(key, n_traj)
                          
    chunks = []
    for k in keys:
        xs = generate_trajectory(k, x0, n_steps, dt, kT) 
        xs = xs[burn_in::stride]                           
        chunks.append(xs)
    
    X = jnp.concatenate(chunks, axis=0)

    E = jax.vmap(lj_energy)(X)
    F = jax.vmap(lj_forces_analytic)(X)

    M = X.shape[0]
    max_dist = jnp.max(jnp.sqrt(jnp.sum(X**2, axis=-1)))
    print(f"dataset size: {M}, max distance from origin: {max_dist}")
    
    return X, E, F

def normalize(E, F):
    e_mean = jnp.mean(E)
    scale = jnp.std(F)
    E_norm = (E - e_mean) / scale
    F_norm = F / scale
    stats = {"e_mean": e_mean, "scale": scale}
    return E_norm, F_norm, stats

def split(key, X, E, F, frac_train=0.8, frac_val=0.1):      
    M = X.shape[0]
    perm = jax.random.permutation(key, M)
    n_train = int(frac_train * M)
    n_val = int(frac_val * M)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    return {
        "train": {"x": X[train_idx], "e": E[train_idx], "f": F[train_idx]},
        "val":   {"x": X[val_idx], "e": E[val_idx], "f": F[val_idx]},
        "test":  {"x": X[test_idx], "e": E[test_idx], "f": F[test_idx]},
        }