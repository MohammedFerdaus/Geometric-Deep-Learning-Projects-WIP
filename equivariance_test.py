import os
import jax
import pickle as pkl
import jax.numpy as jnp
import jax.random as jrand
from lj import initial_cluster
from egnn import (init_egnn,
                  predict_energy,
                  predict_forces_autodiff,
                  predict_forces_direct)
jax.config.update("jax_enable_x64", True)

def random_orthogonal(key, proper=True):
    A = jrand.normal(key, (3, 3))

    Q, R = jnp.linalg.qr(A)

    Q = Q * jnp.sign(jnp.diag(R))

    d = jnp.linalg.det(Q)

    if (d > 0) != proper:
        Q = Q.at[:, 0].set(-Q[:, 0])
    
    return Q

def force_equivariance_error(params, x, R, force_fn):
    f_then_rotate = force_fn(params, x) @ R.T
    rotate_then_f = force_fn(params, x @ R.T)

    return jnp.max(jnp.abs(f_then_rotate - rotate_then_f))

def energy_invariance_error(params, x, R):
    return jnp.abs(predict_energy(params, x @ R.T) - predict_energy(params, x))

def translation_error(params, x, t, force_fn):
    force_translated = force_fn(params, x + t)
    force_original = force_fn(params, x)

    return jnp.max(jnp.abs(force_translated - force_original))

def permutation_error(params, x, perm, force_fn):
    force_translated = force_fn(params, x[perm])
    force_original = force_fn(params, x)[perm]
    
    return jnp.max(jnp.abs(force_translated - force_original))

def run_all_tests(params, x, key, tol=1e-10):
    k1, k2, k3, k4 = jrand.split(key, 4)
    N = x.shape[0]
    R_rot  = random_orthogonal(k1, proper=True)       # a rotation
    R_refl = random_orthogonal(k2, proper=False)      # includes a reflection
    t      = jrand.normal(k3, (3,)) * 5.0             # a big shift
    perm   = jrand.permutation(k4, N)                 # random permutation

    heads = {
        "autodiff": predict_forces_autodiff,
        "direct": predict_forces_direct
    }

    force_tests = {
        "rotation":    lambda fn: force_equivariance_error(params, x, R_rot, fn),
        "reflection":  lambda fn: force_equivariance_error(params, x, R_refl, fn),
        "translation": lambda fn: translation_error(params, x, t, fn),
        "permutation": lambda fn: permutation_error(params, x, perm, fn),
    }

    energy_tests = {
        "rotation":   lambda: energy_invariance_error(params, x, R_rot),
        "reflection": lambda: energy_invariance_error(params, x, R_refl),
    }

    results = {}
    for head_name, force_fn in heads.items():
        results[head_name] = {}
        print(f"--- {head_name} forces ---")
        for test_name, test in force_tests.items():
            err = test(force_fn)
            results[head_name][test_name] = err
            status = "PASS" if err < tol else "FAIL"
            print(f"{test_name:12s} {err:.2e}  {status}")

    print("--- energy invariance ---")
    results["energy"] = {}
    for test_name, test in energy_tests.items():
        err = test()
        results["energy"][test_name] = err
        status = "PASS" if err < tol else "FAIL"
        print(f"{test_name:12s} {err:.2e}  {status}")

    return results

if __name__ == "__main__":
    key = jrand.PRNGKey(7)
    k_init, k_noise, k_t1, k_t2 = jrand.split(key, 4)

    x = initial_cluster()
    x = x + 0.1 * jrand.normal(k_noise, x.shape)

    print("Untrained Model")
    params = init_egnn(k_init, d=32, n_layers=3)
    run_all_tests(params, x, k_t1)

    if os.path.exists("egnn_autodiff.pkl"):
        print("\n Trained Model")
        with open("egnn_autodiff.pkl", "rb") as fh:
            saved = pkl.load(fh)
        run_all_tests(saved["params"], x, k_t2)