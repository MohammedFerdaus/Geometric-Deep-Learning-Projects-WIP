import numpy as np
from fd_solver import (generate_random_initial_condition,
                       apply_dirichlet_bc, fd_rollout)

def build_dataset(n_samples, H, W, n_steps, dt, dx, alpha, rng, boundary_value=0.0):
    X = np.zeros((n_samples, 1, H, W))
    Y = np.zeros((n_samples, 1, H, W))

    for i in range(n_samples):
        init_field = generate_random_initial_condition(H, W, rng)
        init_field = apply_dirichlet_bc(init_field, boundary_value)
        final_field = fd_rollout(init_field, boundary_value, n_steps, dt, dx, alpha)

        X[i, 0] = init_field
        Y[i, 0] = final_field

    return X, Y
 
def train_test_split(X, Y, split_ratio=0.8, rng=None):
    n = X.shape[0]
    indices = np.arange(n)
    rng.shuffle(indices)
    split_point = int(n * split_ratio)
    train_idex = indices[:split_point]
    test_idex = indices[split_point:]
    X_train, Y_train = X[train_idex], Y[train_idex]
    X_test, Y_test = X[test_idex], Y[test_idex]
    
    return X_train, Y_train, X_test, Y_test

def batch_iterator(X, Y, batch_size, rng=None, shuffle=True):
    n = X.shape[0]
    indices = np.arange(n)

    if shuffle:
        rng.shuffle(indices)
    
    for start in range(0, n, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]