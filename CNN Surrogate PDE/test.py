import numpy as np
from model import CNNSurrogate
from train import train
from fd_solver import choose_stable_dt
from dataset import build_dataset, train_test_split
from evaluate import compare_rollout_vs_direct, plot_diffusion_comparison, plot_loss_curves

rng = np.random.default_rng(0)
H = W = 32
dx = 1.0
alpha = 1.0
dt = choose_stable_dt(alpha, dx, safety_factor=0.5)
n_steps = 8
boundary_value = 0.0

X, Y = build_dataset(500, H, W, n_steps, dt, dx, alpha, rng, boundary_value)
print("X min/max:", X.min(), X.max())
print("Y min/max:", Y.min(), Y.max())

X_train, Y_train, X_val, Y_val = train_test_split(X, Y, split_ratio=0.8, rng=rng)

layer_config = [
    (1, 8, 5, 1, 2, True),
    (8, 8, 5, 1, 2, True),
    (8, 8, 5, 1, 2, True),
    (8, 1, 5, 1, 2, False),
]
model = CNNSurrogate(layer_config, rng)

model, history = train(model, X_train, Y_train, X_val, Y_val,
                        epochs=75, batch_size=25, lr=0.03, rng=rng, boundary_value=boundary_value)

initial_field = X_val[0, 0]
ground_truth, pred_field, error_map = compare_rollout_vs_direct(
    model, initial_field, boundary_value, n_steps, dt, dx, alpha)

plot_diffusion_comparison(initial_field, ground_truth, pred_field, error_map)
plot_loss_curves(history)
print("done")

# X min/max: 0.0 1.9176886278904974
# Y min/max: 0.0 1.8322611854136095
# epoch 0: train_loss=0.060842 val_loss=0.005819
# epoch 1: train_loss=0.005994 val_loss=0.002452
# epoch 74: train_loss=0.000388 val_loss=0.000233
# done