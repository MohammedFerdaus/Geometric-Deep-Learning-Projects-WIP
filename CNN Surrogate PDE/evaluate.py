import numpy as np
import matplotlib.pyplot as plt
from autograd import Tensor
from fd_solver import fd_rollout, laplacian_stencil
from train import apply_boundary_to_output

def evaluate_field_error(model, X_test, Y_test, boundary_value=None):
    pred = model.forward(Tensor(X_test, requires_grad=False)).data
    if boundary_value is not None:
        pred = apply_boundary_to_output(pred, boundary_value)

    mse = np.mean((pred - Y_test) ** 2)
    max_abs_err = np.max(np.abs(pred - Y_test))
    rel_l2 = np.linalg.norm(pred - Y_test) / (np.linalg.norm(Y_test) + 1e-12)

    return {'mse': mse, 'max_abs_err': max_abs_err, 'rel_l2': rel_l2}

def extract_effective_kernel(model, alpha, dt, dx):
    assert len(model.layers) == 1
    learned_kernel = model.layers[0]['weight'].data[0, 0]

    true_stencil = np.array([[0, 1, 0], [1, -4, 1],
                            [0, 1, 0]]) * (alpha * dt / dx**2)
    true_stencil[1, 1] += 1.0

    return learned_kernel, true_stencil

def compare_rollout_vs_direct(model, initial_field, boundary_value, n_steps, dt, dx, alpha):
    ground_trth = fd_rollout(initial_field, boundary_value, n_steps, dt, dx, alpha)
    model_input = initial_field.reshape(1, 1, *initial_field.shape)
    
    pred = model.forward(Tensor(model_input, requires_grad=False)).data
    pred = apply_boundary_to_output(pred, boundary_value)
    
    pred_field = pred[0, 0]
    error_map = np.abs(pred_field - ground_trth)

    return ground_trth, pred_field, error_map

def plot_diffusion_comparison(initial_field, ground_truth, pred_field, error_map):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    titles = ['Initial', 'Ground Truth (FD)', 'Predicited (CNN)', 'Absolute Error']
    fields = [initial_field, ground_truth, pred_field, error_map]
    for ax, title, field in zip(axes, titles, fields):
        im = ax.imshow(field, cmap='hot')
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig('diffusion_comparison.png')

def plot_kernel_comparison(learned_kernel, true_kernel):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    im0 = axes[0].imshow(learned_kernel, cmap='coolwarm')
    axes[0].set_title('Learned Kernel')
    fig.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(true_kernel, cmap='coolwarm')
    axes[1].set_title('True Stencil')
    fig.colorbar(im1, ax=axes[1])
    plt.tight_layout()
    plt.show()

def plot_loss_curves(history):
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.xlabel('epoch'); plt.ylabel('MSE loss'); plt.legend()
    plt.show()