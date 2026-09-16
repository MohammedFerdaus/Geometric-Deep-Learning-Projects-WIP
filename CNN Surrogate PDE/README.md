# Project 01 — CNN Surrogate PDE (Heat Equation)

Part of the GDL Foundations series — a structured sequence of geometric Deep learning projects built entirely from scratch in Python using only Numpy and JAX.

**Series:** Phase 1 Foundations
1. CNN — Heat Equation Surrogate ← you are here
2. RNN/LSTM — Reduced-Order Dynamics Model
3. GNN — Interatomic Property Predictor
4. E(3)-Equivariant Network — Rotation-Equivariant Force Field
5. GNN + Equivariance — Machine-Learned Interatomic Potential
6. CNN + RNN — Spatiotemporal PDE Surrogate
7. GNN + Transformer — Long-Range Interaction Model
---
 
## The Problem
 
The 2D heat equation describes how a temperature field diffuses over time on a plate or cross-section:
 
$$\frac{\partial u}{\partial t} = \alpha \nabla^2 u, \quad (x, y) \in [0, H] \times [0, W]$$
 
with fixed (Dirichlet) boundary conditions and a smooth initial temperature distribution. This project trains a CNN to approximate the solution operator — mapping an initial field directly to the field after $N$ timesteps — replacing an explicit finite-difference solver with a single learned forward pass.
 
This project builds it three ways and compares them:
 
1. Explicit finite-difference (FD) solver — the ground-truth physics engine
2. A single-layer linear CNN, used as a correctness check against the known discrete stencil
3. A deep multi-channel CNN surrogate that replaces an 8-step FD rollout
---
 
## Repository Structure
 
```
heat_cnn_surrogate/
├── fd_solver.py     — stability limit, Laplacian stencil, BC, rollout, IC generation
├── autograd.py      — Tensor, ops, im2col/col2im conv2d, SGD (reverse-mode autodiff engine)
├── dataset.py       — build_dataset, train_test_split, batch_iterator
├── model.py         — CNNSurrogate (generic conv stack built on autograd.py)
├── train.py         — training loop, boundary-overwrite helper
├── evaluate.py      — metrics, kernel comparison, rollout comparison, plots
└── gradcheck.py     — numerical gradient checker and full op test suite
```
 
---
 
## Method 1 — Finite-Difference Solver
 
### The Mathematics
 
The heat equation is discretized on a grid using the 5-point Laplacian stencil:
 
$$\nabla^2 u_{i,j} \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{\Delta x^2}$$
 
Advancing one timestep with explicit Euler integration:
 
$$u_{i,j}^{n+1} = u_{i,j}^{n} + \alpha \Delta t \, \nabla^2 u_{i,j}^{n}$$
 
with the boundary re-fixed to the known Dirichlet value after every step. This is itself a convolution — the 5-point stencil is a fixed 3×3 kernel, which is the conceptual bridge this project is built on: a CNN layer applying a learned 3×3 kernel is structurally identical to one physics timestep.
 
### Stability
 
Explicit schemes are only conditionally stable. A von Neumann stability analysis — assuming a Fourier error mode and finding the timestep at which its amplification factor exceeds 1 — gives the timestep bound:
 
$$\Delta t \leq \frac{1}{2\alpha\left(\frac{1}{\Delta x^2} + \frac{1}{\Delta y^2}\right)}$$
 
which for a square grid reduces to the familiar CFL-like limit $\alpha \Delta t / \Delta x^2 \leq 0.25$. `choose_stable_dt` applies a safety factor under this bound before any data generation, since an unstable timestep can produce trajectories with no crash or NaN for many steps while still being numerically invalid — data a CNN would otherwise learn to reproduce.
 
### Initial Conditions
 
Random smooth initial fields are generated as a sum of 2D Gaussian bumps with randomized center, amplitude, and width — width scaled relative to the grid size rather than a fixed constant, so as to produce physically realistic smooth temperature distributions rather than near-delta-function spikes.
 
---
 
## Method 2 — Single-Layer CNN (Correctness Check)
 
### Purpose
 
Before trusting any deep model's output, a single conv layer with no activation function is trained on one-step ($N=1$) FD data. Because one explicit Euler step is itself a single 3×3 convolution, this model has exactly the right capacity to recover the true stencil — no more, no less. If the learned kernel converges toward:
 
$$
K_{\text{true}} = \frac{\alpha \Delta t}{\Delta x^2}
\begin{bmatrix}
0 & 1 & 0 \\
1 & -4 & 1 \\
0 & 1 & 0
\end{bmatrix}
+
\begin{bmatrix}
0 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 0
\end{bmatrix}
$$
 
the entire pipeline — autodiff, conv2d forward/backward, training loop, data generation — is validated end to end before any deeper architecture is trusted.
 
### A Note on Identifiability
 
Low loss does not automatically imply the learned kernel matches $K_{\text{true}}$ exactly. On smooth, low-frequency training data (Gaussian bumps), many kernels close to $K_{\text{true}}$ fit the observed input/output pairs almost equally well — the true stencil is only the uniquely recoverable answer if the training distribution is rich enough in high-frequency content to distinguish it from nearby alternatives. This project confirmed near-zero loss is achievable; exact stencil recovery was not required to consider the pipeline validated.
 
---
 
## The Autograd Engine
 
### Why Build One
 
Rather than hand-deriving and hardcoding a backward pass for one specific architecture, this project implements a small reverse-mode automatic differentiation engine — reused, unmodified, across every later project in the series.
 
### The Graph
 
Every `Tensor` records the parent tensors that produced it (`_prev`) and a closure (`_backward`) encoding the local gradient rule for the operation that created it. `Tensor.backward()` performs a depth-first topological sort of the graph, seeds the output gradient to 1, and walks the sort in reverse, calling each node's `_backward()` to accumulate gradient contributions into its parents via `+=` — accumulation rather than overwrite is what makes tensors correctly handle fan-out (a value reused more than once in the graph).
 
### Primitive Operations
 
| Operation | Forward | Backward |
|-----------|---------|----------|
| $v = a + b$ | $a + b$ | $\bar{a} \mathrel{+}= \bar{v}$, $\bar{b} \mathrel{+}= \bar{v}$ (shape-reduced under broadcasting) |
| $v = a \cdot b$ | $a \cdot b$ | $\bar{a} \mathrel{+}= \bar{v} b$, $\bar{b} \mathrel{+}= \bar{v} a$ |
| $v = a \,@\, b$ | matmul | $\bar{a} \mathrel{+}= \bar{v} b^T$, $\bar{b} \mathrel{+}= a^T \bar{v}$ |
| $v = \text{relu}(a)$ | $\max(0, a)$ | $\bar{a} \mathrel{+}= \bar{v} \cdot \mathbb{1}[a > 0]$ |
| $v = \text{reshape}(a)$ | reshape | $\bar{a} \mathrel{+}= \bar{v}.\text{reshape}(a.\text{shape})$ |
| $v = \text{mean}((a-y)^2)$ | MSE | $\bar{a} \mathrel{+}= \bar{v} \cdot 2(a-y)/n$ |
 
### Convolution via im2col
 
`conv2d` is implemented by unfolding overlapping receptive fields into columns (`im2col`), reducing convolution to a single matrix multiplication, then folding gradients back into the input's spatial layout (`col2im`) by scatter-adding rather than overwriting — necessary because overlapping stride windows mean a single input pixel contributes to multiple output positions, and summation (not replacement) is the correct adjoint of the forward unfold.
 
### Validation
 
Every primitive — including convolution at multiple strides and paddings, and a fan-out case where one tensor is reused three times in a single expression — is checked against central-difference numerical gradients:
 
$$\frac{\partial \mathcal{L}}{\partial \theta_i} \approx \frac{\mathcal{L}(\theta_i + \epsilon) - \mathcal{L}(\theta_i - \epsilon)}{2\epsilon}, \quad \epsilon = 10^{-5}$$
 
All 30 checks (14 test cases across scalar/broadcast/matmul/conv/multichannel/end-to-end configurations) passed with relative error on the order of $10^{-10}$ to $10^{-13}$.
 
---
 
## The Optimizer
 
Plain SGD is used throughout:
 
$$\theta_i \leftarrow \theta_i - \eta \, \bar{\theta}_i$$
 
No momentum or adaptive scaling — sufficient at this project's scale.
 
---
 
## Results
 
| Model | Task | Final Train Loss | Final Val Loss | Notes |
|-------|------|-------------------|-----------------|-------|
| Single-layer CNN | 1-step diffusion | `~3.0e-4` (small-scale test) | — | Learned kernel qualitatively matches the true stencil's sign/magnitude pattern |
| Deep CNN (4 layers, 8 channels, 5×5 kernels) | 8-step diffusion in one forward pass | `3.9e-4` | `2.3e-4` | 75 epochs, `lr=0.03`; predicted field visually matches FD ground truth, error concentrated at low magnitude, no systematic spatial bias |
 
---
 
## Libraries
 
| Purpose | Library |
|---------|---------|
| Arrays / numerics | `numpy` |
| Visualization | `matplotlib` |
 
All tensors, autodiff operations, convolution (forward and backward), the optimizer, the finite-difference solver, and the CNN architecture are implemented from scratch.
 
---
 
## How to Run
 
```bash
python gradcheck.py   # must pass before trusting anything else
python test.py
```
 
Produces `diffusion_comparison.png` (initial field, FD ground truth, CNN prediction, absolute error) and, optionally, a training loss curve.
