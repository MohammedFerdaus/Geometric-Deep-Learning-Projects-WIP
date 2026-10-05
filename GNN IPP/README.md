# Project 03 — GNN Interatomic Property Predictor

Part of the GDL Foundations series a structured sequence of geometric deep learning projects built entirely from scratch in Python using only NumPy and JAX.

**Series:** Phase 1 Foundations
1. CNN — Heat Equation Surrogate
2. RNN/LSTM — Reduced-Order Dynamics Model
3. GNN — Interatomic Property Predictor ← you are here
4. E(3)-Equivariant Network — Rotation-Equivariant Force Field
5. GNN + Equivariance — Machine-Learned Interatomic Potential
6. CNN + RNN — Spatiotemporal PDE Surrogate
7. GNN + Transformer — Long-Range Interaction Model

---

## The Problem

A molecule is naturally a graph: atoms are nodes, and pairs of atoms within some distance cutoff are edges. This project implements a message-passing graph neural network (MPNN) from scratch and trains it to predict a quantum-chemical property atomization energy directly from atomic positions and types, with no bond-order or connectivity information supplied by hand.

The central architectural requirement is permutation invariance: relabeling the atoms of a molecule (atom 5 becomes atom 12, etc.) must not change the predicted property, since atom indices are an artifact of how the data was stored, not a physical property of the molecule. This project builds the architecture so that invariance holds by construction, then verifies it directly rather than assuming it.

---

## Repository Structure

```
GNN Interatomic/
├── data/
│   ├── load_qm9.py       — QM9 download/parsing (via torch_geometric IO only), target selection, splitting
│   └── graph_utils.py    — cutoff-radius edge construction, one-hot atom typing, molecule → graph conversion
├── model/
│   ├── init.py           — parameter initialization (Glorot) for every MLP in the network
│   ├── layers.py         — message / aggregate / update, as explicit vectorized functions
│   └── mpnn.py           — full forward pass: embed → message-passing stack → pool → readout
├── train.py              — loss, Adam (via optax) training loop, checkpointing/resume
├── plot_results.py       — load a checkpoint and produce all plots without retraining
└── visualize.py          — loss curve, parity plot, error histogram, error-vs-molecule-size
```

---

## The Architecture — Message Passing Neural Network

### The Three-Stage Update

For every message-passing layer $t = 1, \dots, T$:

**Message.** For every directed edge $(i, j)$:

$$m_{ij}^{(t)} = \phi_m\left(h_i^{(t-1)}, h_j^{(t-1)}, e_{ij}\right)$$

**Aggregate.** Incoming messages are summed at each destination node via scatter-add:

$$M_i^{(t)} = \sum_{j \in \mathcal{N}(i)} m_{ij}^{(t)}$$

**Update.** The aggregated message and the node's previous state are combined:

$$h_i^{(t)} = \phi_u\left(h_i^{(t-1)}, M_i^{(t)}\right)$$

$\phi_m$ and $\phi_u$ are small MLPs (two linear layers with a ReLU between them), operating on a fixed `hidden_dim=64` after an initial node-embedding layer projects the raw 5-dimensional one-hot atom type (H, C, N, O, F) into that space this keeps every message-passing layer dimensionally uniform, rather than special-casing the first layer.

After $T=3$ layers, all final node states are mean-pooled into a single graph-level vector and passed through a readout MLP to produce one scalar prediction.

### Why This Guarantees Permutation Invariance

Node embedding and the message/update MLPs are applied identically, independently, to each node or edge nothing in them depends on a node's numerical index. Aggregation and readout pooling are both order-independent reductions (sum and mean respectively), so relabeling atom indices changes the order in which rows and edges are processed, but not the result. There is no architectural path by which atom numbering can influence the output.

### A Note on Pooling and Target Scale

Early experiments used sum pooling and trained on raw internal energy (U0, QM9 target index 7), which is dominated by a strong, roughly linear dependence on molecule size. The network badly underfit even its training set under this combination train loss plateaued around 0.33 (normalized) after 40 epochs. Switching to mean pooling fixed exploding initial activations (sum pooling over ~20 node vectors, each already passed through 3 unnormalized message-passing layers, compounds badly), but then discarded direct access to atom count, making the size-dependent component of raw U0 hard to recover. Switching the target to atomization energy (index 12), which removes most of this size-dependent offset, resolved the underfitting entirely see Results below.

---

## Graph Construction

Each molecule's nodes are one-hot encoded over the five elements present in QM9 (H, C, N, O, F). Edges are built by an $O(N^2)$ pairwise-distance cutoff (default 5.0 Å) small enough to be fine for QM9's molecules (≤29 atoms), though at this cutoff most molecules are nearly fully connected, since typical bond lengths are well under 5 Å. Edge features are the raw Euclidean distance between the two atoms; no bond order or explicit connectivity is used.

---

## Training

Targets are standardized (`(y - mean) / std`, statistics computed from the training split only) before the loss is computed; predictions are converted back to eV for all reported metrics.

Training uses Adam (via `optax`) with a simple per-graph forward pass not a vectorized batch since QM9 molecules have variable atom counts and a fully batched implementation (padding, or disjoint-union/block-diagonal batching) was out of scope for this pass. This is the primary performance bottleneck; `jax.jit` is not applied for the same reason. The 50-epoch, 1000-molecule run below took approximately 5 hours on CPU.

Training supports checkpointing (every 3 epochs by default) and resuming from the last checkpoint, since runs at this scale are long enough to require running unattended across multiple sessions.

---

## Results

| Run | Target | Molecules | Epochs | Final Train Loss (normalized) | Test MAE | Test Max Error | Notes |
|-----|--------|-----------|--------|-------------------------------|----------|-----------------|-------|
| Smoke test | U0 (raw), sum pooling | 200 | 40 | `0.33` | `572 eV` | `2028 eV` | Baseline (predict-mean) MAE ≈ `800 eV`; underfit, size-dependence suspected |
| Main run | Atomization energy, mean pooling | 1000 | 50 | `0.0219` | `1.33 eV` | `6.86 eV` | Baseline (predict-mean) MAE ≈ `7.5 eV` (target_std = 9.4 eV) |

**Error distribution** (main run): mean error -1.1 eV, std 1.3 eV — a small but consistent bias toward over-predicting the magnitude of atomization energy, not centered at zero.

**Error vs. molecule size** (main run): no upward trend in absolute error as atom count increases (10–25 atoms) — confirms the switch to atomization energy resolved the size-dependence problem seen in the smoke test. The single largest error in the test set occurs on one of the smallest molecules, not the largest, the opposite of what unresolved size-dependence would predict.

---

## Known Limitations

- **No batched/vectorized forward pass.** Each graph is processed individually in a Python loop; `jax.jit` is not used, since JIT compilation assumes fixed shapes and QM9 molecules vary in atom count. This is the main reason training is slow (~5 hours for 50 epochs on 1000 molecules on CPU). A disjoint-union (block-diagonal) batching scheme would resolve this.
- **Materials Project crystal extension (periodic boundary conditions) was not pursued** — given the runtime cost of the current unbatched implementation, extending to a dataset requiring lattice-image-aware neighbor search was judged not worth the added complexity at this stage.

---

## Libraries

| Purpose | Library |
|---------|---------|
| Arrays / autodiff / GPU | `jax`, `jax.numpy` |
| Optimizer | `optax` |
| Dataset IO only | `torch_geometric.datasets.QM9` (download/parsing; no GNN layers used) |
| Visualization | `matplotlib` |

Message passing (message/aggregate/update), the MPNN architecture, parameter initialization, and the training loop are implemented from scratch. Automatic differentiation uses JAX rather than a hand-written backward pass; the optimizer uses `optax`'s Adam rather than a hand-implemented one both scoped out deliberately, since the focus of this project is the graph-network architecture itself.

---

## How to Run

```bash
python -m train                   # full training run, checkpoints every 3 epochs, resumable
python -m plot_results ae_1000   # load latest checkpoint, produce all plots
```

Produces `<run_name>_loss.png`, `<run_name>_parity.png`, `<run_name>_hist.png`, and `<run_name>_error_vs_size.png`.
