# Project 02 — RNN/LSTM Reduced-Order Dynamics Model (0D Plasma Energy Balance)

Part of the GDL Foundations series — a structured sequence of geometric deep learning projects built entirely from scratch in Python using only NumPy and JAX.

**Series:** Phase 1 Foundations
1. CNN — Heat Equation Surrogate
2. RNN/LSTM — Reduced-Order Dynamics Model ← you are here
3. GNN — Interatomic Property Predictor
4. E(3)-Equivariant Network — Rotation-Equivariant Force Field
5. GNN + Equivariance — Machine-Learned Interatomic Potential
6. CNN + RNN — Spatiotemporal PDE Surrogate
7. GNN + Transformer — Long-Range Interaction Model

---

## The Problem

A simplified 0D tokamak scoping model tracks two coupled state variables — density $n$ and temperature $T$ — driven by an external heating power $P_{heat}(t)$:

$$\frac{dn}{dt} = S_p - \frac{n}{\tau_p}, \qquad \frac{dT}{dt} = \frac{P_{heat}(t)}{n \, c_v} - \frac{T}{\tau_E(n, T)}$$

with the energy confinement time given by an ITER89P-style empirical scaling law:

$$\tau_E(n, T) = C \, I^a B^b n^c T^d$$

Because $\tau_E$ depends on the very states it damps, this system exhibits real nonlinear feedback — higher $T$ shortens $\tau_E$, which increases energy loss, which pulls $T$ back down. Combined with an exogenous, time-varying $P_{heat}(t)$, this is a harder rollout target than an autonomous chaotic system: the model has to learn both the internal feedback and how to correctly incorporate a known external control signal.

This project trains an LSTM — gates implemented explicitly, no `nn.LSTM` — to learn the discrete-time state-transition function $h_t = f(h_{t-1}, x_t)$, where the "hidden state" is physically meaningful: $(n, T)$ evolving under $P_{heat}$. The trained model is then rolled out autoregressively (its own predictions fed back in as the next input) and compared against RK4 ground truth, with a specific focus on where and how fast the LSTM's rollout error compounds.

---

## Repository Structure

```
plasma_lstm/
├── physics.py     — tau_E scaling law, ODE right-hand side, RK4 integrator, default_params
├── forcing.py      — step/ramp/sinusoid P_heat(t) generators, randomized forcing sampler
├── data.py        — trajectory dataset generation, train/val/test split, normalization, windowing
├── lstm.py         — from-scratch LSTM cell (explicit gates), lax.scan forward pass, predict
├── train.py        — MSE loss, optax-based train step, training loop
├── rollout.py      — autoregressive rollout, ground-truth comparison
├── evaluate.py     — trajectory/phase-space/error plots, pooled stiffness-vs-error analysis
└── main.py         — end-to-end orchestration and config
```

---

## Method — Ground Truth, Then Learned Surrogate

### The Physics Model

All physical quantities are kept in normalized, dimensionless units (rather than raw physical scales like $n \sim 10^{19}$) so that both the RK4 trajectories and the LSTM's training targets stay O(1) — this matters directly for gradient-based training, since raw physical units would produce badly-scaled gradients before any model design choice was even in play.

Default parameters place the system's fixed point at $n^* = S_p \tau_p = 1.0$, $T^* = 1.0$, verified directly: a constant-forcing run initialized exactly at $(1,1)$ stays there for the full horizon with zero drift, confirming the equilibrium algebra is correct. Perturbed initial conditions relax back to this fixed point smoothly under constant forcing — real dynamics only emerge under time-varying $P_{heat}(t)$, which is what `forcing.py`'s step/ramp/sinusoid generators are for.

### Integration

`rk4_step` implements classical 4th-order Runge-Kutta; `simulate` calls it in a loop to produce a full $(t, n, T)$ trajectory from an initial condition and a forcing function. `derivatives` returns a `numpy` array (not a tuple), which is required for the elementwise vector arithmetic (`state + dt/2 * k1`, etc.) inside `rk4_step` to behave correctly rather than triggering tuple repetition.

### Forcing Signal Design

$P_{heat}(t)$ is deliberately kept non-negative across all three forcing kinds (`sinusoid_forcing`'s amplitude is sampled strictly below its offset), since negative heating power has no physical meaning and would otherwise hand the model an ambiguous training signal. Step forcing is allowed to default to zero power before $t_{on}$ — this produces the widest excursions in $T$ (down to roughly 0.12–0.17 in some draws) and is kept deliberately, since it adds valuable dynamic-range variety to the training data.

### From-Scratch LSTM

Gate equations are written out explicitly in `lstm.py`:

$$
\begin{aligned}
f_t &= \sigma(x_t W_{fx} + h_{t-1} W_{fh} + b_f) \\
i_t &= \sigma(x_t W_{ix} + h_{t-1} W_{ih} + b_i) \\
\tilde{c}_t &= \tanh(x_t W_{cx} + h_{t-1} W_{ch} + b_c) \\
c_t &= f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \\
o_t &= \sigma(x_t W_{ox} + h_{t-1} W_{oh} + b_o) \\
h_t &= o_t \odot \tanh(c_t)
\end{aligned}
$$

followed by a linear output projection $h_t \to (n, T)$. Sequence iteration uses `jax.lax.scan` rather than a Python `for` loop, so the recurrence jits and traces efficiently. Batching over multiple sequences is handled by wrapping `predict` in `jax.vmap` inside the loss function (`in_axes=(None, 0)` — parameters unbatched, sequences batched) rather than writing batching logic into the LSTM itself.

Gate saturation was checked directly: driving the input to strongly positive/negative extremes with large weights confirmed $c_t$ saturates cleanly to 1.0 / 0.0 respectively, verifying the sign conventions on every gate before any training was trusted.

### Training

One-step-ahead teacher-forced MSE, `optax.adam`, JIT-compiled train step via `jax.value_and_grad`. Loss converges from $O(10^{-2})$ to $O(10^{-6})$ within the first ~5–10 epochs across every run at every dataset scale tested — one-step prediction is not the hard part of this problem.

### Autoregressive Rollout

The rollout loop stays entirely in normalized space (predictions feed back in normalized form directly), denormalizing only once at the end for interpretation — this minimizes the number of normalize/denormalize boundary crossings, each of which is a place a scale mismatch could hide. At each step, $P_{heat}$ is taken from the true, known forcing sequence — it is never predicted, only $(n, T)$ are fed back autoregressively, since $P_{heat}$ is an exogenous control input in this problem, not a system state.

---

## Results

| Config | Train traj. | Epochs | Final train / val loss | Rollout MSE (pooled) | Stiffness correlation (pooled) |
|---|---|---|---|---|---|
| Smoke test | 50 | 50 | `6e-6` / `2e-6` | `0.067` (1 trajectory) | `-0.79` (1 trajectory — confounded, see below) |
| Mid-scale | 50 | 50 | — | `0.040` (8 trajectories) | `-0.31` (pooled) |
| Full run | 150 | 100 | `1e-6` / `1e-6` | `0.162` (10 trajectories) | `-0.34` (pooled) |

**One-step prediction is essentially solved** at every scale tested — training and validation loss both converge to the $10^{-6}$ range well before training ends, with no overfitting gap observed at any dataset size.

**Autoregressive rollout error grows exponentially before saturating.** Plotting per-step MSE on a log scale shows a clean straight-line (exponential) growth phase from roughly step 0.2–0.45 of the rollout horizon, before flattening out — consistent with the classic lstm failure mode this project set out to study: a model that is excellent at one-step prediction can still diverge badly under its own autoregressive feedback loop.

**Naive stiffness-correlation analysis is confounded by rollout time.** An early single-trajectory analysis found a strong correlation ($r = -0.79$) between $|dT/dt|$ and per-step error, suggestive of the model failing specifically in low-stiffness regions. Direct inspection showed this was largely a time confound: within a single trajectory settling toward equilibrium, $|dT/dt|$ naturally decays over the rollout window ($r = -0.96$ between $|dT/dt|$ and step index) while error naturally grows with elapsed steps ($r = +0.57$) — combining to produce a strong negative correlation with no real stiffness relationship required. Pooling the same analysis across 8–10 independent test trajectories (each with different initial conditions and forcing, so the time-confound direction is no longer shared across the pool) brought the correlation down to a stable -0.31 to -0.34 — reproducible across two independently-sized runs, and modest enough to represent a real but secondary effect rather than the dominant driver of rollout failure.

**The dominant failure mode is autoregressive trend overconfidence, not stiffness-localized breakdown.** Phase-space plots show the LSTM rollout departing from the true trajectory's path early and continuing in a self-consistent but incorrect direction — e.g. one run showed predicted $T$ overshooting to ~1.62 while the true trajectory gently declined toward ~1.19, and predicted $n$ overshooting a rising trend before plateauing off the true curve. This is consistent with the model having learned dominant directional trends from training data and extrapolating them autoregressively without a mechanism to self-correct once real dynamics diverge from the learned pattern — a failure mode directly relevant to Neural ODE approaches, where physics-consistency constraints are built into the architecture rather than left implicit.

---

## Libraries

| Purpose | Library |
|---|---|
| Arrays / numerics | `numpy` |
| Autodiff, JIT, `lax.scan`, `vmap` | `jax` |
| Optimizer | `optax` |
| Visualization | `matplotlib` |

The RK4 integrator, forcing signal generators, dataset/windowing pipeline, and every LSTM gate equation are implemented from scratch. Only the optimizer step itself (`optax.adam`) is off-the-shelf.

---

## How to Run

```bash
python main.py
```

Produces `trajectory_comparison.png` (n(t) and T(t), LSTM vs RK4), `phase_space.png` (n-T phase portrait), and `error_vs_time.png` (log-scale rollout error growth), plus printed pooled rollout MSE and stiffness-correlation statistics across the test set.
