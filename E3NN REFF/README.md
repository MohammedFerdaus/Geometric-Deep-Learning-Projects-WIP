Project 04 — E(3)-Equivariant Network (Rotation-Equivariant Force Field)
Part of the GDL Foundations series — a structured sequence of geometric deep learning projects built entirely from scratch in Python using only NumPy and JAX.

Series: Phase 1 Foundations

1. CNN — Heat Equation Surrogate
2. RNN/LSTM — Reduced-Order Dynamics Model
3. GNN — Interatomic Property Predictor
4. E(3)-Equivariant Network — Rotation-Equivariant Force Field ← you are here
5. GNN + Equivariance — Machine-Learned Interatomic Potential
6. CNN + RNN — Spatiotemporal PDE Surrogate
7. GNN + Transformer — Long-Range Interaction Model

## The Problem

Atoms in a cluster interact through the Lennard-Jones potential:

$$V(r) = 4\varepsilon\left[\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^{6}\right], \qquad E(x) = \sum_{i \lt j} V(\|x_i - x_j\|), \qquad F_i = -\nabla_{x_i} E$$

This project trains a network to predict the per-atom forces of a 13-atom LJ cluster (the icosahedral magic number cluster, LJ13) directly from atomic positions, and then verifies numerically that the network respects the symmetries of the physics. Rotating, reflecting, or translating a cluster, or relabeling its atoms, must transform the forces in exactly the corresponding way:

$$E(Rx + t) = E(x), \qquad F(Rx + t) = R\,F(x), \qquad F(\pi x) = \pi F(x)$$

An ordinary network can only learn this approximately, from data. The architecture here makes it hold by construction, and the verification step checks that the guarantee is exact and not approximate.

This project builds it in four parts:

1. An exact Lennard-Jones engine — the ground-truth physics (energy, analytic forces, velocity Verlet dynamics)
2. A minimal EGNN (E(n)-Equivariant Graph Neural Network) with two interchangeable force heads
3. A numerical equivariance test suite (rotation, reflection, translation, permutation)
4. An MD stability benchmark — rolling the learned force field forward as a simulation and checking energy conservation against true LJ dynamics

## Repository Structure

```
egnn_force_field/
├── lj.py                 — LJ energy, analytic + autodiff forces, velocity Verlet, MD rollout, cluster relaxation
├── data.py               — velocity sampling, trajectory generation, dataset build, normalization, splits
├── egnn.py               — EGNN layer, radial basis features, energy head, autodiff and direct force heads
├── train.py              — loss, hand-written Adam, jitted train step, evaluation, training loop
├── equivariance_test.py  — random rotation/reflection/translation/permutation tests
├── md_rollout.py         — MD with learned forces vs true LJ: energy drift, divergence, force error
├── plots.py              — training curves, force parity, energy conservation, rollout quality
└── run.py                — single control file: config, dataset caching, and stage runner
```

## Method 1 — Lennard-Jones Ground Truth

### The Mathematics

Working in reduced units ($\varepsilon = \sigma = 1$), the force on atom $i$ from atom $j$ is a scalar times the displacement vector:

$$F_{ij} = 24\varepsilon\,\frac{2(\sigma/r)^{12} - (\sigma/r)^{6}}{r^{2}}\,(x_i - x_j)$$

The code works entirely with $r^2$ and never takes a square root, which avoids an infinite gradient at zero distance on the diagonal. Self-pairs are handled by adding the identity matrix to $r^2$ before any inversion, then masking the diagonal afterward. A `jnp.where` after the division would still leak NaNs into the backward pass.

### Validation

Analytic forces are checked against `-jax.grad(lj_energy)`: maximum absolute difference $2.7\times10^{-15}$ in float64. The relaxed LJ13 cluster, started from a scaled icosahedron and descended by gradient descent, reaches $E = -44.3268$, the known LJ13 ground state.

### Dynamics

Time evolution uses velocity Verlet, a symplectic, time-reversible integrator whose energy error stays bounded rather than drifting. At $\Delta t = 0.005$ the true LJ dynamics conserve total energy to a maximum deviation of $3.5\times10^{-3}$ (relative $8.8\times10^{-5}$) over 20 time units. The rollout is a `jax.lax.scan`, so the whole trajectory compiles to one fused operation.

### Dataset

Twenty independent MD trajectories (20,000 steps each) start from the relaxed icosahedron with random Maxwell-Boltzmann velocities ($kT = 0.3$, center-of-mass drift removed). After discarding 1,000 burn-in steps and keeping every 20th frame, this gives 19,020 labeled snapshots (positions, true energy, true forces), split 80/10/10 into train/val/test.

Energies and forces are normalized with one shared scale (energies are also mean-shifted). Since $F = -\partial E/\partial x$, rescaling the two by different factors would make the labels inconsistent with each other.

## Method 2 — The EGNN

### The Layer

Each atom carries an invariant feature vector $h_i$ and a position $x_i$. One layer, on the fully connected graph of 13 atoms:

$$m_{ij} = \phi_e\!\left(h_i,\; h_j,\; \mathrm{rbf}(\lVert x_i - x_j \rVert)\right)$$

$$x_i' = x_i + \frac{1}{N-1}\sum_{j\neq i} (x_i - x_j)\,\phi_x(m_{ij})$$

$$h_i' = h_i + \phi_h\!\left(h_i,\; \frac{1}{N-1}\sum_{j\neq i} m_{ij}\right)$$

with $\phi_e, \phi_x, \phi_h$ small MLPs using SiLU activations.

### Why It Is Exactly Equivariant

1. Every input to every MLP — the features $h$ and a function of the distance $\lVert x_i - x_j\rVert$ — is unchanged by rotations, reflections, and translations. So all the scalars the network computes are invariant.
2. Positions enter the output in exactly one way: the vector $(x_i - x_j)$ multiplied by an invariant scalar. Under $x \to Rx + t$, that difference becomes $R(x_i - x_j)$ for any orthogonal $R$, so the update rotates exactly with the input.

Because orthogonal $R$ includes reflections, this is E(3) equivariance and not just SE(3). The guarantee depends only on this structure, not on the weights, so it holds for an untrained network just as well as a trained one.

### Two Force Heads

The same trunk supports two ways of producing forces, and the project trains and compares both:

| Head | Output | Forces | Conservative? |
|---|---|---|---|
| Autodiff | invariant scalar energy from $h$ | $F = -\nabla_x E_\theta$ via `jax.grad` | Yes, by construction |
| Direct | none | $F_i = x_i^{(L)} - x_i^{(0)}$, the net coordinate update | No |

The autodiff head is the NequIP/MACE-style formulation: the gradient of an invariant scalar is automatically an equivariant vector. Training it needs a second derivative (loss → forces → energy), so the activation must be smooth. SiLU is used throughout, since ReLU has a zero second derivative.

### A Note on Distance Features

The first version fed the raw squared distance $r^2$ into $\phi_e$. It trained, but stalled: validation force MAE sat at 0.73 (in units of the force standard deviation) and oscillated there for 30 epochs, because the network was effectively predicting zero forces. The untrained network also started with a loss of 71, since messages are summed over neighbors in several layers.

The MD rollout made the failure concrete. With no learned repulsive wall, atoms passed through each other and the true LJ energy along the model's trajectory reached $\sim10^{14}$.

Two changes fixed it, with no loss of exact equivariance:

- **Radial basis expansion.** The distance is expanded into 16 Gaussians, $\exp\!\left(-(r-\mu_k)^2 / 2\delta^2\right)$, with centers spread over $[0.7, 3.0]$. LJ forces scale like $r^{-13}$, which an MLP struggles to construct from one scalar. The input is still a function of $r^2$ only.
- **Small initial outputs and mean aggregation.** The last layer of each MLP is scaled down at initialization ($0.01$ for $\phi_x$, $0.1$ for $\phi_h$ and the energy head), and messages are averaged instead of summed over neighbors.

With these changes, validation MAE fell from 0.77 to 0.19 within 10 epochs on the same small dataset.

## Training

Hand-written Adam (no optimizer library), batch size 64, learning rate $3\times10^{-4}$, 100 epochs, width 32, 3 layers. The loss is the mean squared error on energy and forces for the autodiff head and force MSE alone for the direct head. `jax.value_and_grad` differentiates the loss through the inner `jax.grad` automatically.

## The Equivariance Test

A random orthogonal matrix is built from a QR decomposition (with signs fixed so it is unique), with the determinant forced to $+1$ for a rotation or $-1$ for a reflection. Each test applies the transformation to the input, runs the model, and compares against transforming the model's output. Tests run on an asymmetric configuration (a perfect icosahedron would map onto itself under some rotations and hide errors), in float64.

| Test (max abs error) | Untrained, autodiff head | Untrained, direct head | Trained autodiff model | Trained direct model |
|---|---|---|---|---|
| Rotation | 2.0e-18 | 1.8e-16 | 3.0e-14 | 3.7e-14 |
| Reflection | 2.0e-18 | 1.5e-16 | 1.1e-14 | 3.9e-14 |
| Translation | 3.9e-18 | 6.7e-16 | 2.7e-14 | 4.3e-14 |
| Permutation | 6.5e-19 | 0.0 | 5.3e-15 | 1.3e-14 |

Energy invariance under rotation and reflection is $\leq 5.3\times10^{-15}$ in every case. Every error is floating-point rounding noise, for both heads, before and after training.

## Results

### Force accuracy

| Model | Final train loss | Final val force MAE (normalized) |
|---|---|---|
| Autodiff head | ~6e-4 | 0.017 |
| Direct head | ~2e-4 | 0.0074 |

Normalized units are fractions of the force standard deviation (3.78 in reduced units). On a fresh trajectory the model never saw, the autodiff head's force parity plot gives RMSE 0.081 against a force standard deviation of 3.78, about 2%. The direct head's validation curve is noticeably noisier than the autodiff head's during training, with sporadic spikes in the middle epochs before it settles.

### MD stability: energy conservation

Both learned force fields and true LJ are rolled out for 4,000 steps ($\Delta t = 0.005$, 20 time units) from the same initial state:

| Energy measured | max $\lvert\Delta E\rvert$ | drift slope | relative |
|---|---|---|---|
| True LJ dynamics | 3.5e-3 | −3.9e-6 | 8.8e-5 |
| Autodiff model — the model's own energy | 4.7e-3 | −4.9e-6 | 1.2e-4 |
| Autodiff model — true LJ energy along the model's path | 5.3e-2 | +1.6e-5 | 1.3e-3 |
| Direct model — true LJ energy along the model's path | 4.6e-2 | +6.4e-4 | 1.1e-3 |

**Findings.**

1. **A conservative learned force field conserves its own energy as well as the real physics does,** however imperfect it is: the autodiff head's own energy is conserved to the same level as true LJ. Velocity Verlet conserves whatever potential it is given.
2. **The true energy along the learned trajectory stays in a bounded band for the autodiff head,** and wanders with a drift slope about 40 times larger for the direct head.
3. **The direct head is the more accurate force predictor** (lower validation MAE, about half the force error along the rollout) **but the worse dynamical model.** Its forces are not the gradient of any energy, so they can do net work on the atoms. This accuracy-versus-conservation tradeoff is the reason machine-learned interatomic potentials predict an energy and differentiate it.
4. Neither head produces a blow-up: the rollouts stay finite, the cluster stays bounded (max radius 1.1–1.23), and the force error along the path stays flat rather than growing.

### Limitations

- The drift comparison comes from a single 20-time-unit trajectory with one random seed. The maximum deviations of the two heads are comparable, and only the drift slope and the shape of the curves separate them. Longer rollouts and several seeds would make the claim firmer.
- The train/validation/test split is shuffled across snapshots, so correlated frames from the same trajectory land in different splits and the validation MAE is somewhat optimistic. The parity and rollout results use fresh seeds but the same starting cluster and temperature, so they measure interpolation within the training distribution, not extrapolation to new clusters or temperatures.
- Trajectory divergence from the true rollout saturates quickly for both heads. MD is chaotic, so this is expected even for a very accurate force field and says little about which head is better.
- The system is one species at a fixed size (13 atoms) with a fully connected graph and no cutoff.

## Libraries

| Purpose | Library |
|---|---|
| Arrays, autodiff, `jit`, `vmap`, `scan` | jax |
| Dataset caching | numpy |
| Visualization | matplotlib |

The LJ engine, the integrator, the EGNN layers and both force heads, the loss, the Adam optimizer, the dataset pipeline, and the equivariance tests are all written from scratch. JAX supplies array operations and automatic differentiation. No graph-learning, equivariance, or optimizer library is used.

## How to Run

```
pip install jax numpy matplotlib
python run.py                          # every stage in CONFIG["stages"]
python run.py train                    # just training
python run.py equivariance rollout plots
```

All settings (dataset size, epochs, learning rate, width, which heads to train) live in the `CONFIG` dictionary at the top of `run.py`. The generated dataset is cached on disk, keyed on the data settings, so retraining does not regenerate it. Float64 is enabled throughout, which the equivariance test needs to resolve errors at the $10^{-14}$ level.

Produces `training.png`, `parity.png`, `energy.png`, and `rollout_quality.png`, plus the equivariance table printed to the terminal.
