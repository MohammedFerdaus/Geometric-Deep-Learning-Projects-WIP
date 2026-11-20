from functools import partial
import jax
import jax.numpy as jnp
import jax.random as jrand
from egnn import (init_egnn,
                  predict_energy,
                  predict_forces_autodiff,
                  predict_forces_direct)

def loss_fn(params, X, E, F, mode, lam_e=1.0, lam_f=1.0):
  in_axes = (None, 0)
  batched_energy = jax.vmap(predict_energy, in_axes=in_axes)

  if mode == "autodiff":
    batched_forces = jax.vmap(predict_forces_autodiff, in_axes=in_axes)
    pred_E = batched_energy(params, X)
    pred_F = batched_forces(params, X)

    loss_e = jnp.mean((pred_E - E) ** 2)
    loss_f = jnp.mean((pred_F - F) ** 2)

    return lam_e * loss_e + lam_f * loss_f
  else:
    batched_forces = jax.vmap(predict_forces_direct, in_axes=in_axes)
    pred_F = batched_forces(params, X)
    loss_f = jnp.mean((pred_F - F) ** 2)

    return loss_f

def adam_init(params):
  return {
      "m": jax.tree_util.tree_map(jnp.zeros_like, params),
      "v": jax.tree_util.tree_map(jnp.zeros_like, params),
      "t": 0,
  }

def adam_update(params, grads, state, lr, b1=0.9, b2=0.999, eps=1e-8):
  t = state["t"] + 1
  m = jax.tree_util.tree_map(
      lambda m_val, g: b1 * m_val + (1 - b1) * g, state["m"], grads
  )
  v = jax.tree_util.tree_map(
      lambda v_val, g: b2 * v_val + (1 - b2) * (g**2), state["v"], grads
  )

  m_hat = jax.tree_util.tree_map(lambda m_val: m_val / (1 - (b1**t)), m)
  v_hat = jax.tree_util.tree_map(lambda v_val: v_val / (1 - (b2**t)), v)

  new_params = jax.tree_util.tree_map(
      lambda p, mh, vh: p - lr * mh / (jnp.sqrt(vh) + eps), params, m_hat, v_hat
  )

  return new_params, {"m": m, "v": v, "t": t}

@partial(jax.jit, static_argnames=("mode",))
def train_step(params, opt_state, X, E, F, lr, mode):
  loss, grads = jax.value_and_grad(loss_fn)(params, X, E, F, mode)
  params, opt_state = adam_update(params, grads, opt_state, lr)

  return params, opt_state, loss

def evaluate(params, data, mode):
  if mode == "autodiff":
    batched_forces = jax.vmap(predict_forces_autodiff, in_axes=(None, 0))
    batched_energy = jax.vmap(predict_energy, in_axes=(None, 0))
    pred_E = batched_energy(params, data["x"])
    pred_F = batched_forces(params, data["x"])
    energy_mae = jnp.mean(jnp.abs(pred_E - data["e"]))
  else:
    batched_forces = jax.vmap(predict_forces_direct, in_axes=(None, 0))
    pred_F = batched_forces(params, data["x"])
    energy_mae = None

  force_mae = jnp.mean(jnp.abs(pred_F - data["f"]))

  return {"force_mae": force_mae, "energy_mae": energy_mae}

def train(
    key, split, mode, d=32, n_layers=3, epochs=50, batch_size=64, lr=1e-3
):
  params = init_egnn(key, d=d, n_layers=n_layers)
  opt_state = adam_init(params)

  history = {"train_loss": [], "val_force_mae": []}
  best_val, best_params = float("inf"), params

  num_train = len(split["train"]["x"])

  for epoch in range(0, epochs):
    lr_epoch = lr * 0.5 * (1 + jnp.cos(jnp.pi * epoch / epochs)) 

    key, subkey = jrand.split(key)
    perm = jrand.permutation(subkey, num_train)

    epoch_losses = []

    for start in range(0, num_train, batch_size):
      idx = perm[start : start + batch_size]

      X_batch = split["train"]["x"][idx]
      E_batch = split["train"]["e"][idx]
      F_batch = split["train"]["f"][idx]

      params, opt_state, loss = train_step(params, opt_state,
                                     X_batch, E_batch, F_batch, lr_epoch, mode)

      epoch_losses.append(loss)

    train_loss = jnp.mean(jnp.array(epoch_losses))
    val = evaluate(params, split["val"], mode)

    history["train_loss"].append(train_loss)
    history["val_force_mae"].append(val["force_mae"])

    if val["force_mae"] < best_val:
      best_val = val["force_mae"]
      best_params = params

    if epoch % 5 == 0:
      print(
          f"epoch: {epoch}, train loss: {train_loss}, val force MAE:"
          f" {val['force_mae']}"
      )

  return best_params, history