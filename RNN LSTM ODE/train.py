import jax
import jax.numpy as jnp
import optax
from lstm import predict

def mse_loss(params, X_batch, Y_batch):
    batched_predict = jax.vmap(predict, in_axes=(None, 0))
    Y_pred = batched_predict(params, X_batch)

    return jnp.mean((Y_pred - Y_batch) ** 2)

def make_train_step(optimizer):
    @jax.jit
    def train_step(params, opt_state, X_batch, Y_batch):
        loss_values, grads = jax.value_and_grad(mse_loss)(params, X_batch, Y_batch)
        updates, new_opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss_values

    return train_step 

def get_batches(X, Y, batch_size, rng):
    n_samples = X.shape[0]
    perm = rng.permutation(n_samples)

    for start in range(0, n_samples, batch_size):
        idx = perm[start : start + batch_size]
        yield X[idx], Y[idx]
    
def train_loop(params, optimizer_state, optimizer, train_data, val_data,
                n_epochs, batch_size, rng):

    train_step = make_train_step(optimizer)
    X_train, Y_train = train_data
    X_val, Y_val = val_data

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):
        epoch_losses = []
        for X_batch, Y_batch in get_batches(X_train, Y_train, batch_size, rng):
            params, optimizer_state, loss_value = train_step(
                params, optimizer_state, X_batch, Y_batch)
            epoch_losses.append(loss_value)
    
        train_loss = jnp.mean(jnp.array(epoch_losses))
        val_loss = mse_loss(params, X_val, Y_val)
    
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
    
        print(f"epoch {epoch}: train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
    
    return params, history
