import numpy as np
from autograd import (Tensor, mse_loss, SGD)
from dataset import batch_iterator

def apply_boundary_to_output(output_data, boundary_value):
    output_data[:, :, 0, :] = boundary_value
    output_data[:, :, -1, :] = boundary_value
    output_data[:, :, :, 0] = boundary_value
    output_data[:, :, :, -1] = boundary_value

    return output_data

def train(model, X_train, Y_train, X_val, Y_val, epochs,
          batch_size, lr, rng, boundary_value=0.0, log_every=1):
    
    optimizer = SGD(model.parameters(), lr)
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        epoch_losses = []
        for X_batch, Y_batch in batch_iterator(X_train, Y_train, batch_size, rng, shuffle=True):
            x_tensor = Tensor(X_batch, requires_grad=False)
            pred = model.forward(x_tensor)
            loss = mse_loss(pred, Y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.data.item())
        
        avg_train_loss = np.mean(epoch_losses)

        val_tensor = Tensor(X_val, requires_grad=False)
        val_pred = model.forward(val_tensor)
        val_pred_data = apply_boundary_to_output(val_pred.data.copy(), boundary_value)
        val_loss = np.mean((val_pred_data - Y_val) ** 2)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)

        if epoch % log_every == 0:
            print(f"epoch {epoch}: train_loss={avg_train_loss:.6f} val_loss={val_loss:.6f}")

    return model, history