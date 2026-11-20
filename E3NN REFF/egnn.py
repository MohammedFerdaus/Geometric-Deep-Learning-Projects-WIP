import jax
import jax.numpy as jnp
import jax.random as jrand

N_RBF = 16
R_MIN = 0.7
R_MAX = 3.0

def init_linear(key, n_in, n_out):
    return {
        "W": jrand.normal(key, shape=(n_in, n_out)) * (1.0 / jnp.sqrt(n_in)),
        "b": jnp.zeros((n_out,)),
    }

def init_mlp(key, sizes, last_scale=1.0):
    num_layers = len(sizes) - 1
    keys = jrand.split(key, num_layers)

    layers = []
    for i in range(num_layers):
        layers.append(init_linear(keys[i], sizes[i], sizes[i + 1]))

    layers[-1] = {"W": layers[-1]["W"] * last_scale, "b": layers[-1]["b"]}
    return layers

def apply_mlp(params, x, act_final=False):
    for i, lin in enumerate(params):
        W = lin["W"]
        b = lin["b"]

        x = x @ W + b

        if i < len(params) - 1 or act_final:
            x = jax.nn.silu(x)

    return x

def init_egnn(key, d, n_layers):
    keys = jrand.split(key, n_layers + 2)

    embed_key = keys[0]
    layer_keys = keys[1:n_layers + 1]
    head_key = keys[-1]

    embed = jrand.normal(embed_key, shape=(d,))
    layers = []

    for l in range(n_layers):
        k1, k2, k3 = jrand.split(layer_keys[l], 3)

        layers.append({
            "phi_e": init_mlp(k1, [2 * d + N_RBF, d, d]),
            "phi_x": init_mlp(k2, [d, d, 1], last_scale=0.01),
            "phi_h": init_mlp(k3, [2 * d, d, d], last_scale=0.1),
        })

    head = init_mlp(head_key, [d, d, 1], last_scale=0.1)

    return {"embed": embed, "layers": layers, "head": head}

def rbf_expand(r):
    mu = jnp.linspace(R_MIN, R_MAX, N_RBF)            
    delta = (R_MAX - R_MIN) / (N_RBF - 1)            
    return jnp.exp(-((r - mu) ** 2) / (2 * delta ** 2))

def egnn_layer(layer_params, h, x):
    N = h.shape[0]
    d = h.shape[1]
    eye = jnp.eye(N)

    diff = x[:, None, :] - x[None, :, :]                       
    r2 = jnp.sum(diff ** 2, axis=-1, keepdims=True)            

    r = jnp.sqrt(r2 + eye[..., None])                           
    rbf = rbf_expand(r)                                        

    h_i = jnp.broadcast_to(h[:, None, :], (N, N, d))
    h_j = jnp.broadcast_to(h[None, :, :], (N, N, d))

    msg_in = jnp.concatenate([h_i, h_j, rbf], axis=-1)          

    m = apply_mlp(layer_params["phi_e"], msg_in, act_final=True)
    m = m * (1 - eye)[..., None]

    w = apply_mlp(layer_params["phi_x"], m, act_final=False)
    x_new = x + (1.0 / (N - 1)) * jnp.sum(diff * w, axis=1)

    m_agg = jnp.sum(m, axis=1) / (N - 1)

    h_in = jnp.concatenate([h, m_agg], axis=-1)
    h_new = h + apply_mlp(layer_params["phi_h"], h_in, act_final=False)

    return h_new, x_new

def egnn_forward(params, x):
    N = x.shape[0]
    d = params["embed"].shape[0]

    h = jnp.broadcast_to(params["embed"], (N, d))
    x_cur = x

    for layer in params["layers"]:
        h, x_cur = egnn_layer(layer, h, x_cur)

    return h, x_cur

def predict_energy(params, x):
    h, _ = egnn_forward(params, x)
    per_atom = apply_mlp(params["head"], h, act_final=False)

    return jnp.sum(per_atom)

def predict_forces_autodiff(params, x):
    grad_fn = jax.grad(predict_energy, argnums=1)

    return -grad_fn(params, x)

def predict_forces_direct(params, x):
    _, x_final = egnn_forward(params, x)

    return x_final - x