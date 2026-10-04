import jax
import jax.numpy as jnp

def glorot_init(key, shape):
    fan_in, fan_out = shape[0], shape[1]
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    
    return jax.random.uniform(key, shape, minval=-limit, maxval=limit)

def init_mlp_params(key, layer_sizes):
    params = []
    keys = jax.random.split(key, len(layer_sizes) - 1)

    for i in range(len(layer_sizes) - 1):
        in_dim, out_dim = layer_sizes[i], layer_sizes[i + 1]
        W = glorot_init(keys[i], (in_dim, out_dim))
        b = jnp.zeros((out_dim,))
        params.append({'W': W, 'b': b})

    return params

def init_mpnn_params(key, node_feat_dim=5, edge_feat_dim=1, hidden_dim=64, num_layers=3, output_dim=1):
    keys = jax.random.split(key, num_layers + 2)

    params = {}
    params['node_embed'] = init_mlp_params(keys[0], [node_feat_dim, hidden_dim])
    params['mp_layers'] = []

    message_input_dim = hidden_dim + hidden_dim + edge_feat_dim
    update_input_dim = hidden_dim + hidden_dim

    for layer_idx in range(num_layers):
        layer_key = keys[layer_idx + 1]
        msg_key, upd_key = jax.random.split(layer_key)

        message_mlp = init_mlp_params(msg_key, [message_input_dim, hidden_dim, hidden_dim])
        update_mlp = init_mlp_params(upd_key, [update_input_dim, hidden_dim, hidden_dim])

        params['mp_layers'].append({'message': message_mlp, 'update': update_mlp})

    params['readout'] = init_mlp_params(keys[-1], [hidden_dim, hidden_dim, output_dim])
    params['readout'][-1]['W'] = params['readout'][-1]['W'] * 0.01
    
    return params

if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    params = init_mpnn_params(key)

    print("node_embed:", [(p['W'].shape, p['b'].shape) for p in params['node_embed']])
    print("num mp_layers:", len(params['mp_layers']))
    print("layer 0 message MLP:", [(p['W'].shape, p['b'].shape) for p in params['mp_layers'][0]['message']])
    print("layer 0 update MLP:", [(p['W'].shape, p['b'].shape) for p in params['mp_layers'][0]['update']])
    print("readout:", [(p['W'].shape, p['b'].shape) for p in params['readout']])