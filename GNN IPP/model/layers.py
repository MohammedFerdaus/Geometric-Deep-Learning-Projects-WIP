import jax
import jax.numpy as jnp

def mlp_forward(params, x):
    for i, layer in enumerate(params):
        x = x @ layer['W'] + layer['b']
        if i < len(params) - 1:
            x = jax.nn.relu(x)
    
    return x

def compute_messages(params_message, h, edge_index, edge_attr):
    h_src = h[edge_index[0]]
    h_dst = h[edge_index[1]]

    message_input = jnp.concatenate([h_src, h_dst, edge_attr], axis=1)
    messages = mlp_forward(params_message, message_input)

    return messages

def aggregate_messages(messages, edge_index, num_nodes):
    aggregated = jnp.zeros((num_nodes, messages.shape[-1])).at[edge_index[1]].add(messages)
    
    return aggregated

def update_nodes(params_update, h, aggregated_messages):
    update_input = jnp.concatenate([h, aggregated_messages], axis=-1)
    h_new = mlp_forward(params_update, update_input)
    
    return h_new

def message_passing_layer(layer_params, h, edge_index, edge_attr, num_nodes):
    messages = compute_messages(layer_params['message'], h, edge_index, edge_attr)
    aggregated = aggregate_messages(messages, edge_index, num_nodes)
    h_new = update_nodes(layer_params['update'], h, aggregated)
    
    return h_new