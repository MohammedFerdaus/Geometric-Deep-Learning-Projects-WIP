import jax.numpy as jnp

from model.layers import mlp_forward, message_passing_layer

def mpnn_forward(params, node_features, edge_index, edge_attr):
    num_nodes = node_features.shape[0]
    h = mlp_forward(params['node_embed'], node_features)

    for layer_params in params['mp_layers']:
        h = message_passing_layer(layer_params, h, edge_index, edge_attr, num_nodes)

    pooled = readout_pool(h)
    prediction = mlp_forward(params['readout'], pooled)

    return prediction

def readout_pool(h, mode='mean'):
    if mode == 'sum':
        return jnp.sum(h, axis=0)
    elif mode == 'mean':
        return jnp.mean(h, axis=0)
    else:
        raise ValueError(f"unknown pooling mode: {mode}")

def batched_mpnn_forward(params, graphs):
    predictions = []
    for g in graphs:
        pred = mpnn_forward(params, g['node_features'], g['edge_index'], g['edge_attr'])
        predictions.append(pred)

    return jnp.stack(predictions)

if __name__ == "__main__":
    import jax
    from model.init import init_mpnn_params
    from data.load_qm9 import load_qm9_subset
    from data.graph_utils import build_dataset_graphs

    key = jax.random.PRNGKey(0)
    params = init_mpnn_params(key)

    molecules = load_qm9_subset(n_molecules=3)
    graphs = build_dataset_graphs(molecules)

    preds = batched_mpnn_forward(params, graphs)
    print("predictions shape:", preds.shape)
    print("predictions:", preds)