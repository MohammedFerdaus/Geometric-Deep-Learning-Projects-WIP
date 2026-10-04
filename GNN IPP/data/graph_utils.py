import numpy as np

def compute_pairwise_distances(pos):
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    
    return dist

def build_edges_from_cutoff(pos, cutoff=5.0):
    dist = compute_pairwise_distances(pos)
    N = pos.shape[0]

    edge_index = []
    edge_attr = []

    for i in range(N):
        for j in range(N):
            if i != j and dist[i, j] <= cutoff:
                edge_index.append([i, j])
                edge_attr.append([dist[i, j]])

    edge_index = np.array(edge_index).T
    edge_attr = np.array(edge_attr)

    return edge_index, edge_attr

def molecule_to_graph(molecule, cutoff=5.0):
    z = molecule['z']
    pos = molecule['pos']
    target = molecule['target']

    node_features = one_hot_encode_atomic_number(z)
    edge_index, edge_attr = build_edges_from_cutoff(pos, cutoff=cutoff)

    return {'node_features': node_features,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'target': target,
            'num_atoms': z.shape[0]}

def one_hot_encode_atomic_number(z, allowed_elements=(1, 6, 7, 8, 9)):
    elements_to_index = {elem: i for i, elem in enumerate(allowed_elements)}
    N = z.shape[0]
    one_hot = np.zeros((N, len(allowed_elements)))

    for i, atomic_num in enumerate(z):
        if atomic_num not in elements_to_index:
            raise ValueError(f"Unexpected atomic number {atomic_num} not in {allowed_elements}")
        one_hot[i, elements_to_index[atomic_num]] = 1.0

    return one_hot

def build_dataset_graphs(molecules, cutoff=5.0):
    graphs = [molecule_to_graph(mol, cutoff=cutoff) for mol in molecules]
    
    return graphs

if __name__ == "__main__":
    from data.load_qm9 import load_qm9_subset

    molecules = load_qm9_subset(n_molecules=3)
    graphs = build_dataset_graphs(molecules, cutoff=5.0)

    for i, g in enumerate(graphs):
        print(f"--- molecule {i} ---")
        print("num_atoms:", g['num_atoms'])
        print("node_features shape:", g['node_features'].shape)
        print("edge_index shape:", g['edge_index'].shape)
        print("edge_attr shape:", g['edge_attr'].shape)
        print("target:", g['target'])
        print()