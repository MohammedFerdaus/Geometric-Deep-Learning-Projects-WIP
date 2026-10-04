import os
import numpy as np

from torch_geometric.datasets import QM9

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(THIS_DIR, 'qm9')

def load_raw_dataset(root=DEFAULT_ROOT):
    dataset = QM9(root=root)
    
    return dataset

def extract_target(qm9_entry, target_index=7):
    
    return float(qm9_entry.y[0, target_index])

def qm9_entry_to_arrays(qm9_entry, target_index=7):
    z = qm9_entry.z.numpy()
    pos = qm9_entry.pos.numpy()
    target = extract_target(qm9_entry, target_index=target_index)
    
    return {'z': z, 'pos': pos, 'target': target}

def load_qm9_subset(root=DEFAULT_ROOT, n_molecules=2000, target_index=7, seed=0):
    dataset = load_raw_dataset(root)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=n_molecules, replace=False)

    molecules = []
    for idx in indices:
        entry = dataset[idx]
        molecules.append(qm9_entry_to_arrays(entry, target_index=target_index))

    return molecules

def train_val_test_split(molecules, val_frac=0.1, test_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    n = len(molecules)
    perm = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_idx = perm[:n_test]
    val_idx = perm[n_test:n_test + n_val]
    train_idx = perm[n_test + n_val:]

    train = [molecules[i] for i in train_idx]
    val = [molecules[i] for i in val_idx]
    test = [molecules[i] for i in test_idx]

    return train, val, test

if __name__ == "__main__":
    ds = load_raw_dataset()
    print(ds[0].y)