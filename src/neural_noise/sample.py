import numpy as np


def sample(arr, percentage, dim=0):
    N = arr.shape[dim]
    total_dims = len(arr.shape)
    idx = np.random.choice(N, int(percentage*N), replace=False)
    
    def _idx(i, idx):
        return idx if i == dim else None    

    slices = [_idx(i, idx) for i, idx in enumerate(total_dims)]
    return arr[slices]




