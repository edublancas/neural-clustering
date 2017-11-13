import numpy as np


def sample(arr, percentage, dim=0):
    N = arr.shape[dim]
    total_dims = len(arr.shape)
    idx = np.random.choice(N, int(percentage*N), replace=False)

    def _idx(i):
        return idx if i == dim else slice(0, arr.shape[i])

    slices = [_idx(i) for i in range(total_dims)]
    return arr[slices]

