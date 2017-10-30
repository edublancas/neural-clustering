import numpy as np
import matplotlib as plt

# load noise data
noise = np.load('/ssd/data/eduardo/noise.npy')
noise.shape
N, _, _ = noise.shape

# sample
percentage = 0.01
idx = np.random.choice(N, int(percentage*N), replace=False)
sample = noise[idx, :, :]
sample.shape

for s in sample[:10,:,:]:
    print(s.shape)