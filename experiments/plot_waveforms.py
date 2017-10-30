import numpy as np
import matplotlib.pyplot as plt

from neural_noise import plot

# load noise data
noise = np.load('/users/edu/data/noise/noise.npy')
noise.shape
N, _, _ = noise.shape

# sample
percentage = 0.01
idx = np.random.choice(N, int(percentage*N), replace=False)
sample = noise[idx, :, :]
sample.shape

plot.waveform(sample[0, :, 0])
plt.show()

plot.waveforms(sample, channels=range(7))
plt.show()