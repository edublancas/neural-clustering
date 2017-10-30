import numpy as np
import matplotlib.pyplot as plt

from neural_noise import plot

# load score noise data
score = np.load('/users/edu/data/noise/noise_score.npy')
score.shape
N, _, _ = score.shape

# sample
percentage = 0.3
idx = np.random.choice(N, int(percentage*N), replace=False)
sample = score[idx, :, :]
sample.shape

plot.score(sample[:, :, 0])
plt.show()

plot.scores(sample)
plt.show()
