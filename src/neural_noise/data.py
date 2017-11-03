"""
Make multivariate normal data

"""
import matplotlib.pyplot as plt
import numpy as np

# n data points
N = 100000

# data point dimension
D = 2


def build_toy_dataset(N):
    M = 10
    # all mixtures have same prob
    pi = np.ones(10)/M

    mus = [[10, -10], [20, 20], [8, -3], [-15, 7], [0.5, 0.5], [-5, 5], [5, 5],
           [10, 10], [-4, -4], [20, 0]]

    stds = [[0.3, 0], [0, 0.3]]

    x = np.zeros((N, D), dtype=np.float32)

    for n in range(N):
        # generate N data points from either of the 2 multivariate normals
        k = np.argmax(np.random.multinomial(1, pi))
        x[n, :] = np.random.multivariate_normal(mus[k], stds)

    return x


x_train = build_toy_dataset(N)

plt.scatter(x_train[:, 0], x_train[:, 1])
plt.show()
