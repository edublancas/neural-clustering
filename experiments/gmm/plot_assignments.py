"""
Plot cluster assignments
"""
import matplotlib.pyplot as plt
from matplotlib import cm

plt.scatter(x_train[:, 0], x_train[:, 1], c=clusters, cmap=cm.bwr)
plt.title("Predicted cluster assignments")
plt.show()