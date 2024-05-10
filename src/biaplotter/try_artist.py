import numpy as np
import matplotlib.pyplot as plt
from biaplotter.artists import Histogram2D

np.random.seed(2)

fig, ax = plt.subplots()
histogram = Histogram2D(ax)

n_samples = 100

def generate_gaussian_data(n_samples):
    """Generate a 2D dataset with two Gaussian clusters."""
    # Gaussian 1
    x1 = np.random.normal(loc=2, scale=1, size=n_samples//2)
    y1 = np.random.normal(loc=2, scale=1, size=n_samples//2)
    # Gaussian 2
    x2 = np.random.normal(loc=-2, scale=0.5, size=n_samples//2)
    y2 = np.random.normal(loc=-2, scale=0.5, size=n_samples//2)
    x_data = np.concatenate([x1, x2])
    y_data = np.concatenate([y1, y2])
    return np.vstack([x_data, y_data]).T

data = generate_gaussian_data(n_samples)
histogram.data = data
fig # show the updated figure