import numpy as np
import matplotlib.pyplot as plt

from biaplotter.artists import (
    Scatter, Histogram2D
)


def test_scatter():
    size = 100
    data = np.random.rand(size, 2)
    fig, ax = plt.subplots()
    scatter = Scatter(ax)
    scatter.data = data
    scatter.visible = True
    # array of increasing repeating integers from 0 to 5
    scatter.color_indices = np.linspace(
        start=0, stop=5, num=size, endpoint=False, dtype=int)

    # Test Scatter properties
    assert scatter.data.shape == (size, 2)
    assert scatter.visible == True
    assert scatter.color_indices.shape == (size,)
    assert scatter.color_indices.dtype == int

    # Test scatter colors
    colors = scatter._scatter.get_facecolors()
    assert np.all(colors[0] == scatter._colormap(0))
    assert np.all(colors[50] == scatter._colormap(2))


def test_histogram2d():
    # Inputs
    threshold = 20
    size = 1000
    bins = 20

    # Expected output
    indices_non_zero_overlay = (
        np.array([8, 9, 10], dtype=int), np.array([8, 9, 7], dtype=int))

    np.random.seed(42)

    # Generate gaussian distribution 2d data
    x = np.random.normal(loc=0, scale=1, size=size)
    y = np.random.normal(loc=0, scale=1, size=size)
    data = np.column_stack([x, y])
    fig, ax = plt.subplots()
    histogram = Histogram2D(ax)
    histogram.data = data
    histogram.visible = True
    histogram.bins = bins

    # Set color indices of data in patches exceeding threshold to 1 (orange color)
    indices = histogram.indices_in_above_threshold_patches(threshold=threshold)
    color_indices = np.zeros(size)
    color_indices[indices] = 1
    histogram.color_indices = color_indices

    # Test Histogram2D properties
    assert histogram.data.shape == (size, 2)
    assert histogram.visible == True
    assert histogram.color_indices.shape == (size,)
    assert histogram.color_indices.dtype == int
    assert histogram.bins == bins

    # Test overlay colors
    overlay_array = histogram._overlay.get_array()
    assert overlay_array.shape == (bins, bins, 4)
    # indices where overlay_array is not zero
    indices = np.where(overlay_array[..., -1] != 0)
    assert np.all(indices[0] == indices_non_zero_overlay[0])
