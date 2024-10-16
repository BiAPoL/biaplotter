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
    # Test Scatter signals
    collected_data_signals = []
    def on_data_changed(data):
        collected_data_signals.append(data)
    scatter.data_changed_signal.connect(on_data_changed)
    assert len(collected_data_signals) == 0
    # Set data
    scatter.data = data
    assert len(collected_data_signals) == 1
    assert np.all(collected_data_signals[0] == data)
    collected_color_indices_signals = []
    def on_color_indices_changed(color_indices):
        collected_color_indices_signals.append(color_indices)
    scatter.color_indices_changed_signal.connect(on_color_indices_changed)
    assert len(collected_color_indices_signals) == 0
    # Set color_indices with an array of increasing repeating integers from 0 to 5
    scatter.color_indices = np.linspace(
        start=0, stop=5, num=size, endpoint=False, dtype=int)
    assert len(collected_color_indices_signals) == 1
    assert np.all(collected_color_indices_signals[0] == scatter.color_indices)
    scatter.visible = True

    # Test Scatter properties
    assert scatter.data.shape == (size, 2)
    assert scatter.visible == True
    assert scatter.color_indices.shape == (size,)
    assert scatter.color_indices.dtype == int

    # Test scatter colors
    colors = scatter._scatter.get_facecolors()
    assert np.all(colors[0] == scatter.categorical_colormap(0))
    assert np.all(colors[50] == scatter.categorical_colormap(2))

    # Test axis limits
    x_margin = 0.05 * (np.max(data[:, 0]) - np.min(data[:, 0]))
    y_margin = 0.05 * (np.max(data[:, 1]) - np.min(data[:, 1]))
    assert np.isclose(ax.get_xlim(), (np.min(data[:, 0]) - x_margin, np.max(data[:, 0]) + x_margin)).all()
    assert np.isclose(ax.get_ylim(), (np.min(data[:, 1]) - y_margin, np.max(data[:, 1]) + y_margin)).all()

    # Test size property
    scatter.size = 5.0
    assert scatter.size == 5.0
    sizes = scatter._scatter.get_sizes()
    assert np.all(sizes == 5.0)

    scatter.size = np.linspace(1, 10, size)
    assert np.all(scatter.size == np.linspace(1, 10, size))
    sizes = scatter._scatter.get_sizes()
    assert np.all(sizes == np.linspace(1, 10, size))

    # Test size reset when new data is set
    new_data = np.random.rand(size, 2)
    scatter.data = new_data
    assert scatter.size == 50.0  # that's the default
    sizes = scatter._scatter.get_sizes()
    assert np.all(sizes == 50.0)


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
    # Test Histogram2D signals
    collected_data_signals = []
    def on_data_changed(data):
        collected_data_signals.append(data)
    histogram.data_changed_signal.connect(on_data_changed)
    assert len(collected_data_signals) == 0
    histogram.data = data
    assert len(collected_data_signals) == 1
    assert np.all(collected_data_signals[0] == data)
    collected_color_indices_signals = []
    def on_color_indices_changed(color_indices):
        collected_color_indices_signals.append(color_indices)
    histogram.color_indices_changed_signal.connect(on_color_indices_changed)
    assert len(collected_color_indices_signals) == 0
    # Set color indices of data in patches exceeding threshold to 1 (orange color)
    indices = histogram.indices_in_above_threshold_patches(threshold=threshold)
    color_indices = np.zeros(size)
    color_indices[indices] = 1
    histogram.color_indices = color_indices
    assert len(collected_color_indices_signals) == 1
    assert np.all(collected_color_indices_signals[0] == color_indices)

    histogram.visible = True
    histogram.bins = bins

    # Test Histogram2D properties
    assert histogram.data.shape == (size, 2)
    assert histogram.visible == True
    assert histogram.color_indices.shape == (size,)
    assert histogram.color_indices.dtype == int
    assert histogram.bins == bins
    assert histogram.cmin == 0

    # Test overlay colors
    overlay_array = histogram._overlay.get_array()
    assert overlay_array.shape == (bins, bins, 4)
    # indices where overlay_array is not zero
    indices = np.where(overlay_array[..., -1] != 0)
    assert np.all(indices[0] == indices_non_zero_overlay[0])
