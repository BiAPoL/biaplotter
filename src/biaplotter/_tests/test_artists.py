import numpy as np
import matplotlib.pyplot as plt
import pytest

from biaplotter.artists import (
    Scatter, Histogram2D
)


def test_scatter():
    # Inputs
    size = 100

    # Generate some random data
    data = np.random.rand(size, 2)
    fig, ax = plt.subplots()
    scatter = Scatter(ax)

    # Test Scatter signals
    ## Test data_changed_signal
    collected_data_signals = []
    def on_data_changed(data):
        collected_data_signals.append(data)
    scatter.data_changed_signal.connect(on_data_changed)
    assert len(collected_data_signals) == 0
    scatter.data = data
    assert len(collected_data_signals) == 1
    assert np.all(collected_data_signals[0] == data)

    ## Test color_indices_changed_signal
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

    # Test Scatter properties
    scatter.visible = True

    assert scatter.data.shape == (size, 2)
    assert scatter.visible == True
    assert scatter.color_indices.shape == (size,)

    # Test scatter colors
    colors = scatter._scatter.get_facecolors()
    assert np.all(colors[0] == scatter.overlay_colormap(0))
    assert np.all(colors[50] == scatter.overlay_colormap(2))

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

    # Test changing overlay_colormap
    assert scatter.overlay_colormap.name == "cat10_modified"
    scatter.overlay_colormap = plt.cm.viridis
    assert scatter.overlay_colormap.name == "viridis"

    # Test scatter color indices after continuous overlay_colormap
    scatter.color_indices = np.linspace(0, 1, size)
    colors = scatter._scatter.get_facecolors()
    assert np.all(colors[0] == plt.cm.viridis(0))

    # Test scatter color_normalization_method
    scatter.color_normalization_method = "log"
    assert scatter.color_normalization_method == "log"


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
    ## Test data_changed_signal
    collected_data_signals = []
    def on_data_changed(data):
        collected_data_signals.append(data)
    histogram.data_changed_signal.connect(on_data_changed)
    assert len(collected_data_signals) == 0
    histogram.data = data
    assert len(collected_data_signals) == 1
    assert np.all(collected_data_signals[0] == data)
    ## Test color_indices_changed_signal
    collected_color_indices_signals = []
    def on_color_indices_changed(color_indices):
        collected_color_indices_signals.append(color_indices)
    histogram.color_indices_changed_signal.connect(on_color_indices_changed)
    assert len(collected_color_indices_signals) == 0
    # Set color indices of data in patches exceeding threshold to 1 (orange color)
    indices = histogram.indices_in_patches_above_threshold(threshold=threshold)
    color_indices = np.zeros(size)
    color_indices[indices] = 1
    histogram.color_indices = color_indices
    assert len(collected_color_indices_signals) == 1
    assert np.all(collected_color_indices_signals[0] == color_indices)

    # Test Histogram2D properties
    histogram.visible = True
    histogram.bins = bins

    assert histogram.data.shape == (size, 2)
    assert histogram.visible == True
    assert histogram.color_indices.shape == (size,)
    assert histogram.bins == bins
    assert histogram.histogram_colormap.name == "magma"
    assert histogram.overlay_colormap.name == "cat10_modified_first_transparent"
    assert histogram.histogram_color_normalization_method == "linear"
    assert histogram.histogram_interpolation == "nearest"
    assert histogram.overlay_interpolation == "nearest"
    assert histogram.overlay_opacity == 1
    assert histogram.overlay_visible == True


    # Test overlay colors
    overlay_array = histogram._overlay_histogram_image.get_array()
    assert overlay_array.shape == (bins, bins, 4)
    # indices where overlay_array is not zero
    indices = np.where(overlay_array[..., -1] != 0)
    assert np.all(indices[0] == indices_non_zero_overlay[0])

    # Test changing histogram_colormap options
    histogram.histogram_colormap = plt.cm.viridis
    assert histogram.histogram_colormap.name == "viridis"

    # Test changing histogram color_normalization_method to "log" 
    histogram.histogram_color_normalization_method = "log"
    assert histogram.histogram_color_normalization_method == "log"

    # Test changing overlay_color_normalization_method to "log" under a categorical overlay_colormap
    histogram.overlay_color_normalization_method = "log"
    assert histogram.overlay_color_normalization_method == "linear" # categorical colormap does not support log normalization

    # Test changing overlay_colormap to a continuous colormap and color_normalization_method to "log"
    histogram.overlay_colormap = plt.cm.viridis
    histogram.overlay_color_normalization_method = "log"
    assert histogram.overlay_colormap.name == "viridis"
    assert histogram.overlay_color_normalization_method == "log" # continuous colormap supports log normalization

    # Test other histogram display options
    histogram.histogram_interpolation = "bilinear"
    histogram.overlay_interpolation = "bilinear"
    histogram.overlay_opacity = 0.5
    histogram.overlay_visible = False
    assert histogram.histogram_interpolation == "bilinear"
    assert histogram.overlay_interpolation == "bilinear"
    assert histogram.overlay_opacity == 0.5
    assert histogram.overlay_visible == False
    histogram.histogram_color_normalization_method = "symlog"
    assert histogram.histogram_color_normalization_method == "symlog"
    histogram.histogram_color_normalization_method = "centered"
    assert histogram.histogram_color_normalization_method == "centered"
    histogram.overlay_color_normalization_method = "symlog"
    assert histogram.overlay_color_normalization_method == "symlog"
    histogram.overlay_color_normalization_method = "centered"
    assert histogram.overlay_color_normalization_method == "centered"

    # Don't draw overlay histogram if color_indices are nan
    histogram.color_indices = np.nan
    assert histogram._overlay_histogram_image is None

# Test calculate_statistic_histogram_method for different statistics
statistics = ["sum", "median", "mean"]
expected_results = [
    np.array([
        [    0., np.nan, np.nan],
        [np.nan,     9., np.nan],
        [np.nan, np.nan,    15.]
       ]),
    np.array([
        [0.    , np.nan, np.nan],
        [np.nan, 2.    , np.nan],
        [np.nan, np.nan,    7.5]
        ]),
    np.array([
        [0.    , np.nan, np.nan],
        [np.nan, 3.    , np.nan],
        [np.nan, np.nan,    7.5]
        ]),

]
@pytest.mark.parametrize("statistic,expected_array", zip(statistics, expected_results), ids=statistics)
def test_calculate_statistic_histogram_method(statistic,expected_array):
    input_xy_data = np.array([
            [1, 2],
            [3, 4],
            [3, 5],
            [4, 5],
            [5, 6],
            [6, 7],
        ])
    bins = 3
    input_features = np.array([0, 1, 2, 6, 7, 8])

    expected_histogram_array = np.array([
        [1., 0., 0.],
       [0., 3., 0.],
       [0., 0., 2.]])
    
    histogram = Histogram2D(data=input_xy_data, bins=bins)
    histogram_array, x_edges, y_edges = histogram.histogram
    assert np.all(histogram_array == expected_histogram_array)
    # Get the bin index for each x value ( -1 to start from index 0 and clip to handle edge cases)
    x_bin_indices = (np.digitize(
        input_xy_data[:, 0], x_edges, right=False) - 1).clip(0, len(x_edges)-2)
    # Get the bin index for each y value ( -1 to start from index 0 and clip to handle edge cases)
    y_bin_indices = (np.digitize(
        input_xy_data[:, 1], y_edges, right=False) - 1).clip(0, len(y_edges)-2)
    result = histogram._calculate_statistic_histogram(x_bin_indices, y_bin_indices, input_features, statistic=statistic)
    assert np.array_equal(result, expected_array, equal_nan=True)
