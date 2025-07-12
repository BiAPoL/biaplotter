import matplotlib.pyplot as plt
import numpy as np
import pytest

from biaplotter.artists import Histogram2D, Scatter


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
        start=0, stop=5, num=size, endpoint=False, dtype=int
    )
    assert len(collected_color_indices_signals) == 1
    assert np.all(collected_color_indices_signals[0] == scatter.color_indices)

    # Test Scatter properties
    scatter.visible = True

    assert scatter.data.shape == (size, 2)
    assert scatter.visible == True
    assert scatter.color_indices.shape == (size,)

    # Test scatter colors
    colors = scatter._mpl_artists["scatter"].get_facecolors()
    assert np.all(colors[0] == scatter.overlay_colormap(0))
    assert np.all(colors[50] == scatter.overlay_colormap(2))

    # Test axis limits
    x_margin = 0.05 * (np.max(data[:, 0]) - np.min(data[:, 0]))
    y_margin = 0.05 * (np.max(data[:, 1]) - np.min(data[:, 1]))
    assert np.isclose(
        ax.get_xlim(),
        (np.min(data[:, 0]) - x_margin, np.max(data[:, 0]) + x_margin),
    ).all()
    assert np.isclose(
        ax.get_ylim(),
        (np.min(data[:, 1]) - y_margin, np.max(data[:, 1]) + y_margin),
    ).all()

    # Test size property
    scatter.size = 5.0
    assert scatter.size == 5.0
    sizes = scatter._mpl_artists["scatter"].get_sizes()
    assert np.all(sizes == 5.0)

    scatter.size = np.linspace(1, 10, size)
    assert np.all(scatter.size == np.linspace(1, 10, size))
    sizes = scatter._mpl_artists["scatter"].get_sizes()
    assert np.all(sizes == np.linspace(1, 10, size))

    # Test size reset when new data is set
    scatter.data = np.random.rand(size // 2, 2)
    assert np.all(scatter.size == scatter.default_size)  # that's the default
    sizes = scatter._mpl_artists["scatter"].get_sizes()
    assert np.all(sizes == scatter.default_size)

    # test alpha
    scatter.alpha = 0.5
    assert np.all(scatter._mpl_artists["scatter"].get_alpha() == 0.5)

    # test alpha reset when new data is set
    scatter.data = np.random.rand(size, 2)
    assert np.all(scatter._mpl_artists["scatter"].get_alpha() == 1.0)

    # Test changing overlay_colormap
    assert scatter.overlay_colormap.name == "cat10_modified"
    scatter.overlay_colormap = plt.cm.viridis
    assert scatter.overlay_colormap.name == "viridis"

    # Test scatter color indices after continuous overlay_colormap
    scatter.color_indices = np.linspace(0, 1, size)
    colors = scatter._mpl_artists["scatter"].get_facecolors()
    assert np.all(colors[0] == plt.cm.viridis(0))

    # Test scatter color_normalization_method
    scatter.color_normalization_method = "log"
    assert scatter.color_normalization_method == "log"

    # test handling NaNs
    data_with_nans = np.copy(scatter.data)
    data_with_nans[0, 0] = np.nan
    scatter.data = data_with_nans

    # test changing axes labels
    scatter.x_label_text = "X-axis"
    scatter.y_label_text = "Y-axis"
    assert scatter.x_label_text == "X-axis"
    assert scatter.y_label_text == "Y-axis"

    # test changing axes labels colors
    scatter.x_label_color = "red"
    scatter.y_label_color = (0, 0, 1, 1)
    assert scatter.x_label_color == "red"
    assert scatter.y_label_color == (0, 0, 1, 1)

    # check highlighted points is empty
    assert scatter.highlighted is None

    # Test highlighted_changed_signal
    collected_highlighted_signals = []
    def on_highlighted_changed(highlighted):
        collected_highlighted_signals.append(highlighted)
    scatter.highlighted_changed_signal.connect(on_highlighted_changed)
    assert len(collected_highlighted_signals) == 0

    # set scatter highlighted to a boolean array same size as data
    highlighted = np.zeros(size, dtype=bool)
    # Highlight the first point
    highlighted[0] = True
    scatter.highlighted = highlighted
    assert np.all(scatter.highlighted == highlighted)
    # check highlighted points are correctly set
    assert scatter.size[0] == scatter.default_size * 3
    # check if edgecolor is magenta
    assert np.array_equal(scatter._mpl_artists["scatter"].get_edgecolors()[0], (1, 0, 1, 1))

    # check highlighted_changed_signal is emitted
    assert len(collected_highlighted_signals) == 1

def test_histogram2d():
    # Inputs
    threshold = 20
    size = 1000
    bins = 20

    # Expected output
    indices_non_zero_overlay = (
        np.array([8, 9, 10], dtype=int),
        np.array([8, 9, 7], dtype=int),
    )

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
    assert (
        histogram.overlay_colormap.name == "cat10_modified_first_transparent"
    )
    assert histogram.histogram_color_normalization_method == "linear"
    assert histogram.histogram_interpolation == "nearest"
    assert histogram.overlay_interpolation == "nearest"
    assert histogram.overlay_opacity == 1
    assert histogram.overlay_visible == True

    assert histogram.cmin == 0

    # Test overlay colors
    overlay_array = histogram._mpl_artists[
        "overlay_histogram_image"
    ].get_array()
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
    assert (
        histogram.overlay_color_normalization_method == "linear"
    )  # categorical colormap does not support log normalization

    # Test changing overlay_colormap to a continuous colormap and color_normalization_method to "log"
    histogram.overlay_colormap = plt.cm.viridis
    histogram.overlay_color_normalization_method = "log"
    assert histogram.overlay_colormap.name == "viridis"
    assert (
        histogram.overlay_color_normalization_method == "log"
    )  # continuous colormap supports log normalization

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
    assert "overlay_histogram_image" not in histogram._mpl_artists.keys()


    # Test highlighted property
    ## Test highlighted_changed_signal
    collected_highlighted_signals = []
    def on_highlighted_changed(highlighted):
        collected_highlighted_signals.append(highlighted)
    histogram.highlighted_changed_signal.connect(on_highlighted_changed)
    assert len(collected_highlighted_signals) == 0

    # Set histogram2D highlighted to a boolean array same size as data
    assert histogram.highlighted is None
    highlighted = np.zeros(size, dtype=bool)
    # Highlight the first point
    highlighted[0] = True
    histogram.highlighted = highlighted
    assert np.all(histogram.highlighted == highlighted)
    # Find the bin of the highlighted point
    x_edges, y_edges = histogram._histogram[1], histogram._histogram[2]
    bin_x = np.digitize(data[0, 0], x_edges) - 1
    bin_y = np.digitize(data[0, 1], y_edges) - 1
    histogram_array = histogram._mpl_artists["histogram_image"].get_array()
    # Check if the highlighted bin alpha is set to 1.0
    assert histogram_array[bin_y, bin_x, -1] == 1.0  # Histogram array is a 3D array colored image (RGBA)), so it is transposed regarding y and x
    # Check if rectangle patches around highlighted bins are drawn
    assert len(histogram._highlighted_bin_patches) > 0
    # Check position of the highlighted bin patch
    highlighted_patch = histogram._highlighted_bin_patches[0]
    assert highlighted_patch.get_x() == x_edges[bin_x]
    assert highlighted_patch.get_y() == y_edges[bin_y]

    # Check signal is emitted
    assert len(collected_highlighted_signals) == 1
