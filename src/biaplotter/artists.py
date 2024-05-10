import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Colormap, Normalize
from matplotlib.collections import QuadMesh
from abc import ABC, abstractmethod
from nap_plot_tools.cmap import cat10_mod_cmap, cat10_mod_cmap_first_transparent
from psygnal import Signal
from typing import Tuple, List
from collections import defaultdict


class Artist(ABC):
    """Abstract base class for artists in the BiAPlotter.

    Parameters
    ----------
    ax : plt.Axes, optional
        axes to plot on, by default None
    data : (N, 2) np.ndarray
        data to be plotted
    categorical_colormap : Colormap, optional
        a colormap to use for the artist, by default cat10_mod_cmap from nap-plot-tools
    color_indices : (N,) np.ndarray, optional
        array of indices to map to the colormap, by default None
    """

    def __init__(self, ax: plt.Axes = None, data: np.ndarray = None, categorical_colormap: Colormap = cat10_mod_cmap, color_indices: np.ndarray = None):
        """Initializes the abstract artist.
        """
        #: Stores data to be plotted
        self._data: np.ndarray = data
        #: Stores axes to plot on
        self.ax: plt.Axes = ax if ax is not None else plt.gca()
        #: Stores visibility of the artist
        self._visible: bool = True
        #: Stores the colormap to use for the artist
        self.categorical_colormap: Colormap = categorical_colormap
        #: Stores the array of indices to map to the colormap
        self._color_indices: np.array = color_indices

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """Abstract property for the artist's data."""
        pass

    @data.setter
    @abstractmethod
    def data(self, value: np.ndarray):
        """Abstract setter for the artist's data."""
        pass

    @property
    @abstractmethod
    def visible(self) -> bool:
        """Abstract property for the artist's visibility."""
        pass

    @visible.setter
    @abstractmethod
    def visible(self, value: bool):
        """Abstract setter for the artist's visibility."""
        pass

    @property
    @abstractmethod
    def color_indices(self) -> np.ndarray:
        """Abstract property for the indices into the colormap."""
        pass

    @color_indices.setter
    @abstractmethod
    def color_indices(self, indices: np.ndarray):
        """Abstract setter for the indices into the colormap."""
        pass

    @abstractmethod
    def draw(self):
        """Abstract method to draw or redraw the artist."""
        pass


class Scatter(Artist):
    """Scatter plot artist for the BiAPlotter.

    Inherits all parameters and attributes from abstract Artist.
    For parameter and attribute details, see the abstract Artist class documentation.

    Parameters
    ----------
    ax : plt.Axes, optional
        axes to plot on, by default None
    data : (N, 2) np.ndarray
        data to be plotted
    categorical_colormap : Colormap, optional
        a colormap to use for the artist, by default cat10_mod_cmap from nap-plot-tools
    color_indices : (N,) np.ndarray[int] or int, optional
        array of indices to map to the colormap, by default None

    Notes
    -----
    **Signals:**

        * **data_changed_signal** emitted when the data is changed.
        * **color_indices_changed_signal** emitted when the color indices are changed.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from biaplotter.artists import Scatter
    >>> data = np.random.rand(100, 2)
    >>> fig, ax = plt.subplots()
    >>> scatter = Scatter(ax)
    >>> scatter.data = data
    >>> scatter.visible = True
    >>> scatter.color_indices = np.linspace(start=0, stop=5, num=100, endpoint=False, dtype=int)
    >>> plt.show()
    """
    #: Signal emitted when the `data` is changed.
    data_changed_signal: Signal = Signal(np.ndarray)
    #: Signal emitted when the `color_indices` are changed.
    color_indices_changed_signal: Signal = Signal(np.ndarray)

    def __init__(self, ax: plt.Axes = None, data: np.ndarray = None, categorical_colormap: Colormap = cat10_mod_cmap, color_indices: np.ndarray = None):
        """Initializes the scatter plot artist.
        """
        super().__init__(ax, data, categorical_colormap, color_indices)
        #: Stores the scatter plot matplotlib object
        self._scatter = None
        self.data = data
        self.draw()  # Initial draw of the scatter plot

    @property
    def data(self) -> np.ndarray:
        """Gets or sets the data associated with the scatter plot.

        Updates colors if color_indices are set. Triggers a draw idle command.

        Returns
        -------
        data : (N, 2) np.ndarray
           data for the artist. Does not respond if set to None or empty array.

        Notes
        -----
        data_changed_signal : Signal
            Signal emitted when the data is changed.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data for the scatter plot, updating the display as needed."""
        if value is None:
            return
        if len(value) == 0:
            return
        self._data = value
        # emit signal
        self.data_changed_signal.emit(self._data)
        if self._scatter is None:
            self._scatter = self.ax.scatter(value[:, 0], value[:, 1])
            self.color_indices = 0  # Set default color index
        else:
            # If the scatter plot already exists, just update its data
            self._scatter.set_offsets(value)
        
        if self._color_indices is None:
            self.color_indices = 0  # Set default color index
        else:
            # Update colors if color indices are set, resize if data shape has changed
            color_indices_size = len(self._color_indices)
            color_indices = np.resize(self._color_indices, self._data.shape[0])
            if len(color_indices) > color_indices_size:
                # fill with zeros where new data is larger
                color_indices[color_indices_size:] = 0
            self.color_indices = color_indices
        self.draw()

    @property
    def visible(self) -> bool:
        """Gets or sets the visibility of the scatter plot.

        Triggers a draw idle command.

        Returns
        -------
        visible : bool
            visibility of the scatter plot.
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        """Sets the visibility of the scatter plot."""
        self._visible = value
        if self._scatter is not None:
            self._scatter.set_visible(value)
        self.draw()

    @property
    def color_indices(self) -> np.ndarray:
        """Gets or sets the current color indices used for the scatter plot.

        Triggers a draw idle command.

        Returns
        -------
        color_indices : (N,) np.ndarray[int] or int
            indices to map to the categorical_colormap. Accepts a scalar or an array of integers.
        
        Notes
        -----
        color_indices_changed_signal : Signal
            Signal emitted when the color indices are changed.

        """
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
        """Sets color indices for the scatter plot and updates colors accordingly."""
        # Check if indices are a scalar
        if np.isscalar(indices):
            indices = np.full(len(self._data), indices)
        # Check if indices data type is float
        if indices.dtype == float:
            indices = indices.astype(int)
        self._color_indices = indices
        if indices is not None and self._scatter is not None:
            # Normalize indices differently based on data type
            if indices.dtype == int:
                norm = Normalize(vmin=0, vmax=self.categorical_colormap.N)
            elif indices.dtype == float:
                norm = Normalize(vmin=np.nanmin(indices), vmax=np.nanmax(indices))
            new_colors = self.categorical_colormap(norm(indices))
            self._scatter.set_facecolor(new_colors)
            self._scatter.set_edgecolor(None)
        # emit signal
        self.color_indices_changed_signal.emit(self._color_indices)
        self.draw()

    def draw(self):
        """Draws or redraws the scatter plot."""
        self.ax.figure.canvas.draw_idle()


class Histogram2D(Artist):
    """2D histogram artist for the BiAPlotter.

    Inherits all parameters and attributes from abstract Artist.
    For parameter and attribute details, see the abstract Artist class documentation.

    Parameters
    ----------
    ax : plt.Axes, optional
        axes to plot on, by default None
    data : (N, 2) np.ndarray
        data to be plotted
    categorical_colormap : Colormap, optional
        a colormap to use for the artist overlay, by default cat10_mod_cmap_first_transparent from nap-plot-tools (first color is transparent)
    color_indices : (N,) np.ndarray[int] or int, optional
        array of indices to map to the categorical_colormap, by default None
    bins : int, optional
        number of bins for the histogram, by default 20
    histogram_colormap : Colormap, optional
        colormap for the histogram, by default plt.cm.magma

    Notes
    -----
    **Signals:**

        * **data_changed_signal** emitted when the data is changed.
        * **color_indices_changed_signal** emitted when the color indices are changed.

    """
    #: Signal emitted when the `data` is changed.
    data_changed_signal: Signal = Signal(np.ndarray)
    #: Signal emitted when the `color_indices` are changed.
    color_indices_changed_signal: Signal = Signal(np.ndarray)

    def __init__(self, ax: plt.Axes = None, data: np.ndarray = None, categorical_colormap: Colormap = cat10_mod_cmap_first_transparent, color_indices: np.ndarray = None, bins=20, histogram_colormap: Colormap = plt.cm.magma):
        super().__init__(ax, data, categorical_colormap, color_indices)
        """Initializes the 2D histogram artist.
        """
        #: Stores the matplotlib histogram2D object
        self._histogram_image = None
        self._bins = bins
        self._histogram_colormap = histogram_colormap
        self._overlay = None
        self.data = data
        self.draw()  # Initial draw of the histogram

    @property
    def data(self) -> np.ndarray:
        """Gets or sets the data associated with the 2D histogram.

        Updates colors if color_indices are set. Triggers a draw idle command.

        Returns
        -------
        data : (N, 2) np.ndarray
            data for the artist. Does not respond if set to None or empty array.

        Notes
        -----
        data_changed_signal : Signal
            Signal emitted when the data is changed.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data for the 2D histogram, updating the display as needed."""
        if value is None:
            return
        if len(value) == 0:
            return
        self._data = value
        # emit signal
        self.data_changed_signal.emit(self._data)
        # Remove the existing histogram to redraw
        if self._histogram_image is not None:
            self._histogram_image.remove()
        # Calculate and draw the new histogram
        self._histogram = np.histogram2d(value[:, 0], value[:, 1], bins=self._bins)
        print(value.dtype)
        counts, x_edges, y_edges = self._histogram
        print(counts)
        self._histogram_rgba = self.array_to_rgba(array=counts, colormap=self._histogram_colormap, data_type=self._data.dtype) # colormap normalizarion follows data type
        self._histogram_image = self.ax.imshow(self._histogram_rgba, origin='lower', extent=[
            x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto', alpha=1, zorder=1)
        if self._color_indices is None:
            self.color_indices = 0  # Set default color index
        else:
            # Update colors if color indices are set, resize if data shape has changed
            color_indices_size = len(self._color_indices)
            color_indices = np.resize(self._color_indices, self._data.shape[0])
            if len(color_indices) > color_indices_size:
                # fill with zeros where new data is larger
                color_indices[color_indices_size:] = 0
            self.color_indices = color_indices
        self.draw()

    @property
    def visible(self) -> bool:
        """Gets or sets the visibility of the 2D histogram.

        Triggers a draw idle command.

        Returns
        -------
        visible : bool
            visibility of the 2D histogram.
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        """Sets the visibility of the 2D histogram."""
        self._visible = value
        if self._histogram_image is not None:
            artist = self._histogram_image
            artist.set_visible(value)
            if self._overlay is not None:
                self._overlay.set_visible(value)
        self.draw()

    @property
    def color_indices(self) -> np.ndarray:
        """Gets or sets the current color indices used for the 2D histogram underlying data.

        Triggers a draw idle command.

        Returns
        -------
        color_indices : (N,) np.ndarray[int] or int
            indices to map to the overlay colormap. Accepts a scalar or an array.
        
        Notes
        -----
        color_indices_changed_signal : Signal
            Signal emitted when the color indices are changed.

        """
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
        """Sets color indices for the 2D histogram underlying data and updates colors accordingly."""
        # Check if indices are a scalar
        if np.isscalar(indices):
            indices = np.full(len(self._data), indices)
        # Check if indices data type is float
        if indices.dtype == float:
            indices = indices.astype(int)
        self._color_indices = indices

        counts, x_edges, y_edges = self._histogram
        x_bin_indices = (np.digitize(self._data[:,0], x_edges, right=False) - 1).clip(0, len(x_edges)-2) # Get the bin index for each x value ( -1 to start from index 0 and clip to handle edge cases)
        y_bin_indices = (np.digitize(self._data[:,1], y_edges, right=False) - 1).clip(0, len(y_edges)-2) # Get the bin index for each y value ( -1 to start from index 0 and clip to handle edge cases)

        statistic_histogram = self._calculate_statistic_histogram(x_bin_indices, y_bin_indices, indices, statistic='median')
        statistic_histogram_rgba = self.array_to_rgba(array=statistic_histogram, colormap=self.categorical_colormap, data_type=indices.dtype) # colormap normalization follows data type
        # Remove the existing overlay to redraw
        if self._overlay is not None:
            self._overlay.remove()
        # Draw the overlay
        self._overlay = self.ax.imshow(statistic_histogram_rgba, origin='lower', extent=[
            x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], aspect='auto', alpha=1, zorder=2)


        # h, xedges, yedges, _ = self._histogram
        # # Create empty overlay
        # overlay_rgba = np.zeros((*h.shape, 4), dtype=float)
        # output_max = np.zeros(h.shape, dtype=float)
        # for i in np.unique(self._color_indices):
        #     # Filter data by class
        #     data_filtered_by_class = self._data[self._color_indices == i]
        #     # Calculate histogram of filtered data while fixing the bins
        #     histogram_filtered_by_class, _, _ = np.histogram2d(
        #         data_filtered_by_class[:, 0], data_filtered_by_class[:, 1], bins=[xedges, yedges])
        #     class_mask = histogram_filtered_by_class > output_max
        #     output_max = np.maximum(histogram_filtered_by_class, output_max)
        #     overlay_rgba[class_mask] = self.categorical_colormap(i)
        # # Remove the existing overlay to redraw
        # if self._overlay is not None:
        #     self._overlay.remove()
        # # Draw the overlay
        # self._overlay = self.ax.imshow(overlay_rgba.swapaxes(0, 1), origin='lower', extent=[
        #     xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', alpha=1, zorder=2)
        # emit signal
        self.color_indices_changed_signal.emit(self._color_indices)
        self.draw()

    @property
    def bins(self) -> int:
        """Gets or sets the number of bins for the histogram.

        Returns
        -------
        bins : int
            number of bins for the histogram.
        """
        return self._bins

    @bins.setter
    def bins(self, value: int):
        """Sets the number of bins for the histogram."""
        self._bins = value
        self.data = self._data

    @property
    def histogram_colormap(self) -> Colormap:
        """Gets or sets the colormap for the histogram.

        Returns
        -------
        histogram_colormap : Colormap
            colormap for the histogram.
        """
        return self._histogram_colormap

    @histogram_colormap.setter
    def histogram_colormap(self, value: Colormap):
        """Sets the colormap for the histogram."""
        self._histogram_colormap = value
        self.data = self._data

    @property
    def histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, QuadMesh]:
        """Returns the 2D histogram array and edges.

        Returns
        -------
        histogram : Tuple[np.ndarray, np.ndarray, np.ndarray]
            2D histogram, x edges, and y edges.
        """
        return self._histogram

    def indices_in_above_threshold_patches(self, threshold: int) -> List[int]:
        """
        Returns the indices of the points in that fall into the bins
        of the 2D histogram exceeding a specified threshold.

        Parameters
        ----------
        threshold : int
            The count threshold to exceed.

        Returns
        -------
        indices : List[int]
            list of indices of points falling into the exceeding bins.
        """
        histogram, x_edges, y_edges = self._histogram

        # Identify bins that exceed the threshold
        exceeding_bins = np.argwhere(histogram > threshold)

        # Prepare to collect indices
        indices = []

        # For each bin exceeding the threshold...
        for bin_x, bin_y in exceeding_bins:
            # Identify the edges of the current bin
            x_min, x_max = x_edges[bin_x], x_edges[bin_x + 1]
            y_min, y_max = y_edges[bin_y], y_edges[bin_y + 1]

            # Find indices of points within these edges
            bin_indices = np.where((self._data[:, 0] >= x_min) & (self._data[:, 0] < x_max) & (
                self._data[:, 1] >= y_min) & (self._data[:, 1] < y_max))[0]
            indices.extend(bin_indices)

        return indices
    
    def _calculate_statistic_histogram(x_indices, y_indices, features, statistic='median'):
        """
        Calculate either the mean or median "histogram" for provided indices and features.
        
        Parameters:
        - x_indices: numpy array of x indices
        - y_indices: numpy array of y indices
        - features: numpy array of feature values
        - statistic: 'mean', 'median', or 'sum', the type of statistic to compute
        
        Returns:
        - 2D numpy array with the calculated statistic
        """
        height = max(x_indices) + 1
        width = max(y_indices) + 1
        statistic_histogram = np.full((height, width), np.nan)  # Initialize with NaNs

        if statistic == 'mean':
            sums = np.zeros((height, width))
            counts = np.zeros((height, width))
            for x, y, feature in zip(x_indices, y_indices, features):
                sums[x, y] += feature
                counts[x, y] += 1
            np.divide(sums, counts, out=statistic_histogram, where=counts != 0)
        elif statistic == 'median':
            feature_lists = defaultdict(list)
            for x, y, feature in zip(x_indices, y_indices, features):
                feature_lists[(x, y)].append(feature)
            for (x, y), feats in feature_lists.items():
                if feats:
                    statistic_histogram[x, y] = np.median(feats)
        elif statistic == 'sum':
            sums = np.zeros((height, width))
            for x, y, feature in zip(x_indices, y_indices, features):
                if np.isnan(sums[x, y]):
                    sums[x, y] = 0  # Initialize sum as 0 when first feature is added
                sums[x, y] += feature
            statistic_histogram = sums

        return statistic_histogram

    def array_to_rgba(array, colormap=plt.cm.viridis, data_type=np.float64):
        """
        Convert a 2D data array to an RGBA image using a matplotlib colormap.
        
        Parameters
        ----------
        array : 2D numpy array
            the data to convert to an image
        colormap : matplotlib colormap
            the colormap to use
        data_type : type
            the data type of the features. It is used to normalize the data.

        Returns
        -------
        colored_image : 2D numpy array
            the RGBA image
        """
        if data_type == int:
            norm = Normalize(vmin=0, vmax=colormap.N)
        elif data_type == float:
            norm = Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        colored_image = colormap(norm(array))
        colored_image[np.isnan(array)] = [0, 0, 0, 0]  # Set NaN values to transparent
        return colored_image.swapaxes(0, 1)

    def draw(self):
        """Draws or redraws the 2D histogram."""
        self.ax.figure.canvas.draw_idle()
