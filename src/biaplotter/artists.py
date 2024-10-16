import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import Colormap
from matplotlib.collections import QuadMesh
from abc import ABC, abstractmethod
from nap_plot_tools.cmap import cat10_mod_cmap, cat10_mod_cmap_first_transparent
from psygnal import Signal
from typing import Tuple, List, Union


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
        self._size = 50  # Default size
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
            self._scatter = self.ax.scatter(value[:, 0], value[:, 1], s=self._size)
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
        self.size = 50

        x_margin = 0.05 * (np.max(value[:, 0]) - np.min(value[:, 0]))
        y_margin = 0.05 * (np.max(value[:, 1]) - np.min(value[:, 1]))
        self.ax.set_xlim(np.min(value[:, 0]) - x_margin, np.max(value[:, 0]) + x_margin)
        self.ax.set_ylim(np.min(value[:, 1]) - y_margin, np.max(value[:, 1]) + y_margin)

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
            new_colors = self.categorical_colormap(indices)
            self._scatter.set_facecolor(new_colors)
            self._scatter.set_edgecolor(None)
        # emit signal
        self.color_indices_changed_signal.emit(self._color_indices)
        self.draw()

    @property
    def size(self) -> Union[float, np.ndarray]:
        """Gets or sets the size of the points in the scatter plot.

        Triggers a draw idle command.

        Returns
        -------
        size : float or (N,) np.ndarray[float]
            size of the points in the scatter plot. Accepts a scalar or an array of floats.
        """
        return self._size

    @size.setter
    def size(self, value: Union[float, np.ndarray]):
        """Sets the size of the points in the scatter plot."""
        self._size = value
        if self._scatter is not None:
            self._scatter.set_sizes(np.full(len(self._data), value) if np.isscalar(value) else value)
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

    def __init__(self, ax: plt.Axes = None, data: np.ndarray = None, categorical_colormap: Colormap = cat10_mod_cmap_first_transparent, color_indices: np.ndarray = None, bins=20, histogram_colormap: Colormap = plt.cm.magma, cmin=0):
        super().__init__(ax, data, categorical_colormap, color_indices)
        """Initializes the 2D histogram artist.
        """
        #: Stores the matplotlib histogram2D object
        self._histogram = None
        self._bins = bins
        self._histogram_colormap = histogram_colormap
        self._overlay = None
        self.data = data
        self.cmin = cmin
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
        if self._histogram is not None:
            self._histogram[-1].remove()
        # Draw the new histogram
        self._histogram = self.ax.hist2d(
            value[:, 0], value[:, 1], bins=self._bins, cmap=self._histogram_colormap, zorder=1, cmin=self.cmin)
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
        if self._histogram is not None:
            artist = self._histogram[-1]
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
        h, xedges, yedges, _ = self._histogram
        # Create empty overlay
        overlay_rgba = np.zeros((*h.shape, 4), dtype=float)
        output_max = np.zeros(h.shape, dtype=float)
        for i in np.unique(self._color_indices):
            # Filter data by class
            data_filtered_by_class = self._data[self._color_indices == i]
            # Calculate histogram of filtered data while fixing the bins
            histogram_filtered_by_class, _, _ = np.histogram2d(
                data_filtered_by_class[:, 0], data_filtered_by_class[:, 1], bins=[xedges, yedges])
            class_mask = histogram_filtered_by_class > output_max
            output_max = np.maximum(histogram_filtered_by_class, output_max)
            overlay_rgba[class_mask] = self.categorical_colormap(i)
        # Remove the existing overlay to redraw
        if self._overlay is not None:
            self._overlay.remove()
        # Draw the overlay
        self._overlay = self.ax.imshow(overlay_rgba.swapaxes(0, 1), origin='lower', extent=[
            xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', alpha=1, zorder=2)
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
        """Returns the 2D histogram matplotlib object.

        Returns
        -------
        histogram : Tuple[np.ndarray, np.ndarray, np.ndarray, QuadMesh]
            2D histogram matplotlib object.
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
        counts, xedges, yedges, _ = self._histogram

        # Identify bins that exceed the threshold
        exceeding_bins = np.argwhere(counts > threshold)

        # Prepare to collect indices
        indices = []

        # For each bin exceeding the threshold...
        for bin_x, bin_y in exceeding_bins:
            # Identify the edges of the current bin
            x_min, x_max = xedges[bin_x], xedges[bin_x + 1]
            y_min, y_max = yedges[bin_y], yedges[bin_y + 1]

            # Find indices of points within these edges
            bin_indices = np.where((self._data[:, 0] >= x_min) & (self._data[:, 0] < x_max) & (
                self._data[:, 1] >= y_min) & (self._data[:, 1] < y_max))[0]
            indices.extend(bin_indices)

        return indices

    def draw(self):
        """Draws or redraws the 2D histogram."""
        self.ax.figure.canvas.draw_idle()
