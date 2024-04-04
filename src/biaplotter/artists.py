from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, ListedColormap
from nap_plot_tools.cmap import make_cat10_mod_cmap

cat10_mod_cmap = make_cat10_mod_cmap()
cat10_mod_cmap_first_opaque = make_cat10_mod_cmap(first_color_transparent=False)

class AbstractArtist(ABC):
    def __init__(self, data: np.ndarray, ax: plt.Axes = None, colormap: Colormap = cat10_mod_cmap_first_opaque, color_indices: np.ndarray = None):
        self._data = data
        self._ax = ax if ax is not None else plt.gca()
        self._visible = True
        self._colormap = colormap
        self._color_indices = color_indices

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



class Scatter(AbstractArtist):
    def __init__(self, data: np.ndarray = None, ax: plt.Axes = None, colormap: Colormap = cat10_mod_cmap_first_opaque, color_indices: np.ndarray = None):
        super().__init__(data, ax, colormap, color_indices)
        self._scatter = None  # Placeholder for the scatter plot object
        self.data = data  # Initialize the scatter plot with data
        self.draw()  # Initial draw of the scatter plot

    @property
    def data(self) -> np.ndarray:
        """Returns the data associated with the scatter plot."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data for the scatter plot, updating the display as needed."""
        if value is None or len(value) == 0:
            return
        self._data = value
        if self._scatter is None:
            # If the scatter plot hasn't been created yet, do so now
            self._scatter = self._ax.scatter(value[:, 0], value[:, 1], facecolors=self._colormap(1), edgecolors=None)  # Default color
        else:
            # If the scatter plot already exists, just update its data
            self._scatter.set_offsets(value)
        if self._color_indices is not None:
            # Update colors if color indices are set
            self.color_indices = self._color_indices
        self.draw()

    @property
    def visible(self) -> bool:
        """Determines if the scatter plot is currently visible."""
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
        """Gets the current color indices used for the scatter plot."""
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
            # normalized_indices = indices / np.max(indices)
            new_colors = self._colormap(indices)
            self._scatter.set_facecolor(new_colors)
        self.draw()

    def draw(self):
        self._ax.figure.canvas.draw_idle()

class Histogram2D(AbstractArtist):
    def __init__(self, data: np.ndarray = None, ax: plt.Axes = None, colormap: Colormap = cat10_mod_cmap, color_indices: np.ndarray = None, bins=20, histogram_colormap: Colormap = plt.cm.viridis):
        super().__init__(data, ax, colormap, color_indices)
        self._histogram = None  # Placeholder for the 2D histogram artist
        self._bins = bins  # Number of bins for the histogram
        self._histogram_colormap = histogram_colormap  # Colormap for the histogram
        self._overlay = None  # Placeholder for the overlay
        self.data = data  # Initialize the histogram with data
        self.draw()  # Initial draw of the histogram

    @property
    def data(self) -> np.ndarray:
        """Returns the data associated with the 2D histogram."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data for the 2D histogram, updating the display as needed."""
        if value is None or len(value) == 0:
            return
        self._data = value
        # Remove the existing histogram to redraw
        if self._histogram is not None:
            for artist in self._histogram[-1]:
                artist.remove()
        # Draw the new histogram
        self._histogram = self._ax.hist2d(value[:, 0], value[:, 1], bins=self._bins, cmap=self._histogram_colormap, zorder=1)
        self.draw()

    @property
    def visible(self) -> bool:
        """Determines if the 2D histogram is currently visible."""
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
        # This property might be less relevant for histograms, as color mapping is often handled differently
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
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
            histogram_filtered_by_class, _, _ = np.histogram2d(data_filtered_by_class[:, 0], data_filtered_by_class[:, 1], bins=[xedges, yedges])
            class_mask = histogram_filtered_by_class > output_max
            output_max = np.maximum(histogram_filtered_by_class, output_max)
            overlay_rgba[class_mask] = self._colormap(i)
        # Draw the overlay
        self.overlay = self._ax.imshow(overlay_rgba.swapaxes(0, 1), origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect='auto', alpha=1, zorder=2)
        self.draw()

    def draw(self):
        """Draws or redraws the 2D histogram."""
        self._ax.figure.canvas.draw_idle()