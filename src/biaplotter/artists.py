from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap

class AbstractArtist(ABC):
    def __init__(self, data: np.ndarray, ax: plt.Axes = None, colormap: Colormap = plt.cm.viridis, color_indices: np.ndarray = None):
        self._data = data
        self._ax = ax if ax is not None else plt.gca()
        self._visible = True
        self._colormap = colormap
        self._color_indices = color_indices
        # Initialize any needed attributes here, but abstract properties won't directly interact with them

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
    def bounds(self) -> tuple:
        """Abstract property for the artist's data bounds."""
        pass

    @bounds.setter
    @abstractmethod
    def bounds(self, value: tuple):
        """Abstract setter for the artist's data bounds."""
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

    def compute_bounds(self, data: np.ndarray) -> tuple:
        xmin, xmax = np.min(data[:, 0]), np.max(data[:, 0])
        ymin, ymax = np.min(data[:, 1]), np.max(data[:, 1])
        return xmin, xmax, ymin, ymax

class CustomScatterArtist(AbstractArtist):
    def __init__(self, data: np.ndarray, ax: plt.Axes = None, colormap: Colormap = plt.cm.viridis, color_indices: np.ndarray = None):
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
        self._data = value
        if self._scatter is None:
            # If the scatter plot hasn't been created yet, do so now
            self._scatter = self._ax.scatter(value[:, 0], value[:, 1], c='b')  # Default color
        else:
            # If the scatter plot already exists, just update its data
            self._scatter.set_offsets(value)
        if self._color_indices is not None:
            # Update colors if color indices are set
            self.color_indices = self._color_indices
        self._ax.figure.canvas.draw_idle()

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
        self._ax.figure.canvas.draw_idle()

    @property
    def bounds(self) -> tuple:
        """Returns the bounds of the scatter plot's data."""
        return super().compute_bounds(self._data)

    @bounds.setter
    def bounds(self, value: tuple):
        """Setting bounds directly may not be common for scatter plots as they are data-driven.
        This implementation is provided to fulfill the abstract base class contract."""
        # No action needed since bounds are determined by data. Override if different behavior is desired.

    @property
    def color_indices(self) -> np.ndarray:
        """Gets the current color indices used for the scatter plot."""
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
        """Sets color indices for the scatter plot and updates colors accordingly."""
        self._color_indices = indices
        if indices is not None and self._scatter is not None:
            normalized_indices = indices / np.max(indices)
            new_colors = self._colormap(normalized_indices)
            self._scatter.set_facecolor(new_colors)
        self._ax.figure.canvas.draw_idle()

    def draw(self):
        self._ax.figure.canvas.draw_idle()


from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

class Custom2DHistogramArtist(AbstractArtist):
    def __init__(self, data: np.ndarray, ax: plt.Axes = None, bins=10, colormap: Colormap = plt.cm.viridis, color_indices: np.ndarray = None):
        super().__init__(data, ax, colormap, color_indices)
        self._bins = bins
        self._histogram_image = None  # Placeholder for the histogram image
        self.draw()  # Initial drawing of the histogram

    @property
    def data(self) -> np.ndarray:
        """Returns the data associated with the scatter plot."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value
        rgba, xedges, yedges = self.generate_histogram_image_and_edges()
        if self._histogram_image is None:
            # Initial draw of the histogram
            self._histogram_image = self._ax.imshow(rgba, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
        else:
            # Update the histogram image with new data or colors
            self._histogram_image.set_data(rgba)
        self.draw()

    @property
    def visible(self) -> bool:
        """Determines if the scatter plot is currently visible."""
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        self._visible = value
        if self._histogram_image is not None:
            self._histogram_image.set_visible(value)
        self._ax.figure.canvas.draw_idle()

    @property
    def bounds(self) -> tuple:
        return super().compute_bounds(self._data)

    @bounds.setter
    def bounds(self, value: tuple):
        # Updating bounds for a histogram might involve recalculating the histogram
        # or adjusting the axes limits. This example assumes bounds are directly tied to the data.
        pass

    @property
    def color_indices(self) -> np.ndarray:
        return self._color_indices
    

    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
        self._color_indices = indices
        if indices is not None:
            rgba, xedges, yedges = self.generate_histogram_image_and_edges()
            if self._histogram_image is None:
                # Initial draw of the histogram
                self._histogram_image = self._ax.imshow(rgba, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower')
            else:
                # Update the histogram image with new data or colors
                self._histogram_image.set_data(rgba)
            self.draw()  # Redraw to update colors based on the new indices

    def generate_histogram_image_and_edges(self):
        # Calculate the 2D histogram
        H, xedges, yedges = np.histogram2d(self._data[:,0], self._data[:,1], bins=self._bins)
        H = H.T  # Transpose H for the correct orientation
        
        # Generate the RGBA image for the histogram
        if self._color_indices is not None:
            # Use color indices if provided
            normalized_indices = self._color_indices / np.max(self._color_indices)
            colors = self._colormap(normalized_indices)
            rgba_colors = colors[np.newaxis, :, :]
            rgba = np.repeat(rgba_colors, H.shape[0], axis=0)
        else:
            # Default coloring based on histogram values
            norm = Normalize(vmin=H.min(), vmax=H.max())
            mappable = ScalarMappable(norm=norm, cmap=self._colormap)
            rgba = mappable.to_rgba(H)
        return rgba, xedges, yedges
        

    def draw(self):       
        self._ax.figure.canvas.draw_idle()



# import numpy as np

# class CustomScatter:
#     def __init__(self, axes, colormap, initial_size=50):
#         self._axes = axes
#         self._colormap = colormap
#         self._scatter_handle = self._axes.scatter([], [], s=initial_size, c="none")
#         self._current_colors = None
#         self._color_indices = None
#         self._selected_color_index = 0

#     def update_scatter(self, x_data=None, y_data=None):
#         if x_data is not None and y_data is not None:
#             # self._scatter_handle.set_offsets(np.column_stack([x_data, y_data]))
#             self._scatter_handle = self._axes.scatter(x_data, y_data)
#             self._update_axes_limits_with_margin(x_data, y_data)
#         # Initialize colors if not already done
#         if self._current_colors is None:
#             # Set color indices with color index 0
#             self.color_indices = 1  # temporary value for testing!!

#     def _update_axes_limits_with_margin(self, x_data, y_data):
#         x_range = max(x_data) - min(x_data)
#         y_range = max(y_data) - min(y_data)
#         x_margin = x_range * 0.05
#         y_margin = y_range * 0.05
#         self._axes.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)
#         self._axes.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
#         self._axes.relim()  # Recalculate the data limits
#         self._axes.autoscale_view()  # Auto-adjust the axes limits
#         self._axes.figure.canvas.draw_idle()

#     @property
#     def data(self):
#         return self._scatter_handle.get_offsets()

#     @data.setter
#     def data(self, xy):
#         x_data, y_data = xy
#         self.update_scatter(x_data, y_data)

#     @property
#     def selected_color_index(self):
#         return self._selected_color_index

#     @selected_color_index.setter
#     def selected_color_index(self, index):
#         self._selected_color_index = index

#     @property
#     def colors(self):
#         return self._scatter_handle.get_facecolor()

#     @colors.setter
#     def colors(self, new_colors):
#         # Store alpha values
#         alpha = self.alphas
#         self._current_colors = new_colors
#         self._scatter_handle.set_facecolor(self._current_colors)
#         if alpha is not None:
#             self.alphas = alpha  # Restore alpha values
#         self._axes.figure.canvas.draw_idle()  # maybe unecessary because alpha updates the canvas

#     @property
#     def color_indices(self):
#         return self._color_indices

#     @color_indices.setter
#     def color_indices(self, indices):
#         # Do nothing if there is no data
#         if len(self.data) == 0:
#             return
#         # Handle scalar indices
#         if np.isscalar(indices):
#             indices = np.full(self.data.shape[0], indices)
#         self._color_indices = indices
#         new_colors = self._colormap(indices)
#         # update scatter colors
#         self.colors = new_colors

#     @property
#     def alphas(self):
#         if self._current_colors is not None:
#             return self._current_colors[:, -1]
#         return None

#     @alphas.setter
#     def alphas(self, alpha_values):
#         if self._current_colors is not None:
#             # Handle scalar alpha value
#             if np.isscalar(alpha_values):
#                 alpha_values = np.full(self._current_colors.shape[0], alpha_values)
#             self._current_colors[:, -1] = alpha_values  # Update alpha values
#             self._scatter_handle.set_facecolor(self._current_colors)
#             self._axes.figure.canvas.draw_idle()


