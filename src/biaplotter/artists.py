import numpy as np
from selectors import CustomLassoSelector

class CustomScatter:
    def __init__(self, axes, colormap, initial_size=50):
        self._axes = axes
        self._colormap = colormap
        self._scatter_handle = self._axes.scatter([], [], s=initial_size, c="none")
        self._current_colors = None
        self._color_indices = None
        self._selected_color_index = 0

    def update_scatter(self, x_data=None, y_data=None):
        if x_data is not None and y_data is not None:
            # self._scatter_handle.set_offsets(np.column_stack([x_data, y_data]))
            self._scatter_handle = self._axes.scatter(x_data, y_data)
            self._update_axes_limits_with_margin(x_data, y_data)
        # Initialize colors if not already done
        if self._current_colors is None:
            # Set color indices with color index 0
            self.color_indices = 1  # temporary value for testing!!

    def _update_axes_limits_with_margin(self, x_data, y_data):
        x_range = max(x_data) - min(x_data)
        y_range = max(y_data) - min(y_data)
        x_margin = x_range * 0.05
        y_margin = y_range * 0.05
        self._axes.set_xlim(min(x_data) - x_margin, max(x_data) + x_margin)
        self._axes.set_ylim(min(y_data) - y_margin, max(y_data) + y_margin)
        self._axes.relim()  # Recalculate the data limits
        self._axes.autoscale_view()  # Auto-adjust the axes limits
        self._axes.figure.canvas.draw_idle()

    @property
    def data(self):
        return self._scatter_handle.get_offsets()

    @data.setter
    def data(self, xy):
        x_data, y_data = xy
        self.update_scatter(x_data, y_data)

    @property
    def selected_color_index(self):
        return self._selected_color_index

    @selected_color_index.setter
    def selected_color_index(self, index):
        self._selected_color_index = index

    @property
    def colors(self):
        return self._scatter_handle.get_facecolor()

    @colors.setter
    def colors(self, new_colors):
        # Store alpha values
        alpha = self.alphas
        self._current_colors = new_colors
        self._scatter_handle.set_facecolor(self._current_colors)
        if alpha is not None:
            self.alphas = alpha  # Restore alpha values
        self._axes.figure.canvas.draw_idle()  # maybe unecessary because alpha updates the canvas

    @property
    def color_indices(self):
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices):
        # Do nothing if there is no data
        if len(self.data) == 0:
            return
        # Handle scalar indices
        if np.isscalar(indices):
            indices = np.full(self.data.shape[0], indices)
        self._color_indices = indices
        new_colors = self._colormap(indices)
        # update scatter colors
        self.colors = new_colors

    @property
    def alphas(self):
        if self._current_colors is not None:
            return self._current_colors[:, -1]
        return None

    @alphas.setter
    def alphas(self, alpha_values):
        if self._current_colors is not None:
            # Handle scalar alpha value
            if np.isscalar(alpha_values):
                alpha_values = np.full(self._current_colors.shape[0], alpha_values)
            self._current_colors[:, -1] = alpha_values  # Update alpha values
            self._scatter_handle.set_facecolor(self._current_colors)
            self._axes.figure.canvas.draw_idle()

    def add_lasso_selector(self):
        self.lasso_selector = CustomLassoSelector(self, self._axes)


