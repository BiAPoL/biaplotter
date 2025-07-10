import warnings
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, Normalize
from nap_plot_tools.cmap import (cat10_mod_cmap,
                                 cat10_mod_cmap_first_transparent)
from scipy.stats import binned_statistic_2d

from biaplotter.colormap import BiaColormap

from .artists_base import Artist


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
    overlay_colormap : Colormap, optional
        a colormap to use for the artist, by default cat10_mod_cmap from nap-plot-tools
    color_indices : (N,) np.ndarray[int] or int, optional
        array of indices to map to the colormap, by default None

    Notes
    -----
    **Signals:**

        * **data_changed_signal** emitted when the data are changed.
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

    def __init__(
        self,
        ax: plt.Axes = None,
        data: np.ndarray = None,
        overlay_colormap: Colormap = cat10_mod_cmap,
        color_indices: np.ndarray = None,
    ):
        """Initializes the scatter plot artist."""
        super().__init__(ax, data, overlay_colormap, color_indices)
        #: Stores the scatter plot matplotlib object
        self._scatter_overlay_rgba = None  # Store precomputed overlay colors
        self._overlay_visible = True
        self._color_normalization_method = "linear"
        self.data = data
        self._alpha = 1  # Default alpha
        self._size = 50  # Default size
        self.draw()  # Initial draw of the scatter plot

    @property
    def alpha(self) -> Union[float, np.ndarray]:
        """Gets or sets the alpha value of the scatter plot.

        Returns
        -------
        alpha : float
            alpha value of the scatter plot.
        """
        return self._mpl_artists["scatter"].get_alpha()

    @alpha.setter
    def alpha(self, value: Union[float, np.ndarray]):
        """Sets the alpha value of the scatter plot."""
        self._alpha = value

        if np.isscalar(value):
            value = np.ones(len(self._data)) * value
        if "scatter" in self._mpl_artists.keys():
            self._mpl_artists["scatter"].set_alpha(value)
        self.draw()

    @property
    def color_normalization_method(self) -> str:
        """Gets or sets the normalization method for the color indices.

        Returns
        -------
        color_normalization_method : str
            the normalization method for the color indices.
        """
        return self._color_normalization_method

    @color_normalization_method.setter
    def color_normalization_method(self, value: str):
        """Sets the normalization method for the color indices."""
        self._color_normalization_method = value
        self._colorize(self._color_indices)

    @property
    def overlay_visible(self) -> bool:
        return self._overlay_visible

    @overlay_visible.setter
    def overlay_visible(self, value: bool):
        self._overlay_visible = value
        self._colorize(self._color_indices)

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
        if "scatter" in self._mpl_artists.keys():
            self._mpl_artists["scatter"].set_sizes(
                np.full(len(self._data), value)
                if np.isscalar(value)
                else value
            )
        self.draw()

    def color_indices_to_rgba(
        self, indices: np.ndarray, is_overlay: bool = True
    ) -> np.ndarray:
        """
        Convert color indices to RGBA colors using the colormap.
        """
        norm = self._get_normalization(indices)
        colormap = self.overlay_colormap.cmap

        rgba = colormap(norm(indices))
        return rgba

    def _colorize(self, indices: np.ndarray):
        """
        Calculate and optionally render the scatter overlay.
        """
        if indices is None:
            return
        # Always calculate and store the overlay colors
        self._scatter_overlay_rgba = self.color_indices_to_rgba(indices)

        # Update the overlay visibility
        if self._overlay_visible:
            self._mpl_artists["scatter"].set_facecolor(
                self._scatter_overlay_rgba
            )
            self._mpl_artists["scatter"].set_edgecolor("white")
        else:
            # Set colors to the first color of the colormap (index 0)
            default_rgba = self.color_indices_to_rgba(
                np.zeros_like(indices)
            )
            self._mpl_artists["scatter"].set_facecolor(default_rgba)
            self._mpl_artists["scatter"].set_edgecolor("white")

    def _get_normalization(self, values: np.ndarray) -> Normalize:
        """Determine the normalization method and return the normalization object."""
        if self.overlay_colormap.categorical:
            self._validate_categorical_colormap()
            return Normalize(vmin=0, vmax=self.overlay_colormap.N)

        norm_dispatch = {
            "log": lambda: self._log_normalization(values),
            "centered": lambda: self._centered_normalization(values),
            "symlog": lambda: self._symlog_normalization(values),
            "linear": lambda: self._linear_normalization(values),
        }

        normalization_func = norm_dispatch.get(
            self._color_normalization_method
        )
        if normalization_func is None:
            raise ValueError(
                f"Unknown color normalization method: {self._color_normalization_method}.\n"
                f"Available methods are: {list(norm_dispatch.keys())}."
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            return normalization_func()

    def _refresh(self, force_redraw: bool = True):
        """Creates the scatter plot with the data and default properties."""
        if force_redraw or self._mpl_artists.get("scatter") is None:
            self._remove_artists()
            # Create a new scatter plot with the updated data
            self._mpl_artists["scatter"] = self.ax.scatter(
                self._data[:, 0], self._data[:, 1]
            )
            self.size = self.default_size

            if "scatter" in self._mpl_artists.keys():
                self._mpl_artists["scatter"].set_linewidth(self.default_edge_width)
            self.alpha = 1  # Default alpha
            self.color_indices = 0
        else:
            self._mpl_artists["scatter"].set_offsets(
                self._data
            )  # Somehow resets the size and alpha
            self.color_indices = self._color_indices
            self.size = self._size
            self.alpha = self._alpha

    @property
    def default_size(self) -> float:
        """rule of thumb for good point size"""
        return min(10, (max(0.1, 8000 / len(self._data)))) * 2

    @property
    def default_edge_width(self) -> float:
        """Calculate the default edge width based on the point size."""
        return np.sqrt(self.default_size / np.pi) / 8

    def _validate_categorical_colormap(self):
        """Validate settings for a categorical colormap."""
        if self._color_indices.dtype != int:
            warnings.warn(
                "Color indices must be integers for categorical colormap. Change `overlay_colormap` to a continuous colormap or set `color_indices` to integers."
            )
        if self._color_normalization_method != "linear":
            warnings.warn(
                "Categorical colormap detected. Setting color normalization method to linear."
            )
            self._color_normalization_method = "linear"


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
    overlay_colormap : Colormap, optional
        a colormap to use for the artist overlay, by default cat10_mod_cmap_first_transparent from nap-plot-tools (first color is transparent)
    color_indices : (N,) np.ndarray[int] or int, optional
        array of indices to map to the overlay_colormap, by default None
    bins : int, optional
        number of bins for the histogram, by default 20
    histogram_colormap : Colormap, optional
        colormap for the histogram, by default plt.cm.magma
    cmin : int, optional
        minimum count for the histogram, by default 0
        Values below cmin are set to NaN (to be transparent).

    Notes
    -----
    **Signals:**

        * **data_changed_signal** emitted when the data are changed.
        * **color_indices_changed_signal** emitted when the color indices are changed.

    """

    def __init__(
        self,
        ax: plt.Axes = None,
        data: np.ndarray = None,
        overlay_colormap: Colormap = cat10_mod_cmap_first_transparent,
        color_indices: np.ndarray = None,
        bins=20,
        histogram_colormap: Colormap = plt.cm.magma,
        cmin=0,
    ):
        """Initializes the 2D histogram artist."""
        super().__init__(ax, data, overlay_colormap, color_indices)
        self._histogram = None
        self._overlay_histogram_rgba = None  # Store precomputed overlay image
        self._bins = bins
        self._histogram_colormap = BiaColormap(histogram_colormap)
        self._histogram_interpolation = "nearest"
        self._overlay_interpolation = "nearest"
        self._overlay_opacity = 1
        self._overlay_visible = True
        self._margins = 0
        self._histogram_color_normalization_method = "linear"
        self._overlay_color_normalization_method = "linear"
        self._cmin = cmin
        self.data = data
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
        self._refresh(force_redraw=False)
        self._colorize(self._color_indices)

    @property
    def cmin(self) -> int:
        """Gets or sets the minimum count for the histogram.

        Values below cmin are set to NaN (to be transparent).

        Returns
        -------
        cmin : int
            minimum count for the histogram.
        """
        return self._cmin

    @cmin.setter
    def cmin(self, value: int):
        """Sets the minimum count for the histogram."""
        self._cmin = value
        self._refresh(force_redraw=False)

    @property
    def histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the 2D histogram array and edges.

        Returns
        -------
        histogram : Tuple[np.ndarray, np.ndarray, np.ndarray]
            2D histogram, x edges, and y edges.
        """
        return self._histogram

    @property
    def histogram_color_normalization_method(self) -> str:
        """Gets or sets the normalization method for the histogram.

        Returns
        -------
        color_normalization_method : str
            the normalization method for the histogram.
        """
        return self._histogram_color_normalization_method

    @histogram_color_normalization_method.setter
    def histogram_color_normalization_method(self, value: str):
        """Sets the normalization method for the histogram."""
        self._histogram_color_normalization_method = value
        # Update histogram image if new color normalization method is set
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
        self._histogram_colormap = BiaColormap(value)
        self._refresh(force_redraw=False)

    @property
    def histogram_interpolation(self) -> str:
        """Gets or sets the interpolation method for the histogram.

        Returns
        -------
        histogram_interpolation : str
            interpolation method for the histogram.
        """
        return self._histogram_interpolation

    @histogram_interpolation.setter
    def histogram_interpolation(self, value: str):
        """Sets the interpolation method for the histogram."""
        self._histogram_interpolation = value
        self._refresh(force_redraw=False)

    @property
    def overlay_color_normalization_method(self) -> str:
        """Gets or sets the normalization method for the overlay histogram.

        Returns
        -------
        overlay_color_normalization_method : str
            the normalization method for the overlay color indices.
        """
        return self._overlay_color_normalization_method

    @overlay_color_normalization_method.setter
    def overlay_color_normalization_method(self, value: str):
        """Sets the normalization method for the overlay histogram."""
        self._overlay_color_normalization_method = value
        # Update overlay histogram image if new color normalization method is set
        self._colorize(self._color_indices)

    @property
    def overlay_interpolation(self) -> str:
        """Gets or sets the interpolation method for the overlay histogram.

        Returns
        -------
        overlay_interpolation : str
            interpolation method for the overlay histogram.
        """
        return self._overlay_interpolation

    @overlay_interpolation.setter
    def overlay_interpolation(self, value: str):
        """Sets the interpolation method for the overlay histogram."""
        self._overlay_interpolation = value
        self._colorize(self._color_indices)

    @property
    def overlay_opacity(self):
        """Gets or sets the opacity of the overlay histogram.

        Triggers a draw idle command.

        Returns
        -------
        overlay_opacity : float
            opacity of the overlay histogram.
        """
        return self._overlay_opacity

    @overlay_opacity.setter
    def overlay_opacity(self, value):
        """Sets the opacity of the overlay histogram."""
        self._overlay_opacity = value
        self._colorize(self._color_indices)

    @property
    def overlay_visible(self):
        """Gets or sets the visibility of the overlay histogram.

        Triggers a draw idle command.

        Returns
        -------
        overlay_visible : bool
            visibility of the overlay histogram.
        """
        return self._overlay_visible

    @overlay_visible.setter
    def overlay_visible(self, value):
        """Sets the visibility of the overlay histogram."""
        self._overlay_visible = value
        self._colorize(self._color_indices)

    def color_indices_to_rgba(
        self, indices, is_overlay: bool = True
    ) -> np.ndarray:
        """
        Convert color indices to RGBA colors using the overlay colormap.
        """
        norm = self._get_normalization(indices, is_overlay=is_overlay)

        if is_overlay:
            colormap = self.overlay_colormap.cmap
        else:
            colormap = self.histogram_colormap.cmap

        rgba = colormap(norm(indices))

        return rgba

    def indices_in_patches_above_threshold(self, threshold: int) -> List[int]:
        """
        Returns the indices of the points that fall into the bins
        of the 2D histogram exceeding a specified threshold counts value.

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
            bin_indices = np.where(
                (self._data[:, 0] >= x_min)
                & (self._data[:, 0] < x_max)
                & (self._data[:, 1] >= y_min)
                & (self._data[:, 1] < y_max)
            )[0]
            indices.extend(bin_indices)

        return indices

    def _colorize(self, indices: np.ndarray):
        """
        Calculate and optionally render the overlay histogram.
        """
        if indices is None:
            return
        # Always calculate and store the overlay image
        _, x_edges, y_edges = self._histogram
        statistic_histogram, _, _, _ = binned_statistic_2d(
            x=self._data[:, 0],
            y=self._data[:, 1],
            values=indices,
            statistic=_median_np,
            bins=[x_edges, y_edges],
        )
        if not np.all(np.isnan(statistic_histogram)):
            self._overlay_histogram_rgba = self.color_indices_to_rgba(
                statistic_histogram.T, is_overlay=True
            )
        
        # Update the overlay visibility
        self._remove_artists(["overlay_histogram_image"])
        if self._overlay_visible and self._overlay_histogram_rgba is not None:
            _, x_edges, y_edges = self._histogram
            self._mpl_artists["overlay_histogram_image"] = self.ax.imshow(
                self._overlay_histogram_rgba,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin="lower",
                zorder=2,
                interpolation=self._overlay_interpolation,
                alpha=self._overlay_opacity,
                aspect="auto",
            )

    def _get_normalization(
        self, values: np.ndarray, is_overlay: bool = True
    ) -> Normalize:
        """
        Get the normalization class for the histogram data.

        Parameters
        ----------
        values : np.ndarray
            2D histogram data array to be converted to RGBA image.

        Returns
        -------
        norm_class : Normalize
            the normalization class to use for the histogram data.
        """
        if is_overlay:
            colormap = self.overlay_colormap
            is_categorical = self._is_categorical_colormap(colormap)
            norm_method = self._overlay_color_normalization_method
        else:
            colormap = self.histogram_colormap
            is_categorical = self._is_categorical_colormap(colormap)
            norm_method = self._histogram_color_normalization_method

        if is_categorical and norm_method != "linear":
            self._overlay_color_normalization_method = "linear"
            norm_method = "linear"

        # norm_dispatch is to be indexed like this:
        # norm_dispatch[is_categorical, color_normalization_method]
        norm_dispatch = {
            (True, "linear"): lambda: self._linear_normalization(
                values, is_categorical
            ),
            (False, "linear"): lambda: self._linear_normalization(values),
            (False, "log"): lambda: self._log_normalization(values),
            (False, "centered"): lambda: self._centered_normalization(values),
            (False, "symlog"): lambda: self._symlog_normalization(values),
        }

        return norm_dispatch.get((is_categorical, norm_method))()

    def _is_categorical_colormap(self, colormap):
        """
        Check if the colormap is categorical.

        Parameters
        ----------
        colormap : BiaColormap or ColormapLike
            colormap to check.

        Returns
        -------
        bool
            True if the colormap is categorical, False otherwise.
        """
        if hasattr(colormap, "categorical"):
            return colormap.categorical
        try:
            bia_colormap = BiaColormap(colormap)
            return bia_colormap.categorical
        except ValueError:
            return False

    def _refresh(self, force_redraw: bool = True):
        """Recalculate and redraw the histogram."""
        self._remove_artists()
        # Calculate and draw the new histogram
        self._histogram = np.histogram2d(
            self._data[:, 0], self._data[:, 1], bins=self._bins
        )
        counts, x_edges, y_edges = self._histogram
        # Replace values below cmin with NaN (to have them transparent)
        counts[counts < self._cmin] = np.nan
        self._histogram_rgba = self.color_indices_to_rgba(
            counts.T, is_overlay=False
        )
        self._mpl_artists["histogram_image"] = self.ax.imshow(
            self._histogram_rgba,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin="lower",
            zorder=1,
            interpolation=self._histogram_interpolation,
            alpha=1,
            aspect="auto",
        )

        if force_redraw:
            self.color_indices = 0  # Set default color index      


def _median_np(arr, method="lower") -> float:
    """Calculate the median of a 1D array.

    Parameters
    ----------
    arr : np.ndarray
        1D array of values.
    method : str, optional
        Method to use for calculating the median, by default 'lower'.

    Returns
    -------
    float
        The median of the array.
    """
    if len(arr) == 0:
        return np.nan
    return np.nanpercentile(arr, 50, method=method)
