import warnings
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import QuadMesh
from matplotlib.colors import (CenteredNorm, Colormap, LogNorm, Normalize,
                               SymLogNorm)
from nap_plot_tools.cmap import (cat10_mod_cmap,
                                 cat10_mod_cmap_first_transparent)
from psygnal import Signal

from biaplotter.colormap import BiaColormap
from scipy.stats import binned_statistic_2d


class Artist(ABC):
    """Abstract base class for artists in the BiAPlotter.

    Parameters
    ----------
    ax : plt.Axes, optional
        axes to plot on, by default None
    data : (N, 2) np.ndarray
        data to be plotted
    overlay_colormap : Colormap, optional
        a colormap to use for the artist, by default cat10_mod_cmap from nap-plot-tools
    color_indices : (N,) np.ndarray, optional
        array of indices to map to the colormap, by default None
    """

    #: Signal emitted when the `data` are changed.
    data_changed_signal: Signal = Signal(np.ndarray)
    #: Signal emitted when the `color_indices` are changed.
    color_indices_changed_signal: Signal = Signal(np.ndarray)

    def __init__(
        self,
        ax: plt.Axes = None,
        data: np.ndarray = None,
        overlay_colormap: Colormap = cat10_mod_cmap,
        color_indices: np.ndarray = None,
    ):
        """Initializes the abstract artist."""
        #: Stores data to be plotted
        self._data: np.ndarray = data
        #: Stores axes to plot on
        self.ax: plt.Axes = ax if ax is not None else plt.gca()
        #: Stores visibility of the artist
        self._visible: bool = True
        #: Stores the colormap to use for the artist
        self._overlay_colormap: Colormap = BiaColormap(overlay_colormap)
        #: Stores the array of indices to map to the colormap
        self._color_indices: np.array = color_indices
        # store handles to mpl artists for modifying plots
        self._mpl_artists: dict = {}
        self._margins = 0.05

    def _update_axes_limits(self):
        """Update the axes limits based on the data range with a margin."""
        x_margin = self._margins * (np.nanmax(self._data[:, 0]) - np.nanmin(self._data[:, 0]))
        y_margin = self._margins * (np.nanmax(self._data[:, 1]) - np.nanmin(self._data[:, 1]))
        self.ax.set_xlim(
            np.nanmin(self._data[:, 0]) - x_margin,
            np.nanmax(self._data[:, 0]) + x_margin,
        )
        self.ax.set_ylim(
            np.nanmin(self._data[:, 1]) - y_margin,
            np.nanmax(self._data[:, 1]) + y_margin,
        )

    @abstractmethod
    def _create_plot(self, force_redraw: bool = True):
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )
    
    #@abstractmethod
    def _draw_selection_on_plot(self, indiecs: np.ndarray):
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )

    def _modify_plot(self):
        """Modify the existing plot with new data or properties."""
        pass

    def _remove_artists(self, keys: List[str] = None):
        """
        Remove all contents from the plot.
        """

        if not keys:
            [artist.remove() for artist in self._mpl_artists.values()]
            self._mpl_artists = {}
        else:
            for key in keys:
                if key in self._mpl_artists.keys():
                    self._mpl_artists[key].remove()
                    del self._mpl_artists[key]

    @property
    def data(self) -> np.ndarray:
        """Gets or sets the data associated with the artist.

        Updates colors if color_indices are set. Triggers a draw idle command.

        Returns
        -------
        data : (N, 2) np.ndarray
           data for the artist. Does not respond if set to None or empty array.

        Notes
        -----
        data_changed_signal : Signal
            Signal emitted when the data are changed.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data for the scatter plot, resetting other properties to defaults."""
        if value is None or len(value) == 0:
            return

        if self._data is not None:
            data_length_changed = len(value) != len(self._data)
        else:
            data_length_changed = True
        self._data = value

        # Emit the data changed signal
        self.data_changed_signal.emit(self._data)
        self._create_plot(force_redraw=data_length_changed)

        # Redraw the plot
        self._update_axes_limits()
        self.draw()

    @property
    def visible(self) -> bool:
        """Gets or sets the visibility of the artist.

        Triggers a draw idle command.

        Returns
        -------
        visible : bool
            visibility of the artist.
        """
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        """Sets the visibility of the scatter plot."""
        self._visible = value
        [a.set_visible(value) for a in self._mpl_artists.values()]
        self.draw()

    @property
    def color_indices(self) -> np.ndarray:
        """Gets or sets the current color indices used for the artist.

        Triggers a draw idle command.

        Returns
        -------
        color_indices : (N,) np.ndarray[int] or int
            indices to map to the overlay_colormap. Accepts a scalar or an array of integers.

        Notes
        -----
        color_indices_changed_signal : Signal
            Signal emitted when the color indices are changed.

        """
        return self._color_indices

    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
        """Sets color indices for the plot and updates colors accordingly."""
        # Check if indices are a scalar
        if np.isscalar(indices):
            indices = np.full(len(self._data), indices)
        self._color_indices = indices

        if indices is not None and self._mpl_artists:
            self._draw_selection_on_plot(indices)

        # emit signal
        self.color_indices_changed_signal.emit(self._color_indices)
        self.draw()

    @property
    def overlay_colormap(self) -> BiaColormap:
        """Gets or sets the overlay colormap for the artist.

        Returns
        -------
        overlay_colormap : BiaColormap
            colormap for the artist with a `categorical` attribute.
        """
        return self._overlay_colormap

    @overlay_colormap.setter
    def overlay_colormap(self, value: Colormap):
        """Sets the overlay colormap for the artist."""
        self._overlay_colormap = BiaColormap(value)
        self.color_indices = self._color_indices

    def draw(self):
        """Draws or redraws the artist."""
        self.ax.figure.canvas.draw_idle()


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
        self._overlay_colormap = BiaColormap(overlay_colormap)
        self._overlay_visible = True
        self._normalization_methods = {
            "linear": Normalize,
            "log": LogNorm,
            "symlog": SymLogNorm,
            "centered": CenteredNorm,
        }
        self._color_normalization_method = "linear"
        self.data = data
        self._alpha = 1  # Default alpha
        self._size = 50  # Default size
        self.draw()  # Initial draw of the scatter plot

    def _create_plot(self, force_redraw: bool = True):
        """Creates the scatter plot with the data and default properties."""

        if force_redraw or self._mpl_artists['scatter'] is None:
            self._remove_artists()
            # Create a new scatter plot with the updated data
            self._mpl_artists['scatter'] = self.ax.scatter(
                self._data[:, 0], self._data[:, 1])
            self.size = 50  # Default size
            self.alpha = 1  # Default alpha
            self.color_indices = 0
        else:
            self._mpl_artists['scatter'].set_offsets(
                self._data
            )  #  somehow resets the size and alpha
            self.color_indices = self._color_indices
            self.size = self._size
            self.alpha = self._alpha

    def _draw_selection_on_plot(self, indices: np.ndarray) -> np.ndarray:
        """
        Add a color to the drawn scatter points
        """
        norm = self._get_normalization(indices)
        rgba_colors = self._get_rgba_colors(indices, norm)
        self._mpl_artists['scatter'].set_facecolor(rgba_colors)
        self._mpl_artists['scatter'].set_edgecolor("white")

        return rgba_colors

    def _get_normalization(self, indices):
        """Determine the normalization method and return the normalization object."""
        norm_class = self._normalization_methods[
            self._color_normalization_method
        ]

        if self.overlay_colormap.categorical:
            self._validate_categorical_colormap()
            return Normalize(vmin=0, vmax=self.overlay_colormap.N)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            if self._color_normalization_method == "log":
                return self._log_normalization(indices, norm_class)
            elif self._color_normalization_method == "centered":
                return norm_class(vcenter=np.nanmean(self._color_indices))
            elif self._color_normalization_method == "symlog":
                return norm_class(
                    vmin=np.nanmin(self._color_indices),
                    vmax=np.nanmax(self._color_indices),
                    linthresh=0.03,
                )
            else:
                return norm_class(
                    vmin=np.nanmin(self._color_indices),
                    vmax=np.nanmax(self._color_indices),
                )

    def _get_normalization_instance(self) -> Normalize:
        """
        Returns the normalization instance for the scatter plot
        based on the current color_indices.

        This method preserves the existing behavior including warnings.
        """
        return self._get_normalization(self._color_indices)

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

    def _log_normalization(self, indices, norm_class):
        """Apply log normalization to indices."""
        if np.nanmin(indices) <= 0:
            warnings.warn(
                f"Log normalization applied to values <= 0. Values below 0 were set to np.nan"
            )
            indices[indices <= 0] = np.nan
        min_value = np.nanmin(indices)

        return norm_class(vmin=min_value, vmax=np.nanmax(indices))

    def _get_rgba_colors(self, indices, norm):
        """Convert normalized data to RGBA colors."""
        sm = ScalarMappable(norm=norm, cmap=self.overlay_colormap.cmap)
        rgba_colors = sm.to_rgba(indices)
        if not self._overlay_visible:
            rgba_colors = cat10_mod_cmap(0)  # Set colors to light gray
        return rgba_colors

    @property
    def overlay_visible(self) -> bool:
        """Gets or sets the visibility of the overlay colormap.

        Returns
        -------
        overlay_visible : bool
           visibility of the overlay colormap.
        """
        return self._overlay_visible

    @overlay_visible.setter
    def overlay_visible(self, value: bool):
        """Sets the visibility of the overlay colormap."""
        self._overlay_visible = value
        self.color_indices = self._color_indices

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
        self.color_indices = self._color_indices

    @property
    def alpha(self) -> Union[float, np.ndarray]:
        """Gets or sets the alpha value of the scatter plot.

        Returns
        -------
        alpha : float
            alpha value of the scatter plot.
        """
        return self._mpl_artists['scatter'].get_alpha()

    @alpha.setter
    def alpha(self, value: Union[float, np.ndarray]):
        """Sets the alpha value of the scatter plot."""
        self._alpha = value

        if np.isscalar(value):
            value = np.ones(len(self._data)) * value
        if self._mpl_artists['scatter'] is not None:
            self._mpl_artists['scatter'].set_alpha(value)
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
        if not self._mpl_artists:
            self._mpl_artists['scatter'].set_sizes(
                np.full(len(self._data), value)
                if np.isscalar(value)
                else value
            )
        self.draw()


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
        super().__init__(ax, data, overlay_colormap, color_indices)
        """Initializes the 2D histogram artist.
        """
        #: Stores the matplotlib histogram2D object
        self._histogram = None
        self._bins = bins
        self._histogram_colormap = BiaColormap(histogram_colormap)
        self._overlay_colormap = BiaColormap(overlay_colormap)
        self._histogram_interpolation = "nearest"
        self._overlay_interpolation = "nearest"
        self._overlay_opacity = 1
        self._overlay_visible = True
        self._normalization_methods = {
            "linear": Normalize,
            "log": LogNorm,
            "symlog": SymLogNorm,
            "centered": CenteredNorm,
        }
        self._margins = 0
        self._histogram_color_normalization_method = "linear"
        self._overlay_color_normalization_method = "linear"
        self.data = data
        self.cmin = cmin
        self.draw()  # Initial draw of the histogram

    def _create_plot(self, force_redraw: bool = True):
        self._remove_artists()
        # Calculate and draw the new histogram
        self._histogram = np.histogram2d(
            self._data[:, 0], self._data[:, 1], bins=self._bins
        )
        counts, x_edges, y_edges = self._histogram
        self._histogram_rgba = self._histogram2D_array_to_rgba(
            counts, is_overlay=False
        )
        self._mpl_artists['histogram_image'] = self.ax.imshow(
            self._histogram_rgba,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin="lower",
            zorder=1,
            interpolation=self._histogram_interpolation,
            alpha=1,
        )

        self.color_indices = 0  # Set default color index


    @color_indices.setter
    def color_indices(self, indices: np.ndarray):
        """Sets color indices for the 2D histogram underlying data and updates colors accordingly."""
        # Check if indices are a scalar
        if np.isscalar(indices):
            indices = np.full(len(self._data), indices)
        self._color_indices = indices

        # Remove the existing overlay to redraw
        self._remove_artists(["overlay_histogram_image"])
        counts, x_edges, y_edges = self._histogram

        # Assign median values to the bins (fill with NaNs if no data in the bin)
        statistic_histogram, _, _, _ = binned_statistic_2d(
            x = self._data[:, 0],
            y= self._data[:, 1],
            values=indices,
            statistic="median",
            bins=[x_edges, y_edges]
        )
        if not np.all(np.isnan(statistic_histogram)):
            # Draw the overlay
            self.overlay_histogram_rgba = self._histogram2D_array_to_rgba(
                statistic_histogram, is_overlay=True
            )
            self._mpl_artists['overlay_histogram_image'] = self.ax.imshow(
                self.overlay_histogram_rgba,
                extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
                origin="lower",
                zorder=2,
                interpolation=self._overlay_interpolation,
                alpha=self._overlay_opacity,
            )

        # emit signal
        self.color_indices_changed_signal.emit(self._color_indices)
        self.draw()

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
        self.color_indices = self._color_indices

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
        self._histogram_colormap = BiaColormap(value)
        self.data = self._data

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
        self.data = self._data

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
        self.data = self._data

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
        self.data = self._data

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
        if 'overlay_histogram_image' in self._mpl_artists:
            self._mpl_artists['overlay_histogram_image'].set_visible(value)
        self.draw()

    @property
    def histogram(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the 2D histogram array and edges.

        Returns
        -------
        histogram : Tuple[np.ndarray, np.ndarray, np.ndarray]
            2D histogram, x edges, and y edges.
        """
        return self._histogram

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

    def _handle_norm_method_for_categorical_colormap(
        self, is_overlay, colormap, dtype
    ):
        """
        Set the normalization method for the histogram data when the colormap is categorical.

        Parameters
        ----------
        is_overlay : bool
            whether the histogram is an overlay or not.
        colormap : BiaColormap
            colormap to use for the image with `categorical` attribute
        dtype : type
            data type of the histogram data, can be int or float. A categorical colormap expects positive integer color indices.

        Returns
        -------
        norm_class : Normalize
            the normalization class to use for the histogram data (linear normalization for categorical colormap).
        """
        if is_overlay:
            color_normalization_method = (
                self._overlay_color_normalization_method
            )
        else:
            color_normalization_method = (
                self._histogram_color_normalization_method
            )

        if dtype != int:
            # Warn user that categorical colormap expects integer color indices
            warnings.warn(
                f'Color indices must be integers for categorical colormap. Change `{"overlay_" if is_overlay else "histogram_"}colormap` to a continuous colormap or set `color_indices` to integers.'
            )
        if color_normalization_method != "linear":
            # Warn user that categorical colormap employs linear normalization
            warnings.warn(
                f'Categorical colormap detected in `{"overlay_" if is_overlay else "histogram_"}colormap`. Setting color normalization method to linear.'
            )
            if is_overlay:
                self._overlay_color_normalization_method = "linear"
            else:
                self._histogram_color_normalization_method = "linear"
        return Normalize(vmin=0, vmax=colormap.N)

    def _handle_norm_method_for_continuous_colormap(
        self,
        is_overlay,
        norm_class,
        histogram_data,
        lintresh_for_symlog=0.03,
        min_value_for_log=0.01,
    ):
        """
        Set the normalization method for the histogram data when the colormap is continuous.

        Parameters
        ----------
        is_overlay : bool
            whether the histogram is an overlay or not.
        norm_class : Normalize
            normalization class to use for the histogram data.
        histogram_data : np.ndarray
            2D histogram data array to be converted to RGBA image.
        lintresh_for_symlog : float, optional
            linear threshold for SymLogNorm, by default 0.03
        min_value_for_log : float, optional
            minimum value for log normalization, by default 0.01

        Returns
        -------
        norm_class : Normalize
            the normalization class to use for the histogram data.
        """
        if is_overlay:
            color_normalization_method = (
                self._overlay_color_normalization_method
            )
        else:
            color_normalization_method = (
                self._histogram_color_normalization_method
            )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"All-NaN slice encountered")
            if color_normalization_method == "log":
                min_value = np.nanmin(histogram_data)
                if min_value <= 0:
                    min_value = min_value_for_log
                norm = norm_class(
                    vmin=min_value, vmax=np.nanmax(histogram_data)
                )
                warnings.warn(
                    f"Log normalization applied to color indices with min value {min_value}. Values below 0.01 were set to 0.01."
                )
                histogram_data[histogram_data <= 0] = min_value
            elif color_normalization_method == "centered":
                norm = norm_class(vcenter=np.nanmean(histogram_data))
            elif color_normalization_method == "symlog":
                norm = norm_class(
                    vmin=np.nanmin(histogram_data),
                    vmax=np.nanmax(histogram_data),
                    linthresh=lintresh_for_symlog,
                )
            else:
                norm = norm_class(
                    vmin=np.nanmin(histogram_data),
                    vmax=np.nanmax(histogram_data),
                )
        return norm

    def _get_normalization_class(self, is_overlay=False):
        """
        Get the normalization class

        Parameters
        ----------
        is_overlay : bool
            whether the histogram is an overlay or not. By default False.

        Returns
        -------
        norm_class : Normalize
            the normalization class to use for the histogram data or overlay data.
        """
        if is_overlay:
            return self._normalization_methods[
                self._overlay_color_normalization_method
            ]
        else:
            return self._normalization_methods[
                self._histogram_color_normalization_method
            ]

    def _select_norm_class(self, is_overlay, histogram_data):
        """
        Select the normalization class for the histogram or overlay histogram data.

        Parameters
        ----------
        is_overlay : bool
            whether the histogram is an overlay or not.
        histogram_data : np.ndarray
            2D histogram data array to be converted to RGBA image.

        Returns
        -------
        norm_class : Normalize
            the normalization class to use for the histogram or overlay histogram data.
        """
        if is_overlay:
            colormap = self.overlay_colormap
            dtype = self._color_indices.dtype
        else:
            colormap = self.histogram_colormap
            dtype = float
        norm_class = self._get_normalization_class(is_overlay)
        if self._is_categorical_colormap(colormap):
            return self._handle_norm_method_for_categorical_colormap(
                is_overlay, colormap, dtype
            )
        else:
            return self._handle_norm_method_for_continuous_colormap(
                is_overlay, norm_class, histogram_data
            )

    def _get_normalization_instance(
        self, histogram_data: np.ndarray = None, overlay: bool = False
    ) -> Normalize:
        """
        Returns the normalization instance for the histogram.

        Parameters
        ----------
        histogram_data : np.ndarray, optional
            The 2D data array used for normalization. If not provided, the method uses the histogram
            counts computed during the latest data update.
        overlay : bool, default False
            If True, the normalization is determined using the overlay color settings.

        Returns
        -------
        norm : Normalize
            The normalization instance with the appropriate settings.

        Raises
        ------
        ValueError
            If no histogram data is provided and the histogram has not yet been computed.
        """
        if histogram_data is None:
            if self._histogram is None:
                raise ValueError(
                    "Histogram has not been computed; please set the data first."
                )
            # Use the counts from the histogram (returned as the first element by np.histogram2d)
            histogram_data = self._histogram[0]
        return self._select_norm_class(overlay, histogram_data)

    def _histogram2D_array_to_rgba(
        self, histogram_data, is_overlay=False
    ):
        """
        Convert a 2D data array to a RGBA image object via pcolormesh using a matplotlib colormap.

        Parameters
        ----------
        ax : plt.Axes
            axes to plot on
        histogram_data : np.ndarray
            2D data array to be converted to RGBA image
        x_edges : np.ndarray
            x bin edges
        y_edges : np.ndarray
            y bin edges
        is_overlay : bool, optional
            whether the histogram is an overlay or not, by default False

        Returns
        -------
        rgba_array : np.ndarray
            RGBA image array
        """
        norm = self._select_norm_class(is_overlay, histogram_data)
        cmap = [self.histogram_colormap.cmap, self.overlay_colormap.cmap][
                is_overlay
            ]
        
        return cmap(norm(histogram_data.T))

