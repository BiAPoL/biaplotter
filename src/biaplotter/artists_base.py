import warnings
from abc import ABC, abstractmethod
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import (CenteredNorm, Colormap, LogNorm, Normalize,
                               SymLogNorm)
from nap_plot_tools.cmap import cat10_mod_cmap
from psygnal import Signal

from biaplotter.colormap import BiaColormap


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
    #: Signal emitted when the `highlighted` data are changed.
    highlighted_changed_signal: Signal = Signal(np.ndarray)

    def __init__(
        self,
        ax: plt.Axes = None,
        data: np.ndarray = None,
        overlay_colormap: Colormap = cat10_mod_cmap,
        color_indices: np.ndarray = None,
    ):
        """Initializes the abstract artist."""
        self._ids = None
        #: Stores data to be plotted
        self._data: np.ndarray = data
        #: Stores axes to plot on
        self.ax: plt.Axes = ax if ax is not None else plt.gca()
        #: Stores visibility of the artist
        self._visible: bool = True
        #: Stores the colormap to use for the artist
        self._overlay_colormap: Colormap = BiaColormap(
            overlay_colormap, categorical=True
        )
        #: Stores the array of indices to map to the colormap
        self._color_indices: np.array = color_indices
        # store handles to mpl artists for modifying plots
        self._mpl_artists: dict = {}
        self._margins = 0.05

        self._normalization_methods = {
            "linear": Normalize,
            "log": LogNorm,
            "symlog": SymLogNorm,
            "centered": CenteredNorm,
        }

    def _update_axes_limits(self):
        """Update the axes limits based on the data range with a margin."""
        x_margin = self._margins * (
            np.nanmax(self._data[:, 0]) - np.nanmin(self._data[:, 0])
        )
        y_margin = self._margins * (
            np.nanmax(self._data[:, 1]) - np.nanmin(self._data[:, 1])
        )
        self.ax.set_xlim(
            np.nanmin(self._data[:, 0]) - x_margin,
            np.nanmax(self._data[:, 0]) + x_margin,
        )
        self.ax.set_ylim(
            np.nanmin(self._data[:, 1]) - y_margin,
            np.nanmax(self._data[:, 1]) + y_margin,
        )

    def reset(self):
        """Reset the artist to its initial state."""
        self._remove_artists()
        self._data = None
        self._color_indices = None
        self._ids = None
        self._visible = True
        self._overlay_colormap = BiaColormap(cat10_mod_cmap, categorical=True)
        self._mpl_artists = {}
        self._margins = 0.05

    @abstractmethod
    def _refresh(self, force_redraw: bool = True):
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )

    @abstractmethod
    def _colorize(self, indices: np.ndarray):
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )
    
    @abstractmethod
    def _highlight_data(self, highlight_mask: np.ndarray):
        """Highlight data points based on the provided boolean mask."""
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )

    @abstractmethod
    def color_indices_to_rgba(self, indices: np.ndarray) -> np.ndarray:
        """Convert color indices to RGBA values using the overlay colormap.

        Parameters
        ----------
        indices : np.ndarray
            Array of color indices.

        Returns
        -------
        np.ndarray
            Array of RGBA values corresponding to the indices.
        """
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )

    def _remove_artists(self, keys: List[str] = None):
        """
        Remove all contents from the plot.
        """

        if not keys:
            # Remove rectangle pacthes if any
            if hasattr(self, "_highlighted_bin_patches"):
                for patch in self._highlighted_bin_patches:
                    patch.remove()
            for artist in self._mpl_artists.values():
                artist.remove()
            self._mpl_artists = {}
        else:
            for key in keys:
                if key in self._mpl_artists.keys():
                    if key == "histogram_image":
                        # Remove rectangle patches if any
                        if hasattr(self, "_highlighted_bin_patches"):
                            for patch in self._highlighted_bin_patches:
                                patch.remove()
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
        """Sets the data for the artist, resetting other properties to defaults."""
        if value is None or len(value) == 0:
            return

        if self._data is not None:
            data_length_changed = len(value) != len(self._data)
        else:
            data_length_changed = True
        self._data = value

        if self._ids is None or len(self._ids) != len(value):
            # If ids are not set or have a different length, create a new array
            self._ids = np.arange(1, len(value) + 1)

        # Emit the data changed signal
        self.data_changed_signal.emit(self._data)
        self._refresh(force_redraw=data_length_changed)

        # Redraw the plot
        self._update_axes_limits()
        self.draw()

    @property
    def ids(self) -> np.ndarray:
        """Gets or sets the IDs associated with the data.

        Returns
        -------
        ids : (N,) np.ndarray[int]
            Array of IDs corresponding to the data points.
        """
        return self._ids

    @ids.setter
    def ids(self, value: np.ndarray):
        """Sets the IDs for the data points.

        Parameters
        ----------
        value : (N,) np.ndarray[int]
            Array of IDs. Must have the same length as the data.
        """
        if self._data is None:
            raise ValueError("Cannot set ids because data is not initialized.")
        if value is not None and len(value) != len(self._data):
            raise ValueError("Length of ids must match the length of data.")
        self._ids = value

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
        """Sets the visibility of the artists."""
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
    def color_indices(self, indices: Union[np.ndarray, int]):
        """Sets color indices for the plot and updates colors accordingly."""
        # Check if indices are a scalar
        if np.isscalar(indices):
            indices = np.full(len(self._data), indices)
        self._color_indices = indices

        if indices is not None and self._mpl_artists:
            self._colorize(indices)

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

    @property
    def x_label_text(self) -> str:
        """Gets or sets the x-axis label.
        
        Returns
        -------
        x_label_text : str
            Text of the x-axis label."""
        return self.ax.xaxis.label.get_text()
    
    @x_label_text.setter
    def x_label_text(self, value: str):
        """Sets the x-axis label."""
        self.ax.xaxis.label.set_text(value)

    @property
    def y_label_text(self) -> str:
        """Gets or sets the y-axis label.
        
        Returns
        -------
        y_label_text : str
            Text of the y-axis label.
        """
        return self.ax.yaxis.label.get_text()
    
    @y_label_text.setter
    def y_label_text(self, value: str):
        """Sets the y-axis label."""
        self.ax.yaxis.label.set_text(value)

    @property
    def x_label_color(self) -> Union[str, tuple]:
        """Gets or sets the x-axis label color.
        
        Returns
        -------
        x_label_color : str or tuple
            Color value for the x-axis label.
            Check more at https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def"""
        return self.ax.xaxis.label.get_color()

    @x_label_color.setter
    def x_label_color(self, value: Union[str, tuple]):
        """Sets the x-axis label color.
        
        Parameters
        ----------
        value : str or tuple
            Color value for the x-axis label.
            Can be anything accepted by matplotlib, e.g., 'red', (1, 0, 0), etc.
            Check more at https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
        """
        self.ax.xaxis.label.set_color(value)

    @property
    def y_label_color(self) -> Union[str, tuple]:
        """Gets or sets the y-axis label color.
        
        Returns
        -------
        y_label_color : str or tuple
            Color value for the y-axis label.
            Check more at https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
        """
        return self.ax.yaxis.label.get_color()
    
    @y_label_color.setter
    def y_label_color(self, value: Union[str, tuple]):
        """Sets the y-axis label color.
        
        Parameters
        ----------
        value : str or tuple
            Color value for the y-axis label.
            Can be anything accepted by matplotlib, e.g., 'red', (1, 0, 0), etc.
            Check more at https://matplotlib.org/stable/users/explain/colors/colors.html#colors-def
        """
        self.ax.yaxis.label.set_color(value)
    
    @property
    def highlighted(self) -> np.ndarray:
        """Gets or sets the highlighted data points.
        
        Returns
        -------
        highlighted : (N,) np.ndarray[bool]
            Boolean array indicating which data points are highlighted.
            If None, no points are highlighted.

        Notes
        -----
        highlighted_changed_signal : Signal
            Signal emitted when the highlighted data points are changed.
        """
        return self._highlighted
    
    @highlighted.setter
    def highlighted(self, value: np.ndarray):
        """Sets the highlighted data points.
        """
        if self._data is None or len(self._data) == 0:
            self._highlighted = None
            return
        self._highlight_data(value)
        self._highlighted = value
        self.highlighted_changed_signal.emit(self._highlighted)

    def highlight_data_by_ids(
        self, ids: Union[int, List[int], None] = None, color: Union[str, tuple] = None, unhighlight: bool = False
    ):
        """
        Highlights or unhighlights data (points/bins) based on their IDs.

        Parameters
        ----------
        ids : int, List[int], or None, optional
            A single ID, a list of IDs to highlight/unhighlight, or None to reset all highlighted data.
        color : str or tuple, optional
            The color to use for the highlighted data (points/bins).
            Not used if histogram artist.
        unhighlight : bool, optional
            If True, removes the specified IDs from the highlighted data.
            Default is False.
        """
        if color is not None and hasattr(self, "_highlight_edgecolor"):
            self._highlight_edgecolor = color

        if ids is None or len(ids) == 0:
            self.highlighted = None
            return

        if isinstance(ids, int):
            ids = [ids]

        # Find the indices of the points corresponding to the given IDs
        highlight_indices = np.isin(self.ids, ids)

        if unhighlight:
            # Remove the specified IDs from the highlighted bins
            if self._highlighted is not None:
                self._highlighted[highlight_indices] = False
        else:
            # Add the specified IDs to the highlighted bins
            if self._highlighted is None:
                self._highlighted = np.zeros(len(self._data), dtype=bool)
            self._highlighted[highlight_indices] = True
        
        self.highlighted = self._highlighted

    def draw(self):
        """Draws or redraws the artist."""
        self.ax.figure.canvas.draw_idle()

    def _log_normalization(self, values: np.ndarray):
        """Log normalization."""
        norm_class = self._normalization_methods["log"]
        min_value = np.nanmin(values)
        if min_value <= 0:
            min_value = 0.01
        warnings.warn(
            f"Log normalization applied to color indices with min value {min_value}. Values below 0.01 were set to 0.01."
        )
        values[values <= 0] = min_value
        return norm_class(vmin=min_value, vmax=np.nanmax(values))

    def _centered_normalization(self, values: np.ndarray):
        """Centered normalization."""
        norm_class = self._normalization_methods["centered"]
        return norm_class(vcenter=np.nanmean(values))

    def _symlog_normalization(self, values: np.ndarray):
        """Symmetric log normalization."""
        norm_class = self._normalization_methods["symlog"]
        return norm_class(
            vmin=np.nanmin(values),
            vmax=np.nanmax(values),
            linthresh=0.03,
        )

    def _linear_normalization(
        self, values: np.ndarray, is_categorical: bool = False
    ):
        """Linear normalization."""
        norm_class = self._normalization_methods["linear"]

        if is_categorical:
            norm = norm_class(
                vmin=0,
                vmax=self.overlay_colormap.cmap.N,
            )
        else:
            norm = norm_class(
                vmin=np.nanmin(values),
                vmax=np.nanmax(values),
            )
        return norm
