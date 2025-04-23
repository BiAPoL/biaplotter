from abc import ABC, abstractmethod
from typing import List, Union
from psygnal import Signal

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import (CenteredNorm, Colormap, LogNorm, Normalize,
                               SymLogNorm)
from nap_plot_tools.cmap import cat10_mod_cmap
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

        self._normalization_methods = {
            "linear": Normalize,
            "log": LogNorm,
            "symlog": SymLogNorm,
            "centered": CenteredNorm,
        }

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
    def _sync_artist_data(self, force_redraw: bool = True):
        raise NotImplementedError(
            "This method should be implemented in the derived class."
        )
    
    @abstractmethod
    def _colorize_artist(self, indices: np.ndarray):
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
        self._sync_artist_data(force_redraw=data_length_changed)

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
    def color_indices(self, indices: Union[np.ndarray, int]):
        """Sets color indices for the plot and updates colors accordingly."""
        # Check if indices are a scalar
        if np.isscalar(indices):
            indices = np.full(len(self._data), indices)
        self._color_indices = indices

        if indices is not None and self._mpl_artists:
            self._colorize_artist(indices)

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