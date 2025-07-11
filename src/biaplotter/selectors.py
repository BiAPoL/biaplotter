from __future__ import annotations  # Only necessary for Python 3.7 to 3.9

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as mplPath
from matplotlib.widgets import (EllipseSelector, LassoSelector,
                                RectangleSelector)
from nap_plot_tools.cmap import (cat10_mod_cmap,
                                 cat10_mod_cmap_first_transparent)
from psygnal import Signal
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication, QCursor

from .artists import Histogram2D, Scatter

if TYPE_CHECKING:
    from biaplotter.plotter import CanvasWidget


class Selector(ABC):
    """Abstract class for creating a selector.

    Parameters
    ----------
    ax : plt.Axes, optional
        axes to which the selector will be applied.
    data : np.ndarray
        data to be selected.
    """

    def __init__(self, ax: plt.Axes = None, data: np.ndarray = None):
        """Initializes the selector."""
        #: Stores the axes to which the selector will be applied.
        self.ax: plt.Axes = ax
        #: Stores the data to be selected
        self._data = data
        #: Stores the selector
        self._selector = None

    @property
    @abstractmethod
    def data(self) -> np.ndarray:
        """Abstract property for the selector's data."""
        pass

    @data.setter
    @abstractmethod
    def data(self, value: np.ndarray):
        """Abstract setter for the selector's data."""
        pass

    @abstractmethod
    def create_selector(self):
        """Abstract method to create a selector."""
        pass

    @abstractmethod
    def on_select(self, vertices: np.ndarray):
        """Abstract method to select points based on the selector's shape."""
        pass

    def remove(self):
        """Removes the selector from the canvas."""
        if self._selector:
            self._selector.clear()
            self._selector.disconnect_events()
            self._selector = None


class _MplRectangleSelector(RectangleSelector):
    """Custom rectangle selector class.

    Sub-class of matplotlib RectangleSelector to ensure, via 'qtpy', the option to draw a square when holding the SHIFT key.
    It also sets the cursor to a cross cursor.
    
    Note: matplotlib RectangleSelector already has this functionality via the 'state_modifier_keys' argument, but it doesn't work if the canvas is used inside napari.
    """
    def __init__(self, ax, onselect, **kwargs):
        self._canvas = ax.figure.canvas
        super().__init__(ax, onselect, **kwargs)
        self._set_cursor()

    def _set_cursor(self):
        # only Qt‐based FigureCanvases have setCursor(), macosx backends might not have this method
        if hasattr(self._canvas, "setCursor"):
            self._canvas.setCursor(QCursor(Qt.CrossCursor))

    def onpress(self, event):
        super().onpress(event)
        self._set_cursor()

    def onrelease(self, event):
        super().onrelease(event)
        self._set_cursor()

    def _onmove(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.add_state("square")
        else:
            if "square" in self._state:
                self._state.remove("square")
        super()._onmove(event)
        self._set_cursor()


class BaseRectangleSelector(Selector):
    """Base class for creating a rectangle selector.

    Inherits all parameters and attributes from Selector.
    For parameter and attribute details, see the Selector class documentation.

    Parameters
    ----------
    ax : plt.Axes
        axes to which the selector will be applied.
    data : (N, 2) np.ndarray
        data to be selected.

    """

    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        """Initializes the rectangle selector."""
        super().__init__(ax, data)
        #: The name of the selector, set to 'Rectangle Selector' by default.
        self.name: str = "Rectangle Selector"
        self.data = data

    def on_select(self, eclick, erelease) -> np.ndarray:
        """Selects points within the rectangle and returns their indices.

        Parameters
        ----------
        eclick : MouseEvent
            The press event.
        erelease : MouseEvent
            The release event.

        Returns
        -------
        np.ndarray
            The indices of the selected points.
        """
        if self._data is None or len(self._data) == 0:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        mask = (
            (self._data[:, 0] >= min(x1, x2))
            & (self._data[:, 0] <= max(x1, x2))
            & (self._data[:, 1] >= min(y1, y2))
            & (self._data[:, 1] <= max(y1, y2))
        )
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        """Gets or sets the data from which points will be selected.

        Returns
        -------
        np.ndarray
            The data from which points will be selected.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data from which points will be selected."""
        self._data = value

    def create_selector(self):
        """Creates a rectangle selector.

        Useblit is set to True to improve performance.
        Left mouse button is used to draw the rectangle.
        Minimum span in x and y is set to 5 pixels.
        Span coordinates are set to pixels.
        Interactive is set to True to allow for interaction.
        Drag from anywhere is set to True to allow for drawing from any point.
        """
        self._selector = _MplRectangleSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            drag_from_anywhere=True,
            props=dict(
                facecolor="#00c18c",
                edgecolor="#00c18c",
                alpha=0.3,
                fill=True,
                linewidth=2.5,
                linestyle="--",
            ),
        )

class _MplEllipseSelector(EllipseSelector):
    """Custom ellipse selector class.

    Sub-class of matplotlib EllipseSelector to ensure, via 'qtpy', the option to draw a circle when holding the SHIFT key.
    It also sets the cursor to a cross cursor.

    Note: matplotlib EllipeseSelector already has this functionality via the 'state_modifier_keys' argument, but it doesn't work if the canvas is used inside napari.
    """
    def __init__(self, ax, onselect, **kwargs):
        self._canvas = ax.figure.canvas
        super().__init__(ax, onselect, **kwargs)
        self._set_cursor()
    
    def _set_cursor(self):
        # only Qt‐based FigureCanvases have setCursor(), macosx backends might not have this method
        if hasattr(self._canvas, "setCursor"):
            self._canvas.setCursor(QCursor(Qt.CrossCursor))

    def onpress(self, event):
        super().onpress(event)
        self._set_cursor()

    def onrelease(self, event):
        super().onrelease(event)
        self._set_cursor()

    def _onmove(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        if modifiers == Qt.ShiftModifier:
            self.add_state("square")
        else:
            if "square" in self._state:
                self._state.remove("square")
        super()._onmove(event)
        self._set_cursor()

class BaseEllipseSelector(Selector):
    """Base class for creating an ellipse selector.

    Inherits all parameters and attributes from Selector.
    For parameter and attribute details, see the Selector class documentation.

    Parameters
    ----------
    ax : plt.Axes
        axes to which the selector will be applied.
    data : np.ndarray
        data to be selected.

    """

    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        """Initializes the ellipse selector."""
        super().__init__(ax, data)
        #: The name of the selector, set to 'Ellipse Selector' by default.
        self.name: str = "Ellipse Selector"
        self.data = data

    def on_select(self, eclick, erelease):
        """Selects points within the ellipse and returns their indices.

        Parameters
        ----------
        eclick : MouseEvent
            The press event.
        erelease : MouseEvent
            The release event.

        Returns
        -------
        np.ndarray
            The indices of the selected points."""
        if self._data is None or len(self._data) == 0:
            return
        center = np.array(
            [
                (eclick.xdata + erelease.xdata) / 2,
                (eclick.ydata + erelease.ydata) / 2,
            ]
        )
        width = abs(eclick.xdata - erelease.xdata)
        height = abs(eclick.ydata - erelease.ydata)
        mask = (
            ((self._data[:, 0] - center[0]) / (width / 2)) ** 2
            + ((self._data[:, 1] - center[1]) / (height / 2)) ** 2
        ) <= 1
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        """Gets or sets the data from which points will be selected.

        Returns
        -------
        np.ndarray
            The data from which points will be selected.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data from which points will be selected."""
        self._data = value

    def create_selector(self):
        """Creates an ellipse selector.

        Useblit is set to True to improve performance.
        Left mouse button is used to draw the ellipse.
        Minimum span in x and y is set to 5 pixels.
        Span coordinates are set to pixels.
        Interactive is set to True to allow for interaction.
        Drag from anywhere is set to True to allow for drawing from any point.
        """
        self._selector = _MplEllipseSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True,
            drag_from_anywhere=True,
            props=dict(
                facecolor="#00c18c",
                edgecolor="#00c18c",
                alpha=0.3,
                fill=True,
                linewidth=2.5,
                linestyle="--",
            ),
        )

class _MplLassoSelector(LassoSelector):
    """Custom lasso selector class.

    Sub-class of matplotlib LassoSelector to draw a lasso with a cross cursor.
    """
    def __init__(self, ax, onselect, **kwargs):
        self._canvas = ax.figure.canvas
        super().__init__(ax, onselect, **kwargs)
        self._set_cursor()

    def _set_cursor(self):
        # only Qt‐based FigureCanvases have setCursor(), macosx backends might not have this method
        if hasattr(self._canvas, "setCursor"):
            self._canvas.setCursor(QCursor(Qt.CrossCursor))

    def onpress(self, event):
        super().onpress(event)
        self._set_cursor()

    def onrelease(self, event):
        super().onrelease(event)
        self._set_cursor()

    def onmove(self, event):
        super().onmove(event)
        self._set_cursor()


class BaseLassoSelector(Selector):
    """Base class for creating a lasso selector.

    Inherits all parameters and attributes from Selector.
    For parameter and attribute details, see the Selector class documentation.

    Parameters
    ----------
    ax : plt.Axes
        axes to which the selector will be applied.
    data : np.ndarray
        data to be selected.

    """

    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        super().__init__(ax, data)
        #: The name of the selector, set to 'Lasso Selector' by default.
        self.name: str = "Lasso Selector"
        self.data = data

    def on_select(self, vertices):
        """Selects points within the lasso and returns their indices.

        Parameters
        ----------
        vertices : np.ndarray
            The vertices of the lasso.

        Returns
        -------
        np.ndarray
            The indices of the selected points.
        """
        if self._data is None or len(self._data) == 0:
            return
        path = mplPath(vertices)
        mask = path.contains_points(self._data)
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        """Gets or sets the data from which points will be selected.

        Returns
        -------
        np.ndarray
            The data from which points will be selected.
        """
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data from which points will be selected."""
        self._data = value

    def create_selector(self):
        """Creates a lasso selector.

        Useblit is set to True to improve performance.
        Left mouse button is used to draw the lasso.
        """
        self._selector = _MplLassoSelector(
            self.ax,
            self.on_select,
            useblit=True,
            button=[1],
            props=dict(
                color="#05ffe2", linestyle="--", linewidth=2.5, alpha=0.6
            ),
        )


class Interactive(Selector):
    """Interactive selector class.

    Inherits all parameters and attributes from Selector.
    To be used as a base class together with a selector to turn selectors interactive.

    Parameters
    ----------
    ax : plt.Axes
        axes to which the selector will be applied.
    canvas_widget : CanvasWidget
        canvas widget to which the selector will be applied.
    data : (N, 2) np.ndarray, optional
        data to be selected.

    Notes
    -----
    **Signals:**

    * **selection_applied_signal** emitted when the `apply_selection` is called. Let's the canvas widget know that color_indices were updated because of a selection.

    **Slots:**

        * **update_class_value** method intended to be connected by the **color_spinbox_value_changed_signal** emitted by the canvas_widget to have class_value synchronized.
        * **update_data** method intended to be connected by the **data_changed_signal** emitted by the active_artist to have the selector data synchronized.
        * **update_active_artist** method intended to be connected by the **artist_changed_signal** emitted by the canvas_widget to have the active_artist synchronized.

    **Signals and Slots:**

        This class automatically connects the following signals to slots:

        * **color_spinbox_value_changed_signal** emitted by the canvas_widget to **update_class_value** slot.
        * **data_changed_signal** emitted by the active_artist to **update_data** slot.

    """

    #: Signal emitted when the `apply_selection` is called. Let's the canvas widget know that color_indices were updated because of a selection.
    selection_applied_signal: Signal = Signal(np.ndarray)

    def __init__(
        self,
        ax: plt.Axes,
        canvas_widget: "CanvasWidget",
        data: np.ndarray = None,
    ):
        """Initializes the interactive selectors."""
        super().__init__(ax, data)
        #: The canvas widget to which the selector will be applied.
        self.canvas_widget: "CanvasWidget" = canvas_widget

        self._selected_indices = None  # To store indices of selected points
        # Get initial class value
        self._class_value = self.canvas_widget.class_spinbox.value
        # Get initial active artist
        self._active_artist = self.canvas_widget.active_artist

        # Connect external signals to internal slots
        # Connect class_spinbox_value_changed signal (emitted by colorspinbox when its value changes) to update current_class_value
        self.canvas_widget.class_spinbox.color_spinbox_value_changed_signal.connect(
            self.update_class_value
        )
        # Connect artist_changed_signal (emitted by canvas widget when the current artist changes) to update active_artist
        self.canvas_widget.artist_changed_signal.connect(
            self.update_active_artist
        )

    @property
    def class_value(self) -> int:
        """Gets or sets the current class value.

        Returns
        -------
        int
            The current class value.
        """
        return self._class_value

    @class_value.setter
    def class_value(self, value: int):
        """Sets the current class value."""
        self._class_value = value

    @property
    def active_artist(self) -> Union[Scatter, Histogram2D]:
        """Gets or sets the active artist.

        Returns
        -------
        Union[Scatter, Histogram2D]
            The active artist.
        """
        return self._active_artist

    @active_artist.setter
    def active_artist(self, value):
        """Sets the active artist."""
        self._active_artist = value

    @property
    def selected_indices(self) -> np.ndarray:
        """Gets or sets the indices of the selected points.

        Returns
        -------
        np.ndarray
            The indices of the selected points.
        """
        return self._selected_indices

    @selected_indices.setter
    def selected_indices(self, value: np.ndarray):
        """Sets the indices of the selected points."""
        self._selected_indices = value

    def create_selector(self):
        """Creates the selector and connects the button press event."""
        super().create_selector()
        self.canvas_widget.canvas.mpl_connect(
            "button_press_event", self.on_button_press
        )

    def remove(self):
        """Removes the selector from the canvas and disconnects the button press event."""
        super().remove()
        self.canvas_widget.canvas.mpl_disconnect(
            self.canvas_widget.canvas.mpl_connect(
                "button_press_event", self.on_button_press
            )
        )

    def apply_selection(self):
        """Applies the selection to the data, updating the colors."""
        if self._selected_indices is None or len(self._selected_indices) == 0:
            self._selected_indices = None
            return

        # Ensure the overlay_colormap of the active artist is set to cat10_mod_cmap if needed
        if not self._active_artist.overlay_colormap.cmap.name.startswith(
            "cat10"
        ):
            # Clear previous color indices to remove previous feature coloring
            self._active_artist.color_indices = 0
            if isinstance(self._active_artist, Scatter):
                self._active_artist.overlay_colormap = cat10_mod_cmap
            elif isinstance(self._active_artist, Histogram2D):
                self._active_artist.overlay_colormap = (
                    cat10_mod_cmap_first_transparent
                )

        # Update color indices for the selected indices
        color_indices = self._active_artist.color_indices
        color_indices[self._selected_indices] = self._class_value
        self._active_artist.color_indices = color_indices

        # Emit signal and reset selected indices
        self.selection_applied_signal.emit(color_indices)
        self._selected_indices = None
        # Remove selector and create a new one
        self.remove()
        self.create_selector()

    def on_button_press(self, event):
        """Handles the button press event. Right-click applies the selection.

        Parameters
        ----------
        event : MouseEvent
            The button press event."""
        if event.button == 3:
            self.apply_selection()

    def update_class_value(self, value: int):
        """Update the class value.

        Notes
        -----
        This slot is connected to the **color_spinbox_value_changed_signal** emitted by the canvas widget.
        """
        self.class_value = value

    def update_data(self, value: np.ndarray):
        """Update the selector data.

        Notes
        -----
        This slot is connected to the **data_changed_signal** emitted by the active artist.
        """
        self.data = value

    def update_active_artist(self):
        """Update the active artist.

        Notes
        -----
        This slot is connected to the **artist_changed_signal** emitted by the canvas widget.
        """
        self.active_artist = self.canvas_widget.active_artist


class InteractiveRectangleSelector(Interactive, BaseRectangleSelector):
    """Interactive rectangle selector class.

    Inherits all parameters and attributes from Interactive and BaseRectangleSelector.
    To be used as an interactive rectangle selector.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget to which the selector will be applied.
    data : (N, 2) np.ndarray, optional
        The data to be selected.

    Other Parameters
    ----------------
    name : str
        The name of the selector, set to 'Interactive Rectangle Selector' by default.
    """

    def __init__(
        self,
        ax: plt.Axes,
        canvas_widget: "CanvasWidget",
        data: np.ndarray = None,
    ):
        """Initializes the interactive rectangle selector."""
        super().__init__(ax, canvas_widget, data)
        #: The name of the selector, set to 'Interactive Rectangle Selector' by default.
        self.name: str = "Interactive Rectangle Selector"

    def on_select(self, eclick, erelease):
        """Selects points within the rectangle and assigns them to selected indices.

        Parameters
        ----------
        eclick : MouseEvent
            The press event.
        erelease : MouseEvent
            The release event.
        """
        self.selected_indices = super().on_select(eclick, erelease)


class InteractiveEllipseSelector(Interactive, BaseEllipseSelector):
    """Interactive ellipse selector class.

    Inherits all parameters and attributes from Interactive and BaseEllipseSelector.
    To be used as an interactive ellipse selector.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget to which the selector will be applied.
    data : (N, 2) np.ndarray, optional
        The data to be selected.

    Other Parameters
    ----------------
    name : str
        The name of the selector, set to 'Interactive Ellipse Selector' by default.
    """

    def __init__(
        self,
        ax: plt.Axes,
        canvas_widget: "CanvasWidget",
        data: np.ndarray = None,
    ):
        """Initializes the interactive ellipse selector."""
        super().__init__(ax, canvas_widget, data)
        #: The name of the selector, set to 'Interactive Ellipse Selector' by default.
        self.name: str = "Interactive Ellipse Selector"

    def on_select(self, eclick, erelease):
        """Selects points within the ellipse and assigns them to selected indices.

        Parameters
        ----------
        eclick : MouseEvent
            The press event.
        erelease : MouseEvent
            The release event.
        """
        self.selected_indices = super().on_select(eclick, erelease)


class InteractiveLassoSelector(Interactive, BaseLassoSelector):
    """Interactive lasso selector class.

    Inherits all parameters and attributes from Interactive and BaseLassoSelector.
    To be used as an interactive lasso selector.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget to which the selector will be applied.
    data : (N, 2) np.ndarray, optional
        The data to be selected.

    Other Parameters
    ----------------
    name : str
        The name of the selector, set to 'Interactive Lasso Selector' by default.
    """

    def __init__(
        self,
        ax: plt.Axes,
        canvas_widget: "CanvasWidget",
        data: np.ndarray = None,
    ):
        """Initializes the interactive lasso selector."""
        super().__init__(ax, canvas_widget, data)
        #: The name of the selector, set to 'Interactive Lasso Selector' by default.
        self.name: str = "Interactive Lasso Selector"

    def on_select(self, vertices: np.ndarray):
        """Selects points within the lasso and assigns them the current class value, updating colors.

        Parameters
        ----------
        vertices : np.ndarray
            The vertices of the lasso.
        """
        self.selected_indices = super().on_select(vertices)
        self.apply_selection()
