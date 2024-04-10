from __future__ import annotations  # Only necessary for Python 3.7 to 3.9

import numpy as np
import matplotlib.pyplot as plt

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector


if TYPE_CHECKING:
    from biaplotter.plotter import CanvasWidget


class Selector(ABC):
    """Abstract class for creating a selector.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    data : np.ndarray
        The data to be selected.

    Attributes
    ----------
    _ax : plt.Axes
        Stroes the axes to which the selector will be applied.
    _data : np.ndarray
        Stores the data to be selected.
    _selector : Any
        Stores the selector object.

    Properties
    ----------
    data : np.ndarray
        Gets or sets the data to be selected.

    Methods
    -------
    on_select(vertices: np.ndarray)
        Abstract method to select points based on the selector's shape.
    create_selector()
        Abstract method to create a selector.
    remove()
        Removes the selector from the canvas.
    """

    def __init__(self, ax: plt.Axes, data: np.ndarray):
        """Initializes the selector.

        Parameters
        ----------
        ax : plt.Axes
            The axes to which the selector will be applied.
        data : np.ndarray
            The data to be selected.
        """
        self._ax = ax
        self._data = data
        self._selector = None

    @abstractmethod
    def on_select(self, vertices: np.ndarray):
        """Abstract method to select points based on the selector's shape."""
        pass

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

    def remove(self):
        """Removes the selector from the canvas."""
        if self._selector:
            self._selector.clear()
            self._selector.disconnect_events()
            self._selector = None


class BaseRectangleSelector(Selector):
    """Base class for creating a rectangle selector.	

    Inherits all parameters and attributes from Selector.
    For parameter and attribute details, see the Selector class documentation.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    data : (N, 2) np.ndarray
        The data to be selected.

    Additonal Attributes
    --------------------
    name : str
        The name of the selector.

    Properties
    ----------
    data : (N, 2) np.ndarray
        Gets or sets the data to be selected.

    Methods
    -------
    on_select(eclick, erelease)
        Selects points within the rectangle and returns their indices.
    create_selector()
        Creates a rectangle selector.    
    """

    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        """Initializes the rectangle selector.

        Parameters
        ----------
        ax : plt.Axes
            The axes to which the selector will be applied.
        data : (N, 2) np.ndarray
            The data to be selected.
        """
        super().__init__(ax, data)
        self.name = 'Rectangle Selector'
        self.data = data

    def on_select(self, eclick, erelease) -> np.ndarray:
        """Selects points within the rectangle and returns their indices."""
        if self._data is None or len(self._data) == 0:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        mask = (self._data[:, 0] >= min(x1, x2)) & (self._data[:, 0] <= max(x1, x2)) & (
            self._data[:, 1] >= min(y1, y2)) & (self._data[:, 1] <= max(y1, y2))
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        """Gets the data from which points will be selected."""
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        """Sets the data from which points will be selected."""
        self._data = value

    def create_selector(self, *args, **kwargs):
        """Creates a rectangle selector.

        Useblit is set to True to improve performance.
        Left mouse button is used to draw the rectangle.
        Minimum span in x and y is set to 5 pixels.
        Span coordinates are set to pixels.
        Interactive is set to True to allow for interaction.
        Drag from anywhere is set to True to allow for drawing from any point.
        """
        self._selector = RectangleSelector(self._ax, self.on_select, useblit=True, button=[
                                           1], minspanx=5, minspany=5, spancoords='pixels', interactive=True, drag_from_anywhere=True)


class BaseEllipseSelector(Selector):
    """Base class for creating an ellipse selector.

    Inherits all parameters and attributes from Selector.
    For parameter and attribute details, see the Selector class documentation.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    data : np.ndarray
        The data to be selected.

    Additional Attributes
    --------------------
    name : str
        The name of the selector.

    Properties
    ----------
    data : np.ndarray
        Gets or sets the data to be selected.

    Methods
    -------
    on_select(eclick, erelease)
        Selects points within the ellipse and returns their indices.
    create_selector()
        Creates an ellipse selector.
    """

    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        """Initializes the ellipse selector.

        Parameters
        ----------
        ax : plt.Axes
            The axes to which the selector will be applied.
        data : np.ndarray
            The data to be selected.
        """
        super().__init__(ax, data)
        self.name = 'Ellipse Selector'
        self.data = data

    def on_select(self, eclick, erelease):
        """Selects points within the ellipse and returns their indices."""
        if self._data is None or len(self._data) == 0:
            return
        center = np.array([(eclick.xdata + erelease.xdata) / 2,
                          (eclick.ydata + erelease.ydata) / 2])
        width = abs(eclick.xdata - erelease.xdata)
        height = abs(eclick.ydata - erelease.ydata)
        mask = (((self._data[:, 0] - center[0]) / (width / 2))**2 +
                ((self._data[:, 1] - center[1]) / (height / 2))**2) <= 1
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        """Gets the data from which points will be selected."""
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
        self._selector = EllipseSelector(self._ax, self.on_select, useblit=True, button=[
            1], minspanx=5, minspany=5, spancoords='pixels', interactive=True, drag_from_anywhere=True)


class BaseLassoSelector(Selector):
    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Lasso Selector'
        self.data = data

    def on_select(self, vertices):
        if self._data is None or len(self._data) == 0:
            return
        path = mplPath(vertices)
        mask = path.contains_points(self._data)
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    def create_selector(self):
        self._selector = LassoSelector(self._ax, self.on_select, useblit=True, button=[
            1], props={'color': 'r', 'linestyle': '--'})


class Interactive(Selector):
    """Interactive selector class.

    Inherits all parameters and attributes from Selector.
    To be used as a base class for interactive selectors.

    Parameters
    ----------
    ax : plt.Axes
        The axes to which the selector will be applied.
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget to which the selector will be applied.
    data : (N, 2) np.ndarray
        The data to be selected.

    Additional Attributes
    ----------
    canvas_widget : biaplotter.plotter.CanvasWidget
        The canvas widget to which the selector is applied.
    _selected_indices : (N,) np.ndarray
        Stores the indices of the selected points.
    _class_value : int
        Stores the current class value.
    _active_artist : biaplotter.artists.Artist
        Stores the active artist.

    Properties
    ----------
    class_value : int
        Gets the current class value.
    selected_indices : (N,) np.ndarray
        Gets or sets the indices of the selected points. 

    Methods
    -------
    on_button_press(event)
        Handles the button press event. Right-click applies the selection.
    apply_selection()
        Applies the selection to the data, updating the colors.
    create_selector()
        Creates the rectangle selector and connects the button press event.
    remove()
        Removes the selector from the canvas and disconnects the button press event.

    Slots
    -----
    update_class_value(value: int)
        Connects to the color_spinbox_value_changed_signal from the canvas widget to update the class value.
    update_data(value: np.ndarray)
        Connects to the data_changed_signal from the active artist to update the selector data. Not connected in this class.
    update_active_artist()
        Connects to the artist_changed_signal from the canvas widget to update the active artist.
        """

    def __init__(self, ax: plt.Axes, canvas_widget: "CanvasWidget", data: np.ndarray = None):
        """Initializes the interactive rectangle selector.

        Parameters
        ----------
        ax : plt.Axes
            The axes to which the selector will be applied.
        canvas_widget : biaplotter.plotter.CanvasWidget
            The canvas widget to which the selector will be applied.
        data : (N, 2) np.ndarray
            The data to be selected.
        """
        super().__init__(ax, data)
        self.canvas_widget = canvas_widget

        self._selected_indices = None  # To store indices of selected points
        # Get initial class value
        self._class_value = self.canvas_widget.class_spinbox.value
        # Get initial active artist
        self._active_artist = self.canvas_widget.get_active_artist()

        # Connect external signals to internal slots
        # Connect class_spinbox_value_changed signal (emitted by colorspinbox when its value changes) to update current_class_value
        self.canvas_widget.class_spinbox.color_spinbox_value_changed_signal.connect(
            self.update_class_value)
        # Connect artist_changed_signal (emitted by canvas widget when the current artist changes) to update active_artist
        self.canvas_widget.artist_changed_signal.connect(
            self.update_active_artist)

    @property
    def class_value(self):
        """Gets the current class value."""
        return self._class_value

    @class_value.setter
    def class_value(self, value: int):
        """Sets the current class value."""
        self._class_value = value

    @property
    def selected_indices(self):
        """Gets the indices of the selected points."""
        return self._selected_indices

    @selected_indices.setter
    def selected_indices(self, value: np.ndarray):
        """Sets the indices of the selected points."""
        if value is None:
            return
        self._selected_indices = value

    def on_button_press(self, event):
        """Handles the button press event. Right-click applies the selection."""
        if event.button == 3:
            self.apply_selection()

    def apply_selection(self):
        """Applies the selection to the data, updating the colors."""
        if self._selected_indices is not None:
            if len(self._selected_indices) > 0:
                color_indices = self._active_artist.color_indices
                color_indices[self._selected_indices] = self._class_value
                self._active_artist.color_indices = color_indices
            self._selected_indices = None
        # Remove selector and create a new one
        self.remove()
        self.create_selector()

    def create_selector(self):
        """Creates the rectangle selector and connects the button press event."""
        super().create_selector()
        self.canvas_widget.canvas.mpl_connect(
            'button_press_event', self.on_button_press)

    def remove(self):
        """Removes the selector from the canvas and disconnects the button press event."""
        super().remove()
        self.canvas_widget.canvas.mpl_disconnect(self.canvas_widget.canvas.mpl_connect(
            'button_press_event', self.on_button_press))

    def update_class_value(self, value: int):
        """Handles the color_spinbox_value_changed_signal from the canvas widget to update the class value."""
        self.class_value = value

    def update_data(self, value: np.ndarray):
        """Handles the data_changed_signal from the active artist to update the selector data."""
        self.data = value

    def update_active_artist(self):
        """Handles the artist_changed_signal from the canvas widget to update the active artist."""
        self._active_artist = self.canvas_widget.get_active_artist()


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
    data : (N, 2) np.ndarray
        The data to be selected.

    Additional Attributes
    --------------------
    name : str
        The name of the selector.

    Methods
    -------
    on_select(eclick, erelease)
        Selects points within the rectangle and assigns them to selected indices.
    """

    def __init__(self, ax: plt.Axes, canvas_widget: "CanvasWidget", data: np.ndarray = None):
        """Initializes the interactive rectangle selector."""
        super().__init__(ax, canvas_widget, data)
        self.name = 'Interactive Rectangle Selector'

    def on_select(self, eclick, erelease):
        """Selects points within the rectangle and assigns them to selected indices."""
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
    data : (N, 2) np.ndarray
        The data to be selected.
    """

    def __init__(self, ax: plt.Axes, canvas_widget: "CanvasWidget", data: np.ndarray = None):
        """Initializes the interactive ellipse selector."""
        super().__init__(ax, canvas_widget, data)
        self.name = 'Interactive Ellipse Selector'

    def on_select(self, eclick, erelease):
        """Selects points within the ellipse and assigns them to selected indices."""
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
    data : (N, 2) np.ndarray
        The data to be selected.
    """

    def __init__(self, ax: plt.Axes, canvas_widget: "CanvasWidget", data: np.ndarray = None):
        """Initializes the interactive lasso selector."""
        super().__init__(ax, canvas_widget, data)
        self.name = 'Interactive Lasso Selector'

    def on_select(self, vertices: np.ndarray):
        """Selects points within the lasso and assigns them the current class value, updating colors."""
        self.selected_indices = super().on_select(vertices)
        self.apply_selection()
