import numpy as np
from abc import ABC, abstractmethod
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from qtpy.QtWidgets import QWidget
import matplotlib.pyplot as plt

class Selector(ABC):
    def __init__(self, ax: plt.Axes, data: np.ndarray):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self._data = data
        self.selector = None
    
    @abstractmethod
    def on_select(self, vertices):
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

    def enable(self):
        if self.selector:
            self.selector.set_active(True)
    
    def disable(self):
        if self.selector:
            self.selector.set_active(False)

    def remove(self):
        if self.selector:
            self.disable()
            self.selector.clear()
            self.selector.disconnect_events()
            self.selector = None

class BaseRectangleSelector(Selector):
    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Rectangle Selector'
        self.data = data

    def on_select(self, eclick, erelease):
        if self._data is None or len(self._data) == 0:
            return
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        mask = (self._data[:,0] >= min(x1, x2)) & (self._data[:,0] <= max(x1, x2)) & (self._data[:,1] >= min(y1, y2)) & (self._data[:,1] <= max(y1, y2))
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    def create_selector(self, *args, **kwargs):
        self.selector = RectangleSelector(self.ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True, drag_from_anywhere=True)

class InteractiveRectangleSelector(BaseRectangleSelector):
    def __init__(self, ax: plt.Axes, canvas_widget: QWidget, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Interactive Rectangle Selector'
        self.canvas_widget = canvas_widget
        self.canvas_widget.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.selected_indices = None  # To store indices of selected points
        # Get initial class value
        self._class_value = self.canvas_widget.class_spinbox.value
        # Get initial active artist
        self._active_artist = self.canvas_widget.get_active_artist()

        # Connect external signals to internal slots
        # Connect class_spinbox_value_changed signal (emitted by colorspinbox when its value changes) to update current_class_value
        self.canvas_widget.class_spinbox.color_spinbox_value_changed_signal.connect(self.set_class_value)
        # Connect artist_changed_signal (emitted by canvas widget when the current artist changes) to update active_artist
        self.canvas_widget.artist_changed_signal.connect(self.set_active_artist)
    
    def on_button_press(self, event):
        if event.button == 3:
            self.apply_selection()

    def apply_selection(self):
        if self.selected_indices is not None:
            # Update color indices only if 'e' is pressed
            color_indices = self._active_artist.color_indices
            color_indices[self.selected_indices] = self._class_value
            self._active_artist.color_indices = color_indices
            self.selected_indices = None
            # Remove selector and create a new one
            self.remove()
            self.create_selector()

    @property
    def class_value(self):
        return self._class_value

    def set_class_value(self, value):
        print('Got signal, updating class value to ' + str(value))
        self._class_value = value

    def set_data(self, value):
        print('Detected data update from artist, updating selector data to match')
        self._data = value

    def set_active_artist(self):
        print('Detected artist change, updating active artist')
        self._active_artist = self.canvas_widget.get_active_artist()

    def on_select(self, eclick, erelease):
        """Selects points within the ellipse and assigns them the current class value, updating colors."""
        self.selected_indices = super().on_select(eclick, erelease)


class BaseEllipseSelector(Selector):
    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Ellipse Selector'
        self.data = data
        # self.create_selector()

    def on_select(self, eclick, erelease):
        if self._data is None or len(self._data) == 0:
            return
        center = np.array([(eclick.xdata + erelease.xdata) / 2, (eclick.ydata + erelease.ydata) / 2])
        width = abs(eclick.xdata - erelease.xdata)
        height = abs(eclick.ydata - erelease.ydata)
        mask = (((self._data[:, 0] - center[0]) / (width / 2))**2 + ((self._data[:, 1] - center[1]) / (height / 2))**2) <= 1
        indices = np.where(mask)[0]
        return indices

    @property
    def data(self) -> np.ndarray:
        return self._data
    
    @data.setter
    def data(self, value: np.ndarray):
        self._data = value

    def create_selector(self, *args, **kwargs):
        self.selector = EllipseSelector(self.ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True, drag_from_anywhere=True)

class InteractiveEllipseSelector(BaseEllipseSelector):
    def __init__(self, ax: plt.Axes, canvas_widget: QWidget, data: np.ndarray = None):
        super().__init__(ax, data)
        
        self.name = 'Interactive Ellipse Selector'
        self.canvas_widget = canvas_widget
        self.canvas_widget.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.selected_indices = None  # To store indices of selected points
        # Get initial class value
        self._class_value = self.canvas_widget.class_spinbox.value
        # Get initial active artist
        self._active_artist = self.canvas_widget.get_active_artist()

        # Connect external signals to internal slots
        # Connect class_spinbox_value_changed signal (emitted by colorspinbox when its value changes) to update current_class_value
        self.canvas_widget.class_spinbox.color_spinbox_value_changed_signal.connect(self.set_class_value)
        # Connect artist_changed_signal (emitted by canvas widget when the current artist changes) to update active_artist
        self.canvas_widget.artist_changed_signal.connect(self.set_active_artist)

    def on_button_press(self, event):
        if event.button == 3:
            self.apply_selection()

    def apply_selection(self):
        if self.selected_indices is not None:
            # Update color indices only if 'e' is pressed
            color_indices = self._active_artist.color_indices
            color_indices[self.selected_indices] = self._class_value
            self._active_artist.color_indices = color_indices
            self.selected_indices = None
            # Remove selector and create a new one
            self.remove()
            self.create_selector()
        
    @property
    def class_value(self):
        return self._class_value
    
    def set_class_value(self, value):
        print('Got signal, updating class value to ' + str(value))
        self._class_value = value

    def set_data(self, value):
        print('Detected data update from artist, updating selector data to match')
        self._data = value
   
    def set_active_artist(self):
        print('Detected artist change, updating active artist')
        self._active_artist = self.canvas_widget.get_active_artist()

    def on_select(self, eclick, erelease):
        """Selects points within the ellipse and assigns them the current class value, updating colors."""
        self.selected_indices = super().on_select(eclick, erelease)

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

    def create_selector(self, *args, **kwargs):
        self.selector = LassoSelector(self.ax, self.on_select, useblit=True, button=[1], props={'color': 'r', 'linestyle': '--'})

class InteractiveLassoSelector(BaseLassoSelector):
    def __init__(self, ax: plt.Axes, canvas_widget: QWidget, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Interactive Lasso Selector'
        self.canvas_widget = canvas_widget
        self.selected_indices = None  # To store indices of selected points
        # Get initial class value
        self._class_value = self.canvas_widget.class_spinbox.value
        # Get initial active artist
        self._active_artist = self.canvas_widget.get_active_artist()

        # Connect external signals to internal slots
        # Connect class_spinbox_value_changed signal (emitted by colorspinbox when its value changes) to update current_class_value
        self.canvas_widget.class_spinbox.color_spinbox_value_changed_signal.connect(self.set_class_value)
        # Connect artist_changed_signal (emitted by canvas widget when the current artist changes) to update active_artist
        self.canvas_widget.artist_changed_signal.connect(self.set_active_artist)

    def apply_selection(self):
        if self.selected_indices is not None:
            # Update color indices only if 'e' is pressed
            color_indices = self._active_artist.color_indices
            color_indices[self.selected_indices] = self._class_value
            self._active_artist.color_indices = color_indices

    @property
    def class_value(self):
        return self._class_value

    def set_class_value(self, value):
        print('Got signal, updating class value to ' + str(value))
        self._class_value = value

    def set_data(self, value):
        print('Detected data update from artist, updating selector data to match')
        self._data = value

    def set_active_artist(self):
        print('Detected artist change, updating active artist')
        self._active_artist = self.canvas_widget.get_active_artist()

    def on_select(self, vertices):
        """Selects points within the lasso and assigns them the current class value, updating colors."""
        self.selected_indices = super().on_select(vertices)
        self.apply_selection()
