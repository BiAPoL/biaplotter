import numpy as np
from abc import ABC
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector
from qtpy.QtWidgets import QWidget
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

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

from matplotlib.widgets import RectangleSelector, EllipseSelector

class BaseRectangleSelector(Selector):
    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Rectangle Selector'
        self.data = data
        self.selector = RectangleSelector(ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        self.disable()

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

class InteractiveRectangleSelector(BaseRectangleSelector):
    def __init__(self, ax: plt.Axes, canvas_widget: QWidget, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Interactive Rectangle Selector'
        self.canvas_widget = canvas_widget
        # Get initial class value
        self._class_value = self.canvas_widget.class_spinbox.value
        # Get initial active artist
        self._active_artist = self.canvas_widget.get_active_artist()

        # Connect external signals to internal slots
        # Connect class_spinbox_value_changed signal (emitted by colorspinbox when its value changes) to update current_class_value
        self.canvas_widget.class_spinbox.color_spinbox_value_changed_signal.connect(self.set_class_value)
        # Connect artist_changed_signal (emitted by canvas widget when the current artist changes) to update active_artist
        self.canvas_widget.artist_changed_signal.connect(self.set_active_artist)
        

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
        """Selects points within the rectangle and assigns them the current class value, updating colors."""
        selected_indices = super().on_select(eclick, erelease)
        color_indices = self._active_artist.color_indices
        color_indices[selected_indices] = self._class_value
        self._active_artist.color_indices = color_indices
        print(selected_indices)


class BaseEllipseSelector(Selector):
    def __init__(self, ax: plt.Axes, data: np.ndarray = None):
        super().__init__(ax, data)
        self.name = 'Ellipse Selector'
        self.data = data
        self.selector = EllipseSelector(ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        self.disable()

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

import napari

class InteractiveEllipseSelector(BaseEllipseSelector):
    def __init__(self, ax: plt.Axes, canvas_widget: QWidget, data: np.ndarray = None):
        super().__init__(ax, data)
        
        self.name = 'Interactive Ellipse Selector'
        self.canvas_widget = canvas_widget
        self.viewer = self.canvas_widget.viewer
        self.viewer.bind_key("d", self._func)
        self.viewer.bind_key("e", self._func2)
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

        # self.canvas_widget.canvas.mpl_connect('key_press_event', self.on_key_press)

    def _func(self, viewer):
        print('d key pressed')
        if self.selected_indices is not None:
            # Update color indices only if 'e' is pressed
            color_indices = self._active_artist.color_indices
            color_indices[self.selected_indices] = self._class_value
            self._active_artist.color_indices = color_indices
            # Reset selected indices to None
            self.selected_indices = None

            self.disable()

    def _func2(self, viewer):
        print('e key pressed')
        # TODO: control this from parent (canvas widget) to make a new selector (and not re-enable the current one)
        self.enable()
        
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
        # # TODO: Update color indices only if ENTER is pressed or something similar
        # color_indices = self._active_artist.color_indices
        # color_indices[self.selected_indices] = self._class_value
        # self._active_artist.color_indices = color_indices
        # print(self.selected_indices)

    # def on_key_press(self, event):
    #     print('press', event.key)

    # @napari.Viewer.bind_key('enter')
    # def print_key(self):
    #     print('pressed enter')

    # def on_key_press(self, event):
    #     """Handles key press events; specifically looking for the ENTER key."""
    #     if event.key == 'enter':
    #         if self.selected_indices is not None:
    #             # Update color indices only if ENTER is pressed
    #             color_indices = self._active_artist.color_indices
    #             color_indices[self.selected_indices] = self._class_value
    #             self._active_artist.color_indices = color_indices
    #             # Reset selected indices to None
    #             self.selected_indices = None

    #             self.disable()
    
        
