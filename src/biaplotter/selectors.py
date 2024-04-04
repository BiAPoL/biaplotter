import numpy as np
from abc import ABC
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector, RectangleSelector, EllipseSelector

from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class Selector(ABC):
    def __init__(self, ax, data):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.data = data
        self.selector = None
    
    @abstractmethod
    def on_select(self, vertices):
        pass
    
    def enable(self):
        if self.selector:
            self.selector.set_active(True)
    
    def disable(self):
        if self.selector:
            self.selector.set_active(False)

from matplotlib.widgets import RectangleSelector, EllipseSelector

class InteractiveRectangleSelector(Selector):
    def __init__(self, ax, data):
        super().__init__(ax, data)
        self.selector = RectangleSelector(ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    
    def on_select(self, eclick, erelease):
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        mask = (self.data[:,0] >= min(x1, x2)) & (self.data[:,0] <= max(x1, x2)) & (self.data[:,1] >= min(y1, y2)) & (self.data[:,1] <= max(y1, y2))
        indices = np.where(mask)[0]
        print("Selected indices:", indices)

class InteractiveEllipseSelector(Selector):
    def __init__(self, ax, data):
        super().__init__(ax, data)
        self.selector = EllipseSelector(ax, self.on_select, useblit=True, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
    
    def on_select(self, eclick, erelease):
        center = np.array([(eclick.xdata + erelease.xdata) / 2, (eclick.ydata + erelease.ydata) / 2])
        width = abs(eclick.xdata - erelease.xdata)
        height = abs(eclick.ydata - erelease.ydata)
        mask = (((self.data[:, 0] - center[0]) / (width / 2))**2 + ((self.data[:, 1] - center[1]) / (height / 2))**2) <= 1
        indices = np.where(mask)[0]
        print("Selected indices:", indices)

# TODO: Implement MultiRectangleSelector
class MultiRectangleSelector:
    def __init__(self, ax, data):
        self.ax = ax
        self.canvas = ax.figure.canvas
        self.data = data
        self.active_rectangle = InteractiveRectangleSelector(self.ax, self.data)
        self.rectangles = [self.active_rectangle]
        self.canvas.mpl_connect('button_press_event', self.on_click)
    
    def add_rectangle_selector(self):
        self.disable_all()
        rect = InteractiveRectangleSelector(self.ax, self.data)
        self.rectangles.append(rect)
        self.active_rectangle = rect

    def disable_all(self):
        for rect in self.rectangles:
            rect.selector.set_active(False)
        self.active_rectangle = None

    def on_click(self, event):
        # check if mouse is on existing rectangle
        for rect in self.rectangles:
            if rect.selector.extents is not None:
                if rect.selector.extents[0] < event.xdata < rect.selector.extents[1] and rect.selector.extents[2] < event.ydata < rect.selector.extents[3]:
                    self.active_rectangle = rect
                    print('click point coords are' + str(event.xdata) + ' ' + str(event.ydata))
                    print('rectangle extents are' + str(rect.selector.extents))
                    break
        if self.active_rectangle is None:
            print('Adding new rectangle selector')
            self.add_rectangle_selector()
        else:
            print('Using existing rectangle selector')
            self.disable_all()
            self.active_rectangle = rect
            self.active_rectangle.enable()


