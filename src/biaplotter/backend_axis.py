from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt

class AbstractAxis(ABC):
    @abstractmethod
    def scatter(self, x, y, **kwargs):
        pass

    @abstractmethod
    def set_xlim(self, left, right):
        pass

    @abstractmethod
    def set_ylim(self, bottom, top):
        pass

    @abstractmethod
    def set_xlabel(self, label):
        pass

    @abstractmethod
    def set_ylabel(self, label):
        pass

    @abstractmethod
    def show(self):
        pass

class MatplotlibAxis(AbstractAxis):
    def __init__(self, ax=None):
        self.ax = ax if ax is not None else plt.gca()

    def plot(self, x, y, **kwargs):
        return self.ax.plot(x, y, **kwargs)[0]

    def set_xlim(self, left, right):
        self.ax.set_xlim(left, right)

    def set_ylim(self, bottom, top):
        self.ax.set_ylim(bottom, top)

    def set_xlabel(self, label):
        self.ax.set_xlabel(label)

    def get_xlabel(self):
        return self.ax.get_xlabel()

    def set_ylabel(self, label):
        self.ax.set_ylabel(label)

    def get_ylabel(self):
        return self.ax.get_ylabel()

    def draw(self):
        self.ax.figure.canvas.draw_idle()

    def show(self):
        plt.show()

    def scatter(self, x, y, **kwargs):
        return self.ax.scatter(x, y, **kwargs)