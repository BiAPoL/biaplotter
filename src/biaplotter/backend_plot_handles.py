from abc import ABC, abstractmethod
import numpy as np
import pyqtgraph as pg

class AbstractPlotArtist(ABC):
    @abstractmethod
    def remove(self):
        pass

    @abstractmethod
    def set_data(self, *args):
        pass

    @abstractmethod
    def set_alpha(self, alpha):
        pass

    @abstractmethod
    def set_visible(self, visible):
        pass

class AbstractScatterArtist(AbstractPlotArtist):
    @abstractmethod
    def set_sizes(self, sizes):
        pass

    @abstractmethod
    def set_edgecolor(self, color):
        pass


class MatplotlibScatterArtist(AbstractScatterArtist):
    def __init__(self, scatter_plot):
        self.scatter_plot = scatter_plot

    def remove(self):
        self.scatter_plot.remove()

    def set_data(self, offsets):
        self.scatter_plot.set_offsets(offsets)

    def set_facecolor(self, rgba):
        self.scatter_plot.set_facecolor(rgba)

    def set_alpha(self, alpha):
        self.scatter_plot.set_alpha(alpha)

    def get_alpha(self):
        return self.scatter_plot.get_alpha()

    def set_sizes(self, sizes):
        self.scatter_plot.set_sizes(sizes)

    def set_edgecolor(self, color):
        self.scatter_plot.set_edgecolor(color)

    def set_linewidth(self, linewidth):
        self.scatter_plot.set_linewidth(linewidth)

    def set_visible(self, visible):
        self.scatter_plot.set_visible(visible)


class PyQtGraphScatterArtist:
    def __init__(self, scatter_plot):
        self.scatter_plot = scatter_plot
        self.symbol_size = 5  # Default size

    def remove(self):
        # In PyQtGraph, we clear the plot item instead of removing it
        self.scatter_plot.clear()

    def set_data(self, offsets):
        """Set the data for the scatter plot.

        Parameters
        ----------
        offsets : array-like, shape (N, 2)
            The x and y coordinates of the scatter points.
        """
        x, y = offsets[:, 0], offsets[:, 1]
        self.scatter_plot.setData(x=x, y=y)

    def set_facecolor(self, rgba):
        """Set the face color (fill color) of the scatter plot points.

        Parameters
        ----------
        rgba : tuple or list
            Color of the points in RGBA format.
        """
        color = pg.mkColor(rgba)
        self.scatter_plot.setBrush(pg.mkBrush(color))

    def set_alpha(self, alpha):
        """Set the transparency (alpha) of the scatter plot points.

        Parameters
        ----------
        alpha : float
            Transparency level, between 0 (fully transparent) and 1 (fully opaque).
        """
        brush = self.scatter_plot.opts['brush']
        if brush is not None:
            color = brush.color()
            color.setAlphaF(alpha)
            self.scatter_plot.setBrush(pg.mkBrush(color))

    def get_alpha(self):
        """Get the current alpha (transparency) level of the points.

        Returns
        -------
        float
            The current alpha level.
        """
        brush = self.scatter_plot.opts['brush']
        if brush is not None:
            return brush.color().alphaF()
        return 1.0  # Default alpha if not set

    def set_sizes(self, sizes):
        """Set the size of the scatter plot points.

        Parameters
        ----------
        sizes : int or array-like
            The size (or sizes) of the points. If it's a scalar, all points will have the same size.
        """
        if np.isscalar(sizes):
            self.symbol_size = sizes
        else:
            # For simplicity, use the first size if sizes is array-like
            self.symbol_size = sizes[0] if len(sizes) > 0 else 5
        self.scatter_plot.setData(symbolSize=self.symbol_size)

    def set_edgecolor(self, color):
        """Set the edge color of the scatter plot points.

        Parameters
        ----------
        color : tuple or string
            The color of the edges of the points.
        """
        pen = pg.mkPen(color=color)
        self.scatter_plot.setPen(pen)

    def set_linewidth(self, linewidth):
        """Set the linewidth of the edges of the scatter plot points.

        Parameters
        ----------
        linewidth : float
            The width of the edges of the points.
        """
        pen = self.scatter_plot.opts['pen']
        if pen is not None:
            pen.setWidth(linewidth)
            self.scatter_plot.setPen(pen)

    def set_visible(self, visible):
        """Set the visibility of the scatter plot points.

        Parameters
        ----------
        visible : bool
            Whether the points are visible.
        """
        self.scatter_plot.setVisible(visible)
