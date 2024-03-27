import numpy as np
from matplotlib.path import Path as mplPath
from matplotlib.widgets import LassoSelector

class CustomLassoSelector:
    def __init__(self, parent, axes):
        self.artist = parent
        self.axes = axes
        self.canvas = axes.figure.canvas

        self.lasso = LassoSelector(axes, onselect=self.onselect)
        self.ind = []
        self.ind_mask = []
        # start disabled
        self.disable()

    def enable(self):
        """Enable the Lasso selector."""
        self.lasso = LassoSelector(self.axes, onselect=self.onselect)

    def disable(self):
        """Disable the Lasso selector."""
        self.lasso.disconnect_events()

    def onselect(self, verts):
        # Get plotted data and color indices
        plotted_data = self.artist.data
        color_indices = self.artist.color_indices
        # Get indices of selected data points
        path = mplPath(verts)
        self.ind_mask = path.contains_points(plotted_data)
        self.ind = np.nonzero(self.ind_mask)[0]
        # Set selected indices with selected color index
        color_indices[self.ind] = self.artist.selected_color_index
        # TODO: Replace this by pyq signal/slot
        self.artist.color_indices = color_indices  # This updates the plot