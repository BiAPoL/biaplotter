__version__ = "0.4.1"
from .artists import Histogram2D, Scatter
from .colormap import BiaColormap
from .plotter import CanvasWidget
from .selectors import (InteractiveEllipseSelector, InteractiveLassoSelector,
                        InteractiveRectangleSelector)

__all__ = (
    "CanvasWidget",
    "Scatter",
    "Histogram2D",
    "InteractiveRectangleSelector",
    "InteractiveEllipseSelector",
    "InteractiveLassoSelector",
)
