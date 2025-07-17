try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

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
