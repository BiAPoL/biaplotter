__version__ = "0.0.1"
from .plotter import CanvasWidget
from .artists import Scatter, Histogram2D
from .selectors import InteractiveRectangleSelector, InteractiveEllipseSelector, InteractiveLassoSelector

__all__ = (
    "CanvasWidget",
)
