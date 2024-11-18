__version__ = "0.0.5alpha.2"
from .plotter import CanvasWidget
from .artists import Scatter, Histogram2D
from .selectors import InteractiveRectangleSelector, InteractiveEllipseSelector, InteractiveLassoSelector

__all__ = (
    "CanvasWidget", "Scatter", "Histogram2D", "InteractiveRectangleSelector", "InteractiveEllipseSelector", "InteractiveLassoSelector"
)
