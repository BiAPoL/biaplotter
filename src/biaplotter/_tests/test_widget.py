import numpy as np

from biaplotter.plotter import (
    CanvasWidget
)


def test_widget(make_napari_viewer):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    widget = CanvasWidget(viewer)

    viewer.window.add_dock_widget(widget)

    # check that the widget exists
    assert widget is not None
