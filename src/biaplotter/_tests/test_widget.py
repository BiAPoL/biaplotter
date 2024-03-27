import numpy as np

from biaplotter.plotter import (
    PlotterWidget
)


# capsys is a pytest fixture that captures stdout and stderr output streams
def test_example_q_widget(make_napari_viewer, capsys):
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()
    # viewer.add_image(np.random.random((100, 100)))

    # create our widget, passing in the viewer
    plotter = PlotterWidget(viewer)

    # Check button callback function
    plotter.on_enable_selector(True)

    # read captured output and check that it's as we expected
    captured = capsys.readouterr()
    assert captured.out == "Selector enabled\n"
