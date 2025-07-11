import pytest
from matplotlib.widgets import (EllipseSelector, LassoSelector,
                                RectangleSelector)

from biaplotter.artists import Histogram2D, Scatter
from biaplotter.plotter import CanvasWidget
from biaplotter.selectors import (InteractiveEllipseSelector,
                                  InteractiveLassoSelector,
                                  InteractiveRectangleSelector)


@pytest.fixture
def canvas_widget(make_napari_viewer):
    """Fixture to create a CanvasWidget instance for testing."""
    # make viewer and add an image layer using our fixture
    viewer = make_napari_viewer()

    # create our widget, passing in the viewer
    widget = CanvasWidget(viewer)
    return widget


def test_initialization(canvas_widget):
    """Test that the CanvasWidget initializes correctly."""
    assert canvas_widget.artists
    assert canvas_widget.selectors
    assert canvas_widget.active_artist == canvas_widget.artists["HISTOGRAM2D"]
    assert canvas_widget.active_selector is None


def test_add_and_remove_artist(canvas_widget):
    """Test adding and removing artists."""
    scatter = Scatter(ax=canvas_widget.axes)
    canvas_widget.add_artist("NEW_SCATTER", scatter)
    assert "NEW_SCATTER" in canvas_widget.artists
    assert canvas_widget.artists["NEW_SCATTER"] == scatter

    canvas_widget.remove_artist("NEW_SCATTER")
    assert "NEW_SCATTER" not in canvas_widget.artists


def test_set_active_artist(canvas_widget):
    """Test setting the active artist."""
    canvas_widget.active_artist = "SCATTER"
    assert canvas_widget.active_artist == canvas_widget.artists["SCATTER"]
    assert canvas_widget.artists["SCATTER"].visible
    assert not canvas_widget.artists["HISTOGRAM2D"].visible


def test_add_and_remove_selector(canvas_widget):
    """Test adding and removing selectors."""
    lasso = InteractiveEllipseSelector(
        ax=canvas_widget.axes, canvas_widget=canvas_widget
    )
    canvas_widget.add_selector("NEW_LASSO", lasso)
    assert "NEW_LASSO" in canvas_widget.selectors
    assert canvas_widget.selectors["NEW_LASSO"] == lasso

    canvas_widget.remove_selector("NEW_LASSO")
    assert "NEW_LASSO" not in canvas_widget.selectors


def test_set_active_selector(canvas_widget):
    """Test setting the active selector."""
    canvas_widget.active_selector = "LASSO"
    assert canvas_widget.active_selector == canvas_widget.selectors["LASSO"]
    assert isinstance(
        canvas_widget.selectors["LASSO"]._selector, LassoSelector
    )

    canvas_widget.active_selector = "ELLIPSE"
    assert canvas_widget.active_selector == canvas_widget.selectors["ELLIPSE"]
    assert not isinstance(
        canvas_widget.selectors["LASSO"]._selector, LassoSelector
    )


def test_deactivate_and_remove_all_selectors(canvas_widget):
    """Test disabling all selectors."""
    canvas_widget.active_selector = "LASSO"
    canvas_widget._deactivate_and_remove_all_selectors()
    assert canvas_widget.active_selector is None
    for selector in canvas_widget.selectors.values():
        assert selector._selector is None


def test_show_color_overlay(canvas_widget):
    """Test the show_color_overlay method."""
    canvas_widget.show_color_overlay = False
    assert not canvas_widget.active_artist.overlay_visible

    canvas_widget.show_color_overlay = True
    assert canvas_widget.active_artist.overlay_visible


def test_signals(canvas_widget, qtbot):
    """Test that signals are emitted correctly."""
    with qtbot.waitSignal(
        canvas_widget.artist_changed_signal, timeout=100
    ) as signal:
        canvas_widget.active_artist = "SCATTER"
    assert signal.args == ["SCATTER"]

    with qtbot.waitSignal(
        canvas_widget.selector_changed_signal, timeout=100
    ) as signal:
        canvas_widget.active_selector = "LASSO"
    assert signal.args == ["LASSO"]

    with qtbot.waitSignal(
        canvas_widget.selector_changed_signal, timeout=100
    ) as signal:
        canvas_widget._deactivate_and_remove_all_selectors()
    assert signal.args == [""]

    with qtbot.waitSignal(
        canvas_widget.show_color_overlay_signal, timeout=100
    ) as signal:
        canvas_widget.show_color_overlay = True
    assert signal.args == [True]
