import matplotlib.pyplot as plt
import numpy as np
import pytest

from biaplotter.plotter import CanvasWidget
from biaplotter.selectors import (
    BaseEllipseSelector, BaseLassoSelector, BaseRectangleSelector,
    InteractiveEllipseSelector, InteractiveLassoSelector, InteractiveRectangleSelector,
    InteractiveClickSelector, BaseClickSelector
)

# --- Parametrized tests for BaseClickSelector threshold functionality ---
@pytest.mark.parametrize(
    "data, click, threshold, expected_indices",
    [
        # Click near [1, 1], within threshold
        (np.array([[0, 0], [1, 1], [2, 2]]), (1.025, 1.025), 0.2, [1]),
        # Click far from all points, outside threshold
        (np.array([[0, 0], [1, 1], [2, 2]]), (10, 10), 0.5, []),
        # Click exactly on [2, 2], within threshold
        (np.array([[0, 0], [1, 1], [2, 2]]), (2, 2), 0.1, [2]),
        # Click between [0,0] and [1,1], closer to [0,0]
        (np.array([[0, 0], [1, 1], [2, 2]]), (0.1, 0.1), 0.5, [0]),
        # Click between [0,0] and [1,1], but threshold too small
        (np.array([[0, 0], [1, 1], [2, 2]]), (0.1, 0.1), 0.05, []),
    ]
)
def test_base_click_selector_threshold(data, click, threshold, expected_indices):
    fig, ax = plt.subplots()
    ax.scatter(data[:, 0], data[:, 1])
    selector = BaseClickSelector(ax)
    selector.data = data
    event = type('Event', (), {'xdata': click[0], 'ydata': click[1]})()
    idx = selector.on_select(event, threshold=threshold)
    if len(expected_indices) == 0:
        assert len(idx) == 0, f"Expected empty array, got {idx}"
    else:
        assert np.array_equal(idx, expected_indices), f"Expected {expected_indices}, got {idx}"


class MockMouseEvent:
    """A mock event class to simulate MouseEvent with necessary attributes."""

    def __init__(self, xdata, ydata):
        self.xdata = xdata
        self.ydata = ydata


@pytest.fixture
def setup_selector(request):
    """Setup for base selector tests."""
    fig, ax = plt.subplots()
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    selector_class = request.param
    selector = selector_class(ax)
    selector.data = data
    selector.create_selector()
    return selector


@pytest.mark.parametrize(
    "setup_selector, eclick_coords, erelease_coords, expected_indices",
    [
        # Test case for BaseEllipseSelector
        (BaseEllipseSelector, (1, 1), (5, 5), [2, 3, 4]),
        # Empty region test case for BaseEllipseSelector
        (BaseEllipseSelector, (0, 3), (1, 4), []),
        # Test case for BaseRectangleSelector
        (BaseRectangleSelector, (1.5, 1), (4.5, 5), [2, 3, 4]),
        # Empty region test case for BaseRectangleSelector
        (BaseRectangleSelector, (0, 3), (1, 4), []),
    ],
    indirect=["setup_selector"],
)
def test_base_ellipse_and_rectangle_selectors(
    setup_selector, eclick_coords, erelease_coords, expected_indices
):
    """Test BaseEllipseSelector and BaseRectangleSelector."""
    selector = setup_selector
    eclick = MockMouseEvent(*eclick_coords)
    erelease = MockMouseEvent(*erelease_coords)
    actual_indices = selector.on_select(eclick, erelease)
    actual_indices.sort()

    assert np.array_equal(
        actual_indices, expected_indices
    ), "Indices of selected points {} do not match expected values {}.".format(
        actual_indices, expected_indices
    )


@pytest.mark.parametrize(
    "setup_selector, vertices, expected_indices",
    [
        (
            BaseLassoSelector,
            [(1.5, 1), (1.5, 5), (4.5, 5), (4.5, 1)],
            [2, 3, 4],
        ),  # Test case for BaseLassoSelector
        # Empty region test case for BaseLassoSelector
        (BaseLassoSelector, [(0, 3), (0, 4), (1, 4)], []),
    ],
    indirect=["setup_selector"],
)
def test_base_lasso_selector(setup_selector, vertices, expected_indices):
    """Test BaseLassoSelector."""
    selector = setup_selector
    actual_indices = selector.on_select(vertices)
    actual_indices.sort()

    assert np.array_equal(
        actual_indices, expected_indices
    ), "Indices of selected points {} do not match expected values {}.".format(
        actual_indices, expected_indices
    )


@pytest.fixture
def selector_class(request):
    """Fixture to dynamically handle the selector class type."""
    return request.param


@pytest.mark.parametrize(
    "selector_class, expected_color_indices",
    [
        # Test case for InteractiveRectangleSelector
        (InteractiveRectangleSelector, [0, 0, 1, 1, 1, 0]),
        # Test case for InteractiveEllipseSelector
        (InteractiveEllipseSelector, [0, 0, 1, 1, 1, 0]),
        # Test case for InteractiveLassoSelector
        (InteractiveLassoSelector, [0, 0, 1, 1, 1, 0]),
        # Test case for InteractiveClickSelector
        (InteractiveClickSelector, [0, 0, 1, 1, 1, 0]),
    ],
    indirect=["selector_class"],
)
def test_interactive_selectors(
    make_napari_viewer, selector_class, expected_color_indices
):
    """Test InteractiveRectangleSelector, InteractiveEllipseSelector, InteractiveLassoSelector, and InteractiveClickSelector."""
    viewer = make_napari_viewer()
    widget = CanvasWidget(viewer)
    selector = selector_class(widget.axes, widget)
    assert selector is not None

    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    # Set artist data to initialize color indices
    artist = widget.active_artist
    artist.data = data
    selector.data = data
    selector.create_selector()
    assert selector._selector is not None

    selector.class_value = 1
    selector.selected_indices = [2, 3, 4]
    # artist color indices are updated based on selected indices
    selector.apply_selection()
    assert np.array_equal(
        artist.color_indices, expected_color_indices
    ), "Color indices {} do not match expected values {}.".format(
        selector.color_indices, expected_color_indices
    )


if __name__ == "__main__":
    pytest.main([__file__])