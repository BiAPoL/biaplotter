import numpy as np
from pathlib import Path
from enum import Enum, auto
from nap_plot_tools import CustomToolbarWidget, QtColorSpinBox, make_cat10_mod_cmap
from napari.layers import Labels, Points, Tracks
from napari_matplotlib.base import SingleAxesWidget
from napari_matplotlib.util import Interval
from qtpy.QtWidgets import QHBoxLayout, QLabel

# from biaplotter.selectors import CustomLassoSelector
from biaplotter.artists import Scatter, Histogram2D

icon_folder_path = (
    Path(__file__).parent / "icons"
)

class PlottingType(Enum):
    HISTOGRAM = auto()
    SCATTER = auto()

class CanvasWidget(SingleAxesWidget):
    # Amount of available input layers
    n_layers_input = Interval(1, None)
    # All layers that have a .features attributes
    input_layer_types = (Labels, Points, Tracks)

    def __init__(self, napari_viewer, parent=None, label_text="Class:"):
        super().__init__(napari_viewer, parent=parent)

        # Add selection tools layout below canvas
        self.selection_tools_layout = self._build_selection_toolbar_layout(label_text=label_text)

        # Add button to selection_toolbar
        self.selection_toolbar.add_custom_button(
            name="Lasso Selection",
            tooltip="Click to enable/disable Lasso selection",
            default_icon_path=icon_folder_path / "lasso.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "lasso_checked.png",
        )
        # Connect button to callback
        self.selection_toolbar.connect_button_callback(
            name="Lasso Selection", callback=self.on_enable_selector
        )

        # Set selection class colormap
        self.colormap = make_cat10_mod_cmap(first_color_transparent=False)

        # Add selection tools layout to main layout below matplotlib toolbar and above canvas
        self.layout().insertLayout(2, self.selection_tools_layout)


        # Create artists
        self.scatter_artist = Scatter(ax=self.axes, colormap=self.colormap)
        self.scatter_artist.visible = False
        self.histogram2d_artist = Histogram2D(ax=self.axes)




    def _build_selection_toolbar_layout(self, label_text="Class:"):
        # Add selection tools layout below canvas
        selection_tools_layout = QHBoxLayout()
        # Add selection toolbar
        self.selection_toolbar = CustomToolbarWidget(self)
        selection_tools_layout.addWidget(self.selection_toolbar)
        # Add class spinbox
        selection_tools_layout.addWidget(QLabel(label_text))
        self.class_spinbox = QtColorSpinBox(first_color_transparent=False)
        selection_tools_layout.addWidget(self.class_spinbox)
        # Add stretch to the right to push buttons to the left
        selection_tools_layout.addStretch(1)
        return selection_tools_layout

    def on_enable_selector(self, checked):
        if checked:
            print("Selector enabled")
        else:
            print("Selector disabled")

    def switch_plotting_type(self, plotting_type):
        if plotting_type == PlottingType.HISTOGRAM:
            self.scatter_artist.visible = False
            self.histogram2d_artist.visible = True
        elif plotting_type == PlottingType.SCATTER:
            self.scatter_artist.visible = True
            self.histogram2d_artist.visible = False
    