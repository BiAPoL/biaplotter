from biaplotter.plotter import CanvasWidget, ArtistType
from typing import TYPE_CHECKING
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QScrollArea,
)
from qtpy.QtCore import Qt
from qtpy import uic
import numpy as np
import pandas as pd
import napari
from napari.layers import Labels
from skimage.util import map_array
from napari.utils import DirectLabelColormap

if TYPE_CHECKING:
    import napari

import matplotlib.pyplot as plt
def colormap_to_dict(colormap, num_colors=10, exclude_first=True):
    """
    Converts a matplotlib colormap into a dictionary of RGBA colors.

    Parameters:
        colormap (matplotlib.colors.Colormap): The colormap to convert.
        num_colors (int): The number of discrete colors to extract from the colormap.
        exclude_first (bool): Whether to exclude the first color in the colormap.

    Returns:
        dict: A dictionary with keys as positive integers and values as RGBA colors.
    """
    color_dict = {}
    start = 0
    if exclude_first:
        start = 1
    for i in range(start, num_colors + start):
        pos = i / (num_colors - 1)
        color = colormap(pos)
        color_dict[i+1-start] = color
    color_dict[None] = (0, 0, 0, 0)
    return color_dict





class PlotterWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        # Load plotter inputs widget from ui file
        self.plotter_inputs_widget = QWidget()
        uic.loadUi(
            "src/biaplotter/ui/plotter_inputs_widget.ui",
            self.plotter_inputs_widget,
        )
        self.canvas_widget = CanvasWidget(napari_viewer)

        # Create a scroll area
        self.scrollArea = QScrollArea(self)
        self.scrollArea.setMinimumWidth(300)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scrollableWidget = QWidget()
        scrollAreaLayout = QVBoxLayout()
        # Add the widgets to the layout
        scrollAreaLayout.addWidget(self.canvas_widget)
        scrollAreaLayout.addWidget(self.plotter_inputs_widget)

        # Set the layout to the scrollable widget
        scrollableWidget.setLayout(scrollAreaLayout)

        # Set the scrollable widget as the widget of scroll area
        self.scrollArea.setWidget(scrollableWidget)
        self.scrollArea.setWidgetResizable(True)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.scrollArea)
        self.setLayout(mainLayout)

        self.viewer.layers.events.inserted.connect(self.reset_layer_choices)
        self.viewer.layers.events.removed.connect(self.reset_layer_choices)

        # Populate plot type combobox
        self.plotter_inputs_widget.plot_type_combobox.addItems(
            [ArtistType.SCATTER.name, ArtistType.HISTOGRAM2D.name]
        )

        # Connect callbacks
        self.plotter_inputs_widget.plot_pushbutton.clicked.connect(self.plot)
        self.plotter_inputs_widget.labels_layer_with_phasor_features_combobox.currentIndexChanged.connect(
            self.on_labels_layer_with_phasor_features_changed)

        # Initialize attributes
        self._labels_layer_with_phasor_features = None
        self._phasors_selected_layer = None
        self._colormap = self.canvas_widget.colormap

    def reset_layer_choices(self):
        self.plotter_inputs_widget.labels_layer_with_phasor_features_combobox.clear()
        self.plotter_inputs_widget.labels_layer_with_phasor_features_combobox.addItems(
            [layer.name for layer in self.viewer.layers if isinstance(
                layer, Labels) and layer.name == "Phasor Features Layer"]
        )
        # self.on_labels_layer_with_phasor_features_changed()

    def on_labels_layer_with_phasor_features_changed(self):
        labels_layer_name = self.plotter_inputs_widget.labels_layer_with_phasor_features_combobox.currentText()
        if labels_layer_name == "":
            self._labels_layer_with_phasor_features = None
            return
        self._labels_layer_with_phasor_features = self.viewer.layers[labels_layer_name]
        self.set_valid_features_columns()

    def set_valid_features_columns(self):
        if self._labels_layer_with_phasor_features is None:
            return []
        # Initialize manual selection column if it does not exist
        if 'MANUAL_SELECTION' not in self._labels_layer_with_phasor_features.features.columns:
            self._labels_layer_with_phasor_features.features['MANUAL_SELECTION'] = np.zeros_like(
                self._labels_layer_with_phasor_features.features['label']).astype(int)
        # Populate comboboxes
        for column in self._labels_layer_with_phasor_features.features.columns:
            if 'SELECTION' in column:
                self.plotter_inputs_widget.hue_combobox.addItem(column)
            elif column != 'label' and column != 'frame':
                self.plotter_inputs_widget.x_axis_combobox.addItem(column)
                self.plotter_inputs_widget.y_axis_combobox.addItem(column)

        # Set initial comboboxes default choices
        for i in range(self.plotter_inputs_widget.x_axis_combobox.count()):
            if self.plotter_inputs_widget.x_axis_combobox.itemText(i) == 'G':
                self.x_axis = 'G'
                break
        for i in range(self.plotter_inputs_widget.y_axis_combobox.count()):
            if self.plotter_inputs_widget.y_axis_combobox.itemText(i) == 'S':
                self.y_axis = 'S'
                break
        for i in range(self.plotter_inputs_widget.hue_combobox.count()):
            if self.plotter_inputs_widget.hue_combobox.itemText(i) == 'MANUAL_SELECTION':
                self.hue = 'MANUAL_SELECTION'
                break

    def get_features(self, x_column='G', y_column='S', hue_column='MANUAL_SELECTION'):
        if self._labels_layer_with_phasor_features is None:
            return None
        # Check if layer contains features
        if self._labels_layer_with_phasor_features.features is None:
            return None
        table = self._labels_layer_with_phasor_features.features
        x_column = table[x_column].values
        y_column = table[y_column].values
        if hue_column in table.columns:
            hue_column = table[hue_column].values
        else:
            hue_column = np.zeros_like(x_column)
        return x_column, y_column, hue_column

    @property
    def x_axis(self):
        if self.plotter_inputs_widget.x_axis_combobox.count() == 0:
            return None
        else:
            return self.plotter_inputs_widget.x_axis_combobox.currentText()

    @x_axis.setter
    def x_axis(self, column: str):
        self.plotter_inputs_widget.x_axis_combobox.setCurrentText(
            column
        )

    @property
    def y_axis(self):
        if self.plotter_inputs_widget.y_axis_combobox.count() == 0:
            return None
        else:
            return self.plotter_inputs_widget.y_axis_combobox.currentText()

    @y_axis.setter
    def y_axis(self, column: str):
        self.plotter_inputs_widget.y_axis_combobox.setCurrentText(
            column
        )

    @property
    def hue(self):
        if self.plotter_inputs_widget.hue_combobox.count() == 0:
            return None
        else:
            return self.plotter_inputs_widget.hue_combobox.currentText()

    @hue.setter
    def hue(self, column: str):
        self.plotter_inputs_widget.hue_combobox.setCurrentText(
            column
        )

    @property
    def plot_type(self):
        return self.plotter_inputs_widget.plot_type_combobox.currentText()

    @plot_type.setter
    def plot_type(self, type):
        self.plotter_inputs_widget.plot_type_combobox.setCurrentText(type)

    def plot(self):
        x_column, y_column, hue_column = self.get_features(
            self.x_axis, self.y_axis, self.hue)
        self.canvas_widget.active_artist = self.canvas_widget.artists[ArtistType[self.plot_type]]
        if x_column is None or y_column is None:
            return
        self.canvas_widget.active_artist.data = np.column_stack(
            (x_column, y_column))
        self.canvas_widget.active_artist.color_indices = hue_column
        self.create_phasors_selected_layer()
        # self.canvas_widget.active_artist.plot()

    def create_phasors_selected_layer(self):
        if self._labels_layer_with_phasor_features is None:
            return
        
        mapped_data = map_array(np.asarray(self._labels_layer_with_phasor_features.data),
                          np.asarray(self._labels_layer_with_phasor_features.features['label'].values),
                          np.array(self._labels_layer_with_phasor_features.features['MANUAL_SELECTION'].values),)
        color_dict = colormap_to_dict(self._colormap, self._colormap.N, exclude_first=True)
        phasors_selected_layer = Labels(
            mapped_data, name='Phasors Selected', scale=self._labels_layer_with_phasor_features.scale,
            colormap=DirectLabelColormap(color_dict=color_dict, name='cat10_mod'))
        self._labels_layer_with_phasor_features.opacity = 0.2
        if self._phasors_selected_layer is None:
            self._phasors_selected_layer = self.viewer.add_layer(phasors_selected_layer)
        else:
            self._phasors_selected_layer.data = mapped_data


label_id = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
G = np.random.rand(10,)
S = np.random.rand(10,)
annotations = np.linspace(0, 4, 10).astype(int)
G_image = G.reshape((2, 5))
S_image = S.reshape((2, 5))
table = pd.DataFrame({'label': label_id, 'G': np.ravel(
    G_image), 'S': np.ravel(S_image), 'MANUAL_SELECTION': annotations})
labels_data = np.arange(1, 11).reshape(2, 5)
viewer = napari.Viewer()
plotter = PlotterWidget(viewer)
viewer.window.add_dock_widget(plotter, area="right")
viewer.add_labels(labels_data, name="labels", features=table)
viewer.add_labels(labels_data, name="Phasor Features Layer", features=table)
napari.run()
