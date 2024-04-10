import numpy as np
from pathlib import Path
from enum import Enum, auto
from nap_plot_tools import CustomToolbarWidget, QtColorSpinBox, make_cat10_mod_cmap
from napari.layers import Labels, Points, Tracks
from napari_matplotlib.base import SingleAxesWidget
from napari_matplotlib.util import Interval
from qtpy.QtWidgets import QHBoxLayout, QLabel
from psygnal import Signal
from qtpy.QtCore import Qt

# from biaplotter.selectors import CustomLassoSelector
from biaplotter.artists import Scatter, Histogram2D
from biaplotter.selectors import InteractiveRectangleSelector, InteractiveEllipseSelector, InteractiveLassoSelector

icon_folder_path = (
    Path(__file__).parent / "icons"
)


class ArtistType(Enum):
    HISTOGRAM2D = auto()
    SCATTER = auto()


class SelectorType(Enum):
    LASSO = auto()
    ELLIPSE = auto()
    RECTANGLE = auto()


class CanvasWidget(SingleAxesWidget):
    # Amount of available input layers
    n_layers_input = Interval(1, None)
    # All layers that have a .features attributes
    input_layer_types = (Labels, Points, Tracks)
    # # Signal emitted when the current artist changes
    artist_changed_signal = Signal(ArtistType)

    def __init__(self, napari_viewer, parent=None, label_text="Class:"):
        super().__init__(napari_viewer, parent=parent)

        # Add selection tools layout below canvas
        self.selection_tools_layout = self._build_selection_toolbar_layout(
            label_text=label_text)

        # Add button to selection_toolbar
        self.selection_toolbar.add_custom_button(
            name=SelectorType.LASSO.name,
            tooltip="Click to enable/disable Lasso selection",
            default_icon_path=icon_folder_path / "lasso.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "lasso_checked.png",
            callback=self.on_enable_selector,
        )
        # Add button to selection_toolbar
        self.selection_toolbar.add_custom_button(
            name=SelectorType.ELLIPSE.name,
            tooltip="Click to enable/disable Ellipse selection",
            default_icon_path=icon_folder_path / "ellipse.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "ellipse_checked.png",
            callback=self.on_enable_selector,
        )
        # Add button to selection_toolbar
        self.selection_toolbar.add_custom_button(
            name=SelectorType.RECTANGLE.name,
            tooltip="Click to enable/disable Rectangle selection",
            default_icon_path=icon_folder_path / "rectangle.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "rectangle_checked.png",
            callback=self.on_enable_selector,
        )

        # Set selection class colormap
        self.colormap = make_cat10_mod_cmap(first_color_transparent=False)

        # Add selection tools layout to main layout below matplotlib toolbar and above canvas
        self.layout().insertLayout(2, self.selection_tools_layout)

        # Create artists
        self.artists = {}
        self.add_artist(ArtistType.SCATTER, Scatter(
            ax=self.axes, colormap=self.colormap))
        self.add_artist(ArtistType.HISTOGRAM2D, Histogram2D(ax=self.axes))
        # Set histogram2d as the default artist
        self.set_active_artist(ArtistType.HISTOGRAM2D)

        # Create selectors
        self.selectors = {}
        self.add_selector(SelectorType.LASSO, InteractiveLassoSelector(
            ax=self.axes, canvas_widget=self))
        self.add_selector(SelectorType.ELLIPSE, InteractiveEllipseSelector(
            ax=self.axes, canvas_widget=self))
        self.add_selector(SelectorType.RECTANGLE,
                          InteractiveRectangleSelector(self.axes, self))
        # Connect data_changed signals from each artist to set data in each selector
        for artist in self.artists.values():
            for selector in self.selectors.values():
                print(artist, ' being connected to ', selector)
                artist.data_changed_signal.connect(selector.update_data)

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
        sender_name = self.sender().text()
        # print(sender_name)
        if checked:
            # If the button is checked, disable all other buttons
            for button_name, button in self.selection_toolbar.buttons.items():
                if button.isChecked() and button_name != sender_name:
                    button.setChecked(False)
            # Remove all selectors
            for selector in self.selectors.values():
                selector.remove()
            # Create the chosen selector
            for selector_type, selector in self.selectors.items():
                if selector_type.name == sender_name:
                    selector.create_selector()
        else:
            # If the button is unchecked, remove the selector
            for selector_type, selector in self.selectors.items():
                if selector_type.name == sender_name:
                    selector.remove()
            

    # def on_current_artist_changed(self, artist_type):
    #     self.current_artist_changed_signal.emit(artist_type)

    def get_active_artist(self):
        for artist in self.artists.values():
            if artist.visible:
                return artist

    def set_active_artist(self, new_artist_type):
        for artist_type, artist in self.artists.items():
            if artist_type == new_artist_type:
                artist.visible = True
            else:
                artist.visible = False
        # Emit signal to notify that the current artist has changed
        self.artist_changed_signal.emit(new_artist_type)

    def add_artist(self, artist_type, artist_instance, visible=False):
        """
        Adds a new artist instance to the artists dictionary.

        Parameters:
        - artist_type (ArtistType): The type of the artist, defined by the ArtistType enum.
        - artist_instance: An instance of the artist class.
        """
        if artist_type in self.artists:
            raise ValueError(f"Artist '{artist_type.name}' already exists.")
        self.artists[artist_type] = artist_instance
        artist_instance.visible = visible

    def add_selector(self, selector_type, selector_instance):
        """
        Adds a new selector instance to the selectors dictionary.

        Parameters:
        - selector_type (SelectorType): The type of the selector, defined by the SelectorType enum.
        - selector_instance: An instance of the selector class.
        """
        if selector_type in self.selectors:
            raise ValueError(
                f"Selector '{selector_type.name}' already exists.")
        self.selectors[selector_type] = selector_instance
