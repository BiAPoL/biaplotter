from __future__ import annotations

from pathlib import Path
from enum import Enum, auto
from nap_plot_tools import CustomToolbarWidget, QtColorSpinBox
from napari.layers import Labels, Points, Tracks
from napari_matplotlib.base import BaseNapariMPLWidget
from napari_matplotlib.util import Interval
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget
from psygnal import Signal
from typing import Union, TYPE_CHECKING, Optional

from biaplotter.artists import Scatter, Histogram2D
from biaplotter.selectors import InteractiveRectangleSelector, InteractiveEllipseSelector, InteractiveLassoSelector

if TYPE_CHECKING:
    import napari

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


class CanvasWidget(BaseNapariMPLWidget):
    """A widget that contains a canvas with matplotlib axes and a selection toolbar.

    The widget includes a selection toolbar with buttons to enable/disable selection tools.
    The selection toolbar includes a color class spinbox to select the class to assign to selections.
    The widget includes artists and selectors to plot data and select points.

    Parameters
    ----------
    napari_viewer : napari.viewer.Viewer
        The napari viewer.
    parent : QWidget, optional
        The parent widget, by default None.
    label_text : str, optional
        The text to display next to the class spinbox, by default "Class:".

    Notes
    -----

    Signals:

        * **artist_changed_signal** emitted when the current artist changes.

    Signals and Slots:

        This class automatically connects the following signals to slots:

        * **data_changed_signal** from each artist to the **update_data** slot in each selector. This allows artists to notify selectors when the data changes. Selectors can then synchronize their data with the artist's data.  
    """

    #: Signal emitted when the current `active_artist` changes
    artist_changed_signal: Signal = Signal(ArtistType)

    def __init__(self, napari_viewer: "napari.viewer.Viewer", parent: Optional[QWidget] = None, label_text: str = "Class:"):
        super().__init__(napari_viewer, parent=parent)
        self.add_single_axes()
        # Add selection tools layout below canvas
        selection_tools_layout, selection_toolbar, class_spinbox = self._build_selection_toolbar_layout(
            label_text=label_text)
        #: The selection tools layout.
        self.selection_tools_layout: QHBoxLayout = selection_tools_layout
        #: The selection toolbar.
        self.selection_toolbar: CustomToolbarWidget = selection_toolbar
        #: The color class spinbox.
        self.class_spinbox: QtColorSpinBox = class_spinbox
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

        # Add selection tools layout to main layout below matplotlib toolbar and above canvas
        self.layout().insertLayout(1, self.selection_tools_layout)

        # Create artists
        #: Stores the active artist.
        self._active_artist: Scatter | Histogram2D = None
        #: Dictionary of artists.
        self.artists: dict = {}
        self.add_artist(ArtistType.SCATTER, Scatter(ax=self.axes))
        self.add_artist(ArtistType.HISTOGRAM2D, Histogram2D(ax=self.axes))
        # Set histogram2d as the default artist
        self.active_artist = self.artists[ArtistType.HISTOGRAM2D]

        # Create selectors
        #: Dictionary of selectors.
        self.selectors: dict = {}
        self.add_selector(SelectorType.LASSO, InteractiveLassoSelector(
            ax=self.axes, canvas_widget=self))
        self.add_selector(SelectorType.ELLIPSE, InteractiveEllipseSelector(
            ax=self.axes, canvas_widget=self))
        self.add_selector(SelectorType.RECTANGLE,
                          InteractiveRectangleSelector(self.axes, self))
        # Connect data_changed signals from each artist to set data in each selector
        for artist in self.artists.values():
            for selector in self.selectors.values():
                artist.data_changed_signal.connect(selector.update_data)

    def _build_selection_toolbar_layout(self, label_text: str = "Class:"):
        """Builds the selection toolbar layout.

        The toolbar starts without any buttons. Add buttons using the add_custom_button method.
        The toolbar includes a color class spinbox to select the class to assign to selections.

        Parameters
        ----------
        label_text : str, optional
            The text to display next to the class spinbox, by default "Class:"

        Returns
        -------
        selection_tools_layout : QHBoxLayout
            The selection tools layout.
        selection_toolbar : CustomToolbarWidget
            The toolbar widget
        class_spinbox : QtColorSpinBox
            The color class spinbox.
        """
        # Add selection tools layout below canvas
        selection_tools_layout = QHBoxLayout()
        # Add selection toolbar
        selection_toolbar = CustomToolbarWidget(self)
        selection_tools_layout.addWidget(selection_toolbar)
        # Add class spinbox
        selection_tools_layout.addWidget(QLabel(label_text))
        # Add color class spinbox
        class_spinbox = QtColorSpinBox(first_color_transparent=False)
        selection_tools_layout.addWidget(class_spinbox)
        # Add stretch to the right to push buttons to the left
        selection_tools_layout.addStretch(1)
        return selection_tools_layout, selection_toolbar, class_spinbox

    def on_enable_selector(self, checked: bool):
        """Enables or disables the selected selector.

        Enabling a selector disables all other selectors.

        Parameters
        ----------
        checked : bool
            Whether the button is checked or not.
        """
        sender_name = self.sender().text()
        if checked:
            # If the button is checked, disable all other buttons
            for button_name, button in self.selection_toolbar.buttons.items():
                if button.isChecked() and button_name != sender_name:
                    button.setChecked(False)
            # Remove all selectors
            for selector in self.selectors.values():
                selector.selected_indices = None
                selector.remove()
            # Create the chosen selector
            for selector_type, selector in self.selectors.items():
                if selector_type.name == sender_name:
                    selector.create_selector()
        else:
            # If the button is unchecked, remove the selector
            for selector_type, selector in self.selectors.items():
                if selector_type.name == sender_name:
                    selector.selected_indices = None
                    selector.remove()

    @property
    def active_artist(self):
        """Sets or returns the active artist.

        If set, makes the selected artist visible and all other artists invisible.

        Returns
        -------
        Scatter or Histogram2D
            The active artist.

        Notes
        -----
        artist_changed_signal : Signal
            Signal emitted when the current artist changes.
        """
        return self._active_artist

    @active_artist.setter
    def active_artist(self, value: Scatter | Histogram2D):
        """Sets the active artist.
        """
        self._active_artist = value
        for artist in self.artists.values():
            if artist == self._active_artist:
                artist.visible = True
            else:
                artist.visible = False
        # Gets artist type
        for artist_type, artist in self.artists.items():
            if artist == value:
                active_artist_type = artist_type
        # Emit signal to notify that the current artist has changed
        self.artist_changed_signal.emit(active_artist_type)

    def add_artist(self, artist_type: ArtistType, artist_instance: Scatter | Histogram2D, visible: bool = False):
        """
        Adds a new artist instance to the artists dictionary.

        Parameters
        ----------
        artist_type : ArtistType
            The type of the artist, defined by the ArtistType enum.
        artist_instance : Scatter or Histogram2D
            An instance of the artist class.
        """
        if artist_type in self.artists:
            raise ValueError(f"Artist '{artist_type.name}' already exists.")
        self.artists[artist_type] = artist_instance
        artist_instance.visible = visible

    def add_selector(self, selector_type: SelectorType, selector_instance: InteractiveRectangleSelector | InteractiveEllipseSelector | InteractiveLassoSelector):
        """
        Adds a new selector instance to the selectors dictionary.

        Parameters
        ----------
        selector_type : SelectorType
            The type of the selector, defined by the SelectorType enum.
        selector_instance : InteractiveRectangleSelector or InteractiveEllipseSelector or InteractiveLassoSelector
            An instance of the selector class.
        """
        if selector_type in self.selectors:
            raise ValueError(
                f"Selector '{selector_type.name}' already exists.")
        self.selectors[selector_type] = selector_instance
