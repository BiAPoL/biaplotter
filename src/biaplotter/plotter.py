from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from nap_plot_tools import (CustomToolbarWidget, CustomToolButton,
                            QtColorSpinBox)
from napari_matplotlib.base import BaseNapariMPLWidget
from psygnal import Signal
from qtpy.QtWidgets import QHBoxLayout, QLabel, QWidget

from biaplotter.artists import Histogram2D, Scatter
from biaplotter.selectors import (InteractiveEllipseSelector,
                                  InteractiveLassoSelector,
                                  InteractiveRectangleSelector)

if TYPE_CHECKING:
    import napari

icon_folder_path = Path(__file__).parent / "icons"


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
        * **selector_changed_signal** emitted when the current selector changes.
        * **show_color_overlay_signal** emitted when the plot overlay is shown or hidden.

    Signals and Slots:

        This class automatically connects the following signals to slots:

        * **data_changed_signal** from each artist to the **update_data** slot in each selector. This allows artists to notify selectors when the data changes. Selectors can then synchronize their data with the artist's data.
    """

    #: Signal emitted when the current `active_artist` changes
    artist_changed_signal: Signal = Signal(str)
    #: Signal emitted when the current `active_selector` changes
    selector_changed_signal: Signal = Signal(str)
    #: Signal emitted when the plot overlay is shown or hidden
    show_color_overlay_signal: Signal = Signal(bool)

    # Constructor and Initialization
    def __init__(
        self,
        napari_viewer: "napari.viewer.Viewer",
        parent: Optional[QWidget] = None,
        label_text: str = "Class:",
    ):
        super().__init__(napari_viewer, parent=parent)
        self.add_single_axes()

        # Initialize UI components
        self._initialize_toolbar(label_text)
        self._initialize_artists()
        self._initialize_selectors()

        # Connect signals
        self._connect_signals()

    def _initialize_toolbar(self, label_text: str):
        """
        Initializes the selection toolbar and layout.
        """
        (
            selection_tools_layout,
            selection_toolbar,
            class_spinbox,
            show_overlay_button,
        ) = self._build_selection_toolbar_layout(label_text=label_text)
        self.selection_tools_layout: QHBoxLayout = selection_tools_layout
        self.selection_toolbar: CustomToolbarWidget = selection_toolbar
        self.class_spinbox: QtColorSpinBox = class_spinbox
        self.show_overlay_button: CustomToolButton = show_overlay_button

        # Add buttons to the toolbar
        self.selection_toolbar.add_custom_button(
            name="LASSO",
            tooltip="Click to enable/disable Lasso selection",
            default_icon_path=icon_folder_path / "lasso.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "lasso_checked.png",
            callback=self.on_enable_selector,
        )
        self.selection_toolbar.add_custom_button(
            name="ELLIPSE",
            tooltip="Click to enable/disable Ellipse selection",
            default_icon_path=icon_folder_path / "ellipse.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "ellipse_checked.png",
            callback=self.on_enable_selector,
        )
        self.selection_toolbar.add_custom_button(
            name="RECTANGLE",
            tooltip="Click to enable/disable Rectangle selection",
            default_icon_path=icon_folder_path / "rectangle.png",
            checkable=True,
            checked_icon_path=icon_folder_path / "rectangle_checked.png",
            callback=self.on_enable_selector,
        )

        # Add selection tools layout to the main layout
        self.layout().insertLayout(1, self.selection_tools_layout)

    def _initialize_artists(self):
        """
        Initializes the artists and sets the default active artist.
        """
        self._active_artist: Scatter | Histogram2D = None
        self.artists: dict = {}
        self.add_artist("SCATTER", Scatter(ax=self.axes))
        self.add_artist("HISTOGRAM2D", Histogram2D(ax=self.axes))
        self.active_artist = "HISTOGRAM2D"

    def _initialize_selectors(self):
        """
        Initializes the selectors.
        """
        self._active_selector: (
            InteractiveRectangleSelector
            | InteractiveEllipseSelector
            | InteractiveLassoSelector
        ) = None
        self.selectors: dict = {}
        self.add_selector(
            "LASSO",
            InteractiveLassoSelector(ax=self.axes, canvas_widget=self),
        )
        self.add_selector(
            "ELLIPSE",
            InteractiveEllipseSelector(ax=self.axes, canvas_widget=self),
        )
        self.add_selector(
            "RECTANGLE",
            InteractiveRectangleSelector(self.axes, self),
        )

    def _connect_signals(self):
        """
        Connects signals between artists and selectors.
        """
        for artist in self.artists.values():
            for selector in self.selectors.values():
                artist.data_changed_signal.connect(selector.update_data)

        for selector in self.selectors.values():
            selector.selection_applied_signal.connect(self._on_finish_drawing)

    def _on_finish_drawing(self, *args):
        """
        Slot to handle the finish drawing signal from selectors.
        """
        self.show_overlay_button.setChecked(True)
        self.active_artist.overlay_visible = True

    # Private Helper Methods
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
        show_overlay_button : CustomToolButton
            The button to show/hide the plot overlay.
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
        class_spinbox.value = 1
        selection_tools_layout.addWidget(class_spinbox)
        # Add customtoolbutton to show/hide overlay
        show_overlay_button = CustomToolButton(
            default_icon_path=str(icon_folder_path / "show_overlay.png"),
            checked_icon_path=str(
                icon_folder_path / "show_overlay_checked.png"
            ),
        )
        show_overlay_button.setIconSize(32)
        show_overlay_button.setText("SHOW_OVERLAY")
        show_overlay_button.setToolTip(
            "Click to show/hide the plot colors overlay"
        )
        selection_tools_layout.addWidget(show_overlay_button)
        show_overlay_button.toggled.connect(self._toggle_show_color_overlay)
        # Add stretch to the right to push buttons to the left
        selection_tools_layout.addStretch(1)
        return (
            selection_tools_layout,
            selection_toolbar,
            class_spinbox,
            show_overlay_button,
        )

    def _set_active_artist(self, artist_name: str):
        """
        Sets the active artist by its name (internal method).

        Parameters
        ----------
        artist_name : str
            The name of the artist (e.g., "scatter", "SCATTER", "Scatter").
        """
        normalized_name = artist_name.upper()
        if normalized_name not in self.artists:
            raise ValueError(f"Artist '{artist_name}' does not exist.")
        self._active_artist = self.artists[normalized_name]
        for artist in self.artists.values():
            artist.visible = artist == self._active_artist
            # Only show overlay of active artist if show_color_overlay is True
            artist.overlay_visible = (
                artist == self._active_artist
                ) and self.show_color_overlay
        # Emit signal to notify that the current artist has changed
        self.artist_changed_signal.emit(normalized_name)

    def _set_active_selector(self, selector_name: str):
        """
        Enables a selector by its name (internal method).

        Parameters
        ----------
        selector_name : str
            The name of the selector (e.g., "lasso", "LASSO", "Lasso").
        """
        normalized_name = selector_name.upper()
        if normalized_name not in self.selectors:
            raise ValueError(f"Selector '{selector_name}' does not exist.")

        # Disable all selectors without emitting the signal
        self._disable_all_selectors(emit_signal=False)

        # Activate the new selector
        self.selectors[normalized_name].create_selector()
        self._active_selector = self.selectors[normalized_name]

        # Emit signal to notify that the current selector has changed
        self.selector_changed_signal.emit(normalized_name)

    def _disable_all_selectors(self, emit_signal: bool = True):
        """
        Disables (removes) all selectors from the canvas.

        Parameters
        ----------
        emit_signal : bool, optional
            Whether to emit the selector_changed_signal, by default True.
        """
        for selector in self.selectors.values():
            selector.selected_indices = None
            selector.remove()
        self._active_selector = None

        # Emit signal only if requested
        if emit_signal:
            self.selector_changed_signal.emit("")

    # Public Properties
    @property
    def active_artist(self) -> Union[Scatter, Histogram2D]:
        """
        Gets or sets the active artist.

        If set, makes the selected artist visible and all other artists invisible.

        Returns
        -------
        Scatter or Histogram2D
            The active artist.
        """
        return self._active_artist

    @active_artist.setter
    def active_artist(self, value: str):
        """
        Sets the active artist by its name.
        """
        self._set_active_artist(value)

    @property
    def active_selector(self):
        """
        Gets or sets the active selector.

        Returns
        -------
        InteractiveRectangleSelector or InteractiveEllipseSelector or InteractiveLassoSelector
            The active selector.
        """
        return self._active_selector

    @active_selector.setter
    def active_selector(self, value: str):
        """
        Sets the active selector by its name.
        """
        self._set_active_selector(value)

    @property
    def show_color_overlay(self) -> bool:
        """
        Gets or sets the visibility of the plot overlay.

        Returns
        -------
        bool
            True if the overlay is visible, False otherwise.
        """
        return self.show_overlay_button.isChecked()

    @show_color_overlay.setter
    def show_color_overlay(self, value: bool):
        """
        Sets the visibility of the plot overlay.

        Parameters
        ----------
        value : bool
            True to show the overlay, False to hide it.
        """
        self.show_overlay_button.setChecked(value)
        self._toggle_show_color_overlay(value)

    # Public Methods
    def add_artist(
        self,
        artist_name: str,
        artist_instance: Scatter | Histogram2D,
        visible: bool = False,
    ):
        """
        Adds a new artist instance to the artists dictionary using a string name.

        Parameters
        ----------
        artist_name : str
            The name of the artist (e.g., "scatter", "SCATTER", "Scatter").
        artist_instance : Scatter or Histogram2D
            An instance of the artist class.
        visible : bool, optional
            Whether the artist should be visible initially, by default False.
        """
        normalized_name = artist_name.upper()
        if normalized_name in self.artists:
            raise ValueError(f"Artist '{artist_name}' already exists.")
        self.artists[normalized_name] = artist_instance
        artist_instance.visible = visible

    def remove_artist(self, artist_name: str):
        """
        Removes an artist from the instance, including from the dictionary.

        Parameters
        ----------
        artist_name : str
            The name of the artist to remove.
        """
        normalized_name = artist_name.upper()
        if normalized_name not in self.artists:
            raise ValueError(f"Artist '{artist_name}' does not exist.")
        del self.artists[normalized_name]

    def add_selector(self, selector_name: str, selector_instance):
        """
        Adds a new selector instance to the selectors dictionary using a string name.

        Parameters
        ----------
        selector_name : str
            The name of the selector (e.g., "lasso", "LASSO", "Lasso").
        selector_instance : InteractiveRectangleSelector or InteractiveEllipseSelector or InteractiveLassoSelector
            An instance of the selector class.
        """
        normalized_name = selector_name.upper()
        if normalized_name in self.selectors:
            raise ValueError(f"Selector '{selector_name}' already exists.")
        self.selectors[normalized_name] = selector_instance

    def remove_selector(self, selector_name: str):
        """
        Removes a selector from the instance, including from the dictionary.

        Parameters
        ----------
        selector_name : str
            The name of the selector to remove.
        """
        normalized_name = selector_name.upper()
        if normalized_name not in self.selectors:
            raise ValueError(f"Selector '{selector_name}' does not exist.")
        self.selectors[normalized_name].remove()
        del self.selectors[normalized_name]

    def on_enable_selector(self, checked: bool):
        """
        Enables or disables the selected selector.

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
            # Set the active selector
            self.active_selector = sender_name
        else:
            # If the button is unchecked, disable all selectors
            self._disable_all_selectors()

    def _toggle_show_color_overlay(self, checked: bool):
        """Show or hide the plot overlay.

        Parameters
        ----------
        checked : bool
            Whether the button is checked or not.
        """
        self.active_artist.overlay_visible = checked
        self.active_artist.draw()
        self.show_color_overlay_signal.emit(checked)

    def hide_color_overlay(self, checked: bool):
        """Deprecated method to hide the color overlay."""
        warnings.warn(
            "hide_color_overlay is deprecated after 0.3.0. Use show_color_overlay setter instead.",
            DeprecationWarning,
            stacklevel=2,
        )
