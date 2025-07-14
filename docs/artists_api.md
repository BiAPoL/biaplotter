# Artists
```{eval-rst}
.. currentmodule:: biaplotter.artists

.. autoclass:: Artist
   :show-inheritance:  

   .. rubric:: Properties Summary

   .. autosummary::

      ~Artist.color_indices
      ~Artist.data
      ~Artist.highlighted
      ~Artist.ids
      ~Artist.overlay_colormap
      ~Artist.visible
      ~Artist.x_label_text
      ~Artist.x_label_color
      ~Artist.y_label_text
      ~Artist.y_label_color

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Artist.ax

   .. rubric:: Methods Summary

   .. autosummary::

      ~Artist.draw
      ~Artist.highlight_data_by_ids

   .. rubric:: Properties Documentation

   .. autoattribute:: color_indices
   .. autoattribute:: data
   .. autoattribute:: highlighted
   .. autoattribute:: ids
   .. autoattribute:: overlay_colormap
   .. autoattribute:: visible
   .. autoattribute:: x_label_text
   .. autoattribute:: x_label_color
   .. autoattribute:: y_label_text
   .. autoattribute:: y_label_color

   .. rubric:: Attributes Documentation

   .. autoattribute:: ax

   .. rubric:: Methods Documentation

   .. automethod:: draw
   .. automethod:: highlight_data_by_ids

.. autoclass:: Scatter
   :show-inheritance:  

   .. rubric:: Properties Summary

   .. autosummary::

      ~Scatter.alpha
      ~Scatter.color_indices
      ~Scatter.color_normalization_method
      ~Scatter.data
      ~Scatter.default_edge_width
      ~Scatter.default_size
      ~Scatter.highlighted
      ~Scatter.ids
      ~Scatter.overlay_visible
      ~Scatter.size
      ~Scatter.visible

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Scatter.ax
      ~Scatter.overlay_colormap

   .. rubric:: Methods Summary

   .. autosummary::

      ~Scatter.color_indices_to_rgba
      ~Scatter.draw


   .. rubric:: Signals Summary

   .. autosummary::

      ~Scatter.data_changed_signal
      ~Scatter.color_indices_changed_signal
      ~Scatter.highlighted_changed_signal

   .. rubric:: Properties Documentation

   .. autoattribute:: alpha
   .. autoattribute:: color_indices
   .. autoattribute:: color_normalization_method
   .. autoattribute:: data
   .. autoattribute:: default_edge_width
   .. autoattribute:: default_size
   .. autoattribute:: highlighted
   .. autoattribute:: ids
   .. autoattribute:: overlay_visible
   .. autoattribute:: size
   .. autoattribute:: visible

   .. rubric:: Attributes Documentation

   .. autoattribute:: ax
   .. autoattribute:: overlay_colormap

   .. rubric:: Methods Documentation

   .. automethod:: color_indices_to_rgba
   .. automethod:: draw

   .. rubric:: Signals Documentation

   .. autoattribute:: data_changed_signal
   .. autoattribute:: color_indices_changed_signal
   .. autoattribute:: highlighted_changed_signal

.. autoclass:: Histogram2D
   :show-inheritance:  

   .. rubric:: Properties Summary

   .. autosummary::

      ~Histogram2D.bin_alpha
      ~Histogram2D.bins
      ~Histogram2D.cmin
      ~Histogram2D.color_indices
      ~Histogram2D.data
      ~Histogram2D.highlighted
      ~Histogram2D.histogram
      ~Histogram2D.histogram_color_normalization_method
      ~Histogram2D.histogram_colormap
      ~Histogram2D.histogram_interpolation
      ~Histogram2D.overlay_color_normalization_method
      ~Histogram2D.overlay_interpolation
      ~Histogram2D.overlay_opacity
      ~Histogram2D.overlay_visible
      ~Histogram2D.visible

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Histogram2D.ax
      ~Histogram2D.overlay_colormap

   .. rubric:: Methods Summary

   .. autosummary::

      ~Histogram2D.color_indices_to_rgba
      ~Histogram2D.draw
      ~Histogram2D.indices_in_patches_above_threshold

   .. rubric:: Signals Summary

   .. autosummary::

      ~Histogram2D.data_changed_signal
      ~Histogram2D.color_indices_changed_signal
      ~Histogram2D.highlighted_changed_signal

   .. rubric:: Properties Documentation

   .. autoattribute:: bin_alpha
   .. autoattribute:: bins
   .. autoattribute:: cmin
   .. autoattribute:: color_indices
   .. autoattribute:: data
   .. autoattribute:: highlighted
   .. autoattribute:: histogram
   .. autoattribute:: histogram_color_normalization_method
   .. autoattribute:: histogram_colormap
   .. autoattribute:: histogram_interpolation
   .. autoattribute:: overlay_color_normalization_method
   .. autoattribute:: overlay_interpolation
   .. autoattribute:: overlay_opacity
   .. autoattribute:: overlay_visible
   .. autoattribute:: visible
   
   .. rubric:: Attributes Documentation

   .. autoattribute:: ax
   .. autoattribute:: overlay_colormap

   .. rubric:: Methods Documentation

   .. automethod:: color_indices_to_rgba
   .. automethod:: draw
   .. automethod:: indices_in_patches_above_threshold

   .. rubric:: Signals Documentation

   .. autoattribute:: data_changed_signal
   .. autoattribute:: color_indices_changed_signal
   .. autoattribute:: highlighted_changed_signal
```
