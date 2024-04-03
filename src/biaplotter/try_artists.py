import numpy as np
import matplotlib.pyplot as plt
from biaplotter.artists import CustomScatter, Custom2DHistogram
from nap_plot_tools.cmap import cat10_mod_cmap

# Generate some random data4
np.random.seed(42)  # For reproducible results
data = np.random.randn(100, 2)  # 100 2D points

# Setup the figure and axes
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Create and display a CustomScatterArtist
scatter_artist = CustomScatter(ax=axs[0])
axs[0].set_title('Custom Scatter Plot')

# Create and display a Custom2DHistogramArtist
histogram_artist = Custom2DHistogram(data=data, ax=axs[1], bins=20)
axs[1].set_title('Custom 2D Histogram')

# Display the initial plots
plt.show()

scatter_artist.data = data  # Update data for the CustomScatterArtist

# Update color_indices for both artists to change colors
color_indices = np.linspace(0, 1, 100)  # Example color indices

# Update and display changes for the CustomScatterArtist
scatter_artist.color_indices = color_indices[:data.shape[0]]  # Assuming one color per point

# Update and display changes for the Custom2DHistogramArtist
histogram_artist.color_indices = color_indices  # Use same indices for simplicity

# After updating color_indices, the figures should be redrawn to show changes
# Since draw_idle is called within the setter, the plots will update automatically.
a=1