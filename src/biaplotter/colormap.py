import warnings

import matplotlib.colors as mcolors
from cmap import Colormap


class BiaColormap:
    """
    Wrapper class for cmap.Colormap objects to be used in biaplotter.

    Parameters
    ----------
    cmap : ColormapLike
        A ColormapLike object from cmap library, a matplotlib.colors.LinearSegmentedColormap or a matplotlib.colors.ListedColormap
    categorical : bool, optional
        If True, the colormap is considered categorical. If False, the colormap is considered continuous. If not specified, the class will try to infer it from the colormap name. Default is False.
    """

    def __init__(self, cmap, categorical=False):
        if not isinstance(
            cmap, (mcolors.LinearSegmentedColormap, mcolors.ListedColormap)
        ):
            try:
                cmap = Colormap(cmap, name=cmap.name)
            except:
                raise ValueError(
                    "cmap must be a LinearSegmentedColormap or ListedColormap object or a ColormapLike object from cmap library"
                )
        self.cmap = cmap
        # if cat or tab or Set in cmap name, set categorical to True
        if categorical == False and (
            "cat" in cmap.name or "tab" in cmap.name or "Set" in cmap.name
        ):
            warnings.warn(
                "Categorical colormap detected. Setting categorical=True. If the colormap is continuous, set categorical=False explicitly."
            )
            categorical = True
        self.categorical = categorical

    def __call__(self, X, alpha=None, bytes=False):
        return self.cmap(X, alpha=alpha, bytes=bytes)

    def __getattr__(self, name):
        return getattr(self.cmap, name)
