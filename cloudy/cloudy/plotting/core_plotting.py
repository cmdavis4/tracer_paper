"""Plotting utilities for atmospheric data visualization and analysis.

This module provides functions for creating faceted plots, animations, legends,
and specialized atmospheric plots like soundings and hodographs.
"""

import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.figure as mplfig
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from tqdm.notebook import tqdm
import matplotlib.animation as mplanim
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import metpy.calc as mpc
from metpy.units import units
from metpy.plots import SkewT, Hodograph
import matplotlib as mpl
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Iterable, Callable
import matplotlib.figure
import matplotlib.axes

from ..types_core import PathLike


def format_t_str(t: Any, strftime: str = "%Y-%m-%d %H:%M:%S") -> Any:
    """
    Format timestamp to string, handling numpy datetime64 objects.

    Args:
        t: Timestamp object (numpy datetime64, datetime, etc.)
        strftime: Format string for datetime formatting

    Returns:
        Formatted string or original object if formatting fails
    """
    try:
        return t.astype("datetime64[s]").item().strftime(strftime)
    except:
        # If it's not coerceable to a datetime, just leave it as is
        return t


def clean_legend(
    ax: matplotlib.axes.Axes,
    include_artists: Optional[List[Any]] = None,
    sort: bool = False,
    use_alphas: bool = False,
    **kwargs: Any,
) -> matplotlib.axes.Axes:
    """
    Create a clean legend with customizable styling and sorting.

    Args:
        ax: Matplotlib axes to add legend to
        include_artists: List of specific artists to include in legend
        sort: Whether to sort legend entries by maximum values
        use_alphas: Whether to apply alpha values to legend text colors
        **kwargs: Additional keyword arguments passed to legend()

    Returns:
        The modified axes object
    """
    if include_artists is None:
        include_artists = []

    # Create a function to filter out artists that aren't in include_artists
    def filter_artists(artists):
        if not include_artists:
            return artists
        else:
            return [artist for artist in artists if artist in include_artists]

    # Want to handle lines, collections (i.e. scatter plots/points), and patches (i.e. histograms/polygons)
    # Get the max values of each line
    all_maxes = {
        line.get_label(): (
            {
                "max": max(line.get_ydata()),
                "color": line.get_color(),
                "alpha": line.get_alpha(),
            }
            if sort
            else {"color": line.get_color(), "alpha": line.get_alpha()}
        )
        for line in filter_artists(ax.get_lines())
    }
    collection_maxes = {}
    has_printed_not_implemented = False
    collections_to_iterate = []
    # for collection in filter_artists(ax.collections):
    #     if collection in collections_to_iterate:
    #         continue
    #     if isinstance(collection, mpl.contour.QuadContourSet):
    #         collections_to_iterate += [x for x in collection.collections]
    #     else:
    #         collections_to_iterate.append(collection)
    # for collection in collections_to_iterate:
    for collection in filter_artists(ax.collections):
        # Get the color; assume we use the edge color if the face is transparent
        has_facecolor = collection.get_facecolor() and collection.get_facecolor()[3]
        color = (
            collection.get_edgecolor() if has_facecolor else collection.get_facecolor()
        )
        try:
            collection_maxes[collection.get_label()] = (
                {
                    "max": max([x[1] for x in collection.get_offsets().data]),
                    "color": color,
                }
                if sort
                else {"color": color}
            )
        except NotImplementedError:
            if not has_printed_not_implemented:
                print(
                    "clean_legend not implemented for some elements of figure, skipping"
                )
                has_printed_not_implemented = True
    all_maxes.update(collection_maxes)
    # Need to iterate through patches to handle the alpha
    patches_dict = {}
    for patch in filter_artists(ax.patches):
        this_color = patch.get_facecolor()  # Works for a normal histogram
        if this_color[3] == 0:  # I.e. if the face is transparent, so a step histogram
            this_color = patch.get_edgecolor()
        patches_dict[patch.get_label()] = (
            {
                "max": max([xy[1] for xy in patch.get_xy()]),
                "color": this_color,
            }
            if sort
            else {"color": this_color}
        )
    all_maxes.update(patches_dict)
    handles, labels = ax.get_legend_handles_labels()
    if sort:
        # Get the right order of line names
        label_order_desc = [
            k
            for k, v in sorted(
                all_maxes.items(), key=lambda item: item[1]["max"], reverse=True
            )
        ]
        # Now get the existing legend
        # Get the indexes that the order we want corresponds to in the existing labels/handles
    else:
        label_order_desc = labels
    order = [labels.index(x) for x in label_order_desc]
    # Need to order the *handles* correctly so matplotlib can connect them to the actual lines,
    # even though we hide them
    labelcolors = [all_maxes[k]["color"] for k in label_order_desc]
    if use_alphas:
        new_labelcolors = []
        for k_ix, k in enumerate(label_order_desc):
            this_color = list(labelcolors[k_ix])
            this_color[3] = all_maxes[k]["alpha"]
            new_labelcolors.append(tuple(this_color))
        labelcolors = new_labelcolors
    legend = ax.legend(
        [handles[ix] for ix in order],
        [labels[ix] for ix in order],  # This is the same as line_order_desc
        # Hide the handles and make the text color match the line color
        handletextpad=0.0,
        handlelength=0.0,
        handleheight=0.0,
        markerscale=0.0,
        labelcolor=labelcolors,
        #         scatterpoints=0,
        **kwargs,
    )
    # Get rid of any remaining little rectangular blips
    for item in legend.legend_handles:
        item.set_visible(False)
    return ax


def contour_legend(contour_set, **kwargs):
    handles = [x for x in contour_set.legend_elements()[0]]
    return contour_set.figure.legend(
        handles=handles,
        # Hide the handles and make the text color match the line color
        handletextpad=0.0,
        handlelength=0.0,
        handleheight=0.0,
        markerscale=0.0,
        # labelcolor='black',
        # facecolor=contour_edgecolors,
        ncol=len(handles),
        loc="upper left",
        **kwargs,
    )


def get_nth_color(n):
    return plt.rcParams["axes.prop_cycle"].by_key()["color"][n]


def get_next_color(ax):
    return next(ax._get_lines.prop_cycler)["color"]


def get_cmap(name):
    """
    Get a matplotlib colormap object for a given colormap name.
    This is admittedly silly to make a function for, but I can never remember
    how to do it from matplotlib directly.

    Args:
        name (str): Name of the colormap

    Returns:
        matplotlib.colors.Colormap: The colormap object
    """
    return mpl.colormaps[name]


def add_row_header(ax, text, pad=None, **kwargs):
    """
    Add a row header to the left of an axes with automatic spacing.

    Args:
        ax: Matplotlib axes object
        text: Text for the row header (can be multi-line)
        pad: Manual padding offset. If None, calculated automatically
        **kwargs: Additional arguments passed to annotate()

    Returns:
        The annotation object
    """
    # Split text into lines to handle multi-line text
    lines = text.split("\n") if isinstance(text, str) else [str(text)]
    n_lines = len(lines)

    # Calculate automatic padding based on various factors
    if pad is None:
        # Base padding from ylabel
        base_pad = ax.yaxis.labelpad if hasattr(ax.yaxis, "labelpad") else 5

        # Additional padding for tick labels
        tick_width = 0

        # For log-scaled axes, we need to handle tick labels differently
        # Force the axis to update its ticks before we calculate widths
        ax.figure.canvas.draw_idle()

        if ax.yaxis.get_ticklabels():
            # Get the actual rendered width of tick labels in display coordinates
            # This works better than counting characters, especially for log scales
            try:
                # Get the bounding boxes of all tick labels
                tick_bboxes = []
                for label in ax.yaxis.get_ticklabels():
                    if label.get_text():  # Only consider non-empty labels
                        # Get bbox in figure coordinates
                        bbox = label.get_window_extent(
                            renderer=ax.figure.canvas.get_renderer()
                        )
                        tick_bboxes.append(bbox)

                if tick_bboxes:
                    # Find the maximum width in points
                    max_width = max(bbox.width for bbox in tick_bboxes)
                    # Convert from pixels to points (assuming 72 DPI for points)
                    tick_width = (
                        max_width * 0.75
                    )  # Scale factor to convert pixels to approximate points
            except (AttributeError, TypeError):
                # Fallback to character counting if rendering fails
                tick_texts = [label.get_text() for label in ax.yaxis.get_ticklabels()]
                if tick_texts:
                    max_tick_len = max(len(str(t)) for t in tick_texts if t)
                    tick_width = (
                        max_tick_len * 8
                    )  # Approximate character width in points

        # Additional padding for ylabel if present
        ylabel_width = 0
        if ax.yaxis.get_label().get_text():
            ylabel_width = 20  # Approximate width for rotated ylabel

        # Extra padding for multi-line text
        multiline_pad = (n_lines - 1) * 10 if n_lines > 1 else 0

        # Calculate total padding
        pad = base_pad + tick_width + ylabel_width + multiline_pad + 15

    # Handle multi-line text positioning
    if n_lines > 1:
        # For multi-line text, adjust vertical alignment
        va = kwargs.get("va", "center")
        # Join lines back together for display
        display_text = "\n".join(lines)
    else:
        va = kwargs.get("va", "center")
        display_text = text

    # Set up annotation arguments with automatic spacing
    annotation_kwargs = {
        "xy": (0, 0.5),
        "xytext": (-pad, 0),
        "xycoords": "axes fraction",  # More reliable than ax.yaxis.label
        "textcoords": "offset points",
        "rotation": 90,
        "fontsize": kwargs.get("fontsize", 16),  # Slightly smaller default
        "ha": "center",
        "va": va,
        "fontweight": "bold",
    }

    # Update with user-provided kwargs
    annotation_kwargs.update(kwargs)

    return ax.annotate(display_text, **annotation_kwargs)


def shifted_colormap(cmap, new_range, n=256):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    colors_list = cmap(np.linspace(new_range[0], new_range[1], n))
    return colors.LinearSegmentedColormap.from_list("new", colors_list)


# Define some shifted colormaps
shifted_blues = shifted_colormap("Blues", (0.2, 1.0))
shifted_greens = shifted_colormap("Greens", (0.3, 1.0))
shifted_oranges = shifted_colormap("Oranges", (0.3, 1.0))


def plot_sounding(ds: xr.Dataset) -> matplotlib.figure.Figure:

    # Exclude the fake level if present
    if ds["z"].values[0] < 0:
        ds = ds.isel(z=slice(1, len(ds["z"])))

    this_ds_mean = ds.squeeze().mean(["x", "y"])

    fig = plt.figure(figsize=(9, 9))
    skewt = SkewT(fig, rotation=30)
    skewt.plot(
        this_ds_mean["P"].values,
        (this_ds_mean["T"].values * units("K")).to("degC").magnitude,
        "r",
    )
    skewt.plot(
        this_ds_mean["P"].values,
        (this_ds_mean["dewpoint"].values * units("K")).to("degC"),
        "blue",
    )
    # fig.suptitle(sounding_time)

    # Calculate and plot parcel profile
    parcel_path = mpc.parcel_profile(
        this_ds_mean["P"].values * units.hPa,
        this_ds_mean["T"].values[0] * units.K,
        this_ds_mean["dewpoint"].values[0] * units.K,
    )
    skewt.plot(
        this_ds_mean["P"].values,
        parcel_path,
        color="grey",
        linestyle="dashed",
        linewidth=2,
    )

    # Create a hodograph
    ax_hod = inset_axes(skewt.ax, "40%", "40%", loc=1)
    h = Hodograph(ax_hod, component_range=35.0)
    h.add_grid(increment=10)
    h.plot_colormapped(
        this_ds_mean["UC"].values,
        this_ds_mean["VC"].values,
        this_ds_mean["z"].values * units("m"),
    )

    skewt.ax.set_xlabel("Temperature (°C)")
    skewt.ax.set_ylabel("Pressure (hPa)")

    ax_hod.set_xlabel("U (m/s)")
    ax_hod.set_ylabel("V (m/s)")

    fig.suptitle("Initial sounding")

    return fig


def fig_multisave(
    fig: matplotlib.figure.Figure,
    name: Union[str, PathLike],
    dirs: Union[PathLike, List[PathLike]],
    no_title_version: bool = False,
    resize_to_width: Optional[float] = None,
    extensions: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """
    Save a figure into multiple directories (with the same filename). That's easily accomplished
    with a for loop on its own, but the no_title_version flag also allows for the creation of two copies
    of the figure in each directory: one with its suptitle (if present), and one without. These copies
    will be suffixed with 'title-yes' and 'title-no' respectively.
    """
    # Make the dirs argument a list if a single directory was passed
    if not isinstance(dirs, list):
        dirs = [dirs]

    # Handle mutable default argument
    if extensions is None:
        extensions = [".pdf", ".png"]

    # Clean up the file extensions
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = [
        "." + extension if extension[0] != "." else extension
        for extension in extensions
    ]

    # Remove any file extensions from name
    name = Path(name).stem

    if resize_to_width:
        # Resize the figure with the same aspect ratio if we should
        current_size = fig.get_size_inches()
        current_ar = current_size[1] / current_size[0]
        fig.set_size_inches(resize_to_width, current_ar * resize_to_width)

    # Ignore the no_title_version argument if the figure doesn't have a suptitle
    if not fig._suptitle:
        no_title_version = False
    for this_dir in dirs:
        suffix = "_title-yes" if no_title_version else ""
        for extension in extensions:
            fig.savefig(this_dir.joinpath(name + suffix + extension))
    # Make a no title version if we're doing that
    if no_title_version:
        fig._suptitle.remove()
        fig._suptitle = None
        for this_dir in dirs:
            for extension in extensions:
                fig.savefig(this_dir.joinpath(name + f"_title-no{extension}"))
    return fig


def sequential_cmap(colors_list, name=None, N=512):
    colors_list = [
        colors_list.to_rgb(color) if isinstance(color, str) else color
        for color in colors_list
    ]
    return colors.LinearSegmentedColormap.from_list(
        name or f"cd_{str(colors)}", colors_list, N=N
    )


def single_color_cmap(color, linear_opacity=False, name=None, N=512):
    if isinstance(color, str):
        color = colors.to_rgb(color)
    start_color = (color[0], color[1], color[2], 0) if linear_opacity else (1, 1, 1, 1)
    return sequential_cmap([start_color, color], name=name, N=N)


def transparent_under_cmap(cmap, bad=True):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    cmap = cmap.copy()
    cmap.set_under((0, 0, 0, 0))
    if bad:
        cmap.set_bad((0, 0, 0, 0))
    return cmap


def share_axes(axs, x=True, y=True):
    # Just share them all with the first one
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    base_ax = axs[0]
    for ax in axs[1:]:
        if x:
            ax.sharex(base_ax)
        if y:
            ax.sharey(base_ax)
    return axs


def scale_axes_ticks(ax, scale=1000, x=True, y=True):
    # Scale the horizontal axes to km rather than m
    ticks = mpl.ticker.FuncFormatter(lambda x, pos: "{0:g}".format(x / scale))
    if x:
        ax.xaxis.set_major_formatter(ticks)
    if y:
        ax.yaxis.set_major_formatter(ticks)


def prepend_axes_letters(
    axs: Union[matplotlib.axes.Axes, np.ndarray],
    start_letter: str = "a",
    format_string: str = "({}) ",
) -> Union[matplotlib.axes.Axes, np.ndarray]:
    """
    Prepend letters to the title of each axes in row-major order.

    Given a set of axes (as returned by plt.subplots), this function prepends
    sequential letters to each axes title. The lettering proceeds in row-major
    order (left-to-right within each row, then top-to-bottom across rows).

    Args:
        axs: Single axes or array of axes from plt.subplots()
        start_letter: The starting letter (default: 'a')
        format_string: Format string for the letter label. Use {} as placeholder
                      for the letter (default: '({}) ' produces '(a) ', '(b) ', etc.)

    Returns:
        The modified axes (same type as input)

    Examples:
        >>> fig, axs = plt.subplots(2, 3)
        >>> prepend_axes_letters(axs)
        # Produces: (a), (b), (c) in first row
        #           (d), (e), (f) in second row

        >>> fig, ax = plt.subplots()
        >>> ax.set_title("My Plot")
        >>> prepend_axes_letters(ax)
        # Produces: (a) My Plot
    """
    # Handle single axes case
    if isinstance(axs, matplotlib.axes.Axes):
        axs_flat = [axs]
    else:
        # Flatten in row-major order (C-style)
        axs_flat = axs.flatten() if isinstance(axs, np.ndarray) else axs

    # Get the starting letter as an integer (a=0, b=1, etc.)
    start_ord = ord(start_letter.lower())

    # Iterate through axes in row-major order
    for i, ax in enumerate(axs_flat):
        # Generate the letter label
        letter = chr(start_ord + i)
        label = format_string.format(letter)

        # Get current title and prepend the letter
        current_title = ax.get_title()
        new_title = label + current_title
        ax.set_title(new_title)

    return axs


def gif_from_pngs(gif_path, pngs_fpaths, fps=24):
    import imageio

    with imageio.get_writer(str(gif_path), mode="I", fps=fps) as writer:
        for png_fpath in tqdm(pngs_fpaths):
            image = imageio.imread(str(png_fpath))
            writer.append_data(image)
