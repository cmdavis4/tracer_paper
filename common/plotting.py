import xarray as xr
import matplotlib.pyplot as plt
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


def format_t_str(t, strftime="%Y-%m-%d %H:%M:%S"):
    try:
        return t.astype("datetime64[s]").item().strftime(strftime)
    except:
        # If it's not coerceable to a datetime, just leave it as is
        return t


def facet_xarray_ds(
    ds_or_da,
    facet_cols,
    value_col=None,
    fig=None,
    col_wrap=3,
    figsize=None,
    standardize_vscale=True,
    subplot_kws={},
    format_datetimes=True,
    **kwargs,
):
    """Create "small multiples" or faceted plots using an xarray DataSet or DataArray. Intended to compliment the
    built-in .plot method of these objects, which doesn't work in all situations—it sometimes doensn't work well
    for faceting a single level, and doesn't support faceting on multiple levels. For hopefully obvious reasons,
    none of the dimensions passed via `facet_cols` should be very large, or you'll have an unmanageable number
    of plots.

    Args:
        ds_or_da (xr.DataSet or xr.DataArray): Dataset to plot. If a DataSet, pulls out `value_col` as the variable
            containing the actual values that are to be plotted.
        facet_cols (str or iterable[str]): Dimensions to facet over. If a string, will only facet over that dimension.
            If an iterable of strings (presently only up to length 2), will facet over each dimension sequentially,
            i.e. subfigures will be created for each of the values in the first dimension, then subplot axes created
            for each value in the second dimension.
        value_col (str, optional): Name of the variable containing the values to be plotted. Only used if `ds_or_da`
            is a DataSet. Will probably be removed at some point in favor of requiring a DataArray for ds_or_da.
            Defaults to None.
        fig (plt.Figure, optional): matplotlib Figure in which this faceted plots should be created. Defaults to None,
            in which case a new Figure is created.
        col_wrap (int, optional): Number of columns in each set of subplots. Defaults to 3.
        figsize (_type_, optional): Size of the overall figure, if `fig` was not passed. Defaults to None.
        standardize_vscale (bool, optional): If True (default), standardizes the value scale (whichever axis that
            may be) across all of the plots such that they are plotted against the same range of values, allowing
            them to be directly compared more easily.

    Raises:
        ValueError: Cannot facet on more than two dimensions at present.
        ValueError: ds_or_da must be either a DataArrray or a Dataset with a str passed in for value_col

    Returns:
        _type_: matplotlib.Figure
    """
    # if facet_cols is a list of only one element, we want to just make it a string
    if len(facet_cols) == 1 and not isinstance(facet_cols, str):
        facet_cols = facet_cols[0]
    if len(facet_cols) > 2 and not isinstance(facet_cols, str):
        raise ValueError("Can't facet on more than two dimensions at present")

    # If ds_or_da is a DataSet, we'll only care about the (value) field we're displaying, so select that to get to a
    # DataArray
    if isinstance(ds_or_da, xr.DataArray):
        da = ds_or_da
    else:
        if isinstance(ds_or_da, xr.Dataset) and isinstance(value_col, str):
            da = ds_or_da[value_col]
        else:
            raise ValueError(
                "ds_or_da must be either a DataArrray or a Dataset with a str passed in for value_col"
            )
    # Need to calculate the global min/max here since _inner_facet won't see all of the values
    vmax = da.max().values
    vmin = da.min().values
    if (kwargs.get("yscale") == "log" and "x" in kwargs and "y" not in kwargs) or (
        kwargs.get("xscale") and "y" in kwargs and "x" not in kwargs
    ):
        vmin = da.where(da > 0).min().values

    # Approach this by defining the procedure we'll use for the innermost facet level, whether or not there is another outer facet level
    def _inner_facet(subda, facet_col, fig=None, figsize=None, **kwargs):
        nonlocal vmin
        nonlocal vmax

        n_facets = len(subda[facet_col].values)
        n_rows = int(n_facets / col_wrap) + (n_facets % col_wrap != 0)
        if fig:
            axs = fig.subplots(nrows=n_rows, ncols=col_wrap, **subplot_kws)
        else:
            fig, axs = plt.subplots(
                nrows=n_rows, ncols=col_wrap, figsize=figsize, **subplot_kws
            )
        axs = axs.flatten()
        for ix, (facet_val, group) in enumerate(list(subda.groupby(facet_col))):
            ax = axs[ix]
            # We scale the values here via the vmin and vmax arguments if it's a pcolormesh, which we infer by neither x nor y being passed
            vscale_kwargs = (
                {"vmax": vmax, "vmin": vmin}
                if ("x" not in kwargs and "y" not in kwargs and standardize_vscale)
                else {}
            )
            # Handle the hue keyword manually so that we can do it with things like multiindexes
            if "hue" in kwargs:
                for hue_ix, (hue, hue_subds) in enumerate(group.groupby(kwargs["hue"])):
                    kwargs_no_hue = {k: v for k, v in kwargs.items() if k != "hue"}
                    hue_subds.plot(
                        ax=ax,
                        label=hue,
                        color=get_nth_color(hue_ix),
                        **vscale_kwargs,
                        **kwargs_no_hue,
                    )
            else:
                group.plot(ax=ax, **vscale_kwargs, **kwargs)
            if format_datetimes and isinstance(facet_val, np.datetime64):
                ax.set_title("{}={}".format(facet_col, format_t_str(facet_val)))
            else:
                ax.set_title("{}={}".format(facet_col, facet_val))
            # If we're supposed to standardize the values and it wasn't a pcolormesh:
            if standardize_vscale:
                if "x" in kwargs and "xlim" not in kwargs:
                    # Then the y axis is the one with the values
                    ax.set_ylim([vmin, vmax])
                elif "y" in kwargs and "ylim" not in kwargs:
                    # Then it's the x axis
                    ax.set_xlim([vmin, vmax])
        n_axes_to_delete = col_wrap * n_rows - n_facets
        if n_axes_to_delete > 0:
            for ax in list(axs[-1 * n_axes_to_delete :]):
                fig.delaxes(ax)
        return fig

    if len(facet_cols) == 2 and not isinstance(facet_cols, str):
        outer_facet_col = facet_cols[0]
        outer_facet_vals = da[outer_facet_col].values
        outer_fig = fig or plt.figure(figsize=figsize)
        # This assumes the facets are relatively large and that we always want them to each be their own row
        subfigs = outer_fig.subfigures(nrows=len(outer_facet_vals))
        for ix, inner_fig in enumerate(subfigs):
            _inner_facet(
                da.sel({outer_facet_col: outer_facet_vals[ix]}),
                facet_cols[1],
                fig=inner_fig,
                figsize=None,
                **kwargs,
            )
            inner_fig.suptitle("{}={}".format(outer_facet_col, outer_facet_vals[ix]))
        return outer_fig
    else:
        return _inner_facet(da, facet_cols, fig=fig, figsize=figsize, **kwargs)


def clean_legend(ax, include_artists=[], sort=False, use_alphas=False, **kwargs):

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
    for item in legend.legendHandles:
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


# def subplots(*args, single_panel_size=(3, 3), max_figsize=() **kwargs):
#     if 'figsize' in kwargs:
#         # Don't do anything in this case
#         return plt.subplots(*args, **kwargs)
#     nrows = kwargs.get('nrows') or 1
#     ncols = kwargs.get('ncols') or 1
#     # Assume the base figure size is for one panel
#     # base_figsize = mpl.rcParams['figure.figsize']
#     # # Calculate the aspect ratio
#     # ar = single_panel_size[1] / single_panel_size[0]
#     # Calculate the figure size from this
#     figsize = (single_panel_size[0] * ncols, single_panel_size[0] * nrows)


def add_row_header(ax, text, pad=4, **kwargs):
    # Allow for any of the arguments to annotate to be overwritten by kwargs
    annotation_kwargs = {
        "xy": (0, 0.5),
        "xytext": (-ax.yaxis.labelpad - pad, 0),
        "xycoords": ax.yaxis.label,
        "textcoords": "offset points",
        "rotation": 90,
        "fontsize": 20,
        "ha": "center",
        "va": "center",
        "fontweight": "bold",
    }

    annotation_kwargs.update(kwargs)
    return ax.annotate(text, **annotation_kwargs)


def animate_xarray_ds(
    ds,
    variables,
    x,
    y,
    animation_dim,
    title=None,
    ncols=None,
    plot_kwargs={},
    blit=False,
    save=None,
    fps=1,
    **kwargs,
):
    # For several of the arguments, we want to allow that they are passed as either a single value (to be applied
    # to all subplots) or a dictionary of values where each key corresponds to the name of a variable; so make
    # a function to make it a dict in both cases, and the code will then just assume a dict
    def maybe_map_to_variables(parameter):
        # The latter part of this if clause is for plot_kwargs, which is a dict; in that case we need to check the
        # keys of the dict, and we require that they all be names of variables
        if not isinstance(parameter, dict) or not all(
            [k in variables for k in parameter.keys()]
        ):
            return {variable: parameter for variable in variables}
        else:
            return parameter

    # Only some variables make sense to be on a variable-by-variable basis; x and y are fine to change for each
    # variable/subplot, and the plot_kwargs can change between them, but we don't allow the dataset to change because
    # the animation dimension size could be different, and we don't allow the animation_dim to change for the same
    # reason. title, ncols, blit, and save are all figure-wide parameters, so they have to be constant too.
    x = maybe_map_to_variables(x)
    y = maybe_map_to_variables(y)
    plot_kwargs = maybe_map_to_variables(plot_kwargs)

    n_variables = len(variables)
    ncols = ncols or min(n_variables, 3)  # Hardcode in a max number of columns
    nrows = n_variables // ncols
    if n_variables % ncols != 0:
        nrows += 1
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, squeeze=False, **kwargs)
    # Since squeeze=False above, axs is always a 2D array, so make it 1D
    axs = axs.flatten()
    # Remove any unneeded axes
    for ax in axs[n_variables:]:
        ax.remove()
    axs = axs[:n_variables]

    # Get variable bounds, either from passed values in plot_kwargs or by calculating them from the data
    # We won't do this if a norm was passed for this variable, since you can't specify vmin/vmax and a norm
    print("Calculating variable bounds...")
    variable_bounds = {}
    for variable in variables:
        if plot_kwargs.get(variable, {}).get("norm"):
            variable_bounds[variable] = {"norm": plot_kwargs[variable]["norm"]}
        else:
            this_variable_bounds = {}
            # We can skip calculating the variable bounds if we were given vmin or vmax
            if plot_kwargs.get(variable, {}).get("vmin") is not None:
                vmin = plot_kwargs[variable]["vmin"]
            else:
                vmin = ds[variable].min().values
            if plot_kwargs.get(variable, {}).get("vmax") is not None:
                vmax = plot_kwargs[variable]["vmax"]
            else:
                vmax = ds[variable].max().values
            variable_bounds[variable] = {"vmin": vmin, "vmax": vmax}
    print("Finished calculating variable bounds.")

    # Do the same for the limits
    xlims = {
        variable: this_kwargs["xlim"]
        for variable, this_kwargs in plot_kwargs.items()
        if "xlim" in this_kwargs
    }
    ylims = {
        variable: this_kwargs["ylim"]
        for variable, this_kwargs in plot_kwargs.items()
        if "ylim" in this_kwargs
    }

    # We're done with a bunch of the keywords, so remove them from plot_kwargs
    plot_kwargs = {
        k: {
            inner_k: inner_v
            for inner_k, inner_v in inner_dict.items()
            if inner_k not in ["vmin", "vmax", "norm", "xlim", "ylim"]
        }
        for k, inner_dict in plot_kwargs.items()
    }

    # Initialize empty "quad" colormesh artist for each axes
    quads = [
        axs[ix].pcolormesh(
            ds[x[variable]].values,
            ds[y[variable]].values,
            np.zeros((len(ds[y[variable]]), len(ds[x[variable]]))),
            **variable_bounds[variable],
            **plot_kwargs.get(variable, {}),
        )
        for ix, variable in enumerate(variables)
    ]
    # Make the colorbar axes
    caxs = [plt.colorbar(quad).ax for quad in quads]

    # Compute the title prefix
    title_prefix = title + ", " if title else ""

    def init():
        # Need to return an iterable of all of the artists to update; this is the quadmeshes themselves and the
        # figure title
        artists = []
        for variable_ix, variable in enumerate(variables):
            ax = axs[variable_ix]
            quad = quads[variable_ix]
            # We have to call set_array in the init method, even though we already initialized it with zeros;
            # it complains otherwise
            quad.set_array(np.zeros((len(ds[y[variable]]), len(ds[x[variable]]))))
            artists.append(quad)
            # Assuming that we don't need to update the x and y grids
            ax.set_title(variable)
            ax.set_xlabel(x[variable])
            ax.set_ylabel(y[variable])
            plt.colorbar(quad, cax=caxs[variable_ix])
            if variable in xlims:
                ax.set_xlim(xlims[variable])
            if variable in ylims:
                ax.set_ylim(ylims[variable])
        title = fig.suptitle("")
        artists.append(title)
        return artists

    def update(grouped):
        frame_val, this_frame_ds = grouped
        for variable_ix, variable in enumerate(variables):
            quads[variable_ix].set_array(this_frame_ds[variable].values)
        title_artist = fig.suptitle(
            title_prefix + f"{animation_dim} = {format_t_str(frame_val)}"
        )
        artists = [x for x in quads] + [title_artist]
        return artists

    print("Starting animation")
    anim = FuncAnimation(
        fig=fig,
        func=update,
        frames=tqdm(ds.groupby(animation_dim)),
        init_func=init,
        blit=blit,
    )

    if save:
        writer = mplanim.PillowWriter(fps=fps, bitrate=-1)
        anim.save(save, writer=writer)

    return anim


def shifted_colormap(cmap, new_range, n=256):
    if isinstance(cmap, str):
        cmap = mpl.colormaps[cmap]
    colors_list = cmap(np.linspace(new_range[0], new_range[1], n))
    return colors.LinearSegmentedColormap.from_list("new", colors_list)


# Define some shifted colormaps
shifted_blues = shifted_colormap("Blues", (0.2, 1.0))
shifted_greens = shifted_colormap("Greens", (0.3, 1.0))
shifted_oranges = shifted_colormap("Oranges", (0.3, 1.0))


def plot_sounding(ds):

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
    fig, name, dirs, no_title_version=False, resize_to_width=None, extensions=[".pdf"]
):
    """
    Save a figure into multiple directories (with the same filename). That's easily accomplished
    with a for loop on its own, but the no_title_version flag also allows for the creation of two copies
    of the figure in each directory: one with its suptitle (if present), and one without. These copies
    will be suffixed with 'title-yes' and 'title-no' respectively.
    """
    # Make the dirs argument a list if a single directory was passed
    if not isinstance(dirs, list):
        dirs = [dirs]

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
