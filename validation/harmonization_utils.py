"""
Shared machinery for the two "setup harmonization" notebooks.

Background
----------
The three storms in the manuscript (isolated convection, squall line, supercell)
were supposed to be set up apples-to-apples apart from a deliberate difference in
low-level moisture (11 g/kg for the isolated convection, 13 g/kg for the other
two). They are not. Three unintended differences crept in:

1. The supercell's pressure profile is anchored at 1000 hPa while the isolated
   convection's and squall line's are anchored at mean sea level, which offsets
   their temperature profiles by roughly 1 K at the same physical height.
2. The isolated convection's and squall line's stratospheric potential
   temperature structure is wrong, because the initial profile they were built
   from was not hydrostatically balanced.
3. The isolated convection's and squall line's low-level moisture is a few
   percent too high, because of an error in the mixing-ratio calculation.

Two independent "harmonizations" were run to test whether any of this matters:

* **Harmonization A ("on the squall line profile")** — rerun the supercell with
  the squall line's literal thermodynamic sounding (keeping the supercell's
  winds). All three storms then share the *paper* profile family, errors and all.
* **Harmonization B ("on the supercell profile")** — rerun the isolated
  convection and squall line on the supercell's (correct, hydrostatically
  balanced, 1000 hPa anchored) profile with the moisture calculation fixed. All
  three storms then share the *corrected* profile family.

The scientific question is not "did the numbers move" — of course they did. It is
whether the setup differences bias each storm *differently*, which would make the
storm-to-storm comparisons in the manuscript unfair. That means the quantities to
watch are the cross-storm ratios, not the per-storm absolute values.

What this module provides
-------------------------
The two notebooks are structurally identical and differ only in which runs make
up the "harmonized" arm, so everything they share lives here:

* A registry of every run directory involved, plus its derived-product paths.
* Initial-state (sounding) reading and comparison, generalized from the block
  that ``compare_wk_sounding.ipynb`` repeats verbatim eight times.
* Two-arm scaffolding for the results analysis, replacing the path-token
  substitution in ``validation_comparison.ipynb`` (which cannot express an arm
  whose runs live in unrelated directories and differ per storm).
* A coarse upward-mass-flux estimator, needed because the harmonized runs were
  written with ``FRQLITE = 0`` and so have no high-cadence lite output.
* Cross-storm summary-table construction, which is what actually answers the
  fairness question.
"""

########################################################################
############################## Imports ##############################
########################################################################

# Native packages
from contextlib import contextmanager
from pathlib import Path
import copy
import sys
import warnings

# Other packages
import matplotlib as mpl
import matplotlib.animation as mplanim
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpc
from metpy.plots import SkewT
from metpy.units import units

# The in-repo `cloudy` and `blt_utils` packages are installed (editable) into the
# pixi environment by pixi.toml. The path entry below is kept only so this module
# still imports `cloudy` from an environment that lacks that install.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _in_repo_package_dir in [
    _PROJECT_ROOT / "cloudy",
]:
    if str(_in_repo_package_dir) not in sys.path:
        sys.path.append(str(_in_repo_package_dir))

# User packages
import xarrams as xrr
import cloudy as cm
import blt_utils as blt

########################################################################
############################## Run registry ##############################
########################################################################

# Root of the RAMS output tree on moonbow. Every run directory below hangs off
# this, so a filesystem move only needs editing here.
RAMS_IO_ROOT = Path("/moonbow/cmdavis4/projects/bl_transport/rams_io")

# The storms, in the order they appear in every figure in the manuscript.
STORM_KEYS_IN_FIGURE_ORDER = ["ic_wk", "sl_wk", "sc_wk"]

# The three runs the manuscript was actually written from. These are the "before"
# arm in both notebooks.
PAPER_RUN_DIRS = {
    "ic_wk": (
        RAMS_IO_ROOT
        / "isolated_convection_wk/all_tracers"
        / "newtracer_forcing-warmbubble4_deltax-500_frqlite-2_nobulku"
    ),
    "sl_wk": (
        RAMS_IO_ROOT
        / "squall_line_wk/all_tracers"
        / "newtracer_blspinup-no_forcing-infinitecoldbubble4_deltax-500_frqlite-2"
    ),
    "sc_wk": (
        RAMS_IO_ROOT
        / "supercell_wk/all_tracers"
        / "newtracer_forcing-grant2014_dx-500_frqlite-2_stratwinds"
    ),
}

# Harmonization A: everything on the squall line's (paper, erroneous) profile.
# Only the supercell is rerun; the other two already use that profile family.
HARMONIZED_ON_SL_PROFILE_RUN_DIRS = {
    "ic_wk": PAPER_RUN_DIRS["ic_wk"],
    "sl_wk": PAPER_RUN_DIRS["sl_wk"],
    "sc_wk": RAMS_IO_ROOT / "from_cumulus/sc_wk/sl_profile",
}

# Harmonization B: everything on the supercell's (correct) profile. Only the
# isolated convection and squall line are rerun.
#
# Provenance, audited against the run directories on 2026-09-08. Do not take this on
# trust after any rerun; the checks are cheap and each one has caught something once.
#
# * The two rerun RAMSINs agree with their own originals on all 308 namelist variables
#   except three groups: file paths, ``FRQLITE`` (2 -> 0, so the reruns have no 2-second
#   lite output, which is why the coarse flux estimator exists), and the sounding itself
#   together with its format flags (``IPSFLG`` 0 -> 1, ``ITSFLG`` 0 -> 2, ``IRTSFLG``
#   3 -> 2). Grid, timestep, microphysics, and the initiating bubble are untouched.
# * The ``PS``/``TS``/``RTS`` arrays in both reruns are byte-identical to the supercell's,
#   and their t=0 output matches the supercell's to 0.001 hPa, 0.000 K and 0.015 J/kg.
# * Each rerun keeps its OWN winds. The format change forced the wind profile to be
#   respecified on height rather than pressure levels, so ``US`` is not byte-identical to
#   the original; it agrees to 0.07 m/s (isolated convection) and 0.21 m/s (squall line),
#   both at the single kink where the shear profile flattens near 3 km. Bulk shear over
#   0-1, 0-3 and 0-6 km agrees to four decimal places.
# * The squall line rerun was restarted from history at t = 110 min (``history_restarts/
#   hr1``) with ``ITRACHIST = 0``. That value is the correct one: ``ITRACHIST = 1`` on a
#   history restart makes ``rdint.f90`` call ``init_tracer``, which would wipe the tracer
#   fields. The accumulated fixed-source emission increments across the restart boundary
#   are identical to those either side of it, which confirms emission was uninterrupted.
# * The reruns sit about 13 hPa lower at the surface than their originals, because the
#   supercell profile is anchored at 1000 hPa rather than at mean sea level. Every mass
#   and particle-count total therefore carries a roughly 1% offset relative to the
#   original arm. Ratios in which the same air mass appears above and below the line are
#   unaffected; absolute totals should not be quoted to better than that.
#
# An earlier version of this harmonization used ``ic_wk/correct_wk_wellmixed_qv_fixed``
# and its squall line counterpart, which reran each storm on its own corrected profile
# rather than on the supercell's. Those directories still exist and are NOT what this
# dict points at. For the record, ``ic_wk/correct_wk_wellmixed`` and
# ``ic_wk/correct_wk_wellmixed_qv_fixed`` have byte-identical RAMSINs on disk but produce
# different output (surface q_v of 11.259 vs 11.015 g/kg), so one of those two RAMSINs no
# longer reflects what was run. Do not resurrect either directory without rechecking that.
HARMONIZED_ON_SC_PROFILE_RUN_DIRS = {
    "ic_wk": RAMS_IO_ROOT / "ic_wk/validation/other_forcing_bthp-2.0_radius-2x",
    "sl_wk": RAMS_IO_ROOT / "sl_wk/validation/sc_profile",
    "sc_wk": PAPER_RUN_DIRS["sc_wk"],
}

# The intended low-level water vapour mixing ratio for each storm, as designed.
# The isolated convection is deliberately drier; the other two are meant to match.
INTENDED_LOW_LEVEL_MIXING_RATIOS_G_KG = {
    "ic_wk": 11.0,
    "sl_wk": 13.0,
    "sc_wk": 13.0,
}

# Where the harmonization notebooks write their figures and figure data. Kept out
# of ``paper_figures/`` so a comparison run can never overwrite a manuscript
# figure.
HARMONIZATION_OUTPUT_ROOT = Path(__file__).parent / "harmonization_output"


def analysis_output_dir(run_dir):
    """Directory holding a run's 5-minute ``a-A-*.h5`` analysis files."""
    return Path(run_dir) / "output"


# The paper runs and the harmonized runs keep their derived products in different
# places, and this asymmetry is easy to get wrong in a way that fails silently.
#
# The harmonized runs each keep theirs in their own ``<run_dir>/derived/``. The
# paper runs do not have a populated ``derived/`` directory at all — theirs live in
# a single in-repo tree, ``replication/data/derived/<storm>/``, which is what the
# committed ``blt_utils.py`` points the manuscript notebook at.
#
# Getting this wrong would not raise: the original arm would simply find no cache,
# silently recompute every derived product from raw output (hours), and write the
# results into the manuscript's run directories. Hence the explicit registry.
#
# Verified rather than assumed: `replication/data/derived/sc_wk/` holds the
# *stratwinds* supercell products (its total_fs_tracer_nc sums to 5.2827e+19,
# matching `rams_io/derived/sc_wk/tracer_totals_profile_ds_stratwinds.nc` exactly
# and differing from the non-stratwinds file's 5.2974e+19), which is the run the
# manuscript used.
PAPER_DERIVED_ROOT = Path(blt.DERIVED_DIR)

PAPER_DERIVED_DIRS = {
    storm_key: PAPER_DERIVED_ROOT / storm_key
    for storm_key in STORM_KEYS_IN_FIGURE_ORDER
}


def derived_dir_for_run(run_dir, storm_key):
    """Where a given run's derived products live.

    Returns the in-repo paper tree for the three manuscript runs and the run's own
    ``derived/`` subdirectory for everything else. See the comment above
    ``PAPER_DERIVED_ROOT`` for why the two differ.
    """
    run_dir = Path(run_dir)
    if run_dir in {Path(p) for p in PAPER_RUN_DIRS.values()}:
        return PAPER_DERIVED_DIRS[storm_key]
    return run_dir / "derived"


def tracer_totals_profile_filepath(run_dir, storm_key):
    """Path to a run's in-updraft tracer vertical-profile dataset.

    This is the expensive derived product behind Figures 7 and 9 and the tracer
    entrainment statistics. The paper runs have theirs on disk already; for anything
    else ``read_or_compute_tracer_totals_datasets`` computes it and writes it here on
    first use.
    """
    return (
        derived_dir_for_run(run_dir, storm_key) / "tracer_totals_profile_ds.nc"
    )


def development_dataframe_filepath(run_dir, storm_key):
    """Path to a run's storm-development dataframe (Figure 10).

    Unlike the tracer profiles, this does *not* yet exist for the harmonized
    runs, so the notebooks compute and cache it here on first use.
    """
    return derived_dir_for_run(run_dir, storm_key) / "development_dataframe.pkl"


def upper_level_derived_dir(run_dir, storm_key):
    """Per-run cache directory for the upper-level derived products.

    ``blt`` caches the upper-level tracer totals and mass fluxes under a single
    ``blt.DERIVED_DIR`` whose path carries no run identity, so without redirecting
    it the second arm would silently read the first arm's cache. Giving each run
    its own subdirectory makes that mistake impossible rather than merely unlikely.
    """
    return derived_dir_for_run(run_dir, storm_key) / "upper_level"


########################################################################
############################## Initial state ##############################
########################################################################

# Reading a t=0 file and running the CAPE integration takes a few seconds, and the
# notebooks ask for the same runs repeatedly, so memoize on the run directory.
_INITIAL_SOUNDING_CACHE = {}


def read_initial_sounding_dataframe(
    run_dir, cape_z_max=4000, cape_stride=1, use_cache=True
):
    """Read a run's t=0 initial state as a sounding dataframe with derived vars.

    Mechanism: the only trustworthy record of what a run actually started with is
    its own t=0 analysis file, not its RAMSIN — one of the RAMSINs in this study
    demonstrably no longer matches its output. So we read
    ``a-A-*-210000-g1.h5``, let ``xarrams`` take the horizontal mean (the initial
    state is horizontally homogeneous, so this is exact rather than a
    simplification), and convert to the (z, PS, TS, RTS, US, VS) sounding frame.

    ``calculate_sounding_derived_vars`` then adds q_v, theta, theta_v, theta_e,
    lapse rates, LCL, and a CAPE/CIN profile. The CAPE profile is the expensive
    part: it runs one moist-adiabat parcel integration per *starting* level, so we
    restrict starting levels to ``z <= cape_z_max`` (the integration itself always
    runs to the top of the column, so this does not truncate the positive area).

    Args:
        run_dir: A run directory (the parent of ``output/``).
        cape_z_max: Highest parcel starting height for the CAPE profile, in m.
        cape_stride: Subsample factor over parcel starting levels.
        use_cache: Reuse a previously read result for the same run directory.

    Returns:
        A pandas DataFrame, one row per model level, ascending in height.
    """
    run_dir = Path(run_dir)
    if use_cache and run_dir in _INITIAL_SOUNDING_CACHE:
        return _INITIAL_SOUNDING_CACHE[run_dir].copy()

    # Grab the single t=0 analysis file. Simulations all start at 21:00:00.
    initial_analysis_filepaths = sorted(
        analysis_output_dir(run_dir).glob("a-A-*-210000-g1.h5")
    )
    if not initial_analysis_filepaths:
        raise FileNotFoundError(f"No t=0 analysis file found under {run_dir}")

    # Read it, then collapse to a single column.
    initial_state_ds = xrr.read_rams_output(
        initial_analysis_filepaths[:1], silent=True
    )
    sounding_df = xrr.to_sounding_df(initial_state_ds.squeeze())

    # Add the thermodynamic diagnostics we actually compare on.
    sounding_df = xrr.calculate_sounding_derived_vars(
        sounding_df, cape=True, cape_z_max=cape_z_max, cape_stride=cape_stride
    )

    if use_cache:
        _INITIAL_SOUNDING_CACHE[run_dir] = sounding_df.copy()
    return sounding_df


def read_initial_soundings_for_arm(run_dirs, **kwargs):
    """Read t=0 soundings for every storm in one arm.

    Args:
        run_dirs: Mapping of storm key -> run directory.

    Returns:
        Mapping of storm key -> sounding DataFrame, in figure order.
    """
    return {
        storm_key: read_initial_sounding_dataframe(
            run_dirs[storm_key], **kwargs
        )
        for storm_key in STORM_KEYS_IN_FIGURE_ORDER
        if storm_key in run_dirs
    }


# The sounding fields worth comparing between setups, with display labels. These
# are the columns produced by ``xrr.calculate_sounding_derived_vars`` plus the raw
# state variables, chosen to expose each of the three suspected errors:
#   PS          -> the 1000 hPa vs MSL pressure anchoring
#   TS, theta   -> the ~1 K temperature offset and the stratospheric theta error
#   q_v, RTS    -> the few-percent low-level moisture error
#   cape, cin   -> whether any of it changes the storm's energy budget
SOUNDING_COMPARISON_VARIABLES = {
    "PS": "Pressure (hPa)",
    "TS": "Temperature (°C)",
    "theta": r"$\theta$ (K)",
    "theta_e": r"$\theta_e$ (K)",
    "q_v": r"$q_v$ (g/kg)",
    "RTS": "RH (%)",
    "dewpoint": "Dewpoint (°C)",
    "theta_lapse_rate": r"$\Delta\theta$ (K/km)",
    "cape": "CAPE (J/kg)",
    "cin": "CIN (J/kg)",
    "lcl": "LCL height (m)",
}

# Height windows each comparison figure is drawn over. Full column first, then the
# depth the manuscript's analysis actually lives in, then the boundary layer.
DEFAULT_SOUNDING_Z_CUTOFFS_M = [np.inf, 10_000, 2_000]


def plot_sounding_comparison(
    labelled_sounding_dfs,
    variables_to_plot=None,
    z_cutoffs=None,
    panel_height=2.4,
    panel_width=3.6,
    suptitle=None,
    y_variable="z",
):
    """Overlay several soundings, one row per variable, one column per z window.

    Mechanism: this is the generalization of the plotting block that
    ``compare_wk_sounding.ipynb`` repeats verbatim for each pair of runs. Rather
    than one call per pair, it takes an arbitrary number of labelled soundings and
    overlays them all, so the three storms in an arm can be read against each
    other in a single figure — which is the comparison this study actually needs.

    Reading it across a row shows *where* in the column two setups disagree; the
    narrowing z windows separate a stratospheric-only difference (visible only in
    the first column) from one that reaches into the storm-bearing layer.

    **Choosing the vertical coordinate matters here.** With ``y_variable="z"`` the
    profiles are compared at the same *physical height*, which is what makes the
    pressure-anchoring error visible: anchoring one storm at 1000 hPa and another
    at mean sea level offsets their temperature by ~1 K at a given height. With
    ``y_variable="PS"`` the profiles are instead compared at the same *pressure*,
    which removes that offset by construction — so any difference that survives is
    a genuine thermodynamic difference over and above the anchoring. Running both
    is how you separate the two.

    Note that the height windows still filter on ``z`` regardless of the vertical
    coordinate plotted, so "z ≤ 10 km" means the same set of levels in both views
    and the panels stay directly comparable.

    Args:
        labelled_sounding_dfs: Mapping of display label -> sounding DataFrame.
        variables_to_plot: Mapping of column name -> axis label. Defaults to
            ``SOUNDING_COMPARISON_VARIABLES``.
        z_cutoffs: Upper height bounds, one per column of panels.
        panel_height: Height of a single panel, in inches.
        panel_width: Width of a single panel, in inches.
        suptitle: Optional figure title.
        y_variable: Sounding column to use as the vertical coordinate, ``"z"``
            (metres) or ``"PS"`` (hPa). A pressure axis is inverted automatically.

    Returns:
        (fig, axs) with axs shaped (n_variables, n_z_cutoffs).
    """
    variables_to_plot = variables_to_plot or SOUNDING_COMPARISON_VARIABLES
    z_cutoffs = z_cutoffs or DEFAULT_SOUNDING_Z_CUTOFFS_M

    # Plotting the vertical coordinate against itself is a trivial monotonic line,
    # so drop it from the rows rather than waste a panel on it.
    plotting_against_pressure = y_variable == "PS"
    if y_variable in variables_to_plot:
        variables_to_plot = {
            variable: label
            for variable, label in variables_to_plot.items()
            if variable != y_variable
        }

    fig, axs = plt.subplots(
        nrows=len(variables_to_plot),
        ncols=len(z_cutoffs),
        figsize=(
            panel_width * len(z_cutoffs),
            panel_height * len(variables_to_plot),
        ),
        layout="constrained",
        squeeze=False,
    )

    # Setups in this study very often lie *exactly* on top of one another — that is
    # the whole point of a harmonization — and a solid line drawn over another solid
    # line is indistinguishable from a single setup. Give each series its own dash
    # pattern and taper the width, so a hidden curve reads as "coincident with the
    # one above it" rather than as "missing".
    line_styles = ["solid", (0, (6, 2)), (0, (2, 2)), (0, (4, 2, 1, 2))]
    n_series = len(labelled_sounding_dfs)

    for series_ix, (label, full_sounding_df) in enumerate(
        labelled_sounding_dfs.items()
    ):
        line_style = line_styles[series_ix % len(line_styles)]
        line_width = 2.6 - 0.7 * series_ix / max(n_series - 1, 1)
        for variable_ix, variable in enumerate(variables_to_plot):
            # A variable missing from one sounding shouldn't kill the whole figure.
            if variable not in full_sounding_df.columns:
                continue
            for cutoff_ix, z_cutoff in enumerate(z_cutoffs):
                ax = axs[variable_ix, cutoff_ix]
                # Restrict to this column's height window before plotting so the
                # x limits autoscale to the window rather than to the full column.
                # Window on height even when plotting against pressure, so the same
                # set of model levels appears in both views of the same figure.
                windowed_df = full_sounding_df.loc[
                    full_sounding_df["z"] <= z_cutoff
                ]
                ax.plot(
                    windowed_df[variable],
                    windowed_df[y_variable],
                    label=label,
                    linestyle=line_style,
                    linewidth=line_width,
                )

    # Label once the data are down, so titles reflect the requested windows.
    for variable_ix, (variable, axis_label) in enumerate(
        variables_to_plot.items()
    ):
        for cutoff_ix, z_cutoff in enumerate(z_cutoffs):
            ax = axs[variable_ix, cutoff_ix]
            window_label = (
                "full column"
                if not np.isfinite(z_cutoff)
                else f"z ≤ {z_cutoff/1000:g} km"
            )
            ax.set_title(f"{axis_label} — {window_label}", fontsize=11)
            if cutoff_ix == 0:
                ax.set_ylabel(
                    "p (hPa)" if plotting_against_pressure else "z (m)"
                )
            # Pressure decreases upward, so a pressure axis reads correctly only
            # when inverted. Invert once per axes, after all series are drawn.
            if plotting_against_pressure and not ax.yaxis_inverted():
                ax.invert_yaxis()

    # One legend for the whole figure; the same lines appear in every panel.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper right", frameon=False)
    if suptitle:
        fig.suptitle(suptitle, fontsize=16)

    return fig, axs


# Variables the difference-profile figure draws, with the unit a *difference* in each
# carries. Deliberately shorter than SOUNDING_COMPARISON_VARIABLES: the coordinate itself
# and the parcel diagnostics (CAPE, CIN, LCL) are not per-level thermodynamic state, so
# differencing them level by level would not mean anything.
SOUNDING_DIFFERENCE_VARIABLES = {
    "TS": "ΔT (K)",
    "theta": r"$\Delta\theta$ (K)",
    "theta_e": r"$\Delta\theta_e$ (K)",
    "q_v": r"$\Delta q_v$ (g/kg)",
    "RTS": "ΔRH (%)",
    "dewpoint": "ΔT$_d$ (K)",
}


def plot_sounding_difference_profiles(
    labelled_sounding_dfs,
    reference_label,
    variables_to_plot=None,
    pressure_limits_hPa=(1050.0, 100.0),
    x_scale_pressure_window_hPa=(1000.0, 200.0),
    panel_width=3.0,
    panel_height=5.4,
    suptitle=None,
):
    """Plot each sounding minus a reference sounding, against pressure.

    Mechanism, and why this figure is needed on top of the overlay figures. An overlay
    plots absolute profiles, so its x axis has to span the whole range the variable takes
    over the column — around 87 K for temperature, and around 60 K for potential
    temperature. Differences between these setups are a few tenths of a Kelvin over most of
    the troposphere. A few tenths out of 87 is thinner than the line used to draw it, so an
    overlay reports "identical" for a difference that is worth hundreds of joules per
    kilogram of CAPE. Differencing first and *then* plotting lets the x axis autoscale to
    the difference, which is the only way tenths of a Kelvin become visible.

    Every sounding is interpolated onto the reference's **pressure** levels before
    differencing, not its height levels. That choice is what separates the two error
    mechanisms in this study. Two profiles built from one recipe but anchored at different
    surface pressures differ at constant height and agree at constant pressure. So anything
    that survives this figure is a genuine thermodynamic difference, over and above the
    anchoring. ``np.interp`` needs increasing sample points and pressure falls with height,
    so both arrays are reversed going in and the result is reversed coming out.

    Args:
        labelled_sounding_dfs: Mapping of display label -> sounding DataFrame. The
            reference is plotted as a flat zero line, which is a useful visual anchor.
        reference_label: Which entry everything is differenced against.
        variables_to_plot: Mapping of column name -> x-axis label. Defaults to
            ``SOUNDING_DIFFERENCE_VARIABLES``.
        pressure_limits_hPa: (bottom, top) crop of the vertical axis.
        x_scale_pressure_window_hPa: (bottom, top) layer the horizontal scale is set from.
            This is separate from the vertical crop on purpose. These setups disagree by
            more than 15 K in the stratosphere and by a few tenths of a Kelvin through the
            troposphere. Letting the stratosphere set the scale reduces the troposphere to a
            flat line again, which reintroduces the exact problem the figure exists to
            remove. So the whole column is still drawn, and the curves are allowed to run
            off the side of the panel above the window. Pass None to autoscale on everything.
        panel_width: Width of one variable's panel, in inches.
        panel_height: Height of the panel row, in inches.
        suptitle: Optional figure title.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (1, number of variables).
    """
    variables_to_plot = variables_to_plot or SOUNDING_DIFFERENCE_VARIABLES
    reference_df = labelled_sounding_dfs[reference_label]
    reference_pressures_hPa = np.asarray(reference_df["PS"].values, dtype=float)

    fig, axs = plt.subplots(
        nrows=1,
        ncols=len(variables_to_plot),
        figsize=(panel_width * len(variables_to_plot), panel_height),
        layout="constrained",
        squeeze=False,
        sharey=True,
    )

    # Same dash-and-taper convention the overlay figures use, so a curve hidden underneath
    # another reads as "coincident with it" rather than as "missing".
    line_styles = ["solid", (0, (6, 2)), (0, (2, 2)), (0, (4, 2, 1, 2))]
    n_series = len(labelled_sounding_dfs)

    for series_ix, (label, sounding_df) in enumerate(
        labelled_sounding_dfs.items()
    ):
        line_style = line_styles[series_ix % len(line_styles)]
        line_width = 2.6 - 0.7 * series_ix / max(n_series - 1, 1)
        for variable_ix, variable in enumerate(variables_to_plot):
            if variable not in sounding_df.columns:
                continue
            ax = axs[0, variable_ix]
            if label == reference_label:
                # The reference against itself is identically zero. Draw it anyway, so the
                # legend lists every setup and the zero line is visibly the reference.
                difference = np.zeros_like(reference_pressures_hPa)
            else:
                on_reference_pressures = np.interp(
                    reference_pressures_hPa[::-1],
                    np.asarray(sounding_df["PS"].values, dtype=float)[::-1],
                    np.asarray(sounding_df[variable].values, dtype=float)[::-1],
                )[::-1]
                difference = on_reference_pressures - np.asarray(
                    reference_df[variable].values, dtype=float
                )
            ax.plot(
                difference,
                reference_pressures_hPa,
                label=label,
                linestyle=line_style,
                linewidth=line_width,
            )

    # Levels the horizontal scale is allowed to see, chosen once and reused for every panel.
    if x_scale_pressure_window_hPa is None:
        inside_scaling_window = np.ones_like(
            reference_pressures_hPa, dtype=bool
        )
    else:
        scaling_bottom_hPa, scaling_top_hPa = x_scale_pressure_window_hPa
        inside_scaling_window = (
            reference_pressures_hPa <= scaling_bottom_hPa
        ) & (reference_pressures_hPa >= scaling_top_hPa)

    for variable_ix, (variable, axis_label) in enumerate(
        variables_to_plot.items()
    ):
        ax = axs[0, variable_ix]
        # Zero line, so "no difference" is a place on the page rather than a number.
        ax.axvline(0.0, color="0.4", linewidth=0.8, zorder=0)
        ax.set_xlabel(axis_label, fontsize=10)
        ax.tick_params(labelsize=8)
        ax.set_ylim(*pressure_limits_hPa)
        if variable_ix == 0:
            ax.set_ylabel("p (hPa)")

        # Set the scale from the window rather than from the drawn data. Always keep zero
        # in view, so a curve that never leaves the reference is still recognisable as one
        # that sits on the zero line rather than as one whose panel happens to be blank.
        differences_inside_window = np.concatenate(
            [
                np.asarray(line.get_xdata(), dtype=float)[inside_scaling_window]
                for line in ax.get_lines()
                if len(line.get_xdata()) == len(reference_pressures_hPa)
            ]
            or [np.zeros(1)]
        )
        finite_differences = differences_inside_window[
            np.isfinite(differences_inside_window)
        ]
        if finite_differences.size:
            lower_limit = min(float(finite_differences.min()), 0.0)
            upper_limit = max(float(finite_differences.max()), 0.0)
            # A margin proportional to the range, with a floor so an all-zero panel still
            # gets a sensible width instead of collapsing to a single point.
            margin = max(0.08 * (upper_limit - lower_limit), 1e-3)
            ax.set_xlim(lower_limit - margin, upper_limit + margin)

    # Legend below the panels. Above, it collides with the suptitle under constrained layout.
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=n_series,
        frameon=False,
    )
    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    return fig, axs


# Poisson exponent R_d / c_p, which is what converts a potential-temperature difference at a
# fixed pressure into a temperature difference at that same pressure.
_POISSON_EXPONENT = 0.2854

# Reference pressure of the potential-temperature definition, in hPa.
_POTENTIAL_TEMPERATURE_REFERENCE_PRESSURE_HPA = 1000.0


def anchoring_versus_theta_profile_table(
    labelled_sounding_dfs,
    reference_label,
    pressure_window_hPa=(850.0, 200.0),
):
    """Split the temperature difference at constant pressure into its two causes.

    The problem this solves. Two of the setups in this study differ in two ways at once,
    and the two push temperature in opposite directions, so the net difference understates
    both. The first cause is **anchoring**: one profile's surface sits at 1013.25 hPa and
    the other's at 1000 hPa, so a given pressure corresponds to a different height in each,
    and the profiles are sampled at different points along the same curve. The second cause
    is a genuinely different **potential-temperature profile** theta(z) — the real
    thermodynamic difference, and the one that moves CAPE.

    The mechanism of the split. Let z_c and z_r be the heights at which the comparison and
    reference soundings reach one common pressure p. Poisson's equation ties temperature to
    potential temperature at fixed pressure, so the temperature difference at p is

        dT|_p = [theta_c(z_c) - theta_r(z_r)] * (p / 1000 hPa) ** (R_d / c_p)

    Adding and subtracting the reference profile evaluated at the *comparison's* height,
    theta_r(z_c), splits that bracket into two terms that are separately meaningful:

        theta_c(z_c) - theta_r(z_c)     the profiles genuinely differ at one height
        theta_r(z_c) - theta_r(z_r)     the same profile, read at two different heights

    The first term is the theta(z) difference. The second is the anchoring, because z_c and
    z_r differ only because the two columns reach a given pressure at different heights.
    Both are converted to Kelvin of temperature by the same Poisson factor, so they add to
    the observed difference exactly, and the residual column below is a check on that rather
    than a third effect.

    Why the sign of each term matters more than its size. If the two terms have opposite
    signs, then removing the anchoring alone would make the total *larger*, not smaller,
    and any argument that treats the observed difference as an upper bound on the real
    thermodynamic difference is wrong.

    Args:
        labelled_sounding_dfs: Mapping of display label -> sounding DataFrame. Each needs
            ``PS`` in hPa, ``z`` in metres and ``theta`` in Kelvin.
        reference_label: Which entry the others are differenced against.
        pressure_window_hPa: (bottom, top) layer the means are taken over. Defaults to the
            free troposphere, which is where the CAPE integral has most of its weight and
            where the boundary layer's own differences do not contaminate the answer.

    Returns:
        A DataFrame indexed by comparison label, with the mean observed difference over the
        window and the two terms it splits into, all in Kelvin.
    """
    reference_df = labelled_sounding_dfs[reference_label]
    reference_pressures_hPa = np.asarray(reference_df["PS"].values, dtype=float)
    reference_heights_m = np.asarray(reference_df["z"].values, dtype=float)
    reference_theta_K = np.asarray(reference_df["theta"].values, dtype=float)

    # Reference theta as a function of height, so it can be re-read at the comparison's
    # heights. Sample points must increase, and height already does.
    def reference_theta_at_height(heights_m):
        return np.interp(heights_m, reference_heights_m, reference_theta_K)

    bottom_pressure_hPa, top_pressure_hPa = pressure_window_hPa
    inside_window = (reference_pressures_hPa <= bottom_pressure_hPa) & (
        reference_pressures_hPa >= top_pressure_hPa
    )

    # Poisson factor at each reference pressure level, applied to every theta difference.
    poisson_factor = (
        reference_pressures_hPa / _POTENTIAL_TEMPERATURE_REFERENCE_PRESSURE_HPA
    ) ** _POISSON_EXPONENT

    table_rows = []
    for label, sounding_df in labelled_sounding_dfs.items():
        if label == reference_label:
            continue
        comparison_pressures_hPa = np.asarray(
            sounding_df["PS"].values, dtype=float
        )

        # The comparison sounding, resampled onto the reference's pressure levels. Reverse
        # both arrays going in because pressure decreases with height.
        def on_reference_pressures(values):
            return np.interp(
                reference_pressures_hPa[::-1],
                comparison_pressures_hPa[::-1],
                np.asarray(values, dtype=float)[::-1],
            )[::-1]

        comparison_theta_K = on_reference_pressures(sounding_df["theta"].values)
        comparison_heights_m = on_reference_pressures(sounding_df["z"].values)

        # Term 1: the profiles genuinely disagree, compared at one common height.
        theta_profile_difference_K = (
            comparison_theta_K - reference_theta_at_height(comparison_heights_m)
        )
        # Term 2: the same reference profile, read at the two columns' different heights.
        anchoring_difference_K = (
            reference_theta_at_height(comparison_heights_m) - reference_theta_K
        )
        # What is actually observed at constant pressure.
        observed_difference_K = (
            comparison_theta_K - reference_theta_K
        ) * poisson_factor

        table_rows.append({
            "comparison": label,
            "observed_dT_K": float(
                np.nanmean(observed_difference_K[inside_window])
            ),
            "theta_profile_term_K": float(
                np.nanmean(
                    (theta_profile_difference_K * poisson_factor)[inside_window]
                )
            ),
            "anchoring_term_K": float(
                np.nanmean(
                    (anchoring_difference_K * poisson_factor)[inside_window]
                )
            ),
            "height_offset_m": float(
                np.nanmean(
                    (comparison_heights_m - reference_heights_m)[inside_window]
                )
            ),
        })

    table = pd.DataFrame(table_rows).set_index("comparison")
    # Residual, which should be numerical noise. It is kept in the table rather than
    # asserted on, because a non-zero value here means the interpolation has gone wrong and
    # the reader should see that rather than have the function raise in the middle of a run.
    table["residual_K"] = (
        table["observed_dT_K"]
        - table["theta_profile_term_K"]
        - table["anchoring_term_K"]
    )
    return table


def interpolate_sounding_to_heights(sounding_df, target_heights_m, variables):
    """Resample a sounding onto a common height grid so setups can be differenced.

    All the runs in this study use the same 108-level vertical grid, so this is
    usually a no-op. Doing it anyway means the difference tables stay correct if a
    run with a different grid is ever added, instead of silently differencing
    mismatched levels.
    """
    resampled = {"z": np.asarray(target_heights_m, dtype=float)}
    for variable in variables:
        if variable not in sounding_df.columns:
            continue
        resampled[variable] = np.interp(
            target_heights_m,
            sounding_df["z"].values,
            np.asarray(sounding_df[variable].values, dtype=float),
        )
    return pd.DataFrame(resampled)


def sounding_difference_table(
    labelled_sounding_dfs,
    reference_label,
    z_max=np.inf,
    variables=None,
):
    """Tabulate how far each sounding sits from a reference, over a height window.

    Mechanism: profile overlays show *where* setups differ but are a poor way to
    answer "by how much". This differences every sounding against one chosen
    reference on a shared height grid (the reference's own levels) and reports,
    per variable, the maximum absolute difference, the height at which it occurs,
    the mean absolute difference, and the difference at the lowest model level.

    The height at the maximum is the load-bearing column: a 35 K theta difference
    is alarming at 5 km and irrelevant at 30 km, and only this table distinguishes
    the two.

    Args:
        labelled_sounding_dfs: Mapping of display label -> sounding DataFrame.
        reference_label: Key in that mapping to difference everything against.
        z_max: Only consider levels at or below this height, in m.
        variables: Column names to tabulate. Defaults to the comparison set.

    Returns:
        A DataFrame indexed by (label, variable).
    """
    variables = list(variables or SOUNDING_COMPARISON_VARIABLES)
    reference_df = labelled_sounding_dfs[reference_label]

    # Use the reference's own levels within the window as the common grid.
    common_heights = reference_df.loc[reference_df["z"] <= z_max, "z"].values
    reference_on_grid = interpolate_sounding_to_heights(
        reference_df, common_heights, variables
    )

    table_rows = []
    for label, sounding_df in labelled_sounding_dfs.items():
        if label == reference_label:
            continue
        comparison_on_grid = interpolate_sounding_to_heights(
            sounding_df, common_heights, variables
        )
        for variable in variables:
            if variable not in comparison_on_grid.columns:
                continue
            difference = (
                comparison_on_grid[variable].values
                - reference_on_grid[variable].values
            )
            # CAPE/CIN are NaN above the parcel-start cutoff, so ignore those levels.
            finite = np.isfinite(difference)
            if not finite.any():
                continue
            absolute_difference = np.abs(np.where(finite, difference, np.nan))
            height_of_max_difference_ix = int(np.nanargmax(absolute_difference))
            table_rows.append({
                "comparison": label,
                "variable": variable,
                "max_abs_diff": float(np.nanmax(absolute_difference)),
                "z_of_max_abs_diff_m": float(
                    common_heights[height_of_max_difference_ix]
                ),
                "mean_abs_diff": float(np.nanmean(absolute_difference)),
                "diff_at_lowest_level": float(difference[np.argmax(finite)]),
            })

    return pd.DataFrame(table_rows).set_index(["comparison", "variable"])


def low_level_moisture_table(
    labelled_sounding_dfs, intended_mixing_ratios_g_kg, sample_height_m=100.0
):
    """Compare each setup's actual low-level q_v against what it was meant to be.

    The moisture bug is a few-percent error, which is invisible on a profile
    overlay and obvious in a table. Sampling just above the surface (rather than
    at the lowest model level) avoids any surface-layer artefact while staying
    well inside the well-mixed boundary layer, where q_v is constant with height.

    Args:
        labelled_sounding_dfs: Mapping of storm key -> sounding DataFrame.
        intended_mixing_ratios_g_kg: Mapping of storm key -> design value in g/kg.
        sample_height_m: Height at which to read q_v, in m.

    Returns:
        A DataFrame indexed by storm key.
    """
    table_rows = []
    for storm_key, sounding_df in labelled_sounding_dfs.items():
        # Interpolate rather than take the nearest level, so the sample height
        # means the same thing regardless of where grid levels happen to fall.
        actual_mixing_ratio = float(
            np.interp(
                sample_height_m,
                sounding_df["z"].values,
                sounding_df["q_v"].values,
            )
        )
        intended_mixing_ratio = intended_mixing_ratios_g_kg.get(
            storm_key, np.nan
        )
        table_rows.append({
            "storm": storm_key,
            "intended_q_v_g_kg": intended_mixing_ratio,
            "actual_q_v_g_kg": actual_mixing_ratio,
            "abs_error_g_kg": actual_mixing_ratio - intended_mixing_ratio,
            "percent_error": (
                100.0
                * (actual_mixing_ratio - intended_mixing_ratio)
                / intended_mixing_ratio
            ),
        })
    return pd.DataFrame(table_rows).set_index("storm")


def temperature_offset_at_fixed_height_table(
    labelled_sounding_dfs,
    reference_label,
    sample_heights_m=(100, 1000, 3000, 5000),
):
    """Tabulate the temperature offset between setups at fixed physical heights.

    This is the diagnostic for the pressure-anchoring error specifically. Anchoring
    one storm's profile at 1000 hPa and another's at mean sea level leaves their
    *potential* temperatures nearly identical while shifting actual temperature at
    a given height — so comparing theta hides the error and comparing T exposes it.
    Sampling at several heights shows whether the offset is a constant shift (a
    pure pressure-anchor effect) or grows with height (a lapse-rate difference).
    """
    reference_df = labelled_sounding_dfs[reference_label]
    table_rows = []
    for label, sounding_df in labelled_sounding_dfs.items():
        row = {"comparison": label}
        for sample_height in sample_heights_m:
            temperature_here = np.interp(
                sample_height, sounding_df["z"].values, sounding_df["TS"].values
            )
            reference_temperature_here = np.interp(
                sample_height,
                reference_df["z"].values,
                reference_df["TS"].values,
            )
            row[f"dT_at_{sample_height}m_K"] = float(
                temperature_here - reference_temperature_here
            )
        table_rows.append(row)
    return pd.DataFrame(table_rows).set_index("comparison")


########################################################################
############################## Results scaffolding ##############################
########################################################################

# The ``blt`` module-level constants that point at run-specific data. Both arms
# need these swapped, and ``blt`` is imported by other notebooks in this repo, so
# we snapshot the originals and always restore them.
# Note that blt.DERIVED_DIR is deliberately absent: blt's constants derived from it
# (UL_TRACER_TOTALS_FILEPATHS, UL_FLUXES_DIR) are bound at import time, so rebinding
# it per arm would change nothing. That is precisely why this module resolves every
# per-run derived path explicitly instead of routing them through blt.DERIVED_DIR.
_BLT_RUN_SPECIFIC_PATH_ATTRIBUTES = [
    "RAMS_OUTPUT_DIRS",
    "TRACER_TOTALS_PROFILE_FILEPATHS",
    "DEVELOPMENT_DATAFRAME_FILEPATHS",
    "FIGURE_DATA_DIR",
    "STORM_NAMES",
]


@contextmanager
def blt_paths_for_arm(run_dirs, figure_data_dir=None):
    """Temporarily point ``blt``'s path constants at one arm's runs.

    Mechanism: ``validation_comparison.ipynb`` switches arms by substituting a
    single token in every path (``fixed-False`` -> ``fixed-True``). That works
    only when the two arms differ by one uniform path component. Here each arm is
    an arbitrary per-storm mapping — Harmonization A changes only the supercell,
    and to a directory that shares no path structure with the others — so token
    substitution cannot express it. This sets the paths explicitly instead.

    Restoring on exit (including on exception) matters because ``blt_utils.py`` is
    imported by the manuscript notebook in the same repo; leaving it mutated is how
    an analysis silently reads the wrong run.

    Args:
        run_dirs: Mapping of storm key -> run directory for this arm.
        figure_data_dir: Where figure-data pickles should be written. Defaults to
            leaving ``blt.FIGURE_DATA_DIR`` untouched.

    Yields:
        None; ``blt``'s constants are correct for the duration of the block.
    """
    original_attribute_values = {
        attribute: copy.deepcopy(getattr(blt, attribute))
        for attribute in _BLT_RUN_SPECIFIC_PATH_ATTRIBUTES
    }
    try:
        # The working copy of blt_utils.py has STORM_NAMES trimmed to one storm
        # for an unrelated debugging session; the comparison always wants all three.
        blt.STORM_NAMES = list(STORM_KEYS_IN_FIGURE_ORDER)
        blt.RAMS_OUTPUT_DIRS = {
            storm_key: analysis_output_dir(run_dir)
            for storm_key, run_dir in run_dirs.items()
        }
        blt.TRACER_TOTALS_PROFILE_FILEPATHS = {
            storm_key: tracer_totals_profile_filepath(run_dir, storm_key)
            for storm_key, run_dir in run_dirs.items()
        }
        blt.DEVELOPMENT_DATAFRAME_FILEPATHS = {
            storm_key: development_dataframe_filepath(run_dir, storm_key)
            for storm_key, run_dir in run_dirs.items()
        }
        if figure_data_dir is not None:
            Path(figure_data_dir).mkdir(parents=True, exist_ok=True)
            blt.FIGURE_DATA_DIR = Path(figure_data_dir)
        yield
    finally:
        for attribute, value in original_attribute_values.items():
            setattr(blt, attribute, value)


def read_storm_datasets(run_dirs, analysis_filename_glob="a-A-*.h5"):
    """Read and postprocess the 5-minute analysis output for one arm's storms.

    Mechanism: identical to the "5 minute RAMS output" cell in
    ``validation_comparison.ipynb`` — read every ``a-A-*.h5`` for a storm, then run
    ``blt.postprocess_storm_ds`` to attach the derived thermodynamics, the
    in-updraft and in-anvil flags, the per-storm ``in_analysis`` flag (which for the
    supercell is the right-mover mask), and the number-concentration tracer
    variables.

    Each arm reads its own datasets even where the two arms point at the same run,
    rather than sharing one object between them. That looks wasteful and is not:
    ``read_rams_output`` is dask-backed and lazy, so the duplicated work is file
    metadata, not data. Sharing the object instead would hand both arms the same
    mutable Dataset, which later gets written into in-place (storm centroid
    positions), and a shared mutable dataset across arms is exactly the kind of
    cross-contamination this comparison is supposed to rule out.

    Args:
        run_dirs: Mapping of storm key -> run directory.
        analysis_filename_glob: Glob for the analysis files to read.

    Returns:
        Mapping of storm key -> postprocessed xarray Dataset.
    """
    storm_dss = {}
    for storm_key in STORM_KEYS_IN_FIGURE_ORDER:
        if storm_key not in run_dirs:
            continue
        output_dir = analysis_output_dir(run_dirs[storm_key])

        analysis_filepaths = sorted(output_dir.glob(analysis_filename_glob))
        if not analysis_filepaths:
            raise FileNotFoundError(
                f"No files matching {analysis_filename_glob} in {output_dir}"
            )

        rams_ds = cm.rams.read_rams_output(
            input_filenames=analysis_filepaths,
            time_dim_name="time",
            units=False,
        )
        # Attaches derived thermodynamics, in_updraft, in_analysis, tracer _nc vars.
        storm_dss[storm_key] = blt.postprocess_storm_ds(rams_ds, storm_key)

    return storm_dss


def read_or_cache_dataframe(
    cache_path, compute_dataframe, *args, expected_row_count=None, **kwargs
):
    """Read a cached DataFrame if present, otherwise compute it and cache it.

    Why this rather than ``cm.utils.read_or_cache_to``: that decorator cannot
    actually write ``.pkl`` files. ``cm.utils.write_file`` dispatches ``.pkl`` to a
    ``to_pkl`` attribute that neither dict nor DataFrame has (pandas spells it
    ``to_pickle``), leaves its ``wrote`` flag False, and raises — and
    ``read_or_cache_to`` swallows that with ``fail_ok=True``. The result is a cache
    that silently never populates, so every rerun recomputes from scratch while
    printing that it wrote the file.

    Using pandas' own pickling directly avoids the whole path. Reading stays
    compatible with the caches the manuscript analysis already wrote.

    Staleness: these caches are keyed only by a file path, so a cache written while
    a run was still integrating stays on disk and keeps being read after the run
    finishes and grows more output times. That is not a hypothetical — the
    supercell-on-squall-line-profile run cached a 9-time upper-level dataframe
    partway through a 37-time simulation, and every consumer downstream then tried
    to divide a 9-element series by a 37-element one. Pass *expected_row_count* to
    make that condition self-correcting: a cache whose row count disagrees is
    discarded and recomputed rather than returned. It is a cheap check because the
    row count of every dataframe cached here is the run's number of output times.

    Args:
        cache_path: Where to read/write the pickled DataFrame.
        compute_dataframe: Callable producing the DataFrame if the cache is cold.
        expected_row_count: Number of rows the caller knows the result must have,
            normally the run's output-time count. ``None`` skips the check and
            trusts whatever is on disk.
        *args, **kwargs: Passed through to *compute_dataframe*.
    """
    cache_path = Path(cache_path)

    # Use the cache only when it exists AND matches what the caller expects.
    if cache_path.exists():
        cached_dataframe = pd.read_pickle(cache_path)
        cache_is_stale = (
            expected_row_count is not None
            and len(cached_dataframe) != expected_row_count
        )
        if not cache_is_stale:
            print(f"Read cached result from {cache_path}")
            return cached_dataframe
        # Fall through to recompute. Say why, because a silent recompute of an
        # expensive reduction looks like the cache is simply broken.
        print(
            f"Stale cache at {cache_path}: it holds {len(cached_dataframe)}"
            f" rows but this run has {expected_row_count}. Recomputing and"
            " overwriting it."
        )
    else:
        print(f"No cache at {cache_path}, computing...")

    result = compute_dataframe(*args, **kwargs)

    # Refuse to cache a result that already fails the caller's own expectation,
    # so a bad compute cannot poison the next run the way the stale file did.
    if expected_row_count is not None and len(result) != expected_row_count:
        raise ValueError(
            f"{compute_dataframe.__name__} returned {len(result)} rows but"
            f" {expected_row_count} were expected for {cache_path}. Not caching"
            " it."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(cache_path)
    print(f"Wrote result to {cache_path}")
    return result


def analysis_output_time_count(run_dir):
    """How many output times a run has on disk, counted without opening anything.

    Mechanism: RAMS writes one ``a-A-<timestamp>-g1.h5`` per analysis time, so the file
    count is the time count. Counting files rather than opening the dataset is what makes
    this usable as a cheap staleness guard on every cached derived product, including the
    ones that are read without the raw run being loaded at all.

    Args:
        run_dir: A run directory (the parent of ``output/``).

    Returns:
        The number of analysis output times, or None if the output directory is absent.
    """
    output_dir = analysis_output_dir(run_dir)
    if not output_dir.is_dir():
        return None
    return len(list(output_dir.glob("a-A-*-g1.h5")))


def read_or_compute_tracer_totals_datasets(run_dirs, storm_dss=None):
    """Load each storm's in-updraft tracer profile dataset, computing it where missing.

    Mechanism: this is the derived product behind Figures 7 and 9 and the tracer
    entrainment statistics. For every output time it masks the run down to in-analysis,
    in-updraft gridpoints, sums each tracer over x and y, and keeps the result as a
    function of height — so a (time, z, y, x) field per tracer collapses to (time, z).
    ``blt.calculate_tracer_profile_ds`` does that reduction; this function only decides
    whether it has to be done at all.

    The paper runs have theirs on disk already. Reruns generally do not, so a missing file
    is computed here and written into that run's own ``derived/`` directory, matching where
    the paper runs keep theirs. Expect the first pass over a new run to be slow: it touches
    every gridpoint at every output time, which is the whole dataset.

    Passing ``storm_dss=None`` restores the old read-only behaviour, and a missing file
    then raises rather than starting a long computation the caller did not ask for.

    Args:
        run_dirs: Mapping of storm key -> run directory.
        storm_dss: Mapping of storm key -> postprocessed dataset, as returned by
            ``read_storm_datasets``. Needed only for runs whose file is missing.

    Returns:
        (tracer_totals_profile_dss, tracer_totals_dss) — the vertical profiles and
        their column sums, both keyed by storm.
    """
    tracer_totals_profile_dss = {}
    for storm_key in STORM_KEYS_IN_FIGURE_ORDER:
        if storm_key not in run_dirs:
            continue
        profile_filepath = tracer_totals_profile_filepath(
            run_dirs[storm_key], storm_key
        )

        if profile_filepath.exists():
            # A cache written while the run was still integrating holds fewer times than
            # the finished run, and nothing downstream would notice: every figure and
            # table would quietly describe a truncated simulation. That has happened on
            # this project before, to the upper-level cache. So check the time count
            # against the output files on disk before trusting the file.
            cached_profile_ds = xr.open_dataset(profile_filepath)
            expected_time_count = analysis_output_time_count(
                run_dirs[storm_key]
            )
            cache_is_stale = (
                expected_time_count is not None
                and cached_profile_ds.sizes.get("time") != expected_time_count
            )
            if not cache_is_stale:
                print(f"Read tracer profile dataset from {profile_filepath}")
                tracer_totals_profile_dss[storm_key] = cached_profile_ds
                continue
            cached_profile_ds.close()
            print(
                f"Stale tracer profile cache at {profile_filepath}: it holds"
                f" {cached_profile_ds.sizes.get('time')} times but the run has"
                f" {expected_time_count}. Recomputing and overwriting it."
            )

        if storm_dss is None or storm_key not in storm_dss:
            raise FileNotFoundError(
                f"Missing tracer profile dataset at {profile_filepath}, and no"
                f" storm dataset was supplied for '{storm_key}' to compute it"
                " from."
            )

        print(f"No tracer profile dataset at {profile_filepath}, computing...")
        tracer_totals_profile_ds = blt.calculate_tracer_profile_ds(
            storm_dss[storm_key]
        )

        # Write it out, then reopen from disk rather than keeping the in-memory object.
        # Doing that puts the first run and every later run on exactly the same code path,
        # so a bug that only shows up when the dataset comes back off disk — a dropped
        # attribute, a coordinate that round-trips as a different dtype — surfaces on the
        # run that created the file instead of the one after it.
        profile_filepath.parent.mkdir(parents=True, exist_ok=True)
        tracer_totals_profile_ds.to_netcdf(profile_filepath)
        print(f"Wrote tracer profile dataset to {profile_filepath}")
        tracer_totals_profile_dss[storm_key] = xr.open_dataset(profile_filepath)

    # Sum over the vertical, skipping the model's below-ground level at index 0.
    tracer_totals_dss = {
        storm_key: (
            profile_ds.isel({"z": slice(1, len(profile_ds["z"]))}).sum("z")
        )
        for storm_key, profile_ds in tracer_totals_profile_dss.items()
    }
    return tracer_totals_profile_dss, tracer_totals_dss


# Former name, kept so older notebooks and scripts keep working. It forwards without a
# storm dataset, so it still raises on a missing file rather than silently starting a
# multi-hour computation in code that was written expecting pure I/O.
def read_tracer_totals_datasets(run_dirs):
    """Deprecated alias for ``read_or_compute_tracer_totals_datasets``, read-only."""
    return read_or_compute_tracer_totals_datasets(run_dirs, storm_dss=None)


def read_or_compute_development_dataframes(run_dirs, storm_dss):
    """Load each storm's development dataframe, computing it where it's missing.

    Mechanism: the development dataframe holds the max-updraft and rainfall-area
    time series behind Figure 10. It exists on disk for the paper runs but *not*
    for any of the harmonized runs, so this computes it from the 5-minute output
    via ``blt.calculate_development_dataframe`` and caches it into the run's own
    ``derived/`` directory, matching where the paper runs keep theirs.

    That computation is the one genuinely expensive step in the results half of
    these notebooks: it reduces over the full (time, z, y, x) updraft and rainfall
    fields, so expect it to take a while on first run for each new storm.

    As in the manuscript notebook, the dataframe is computed over in-analysis
    gridpoints only, which for the supercell restricts to the right mover.

    Returns:
        Mapping of storm key -> development DataFrame.
    """
    development_dfs = {}
    for storm_key in STORM_KEYS_IN_FIGURE_ORDER:
        if storm_key not in run_dirs:
            continue
        dataframe_filepath = development_dataframe_filepath(
            run_dirs[storm_key], storm_key
        )

        cached_development_df = (
            pd.read_pickle(dataframe_filepath)
            if dataframe_filepath.exists()
            else None
        )
        # Same staleness guard as the tracer profile cache: one row per output time, so a
        # row count that disagrees with the run means the cache predates the run finishing.
        expected_time_count = analysis_output_time_count(run_dirs[storm_key])
        if cached_development_df is not None and (
            expected_time_count is None
            or len(cached_development_df) == expected_time_count
        ):
            development_df = cached_development_df
            print(f"Read development dataframe from {dataframe_filepath}")
        else:
            if cached_development_df is not None:
                print(
                    f"Stale development dataframe at {dataframe_filepath}: it"
                    f" holds {len(cached_development_df)} rows but the run has"
                    f" {expected_time_count}. Recomputing and overwriting it."
                )
            else:
                print(
                    f"No development dataframe at {dataframe_filepath},"
                    " computing..."
                )
            this_storm_ds = storm_dss[storm_key]
            development_df = blt.calculate_development_dataframe(
                this_storm_ds.where(this_storm_ds["in_analysis"])
            )
            dataframe_filepath.parent.mkdir(parents=True, exist_ok=True)
            development_df.to_pickle(dataframe_filepath)
            print(f"Wrote development dataframe to {dataframe_filepath}")

        development_dfs[storm_key] = development_df

        # Attach storm centroid positions onto the storm dataset, as the
        # manuscript notebook does, since later figures index off them.
        for coordinate in ["x", "y"]:
            storm_dss[storm_key][f"storm_position_{coordinate}"] = xr.DataArray(
                development_df[f"storm_centroid_{coordinate}"].values,
                coords={
                    "time": [
                        blt.minutes_to_simulation_time(minutes)
                        for minutes in development_df["t_minutes"].values
                    ]
                },
            )

    return development_dfs


########################################################################
############################## Upper-level fluxes ##############################
########################################################################


def calculate_cumulative_mass_flux_from_analysis_output(
    storm_ds,
    upper_level_z_threshold_m,
    output_interval_seconds=300.0,
    horizontal_grid_spacing_m=500.0,
):
    """Cumulative up/down mass flux through a level, from 5-minute analysis output.

    Why this exists: the manuscript computes this flux from the 2-second lite files
    (``a-L-*.h5``). Every harmonized run was submitted with ``FRQLITE = 0`` and so
    wrote no lite output at all, which makes the manuscript's exact flux
    uncomputable for the harmonized arm.

    Mechanism: the same construction, but sampling the vertical velocity field at
    the 5-minute analysis cadence instead. Each output time contributes
    ``w * rho * dx^2 * dt`` with ``dt`` the analysis interval, split into the
    upward and downward contributions by the sign of w, and accumulated in time.

    This is a *biased* estimator of the true flux, and knowingly so: 5-minute
    sampling aliases the convective-scale w field badly, and there is no reason to
    expect it to recover the 2-second answer. It is fit for purpose here only
    because both arms are estimated the same way, so the arm-to-arm comparison
    stays meaningful even though the absolute values do not match the manuscript's.
    The notebooks quantify that bias directly by computing both estimators on the
    paper runs, which do have lite output.

    Air density is derived from P, T and mixing ratio rather than taken from
    ``DN0``, matching how the manuscript's lite-file path does it (lite files carry
    no ``DN0``), so the two estimators differ only in time sampling.

    Args:
        storm_ds: A postprocessed 5-minute storm dataset.
        upper_level_z_threshold_m: Height of the flux level, in m.
        output_interval_seconds: Spacing of the analysis output, in s.
        horizontal_grid_spacing_m: Horizontal grid spacing, in m.

    Returns:
        A Dataset with ``cumu_upward_mass_flux`` and ``cumu_downward_mass_flux``.
    """
    # Take the single model level nearest the requested threshold height.
    level_ds = storm_ds.sel(
        {"z": upper_level_z_threshold_m}, method="nearest", drop=False
    )

    # Moist air density from the state variables, matching the lite-file path.
    level_ds["mixing_ratio"] = mpc.mixing_ratio(
        partial_press=level_ds["vapor_pressure"] * units("hPa"),
        total_press=level_ds["P"] * units("hPa"),
    ).pint.dequantify()
    level_ds["air_density"] = (
        mpc.density(
            pressure=level_ds["P"] * units("hPa"),
            temperature=level_ds["T"] * units("K"),
            mixing_ratio=level_ds["mixing_ratio"],
        )
        .pint.to("kg/m^3")
        .pint.dequantify()
    )

    # Mass through each cell of the level over one output interval:
    #   m/s * kg/m^3 * m^2 * s = kg
    per_cell_mass_flux = (
        level_ds["WC"]
        * level_ds["air_density"]
        * horizontal_grid_spacing_m**2
        * output_interval_seconds
    )

    # Split by sign of w, then accumulate in time.
    is_upward = level_ds["WC"] > 0
    upward_flux = per_cell_mass_flux.where(is_upward).sum(["x", "y"])
    total_flux = per_cell_mass_flux.sum(["x", "y"])
    downward_flux = total_flux - upward_flux

    cumulative_upward = upward_flux.cumsum("time")
    cumulative_downward = downward_flux.cumsum("time")
    cumulative_upward.name = "cumu_upward_mass_flux"
    cumulative_downward.name = "cumu_downward_mass_flux"

    return xr.merge([cumulative_upward, cumulative_downward])


def calculate_upper_level_tracer_dataframe(storm_ds, upper_level_z_threshold_m):
    """Total tracer content above a level, as a time series.

    Mechanism: lifted from the "Calculate upper-level tracer concentrations" cell
    in ``validation_comparison.ipynb``. Everything above the threshold height is
    summed over all three spatial dimensions, deliberately *without* the
    ``in_analysis`` mask, so the supercell's total includes both the right and left
    movers — as the manuscript describes.

    The accumulated (``ACC*``) tracers are surface emission totals rather than 3D
    fields, so they are converted to number concentrations using the air mass of
    the lowest level at the first time, not the local air mass.
    """
    upper_level_ds = storm_ds.sel(
        {"z": slice(upper_level_z_threshold_m, None, None)}
    ).copy()

    # Number-concentration versions of every current tracer.
    all_current_tracer_vars = (
        blt.FS_TRACERS + blt.RS_TRACERS + ["total_fs_tracer"]
    )
    for tracer in all_current_tracer_vars:
        upper_level_ds[tracer + "_nc"] = (
            upper_level_ds[tracer] * upper_level_ds["air_mass"]
        )

    # Accumulated emission totals exist only for the rain-sourced tracers.
    for tracer in blt.RS_TRACERS:
        upper_level_ds["ACC" + tracer + "_nc"] = (
            upper_level_ds["ACC" + tracer]
            * upper_level_ds.isel({"z": 1, "time": 0})["air_mass"]
        )

    return (
        upper_level_ds[
            [x for x in upper_level_ds.data_vars if "tracer" in x.lower()]
            + ["air_mass"]
        ]
        .sum(["x", "y", "z"])
        .to_dataframe()
        .reset_index()
    )


########################################################################
############################## Summary tables ##############################
########################################################################


def collect_headline_metrics(
    tracer_totals_dss, upper_level_tracer_dfs=None, development_dfs=None
):
    """Reduce one arm's results to a small set of per-storm headline numbers.

    Mechanism: the manuscript's cross-storm claims rest on a handful of scalars —
    how much rain-sourced tracer each storm entrains, how efficiently it does so,
    and how much reaches the upper levels. Reducing each arm to those scalars is
    what makes the arms directly comparable; the figures show the time evolution
    but cannot be differenced.

    Metrics use the 10 mm/hr rain-sourced tracer (``TRACERP003``) as the
    representative threshold, matching how the manuscript quotes single numbers.

    Args:
        tracer_totals_dss: Mapping of storm key -> column-summed tracer dataset.
        upper_level_tracer_dfs: Optional mapping of storm key -> upper-level
            tracer DataFrame.
        development_dfs: Optional mapping of storm key -> development DataFrame.

    Returns:
        A DataFrame indexed by storm key, one column per metric.
    """
    # The 10 mm/hr rain threshold, the middle of the five and the one the
    # manuscript leads with.
    representative_rs_tracer = blt.RS_TRACERS[2]

    metric_rows = {}
    for storm_key, tracer_ds in tracer_totals_dss.items():
        # Restrict to times when an updraft actually exists, or the per-kg means
        # divide by zero.
        in_updraft_ds = tracer_ds.where(tracer_ds["air_mass"] > 0)

        rs_mixing_ratio = (
            in_updraft_ds[representative_rs_tracer + "_nc"]
            / in_updraft_ds["air_mass"]
        )
        fs_mixing_ratio = (
            in_updraft_ds["total_fs_tracer_nc"] / in_updraft_ds["air_mass"]
        )
        rs_entrainment_efficiency = (
            in_updraft_ds[representative_rs_tracer + "_nc"]
            / in_updraft_ds["ACC" + representative_rs_tracer + "_nc"]
        )
        rs_fraction_of_total = (
            in_updraft_ds[representative_rs_tracer + "_nc"]
            / in_updraft_ds["total_fs_tracer_nc"]
        )

        metrics = {
            "mean_rs_mixing_ratio_per_kg": float(
                rs_mixing_ratio.mean(...).values
            ),
            "max_rs_mixing_ratio_per_kg": float(
                rs_mixing_ratio.max(...).values
            ),
            "eos_rs_mixing_ratio_per_kg": float(
                rs_mixing_ratio.isel({"time": -1}).values
            ),
            "mean_fs_mixing_ratio_per_kg": float(
                fs_mixing_ratio.mean(...).values
            ),
            "max_rs_entrainment_efficiency": float(
                rs_entrainment_efficiency.max(...).values
            ),
            "mean_rs_entrainment_efficiency": float(
                rs_entrainment_efficiency.mean(...).values
            ),
            "mean_rs_fraction_of_total": float(
                rs_fraction_of_total.mean(...).values
            ),
            "max_updraft_mass_kg": float(
                in_updraft_ds["air_mass"].max(...).values
            ),
        }

        # Upper-level lofting, if the caller computed it.
        if upper_level_tracer_dfs and storm_key in upper_level_tracer_dfs:
            upper_level_df = upper_level_tracer_dfs[storm_key]
            for column, metric_name in [
                ("ul_fs_tracer_normalized-flux", "eos_ul_fs_per_upward_flux"),
                (
                    f"ul_{representative_rs_tracer}_nc_normalized-flux",
                    "eos_ul_rs_per_upward_flux",
                ),
                (
                    f"ul_{representative_rs_tracer}_nc_normalized-acc",
                    "eos_ul_rs_per_emitted",
                ),
            ]:
                if column in upper_level_df.columns:
                    metrics[metric_name] = float(
                        upper_level_df[column].iloc[-1]
                    )

        # Storm development, if the caller computed it.
        if development_dfs and storm_key in development_dfs:
            development_df = development_dfs[storm_key]
            metrics["max_updraft_speed_m_s"] = float(
                development_df["max_updraft"].max()
            )
            # Rain area at the 10 mm/hr threshold, converted to square kilometres.
            metrics["max_rain_area_10mmhr_km2"] = float(
                development_df["rain_threshold2"].max() * 500**2 / 1e6
            )

        metric_rows[storm_key] = metrics

    return pd.DataFrame(metric_rows).T


def compare_arms_table(
    headline_metrics_by_arm, original_arm_label, harmonized_arm_label
):
    """Per storm, show each headline metric before and after harmonization.

    This answers "did the numbers move", which is the easy half of the question.

    Returns:
        A DataFrame indexed by (storm, metric).
    """
    original_metrics = headline_metrics_by_arm[original_arm_label]
    harmonized_metrics = headline_metrics_by_arm[harmonized_arm_label]

    table_rows = []
    for storm_key in original_metrics.index:
        for metric_name in original_metrics.columns:
            original_value = original_metrics.loc[storm_key, metric_name]
            harmonized_value = harmonized_metrics.loc[storm_key, metric_name]
            table_rows.append({
                "storm": storm_key,
                "metric": metric_name,
                original_arm_label: original_value,
                harmonized_arm_label: harmonized_value,
                "percent_change": (
                    100.0 * (harmonized_value - original_value) / original_value
                    if original_value
                    else np.nan
                ),
            })
    return pd.DataFrame(table_rows).set_index(["storm", "metric"])


def cross_storm_ratio_table(headline_metrics_by_arm, storm_pairs=None):
    """Compare the *ratios between storms* across arms — the fairness test itself.

    Mechanism: this is the table the whole exercise exists to produce. A setup
    error that shifts all three storms by the same factor is harmless to the
    manuscript, because every claim in it is comparative ("the supercell entrains
    more rain-sourced tracer than the isolated convection"). An error that shifts
    the storms by *different* factors is what would make those claims artefacts of
    the setup rather than of storm morphology.

    So for each pair of storms we form the ratio of each headline metric, in each
    arm, and report how much that ratio moves. A ratio that survives harmonization
    means the cross-storm claim it supports is robust to the setup differences;
    one that flips sign or changes by a large factor means it is not.

    Args:
        headline_metrics_by_arm: Mapping of arm label -> headline metric DataFrame.
        storm_pairs: Pairs to form ratios over. Defaults to every ordered pair
            among the three storms that the manuscript compares.

    Returns:
        A DataFrame indexed by (storm_pair, metric), one column per arm plus the
        change in the ratio.
    """
    storm_pairs = storm_pairs or [
        ("sl_wk", "ic_wk"),
        ("sc_wk", "ic_wk"),
        ("sc_wk", "sl_wk"),
    ]
    arm_labels = list(headline_metrics_by_arm)

    table_rows = []
    for numerator_storm, denominator_storm in storm_pairs:
        pair_label = f"{numerator_storm} / {denominator_storm}"
        # Only metrics present in every arm can be compared across arms.
        shared_metrics = set.intersection(*[
            set(metrics.columns) for metrics in headline_metrics_by_arm.values()
        ])
        for metric_name in sorted(shared_metrics):
            row = {"storm_pair": pair_label, "metric": metric_name}
            for arm_label, metrics in headline_metrics_by_arm.items():
                denominator_value = metrics.loc[denominator_storm, metric_name]
                row[arm_label] = (
                    metrics.loc[numerator_storm, metric_name]
                    / denominator_value
                    if denominator_value
                    else np.nan
                )
            # How far the cross-storm ratio moved between the two arms.
            first_arm_ratio, last_arm_ratio = (
                row[arm_labels[0]],
                row[arm_labels[-1]],
            )
            row["ratio_percent_change"] = (
                100.0 * (last_arm_ratio - first_arm_ratio) / first_arm_ratio
                if first_arm_ratio
                else np.nan
            )
            table_rows.append(row)

    return pd.DataFrame(table_rows).set_index(["storm_pair", "metric"])


def style_percent_change(
    table, percent_change_column="percent_change", threshold=10.0
):
    """Highlight rows whose change exceeds a threshold, for quick visual triage.

    Returns a pandas Styler; falls back to the plain table if styling is
    unavailable in the current environment.
    """
    try:
        return table.style.map(
            lambda value: (
                "background-color: #ffdddd"
                if isinstance(value, float)
                and np.isfinite(value)
                and abs(value) > threshold
                else ""
            ),
            subset=[percent_change_column],
        ).format(precision=4)
    except Exception:
        warnings.warn("Styling unavailable; returning the unstyled table")
        return table


########################################################################
############################## Skew-T diagrams ##############################
########################################################################

# The pressure window every skew-T in these notebooks is drawn over. The bottom is
# below the lowest surface pressure in the study (~1013 hPa) so no setup's surface
# sits on the frame; the top is above the level where the parcel curves converge.
SKEWT_PRESSURE_LIMITS_HPA = (1050, 100)

# Temperature window at the 1000 hPa reference line. The skew shifts the plotted
# range with height, so this is wide enough to keep the whole dewpoint curve on the
# canvas without wasting half the panel on empty space. The soundings in this study
# are all warm and moist, so the cold half of a conventional skew-T is dead space.
SKEWT_TEMPERATURE_LIMITS_C = (-30, 45)

# Hatch patterns for overlay figures. Three CAPE regions filled with plain alpha blend
# into one indistinguishable smear, so each setup's positive area also carries its own
# hatch and the overlaps stay separable.
SKEWT_OVERLAY_HATCHES = ["//", "\\\\", "xx", ".."]

# Skew angle in degrees. 45 is the conventional value and makes dry adiabats and
# isotherms roughly perpendicular, which is what makes positive area readable.
SKEWT_ROTATION_DEGREES = 45

# Colours used consistently across every skew-T figure below.
SKEWT_ENVIRONMENT_TEMPERATURE_COLOR = "tab:red"
SKEWT_ENVIRONMENT_DEWPOINT_COLOR = "tab:green"
SKEWT_PARCEL_COLOR = "black"
SKEWT_CAPE_SHADE_COLOR = "tab:red"
SKEWT_CIN_SHADE_COLOR = "tab:blue"

# Colour cycle for overlay figures, one entry per sounding drawn on a shared axes.
SKEWT_OVERLAY_COLORS = ["tab:blue", "tab:orange", "tab:purple", "tab:brown"]

# Dash patterns for overlay figures. Harmonized setups are frequently *identical* to
# float32 roundoff, and a solid line drawn over another solid line reads as a single
# setup, so every series gets its own dash pattern.
SKEWT_OVERLAY_LINE_STYLES = [
    "solid",
    (0, (6, 2)),
    (0, (2, 2)),
    (0, (4, 2, 1, 2)),
]

# Dry-air gas constant, used to convert a shaded area back into J/kg. Matches the
# value metpy integrates CAPE with, so the two agree to the vertical grid's accuracy.
_DRY_AIR_GAS_CONSTANT_J_PER_KG_K = 287.047


def calculate_parcel_analysis(sounding_df, parcel_start_index=0):
    """Lift one parcel from a sounding and return everything a skew-T needs.

    Mechanism: this repeats, for a single starting level, exactly the computation
    ``xarrams.calculate_sounding_derived_vars`` runs for every starting level — a
    moist-adiabat parcel integration with ``metpy.calc.parcel_profile`` followed by
    ``metpy.calc.cape_cin``. Using the same two calls means the CAPE printed on a
    skew-T here always matches the ``cape`` column the tables elsewhere in these
    notebooks are built from, rather than being a second, subtly different number.

    Why the virtual temperature appears
    -----------------------------------
    The non-obvious part is that this returns the CAPE *integrand*, not just the
    scalar. CAPE is the integral

        CAPE = R_d * integral of (Tv_parcel - Tv_env) with respect to -ln(p)

    over the positive-buoyancy layer between the level of free convection and the
    equilibrium level. On a skew-T the horizontal distance between two curves at a
    fixed pressure is exactly their temperature difference, and the vertical axis is
    -ln(p), so **the plotted area between two curves is proportional to that
    integral**. Returning the integrand — masked to zero outside the positive area —
    is what lets the difference figure shade a region whose area is proportional to a
    *difference* in CAPE, instead of drawing two CAPE regions and leaving the reader
    to subtract them by eye.

    Buoyancy depends on density, so the integrand is a *virtual* temperature
    difference, not a plain one. ``metpy.calc.cape_cin`` applies that correction
    internally even though its arguments are plain temperature and dewpoint, and it
    is worth about 6% of CAPE for these soundings. This function therefore rebuilds
    metpy's virtual profiles step for step — including taking the parcel's mixing
    ratio from the surface dewpoint below the LCL and from saturation above it — so
    that a shaded area and a printed CAPE never disagree. Verified against
    ``cape_cin`` on all six setups in this study: agreement to better than 0.15%,
    the residual being the partial layers at the LFC and EL that a level-by-level
    trapezoid cannot resolve.

    Args:
        sounding_df: A sounding DataFrame from ``read_initial_sounding_dataframe``,
            ascending in height (so descending in pressure).
        parcel_start_index: Row index the parcel is lifted from. 0 is the surface,
            which is what "SBCAPE" means everywhere else in these notebooks.

    Returns:
        A dict holding

        * the conventional plotted curves — ``pressure_hPa``,
          ``environment_temperature_C``, ``environment_dewpoint_C``,
          ``parcel_temperature_C``;
        * the density-corrected pair the shading is built from —
          ``environment_virtual_temperature_C``, ``parcel_virtual_temperature_C``;
        * the integrand — ``virtual_temperature_excess_C`` and its positive-area mask
          ``cape_virtual_temperature_excess_C``;
        * the scalars — ``cape_j_kg``, ``cin_j_kg``;
        * the characteristic levels — ``lcl_pressure_hPa``, ``lfc_pressure_hPa``,
          ``el_pressure_hPa``.
    """
    # Everything from the parcel's starting level upward. Levels below it play no
    # part in either the lift or the integral.
    column_df = sounding_df.iloc[parcel_start_index:]

    pressure = column_df["PS"].values * units("hPa")
    temperature = column_df["TS"].values * units("degC")
    dewpoint = column_df["dewpoint"].values * units("degC")

    # Lift the parcel: dry adiabatic to the LCL, then moist adiabatic above it.
    parcel_temperature = mpc.parcel_profile(
        pressure=pressure, temperature=temperature[0], dewpoint=dewpoint[0]
    ).to("degC")

    # The scalars, from metpy directly, so they match every other CAPE in these
    # notebooks exactly rather than approximately.
    cape, cin = mpc.cape_cin(
        pressure=pressure,
        temperature=temperature,
        dewpoint=dewpoint,
        parcel_profile=parcel_temperature,
    )

    # The lifting condensation level, which also splits the parcel's moisture history.
    lcl_pressure, _ = mpc.lcl(
        pressure=pressure[0], temperature=temperature[0], dewpoint=dewpoint[0]
    )

    # Rebuild metpy's internal virtual-temperature profiles. Below the LCL the parcel
    # keeps the mixing ratio it started with, set by the surface dewpoint; above the
    # LCL it is saturated at its own temperature, having rained out the difference.
    below_lcl = pressure > lcl_pressure
    parcel_mixing_ratio = np.where(
        below_lcl,
        mpc.saturation_mixing_ratio(pressure[0], dewpoint[0]),
        mpc.saturation_mixing_ratio(pressure, parcel_temperature),
    )
    environment_virtual_temperature = mpc.virtual_temperature_from_dewpoint(
        pressure, temperature, dewpoint
    )
    parcel_virtual_temperature = mpc.virtual_temperature(
        parcel_temperature, parcel_mixing_ratio
    )

    # The LFC and EL bound the integral, and metpy locates them on the *virtual*
    # profiles, so locate them the same way here. Either can be undefined for a
    # stable sounding, which is a legitimate outcome rather than an error.
    lfc_pressure_hPa = _first_level_pressure_or_nan(
        mpc.lfc,
        pressure,
        environment_virtual_temperature,
        dewpoint,
        parcel_virtual_temperature,
    )
    el_pressure_hPa = _first_level_pressure_or_nan(
        mpc.el,
        pressure,
        environment_virtual_temperature,
        dewpoint,
        parcel_virtual_temperature,
    )

    pressure_hPa = pressure.to("hPa").magnitude
    virtual_temperature_excess_C = (
        (parcel_virtual_temperature - environment_virtual_temperature)
        .to("delta_degC")
        .magnitude
    )

    # Mask the excess down to the layer CAPE is actually integrated over: buoyant, at
    # or above the LFC, at or below the EL. Outside that layer the integrand is zero,
    # which is what makes the shaded region's area equal to CAPE.
    inside_positive_area = virtual_temperature_excess_C > 0
    if np.isfinite(lfc_pressure_hPa):
        inside_positive_area &= pressure_hPa <= lfc_pressure_hPa
    if np.isfinite(el_pressure_hPa):
        inside_positive_area &= pressure_hPa >= el_pressure_hPa
    cape_virtual_temperature_excess_C = np.where(
        inside_positive_area, virtual_temperature_excess_C, 0.0
    )

    return {
        "pressure_hPa": pressure_hPa,
        "environment_temperature_C": temperature.to("degC").magnitude,
        "environment_dewpoint_C": dewpoint.to("degC").magnitude,
        "parcel_temperature_C": parcel_temperature.to("degC").magnitude,
        "environment_virtual_temperature_C": (
            environment_virtual_temperature.to("degC").magnitude
        ),
        "parcel_virtual_temperature_C": (
            parcel_virtual_temperature.to("degC").magnitude
        ),
        "virtual_temperature_excess_C": virtual_temperature_excess_C,
        "cape_virtual_temperature_excess_C": cape_virtual_temperature_excess_C,
        "cape_j_kg": float(cape.to("J/kg").magnitude),
        "cin_j_kg": float(cin.to("J/kg").magnitude),
        "lcl_pressure_hPa": float(lcl_pressure.to("hPa").magnitude),
        "lfc_pressure_hPa": lfc_pressure_hPa,
        "el_pressure_hPa": el_pressure_hPa,
    }


def _first_level_pressure_or_nan(
    metpy_level_function, pressure, temperature, dewpoint, parcel_temperature
):
    """Call ``metpy.calc.lfc``/``el`` and reduce the result to a plain hPa float.

    Both functions return a (pressure, temperature) pair, return NaN for a sounding
    with no such level, and can return an array when several candidate levels exist.
    A stable or otherwise degenerate profile is a legitimate outcome here, not an
    error, so this collapses all of those cases to a single float or NaN.
    """
    try:
        level_pressure, _ = metpy_level_function(
            pressure=pressure,
            temperature=temperature,
            dewpoint=dewpoint,
            parcel_temperature_profile=parcel_temperature,
        )
    except Exception:
        return np.nan
    level_pressure_hPa = np.atleast_1d(level_pressure.to("hPa").magnitude)
    if level_pressure_hPa.size == 0:
        return np.nan
    return float(level_pressure_hPa[0])


def cape_from_temperature_excess(pressure_hPa, virtual_temperature_excess_C):
    """Integrate a virtual-temperature-excess profile back into a CAPE in J/kg.

    This exists to check the claim every figure below rests on: that the area shaded
    between two curves really is proportional to a CAPE. It applies the same
    trapezoidal integral in -ln(p) that ``metpy.calc.cape_cin`` applies internally, so
    the two agree to the resolution of the vertical grid.
    """
    return _DRY_AIR_GAS_CONSTANT_J_PER_KG_K * float(
        np.trapezoid(virtual_temperature_excess_C, -np.log(pressure_hPa))
    )


def _new_skewt_axes(fig, subplot_spec=None, background_alpha=0.4):
    """Create one skew-T axes with the study's shared background and limits.

    Keeping the background in one place matters more than it looks: dry adiabats and
    isotherms are the reference grid a reader judges positive area against, so two
    panels drawn with different backgrounds cannot be compared at a glance.

    Args:
        fig: Figure to draw into.
        subplot_spec: Gridspec cell for this panel, or None for a whole-figure axes.
        background_alpha: Opacity of the adiabat and mixing-line grid. Lower it on
            figures that overlay several soundings, where the grid competes with the
            profiles instead of supporting them.
    """
    skew = SkewT(fig, rotation=SKEWT_ROTATION_DEGREES, subplot=subplot_spec)

    # The reference grid. Drawn first, and thin, so the profiles sit on top of it.
    skew.plot_dry_adiabats(linewidth=0.5, alpha=background_alpha)
    skew.plot_moist_adiabats(linewidth=0.5, alpha=background_alpha)
    skew.plot_mixing_lines(linewidth=0.5, alpha=background_alpha)

    # Identical limits on every panel in every figure, so areas are comparable.
    skew.ax.set_ylim(*SKEWT_PRESSURE_LIMITS_HPA)
    skew.ax.set_xlim(*SKEWT_TEMPERATURE_LIMITS_C)
    skew.ax.set_xlabel("T (°C)")
    skew.ax.set_ylabel("p (hPa)")
    return skew


def _scaled_font_size(size_relative_to_base):
    """Return a font size as a multiple of whatever ``font.size`` is currently active.

    Why the figure routines here do not just hardcode point sizes. These panels carry a lot
    of small text — level labels, a legend of seven entries, a numeric caption — and the sizes
    that work on screen are not the sizes that work in a slide or a printed figure. Reading
    every size off ``plt.rcParams["font.size"]`` means one ``plt.rc_context`` block around a
    call scales all of them together, instead of the caller having to pass half a dozen
    separate size arguments that would then have to stay in proportion by hand.

    Args:
        size_relative_to_base: Multiplier on the active base font size. 1.0 matches body
            text; the annotation sizes here sit between 0.6 and 1.05.

    Returns:
        A font size in points.
    """
    return plt.rcParams["font.size"] * size_relative_to_base


def _annotate_characteristic_levels(skew, parcel_analysis, color="0.35"):
    """Mark the LCL, LFC and EL on a skew-T as short labelled ticks.

    These three levels are what turn a shaded blob into a diagnosis: a CAPE change
    driven by a lifted LFC means something different from one driven by a raised EL.
    """
    labelled_levels = [
        ("LCL", parcel_analysis["lcl_pressure_hPa"]),
        ("LFC", parcel_analysis["lfc_pressure_hPa"]),
        ("EL", parcel_analysis["el_pressure_hPa"]),
    ]
    for level_name, level_pressure_hPa in labelled_levels:
        if not np.isfinite(level_pressure_hPa):
            continue
        # Place the tick at the environment temperature of that level, so the label
        # sits on the sounding rather than floating in the middle of the panel.
        # Interpolating in -ln(p) rather than in p keeps the sample points increasing,
        # because pressure descends with height, and it matches the skew-T's own axis.
        level_temperature_C = np.interp(
            -np.log(level_pressure_hPa),
            -np.log(parcel_analysis["pressure_hPa"]),
            parcel_analysis["environment_temperature_C"],
        )
        skew.ax.text(
            level_temperature_C,
            level_pressure_hPa,
            f" {level_name}",
            color=color,
            fontsize=_scaled_font_size(0.62),
            va="center",
            ha="left",
        )
        skew.ax.plot(
            level_temperature_C,
            level_pressure_hPa,
            marker="_",
            markersize=9,
            color=color,
        )


def _zoom_skewt_axes_to_curves(
    skew,
    pressure_hPa,
    temperature_curves_C,
    pressure_pad_fraction=0.04,
    temperature_pad_C=3.0,
):
    """Shrink a skew-T's limits until the supplied curves fill the panel.

    Why this needs its own routine rather than a call to ``set_xlim``: on a skew-T the
    horizontal axis is not the temperature axis. metpy's ``SkewXAxes`` first maps data
    to the unit square using the axis limits, and *then* shears that square sideways by
    ``tan(rotation)`` times the height. A point at (T, p) therefore lands at

        x_in_axes = (T - x_min) / (x_max - x_min)  +  tan(rotation) * y_in_axes

    where ``y_in_axes`` is the point's fractional height in ``-ln(p)``. The shear is
    measured in *axes fractions*, so how far a point slides sideways depends on the
    temperature span you chose — which is the very thing being solved for. Widening the
    window to fit a point also moves that point, so the two cannot be solved in one step.

    The fix is a short fixed-point iteration. Guess a temperature span, shear the curves
    by it, read off the span the sheared curves actually need, and feed that back in.
    The map is a contraction here because the shear contributes at most one panel height
    of slide, so a handful of passes converges to well under a tenth of a Kelvin.

    Pressure limits are set first and independently, because the shear depends on them.
    They are padded in ``-ln(p)``, which is the axis's own coordinate, so the padding
    looks the same at the top of the panel as at the bottom.

    Args:
        skew: The metpy ``SkewT`` whose axes should be re-limited.
        pressure_hPa: Pressure levels the curves are defined on.
        temperature_curves_C: Sequence of temperature arrays, all on ``pressure_hPa``.
            Curves may contain NaN; NaN samples are ignored, which is how a caller
            excludes a curve outside the layer of interest.
        pressure_pad_fraction: Padding above and below, as a fraction of the panel's
            depth in ``-ln(p)``.
        temperature_pad_C: Padding left and right, in Kelvin of temperature.

    Returns:
        ``(pressure_limits_hPa, temperature_limits_C)`` as actually applied.
    """
    pressure_hPa = np.asarray(pressure_hPa, dtype=float)
    stacked_temperatures_C = np.vstack(
        [np.asarray(curve_C, dtype=float) for curve_C in temperature_curves_C]
    )

    # Levels where at least one curve has something to show. Everything else is
    # irrelevant to the zoom, and including it would undo the zoom entirely.
    level_is_used = np.any(np.isfinite(stacked_temperatures_C), axis=0)
    if not np.any(level_is_used):
        return skew.ax.get_ylim(), skew.ax.get_xlim()

    ########## Pressure limits, padded in the axis's own -ln(p) coordinate ##########
    used_negative_log_pressure = -np.log(pressure_hPa[level_is_used])
    negative_log_pressure_span = (
        used_negative_log_pressure.max() - used_negative_log_pressure.min()
    )
    negative_log_pressure_pad = pressure_pad_fraction * max(
        negative_log_pressure_span, 1e-6
    )
    bottom_pressure_hPa = np.exp(
        -(used_negative_log_pressure.min() - negative_log_pressure_pad)
    )
    top_pressure_hPa = np.exp(
        -(used_negative_log_pressure.max() + negative_log_pressure_pad)
    )
    skew.ax.set_ylim(bottom_pressure_hPa, top_pressure_hPa)

    ########## Temperature limits, solved for against the shear ##########
    # Fractional height of every level in the panel, which is what the shear scales.
    fractional_height = (np.log(pressure_hPa) - np.log(bottom_pressure_hPa)) / (
        np.log(top_pressure_hPa) - np.log(bottom_pressure_hPa)
    )
    shear_per_panel_height = np.tan(np.radians(SKEWT_ROTATION_DEGREES))

    # Seed the iteration with the unsheared span, then let the shear correct it.
    finite_temperatures_C = stacked_temperatures_C[
        np.isfinite(stacked_temperatures_C)
    ]
    temperature_span_C = max(
        float(finite_temperatures_C.max() - finite_temperatures_C.min()), 1.0
    )
    minimum_temperature_C = float(finite_temperatures_C.min())
    for _ in range(50):
        # Where each sample sits horizontally, expressed back in temperature units.
        sheared_temperature_C = stacked_temperatures_C + (
            shear_per_panel_height
            * fractional_height[None, :]
            * temperature_span_C
        )
        left_edge_C = np.nanmin(sheared_temperature_C)
        right_edge_C = np.nanmax(sheared_temperature_C)
        updated_span_C = (right_edge_C - left_edge_C) + 2.0 * temperature_pad_C
        if abs(updated_span_C - temperature_span_C) < 1e-3:
            temperature_span_C = updated_span_C
            minimum_temperature_C = left_edge_C
            break
        temperature_span_C = updated_span_C
        minimum_temperature_C = left_edge_C

    temperature_limits_C = (
        minimum_temperature_C - temperature_pad_C,
        minimum_temperature_C - temperature_pad_C + temperature_span_C,
    )
    skew.ax.set_xlim(*temperature_limits_C)
    return (bottom_pressure_hPa, top_pressure_hPa), temperature_limits_C


def plot_skewt_grid(
    labelled_soundings_by_arm,
    parcel_start_index=0,
    panel_width=4.2,
    panel_height=5.2,
    suptitle=None,
):
    """Draw one skew-T per (arm, storm), with CAPE and CIN shaded.

    Mechanism: rows are arms and columns are storms, so reading *down* a column shows
    what harmonization did to one storm, and reading *across* a row shows how the three
    storms differ within one setup family. Every panel shares the same temperature and
    pressure limits and the same adiabat background, which is what makes the shaded
    areas comparable by eye rather than only by their printed values.

    The shaded positive area is bounded by the *virtual* temperature curves rather
    than by the plain temperature and parcel curves, for the reason given in
    ``calculate_parcel_analysis``: buoyancy depends on density, and the plain-T area
    understates CAPE by about 6% here. Both pairs of curves are drawn — the plain ones
    boldly, because they are what a reader expects to see on a skew-T, and the virtual
    ones faintly, so it is visible where the shading boundary actually lies.

    Args:
        labelled_soundings_by_arm: Mapping of arm label -> {storm label -> sounding}.
        parcel_start_index: Row index the parcel is lifted from; 0 is the surface.
        panel_width: Width of a single skew-T panel, in inches.
        panel_height: Height of a single skew-T panel, in inches.
        suptitle: Optional figure title.

    Returns:
        (fig, parcel_analyses) where ``parcel_analyses`` maps
        (arm label, storm label) -> the dict from ``calculate_parcel_analysis``.
    """
    arm_labels = list(labelled_soundings_by_arm)
    # Column order follows the first arm; every arm carries the same storms.
    storm_labels = list(labelled_soundings_by_arm[arm_labels[0]])

    fig = plt.figure(
        figsize=(
            panel_width * len(storm_labels),
            panel_height * len(arm_labels),
        )
    )
    grid_spec = fig.add_gridspec(nrows=len(arm_labels), ncols=len(storm_labels))

    parcel_analyses = {}
    for arm_ix, arm_label in enumerate(arm_labels):
        for storm_ix, storm_label in enumerate(storm_labels):
            sounding_df = labelled_soundings_by_arm[arm_label][storm_label]
            parcel_analysis = calculate_parcel_analysis(
                sounding_df, parcel_start_index
            )
            parcel_analyses[(arm_label, storm_label)] = parcel_analysis

            skew = _new_skewt_axes(fig, grid_spec[arm_ix, storm_ix])

            # The sounding itself: environment temperature and dewpoint.
            skew.plot(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["environment_temperature_C"],
                color=SKEWT_ENVIRONMENT_TEMPERATURE_COLOR,
                linewidth=2.0,
                label="T",
            )
            skew.plot(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["environment_dewpoint_C"],
                color=SKEWT_ENVIRONMENT_DEWPOINT_COLOR,
                linewidth=2.0,
                label="$T_d$",
            )
            # The lifted parcel, whose separation from T is the buoyancy.
            skew.plot(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["parcel_temperature_C"],
                color=SKEWT_PARCEL_COLOR,
                linewidth=1.4,
                linestyle="dashed",
                label="parcel",
            )
            # The pair the shading is actually bounded by.
            skew.plot(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["environment_virtual_temperature_C"],
                color="0.45",
                linewidth=0.8,
                linestyle="dotted",
                label="$T_v$ (shading edge)",
            )
            skew.plot(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["parcel_virtual_temperature_C"],
                color="0.45",
                linewidth=0.8,
                linestyle="dotted",
            )

            # Positive area (CAPE), between the two virtual curves, over the layer
            # bounded by the LFC and the EL.
            skew.ax.fill_betweenx(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["environment_virtual_temperature_C"],
                parcel_analysis["environment_virtual_temperature_C"]
                + parcel_analysis["cape_virtual_temperature_excess_C"],
                color=SKEWT_CAPE_SHADE_COLOR,
                alpha=0.30,
                linewidth=0,
            )
            # Negative area (CIN), below the LFC where the parcel is colder than its
            # environment. metpy's own helper draws this one, since it is bounded by
            # the surface rather than by an integrand mask.
            skew.shade_cin(
                parcel_analysis["pressure_hPa"] * units("hPa"),
                parcel_analysis["environment_virtual_temperature_C"]
                * units("degC"),
                parcel_analysis["parcel_virtual_temperature_C"] * units("degC"),
                parcel_analysis["environment_dewpoint_C"] * units("degC"),
                color=SKEWT_CIN_SHADE_COLOR,
                alpha=0.30,
            )

            _annotate_characteristic_levels(skew, parcel_analysis)

            skew.ax.set_title(f"{storm_label}\n{arm_label}", fontsize=11)
            # Print the numbers as well as shading them: an area difference of a few
            # percent is invisible, and these figures are read alongside the tables.
            skew.ax.text(
                0.02,
                0.02,
                f"CAPE {parcel_analysis['cape_j_kg']:.0f} J/kg\n"
                f"CIN {parcel_analysis['cin_j_kg']:.0f} J/kg",
                transform=skew.ax.transAxes,
                fontsize=9,
                va="bottom",
                ha="left",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.7"},
            )
            if arm_ix == 0 and storm_ix == 0:
                skew.ax.legend(loc="upper left", fontsize=8, frameon=False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=15)
    fig.tight_layout()
    return fig, parcel_analyses


def plot_skewt_overlay(
    labelled_sounding_dfs,
    parcel_start_index=0,
    shade_cape=True,
    figsize=(7.5, 8.0),
    title=None,
):
    """Overlay several soundings on one skew-T, each with its own CAPE shaded.

    Mechanism: one axes, one colour per setup, temperature solid and dewpoint thinner
    in that colour, and each setup's positive area filled in the same colour at low
    opacity. Where two setups have nearly the same CAPE their fills overlap and read as
    one darker region; where they differ, the excess of the larger one stands out as a
    single-colour wedge. Each setup also gets its own dash pattern, because two
    harmonized setups in this study are frequently identical to float32 roundoff and a
    solid line hidden under another solid line reads as a missing setup.

    Use this figure to answer "how do these setups rank"; use
    ``plot_skewt_cape_difference`` to answer "by how much".

    Args:
        labelled_sounding_dfs: Mapping of display label -> sounding DataFrame.
        parcel_start_index: Row index the parcel is lifted from; 0 is the surface.
        shade_cape: Fill each setup's positive area. Turn this off when four or more
            setups are overlaid and the fills stop being separable.
        figsize: Figure size in inches.
        title: Optional axes title.

    Returns:
        (fig, skew, parcel_analyses) where ``parcel_analyses`` maps label -> the dict
        from ``calculate_parcel_analysis``.
    """
    fig = plt.figure(figsize=figsize)
    # Three soundings plus three fills is already a lot of ink, so mute the adiabat
    # grid further than the single-sounding panels do.
    skew = _new_skewt_axes(fig, background_alpha=0.22)

    parcel_analyses = {}
    for series_ix, (label, sounding_df) in enumerate(
        labelled_sounding_dfs.items()
    ):
        parcel_analysis = calculate_parcel_analysis(
            sounding_df, parcel_start_index
        )
        parcel_analyses[label] = parcel_analysis

        color = SKEWT_OVERLAY_COLORS[series_ix % len(SKEWT_OVERLAY_COLORS)]
        line_style = SKEWT_OVERLAY_LINE_STYLES[
            series_ix % len(SKEWT_OVERLAY_LINE_STYLES)
        ]
        # Taper the width across the series so a curve hidden underneath an earlier,
        # thicker one still reads as "coincident" rather than as "missing".
        line_width = 2.6 - 0.8 * series_ix / max(
            len(labelled_sounding_dfs) - 1, 1
        )

        skew.plot(
            parcel_analysis["pressure_hPa"],
            parcel_analysis["environment_temperature_C"],
            color=color,
            linestyle=line_style,
            linewidth=line_width,
            label=f"{label} — CAPE {parcel_analysis['cape_j_kg']:.0f} J/kg",
        )
        skew.plot(
            parcel_analysis["pressure_hPa"],
            parcel_analysis["environment_dewpoint_C"],
            color=color,
            linestyle=line_style,
            linewidth=line_width * 0.7,
            alpha=0.75,
        )
        skew.plot(
            parcel_analysis["pressure_hPa"],
            parcel_analysis["parcel_temperature_C"],
            color=color,
            linestyle="dotted",
            linewidth=1.1,
            alpha=0.9,
        )

        if shade_cape:
            # Fill the positive area, bounded by the virtual-temperature pair so the
            # shaded area matches the CAPE in the legend. Built from the same masked
            # integrand the difference figure integrates, so the two figures agree.
            #
            # Hatch only, with no face colour. These three positive areas overlap
            # almost completely, so any tinted fill blends them into a single patch and
            # destroys exactly the information the figure exists to show. Three hatch
            # directions stay separable no matter how much they overlap.
            skew.ax.fill_betweenx(
                parcel_analysis["pressure_hPa"],
                parcel_analysis["environment_virtual_temperature_C"],
                parcel_analysis["environment_virtual_temperature_C"]
                + parcel_analysis["cape_virtual_temperature_excess_C"],
                facecolor="none",
                edgecolor=color,
                hatch=SKEWT_OVERLAY_HATCHES[
                    series_ix % len(SKEWT_OVERLAY_HATCHES)
                ],
                alpha=0.45,
                linewidth=0,
            )

    # Colour identifies the setup, so line *style* has to carry which curve is which.
    # Spell that out rather than expecting the reader to infer it from three
    # near-parallel curves in the same colour.
    line_style_key_handles = [
        mpl.lines.Line2D([], [], color="0.3", linewidth=2.2, label="T"),
        mpl.lines.Line2D([], [], color="0.3", linewidth=1.3, label="$T_d$"),
        mpl.lines.Line2D(
            [],
            [],
            color="0.3",
            linewidth=1.1,
            linestyle="dotted",
            label="parcel",
        ),
    ]
    setup_legend = skew.ax.legend(
        loc="upper left", fontsize=8, frameon=True, framealpha=0.85
    )
    skew.ax.add_artist(setup_legend)
    skew.ax.legend(
        handles=line_style_key_handles,
        loc="lower right",
        fontsize=8,
        frameon=True,
        framealpha=0.85,
        title="curve",
        title_fontsize=8,
    )
    if title:
        skew.ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig, skew, parcel_analyses


def decompose_cape_difference(reference_analysis, comparison_analysis):
    """Split a CAPE difference into a parcel term, an environment term and a remainder.

    Why this exists: looking at a difference figure, the intuitive reading is that the
    change in CAPE should be the area between the two setups' *parcel* curves. It is
    not, and the gap is large — for the isolated convection in this study the
    parcel-to-parcel area is about -152 J/kg against a true change of -252 J/kg.

    The reason follows from the definition. CAPE integrates the gap between the parcel
    and its environment, so a difference of two CAPEs has two terms:

        CAPE_B - CAPE_A = integral of (Tv_parcel,B - Tv_parcel,A) d(-ln p)
                        - integral of (Tv_env,B    - Tv_env,A)    d(-ln p)

    The first term is the area between the parcel curves. The second is the area
    between the environment curves, entering with the opposite sign. They coincide only
    when the two setups share one environment, which is precisely what a harmonization
    changes. This function reports both, so a CAPE change caused by a colder parcel can
    be told apart from one caused by a warmer environment.

    Both terms are integrated over the *reference* setup's positive-buoyancy layer,
    because a shared layer is the only way the two are comparable. The two setups do not
    have the same LFC and EL, though, so those two terms do not account for the whole
    difference. Whatever they miss is returned as ``layer_change_term_j_kg``, defined as
    the exact remainder rather than as an independent integral, so the three columns
    always sum to the reported difference. A large remainder is itself a finding: it
    means the harmonization moved the *depth* of the buoyant layer rather than its
    strength.

    Args:
        reference_analysis: A ``calculate_parcel_analysis`` dict for the baseline setup.
        comparison_analysis: The same for the setup compared against it.

    Returns:
        A dict with ``parcel_term_j_kg``, ``environment_term_j_kg`` and
        ``layer_change_term_j_kg``.
    """
    shared_pressure_hPa = reference_analysis["pressure_hPa"]
    integration_weight = -np.log(shared_pressure_hPa)

    def on_shared_grid(profile_C):
        """Interpolate one of the comparison setup's profiles onto the reference levels.

        np.interp needs increasing sample points and pressure decreases with height, so
        reverse both arrays going in and reverse the result coming out.
        """
        return np.interp(
            shared_pressure_hPa[::-1],
            comparison_analysis["pressure_hPa"][::-1],
            profile_C[::-1],
        )[::-1]

    # Integrate over the reference setup's positive-buoyancy layer only.
    inside_reference_positive_area = (
        reference_analysis["cape_virtual_temperature_excess_C"] > 0
    )

    def integrate(difference_profile_C):
        return _DRY_AIR_GAS_CONSTANT_J_PER_KG_K * float(
            np.trapezoid(
                np.where(
                    inside_reference_positive_area, difference_profile_C, 0.0
                ),
                integration_weight,
            )
        )

    # How much of the change came from the parcel itself being warmer or colder.
    parcel_term_j_kg = integrate(
        on_shared_grid(comparison_analysis["parcel_virtual_temperature_C"])
        - reference_analysis["parcel_virtual_temperature_C"]
    )
    # How much came from the environment moving instead. Opposite sign: a warmer
    # environment reduces buoyancy and therefore reduces CAPE.
    environment_term_j_kg = -integrate(
        on_shared_grid(comparison_analysis["environment_virtual_temperature_C"])
        - reference_analysis["environment_virtual_temperature_C"]
    )

    cape_difference_j_kg = (
        comparison_analysis["cape_j_kg"] - reference_analysis["cape_j_kg"]
    )
    return {
        "parcel_term_j_kg": parcel_term_j_kg,
        "environment_term_j_kg": environment_term_j_kg,
        "layer_change_term_j_kg": (
            cape_difference_j_kg - parcel_term_j_kg - environment_term_j_kg
        ),
    }


def plot_skewt_cape_difference(
    reference_label,
    reference_sounding_df,
    comparison_label,
    comparison_sounding_df,
    parcel_start_index=0,
    figsize=(7.5, 8.0),
    title=None,
    zoom_to_cape=True,
    zoom_temperature_pad_C=3.0,
):
    """Overlay two setups on one skew-T and shade the *difference* in their CAPE.

    Mechanism, and why this is not simply two CAPE regions drawn on top of each other:
    on a skew-T the horizontal gap between two curves at a fixed pressure is exactly
    their temperature difference, and the vertical axis is -ln(p). The area between two
    curves is therefore proportional to an integral of a temperature difference over
    -ln(p), which is the form CAPE itself takes:

        CAPE = R_d * integral of (Tv_parcel - Tv_env) with respect to -ln(p)

    So if both setups' *positive-area integrands* are drawn against one shared baseline
    — here the reference setup's environment virtual temperature — then the region
    between the two resulting curves has area proportional to
    CAPE_comparison - CAPE_reference, exactly. That is the region this function shades,
    in two colours by sign:

    * red where the reference setup has the larger positive area
    * blue where the comparison setup does

    The signed sum of the two is the net CAPE change. It is printed on the panel next
    to the value recovered by integrating the shaded region, so the reader can check
    the eye against the number. Both setups' environment, dewpoint and parcel curves
    are drawn as well, so the *cause* of the area change — a warmer environment, a
    moister parcel, a displaced LFC — is visible alongside its size.

    The comparison setup's integrand is interpolated onto the reference's pressure
    levels before differencing, with zero fill outside its range. That matters here:
    the setups in this study are anchored at different pressures, so their columns do
    not start at the same surface pressure, and a level present in only one of them
    must contribute its own excess and nothing from the other.

    Args:
        reference_label: Display label of the setup treated as the baseline.
        reference_sounding_df: Its sounding DataFrame.
        comparison_label: Display label of the setup compared against it.
        comparison_sounding_df: Its sounding DataFrame.
        parcel_start_index: Row index the parcel is lifted from; 0 is the surface.
        figsize: Figure size in inches.
        title: Optional axes title.
        zoom_to_cape: Crop the panel to the buoyant layer instead of using the study's
            shared limits. The shared limits exist so panels can be compared by eye, but
            the difference band is only a few Kelvin wide, and on a full-depth panel it
            collapses to a sliver. This figure is read one panel at a time, so the crop
            is worth more here than the comparability is. The dewpoint curves run far to
            the left of the buoyant layer and are allowed to leave the panel.
        zoom_temperature_pad_C: Margin left and right of the cropped region, in Kelvin.

    Returns:
        (fig, skew, difference_summary) where ``difference_summary`` is a dict of the
        two CAPE values, their difference, and the difference recovered by integrating
        the shaded region — which should match to within the vertical grid's accuracy.
    """
    reference_analysis = calculate_parcel_analysis(
        reference_sounding_df, parcel_start_index
    )
    comparison_analysis = calculate_parcel_analysis(
        comparison_sounding_df, parcel_start_index
    )

    # Put both integrands on one pressure grid so they can be differenced. np.interp
    # needs increasing sample points and pressure decreases with height, so reverse
    # both arrays going in and reverse the result coming out.
    shared_pressure_hPa = reference_analysis["pressure_hPa"]
    comparison_excess_on_shared_grid = np.interp(
        shared_pressure_hPa[::-1],
        comparison_analysis["pressure_hPa"][::-1],
        comparison_analysis["cape_virtual_temperature_excess_C"][::-1],
        left=0.0,
        right=0.0,
    )[::-1]

    # The shared baseline both integrands are drawn against. Choosing the reference
    # environment means the reference setup's own shading edge lands exactly on its
    # parcel virtual-temperature curve, so the shaded region sits where its CAPE is.
    baseline_temperature_C = reference_analysis[
        "environment_virtual_temperature_C"
    ]
    reference_edge_C = (
        baseline_temperature_C
        + reference_analysis["cape_virtual_temperature_excess_C"]
    )
    comparison_edge_C = (
        baseline_temperature_C + comparison_excess_on_shared_grid
    )

    fig = plt.figure(figsize=figsize)
    # A quieter reference grid than the default. Once the panel is cropped, the dry and
    # moist adiabats crowd together, and at the default opacity they compete with the
    # parcel paths rather than supporting them.
    skew = _new_skewt_axes(fig, background_alpha=0.18)

    # A white outline laid under every profile line. The background grid runs diagonally
    # across the panel at a similar weight to a profile, so a thin coloured line on top of
    # it reads as one more adiabat. The halo cuts a gap through whatever the line crosses,
    # which is what separates the profiles from the grid without darkening either.
    white_halo_effects = [
        mpatheffects.withStroke(linewidth=3.2, foreground="white")
    ]
    # The parcel path gets a *narrower* halo than the environment curves, not a wider one,
    # despite being the heavier line. The reference setup's parcel curve runs within a few
    # hundredths of a Kelvin of the right-hand edge of the shaded difference band near the
    # equilibrium level, because the band edge is the parcel's virtual temperature and the
    # drawn line is its plain temperature, and the gap between those two shrinks with the
    # parcel's mixing ratio. A wide halo there paints white over the widest part of the
    # band. Measured on these soundings: the gap is 1.93 K at 773 hPa but only 0.09 K at
    # 307 hPa, while a 4.6 pt halo covers 0.42 K.
    parcel_halo_effects = [
        mpatheffects.withStroke(linewidth=2.6, foreground="white")
    ]

    # Both soundings, drawn before the shading so the fills read as background.
    for label, analysis, color, line_style, parcel_marker in [
        (
            reference_label,
            reference_analysis,
            SKEWT_OVERLAY_COLORS[0],
            "solid",
            "o",
        ),
        (
            comparison_label,
            comparison_analysis,
            SKEWT_OVERLAY_COLORS[1],
            (0, (6, 2)),
            "s",
        ),
    ]:
        skew.plot(
            analysis["pressure_hPa"],
            analysis["environment_temperature_C"],
            color=color,
            linestyle=line_style,
            linewidth=2.2,
            path_effects=white_halo_effects,
            label=f"{label} — CAPE {analysis['cape_j_kg']:.0f} J/kg",
        )
        skew.plot(
            analysis["pressure_hPa"],
            analysis["environment_dewpoint_C"],
            color=color,
            linestyle=line_style,
            linewidth=1.5,
            alpha=0.75,
            path_effects=white_halo_effects,
        )
        # The parcel path, deliberately the heaviest line in the panel. It is the curve
        # that sets where the shaded area is, so it has to survive being crossed by both
        # environment curves and the whole adiabat grid.
        #
        # Markers are placed by index rather than at every level. The soundings carry
        # about a hundred levels, so a marker per level would fuse into a solid band and
        # destroy the dashes that identify which setup the curve belongs to.
        parcel_marker_stride = max(1, len(analysis["pressure_hPa"]) // 14)
        skew.plot(
            analysis["pressure_hPa"],
            analysis["parcel_temperature_C"],
            color=color,
            linestyle="dashdot",
            linewidth=2.4,
            marker=parcel_marker,
            markevery=parcel_marker_stride,
            markersize=6.0,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.6,
            path_effects=parcel_halo_effects,
            label=f"{label} — parcel path",
        )

    # The reference setup's *entire* positive area, in light grey, purely to give the
    # difference a sense of scale. Without it a 350 J/kg sliver looks the same whether
    # the storm started with 500 J/kg or 2000.
    skew.ax.fill_betweenx(
        shared_pressure_hPa,
        baseline_temperature_C,
        reference_edge_C,
        color="0.55",
        alpha=0.15,
        linewidth=0,
        label=f"CAPE of {reference_label}",
    )

    # The difference itself, split by sign so gains and losses are distinguishable.
    # zorder above the profile lines, which default to 2. The band is only about a Kelvin
    # wide, and its right-hand edge coincides with the reference parcel curve, so leaving the
    # fill underneath lets a line and its halo erase the band exactly where it is widest.
    # Translucent at 0.4, so the lines it covers still read through it.
    for keep_where, shade_color, shade_label in [
        (
            reference_edge_C >= comparison_edge_C,
            SKEWT_CAPE_SHADE_COLOR,
            f"CAPE lost relative to {reference_label}",
        ),
        (
            reference_edge_C < comparison_edge_C,
            SKEWT_CIN_SHADE_COLOR,
            f"CAPE gained relative to {reference_label}",
        ),
    ]:
        skew.ax.fill_betweenx(
            shared_pressure_hPa,
            comparison_edge_C,
            reference_edge_C,
            where=keep_where,
            color=shade_color,
            alpha=0.4,
            linewidth=0,
            label=shade_label,
            interpolate=True,
            zorder=4,
        )

    # Both edges of the shaded region, drawn explicitly. Without these the band is
    # bounded by one plotted curve and one invisible one, and the natural (wrong) reading
    # is that the difference should be the area between the two *parcel* curves instead.
    # It should not: the environment moved as well as the parcel, and this transplanted
    # curve is what carries both effects at once.
    # Draw them only over the layer where at least one setup is buoyant. Outside it both
    # integrands are zero, so both edges collapse onto the baseline and would otherwise
    # trail a pair of meaningless lines the full depth of the panel.
    inside_either_positive_area = (
        reference_analysis["cape_virtual_temperature_excess_C"] > 0
    ) | (comparison_excess_on_shared_grid > 0)
    visible_edge = lambda edge_C: np.where(
        inside_either_positive_area, edge_C, np.nan
    )
    # skew.ax.plot(
    #     visible_edge(reference_edge_C),
    #     shared_pressure_hPa,
    #     color=SKEWT_OVERLAY_COLORS[0],
    #     linewidth=1.0,
    #     alpha=0.9,
    #     label=f"{reference_label} positive-area edge",
    # )
    # skew.ax.plot(
    #     visible_edge(comparison_edge_C),
    #     shared_pressure_hPa,
    #     color=SKEWT_OVERLAY_COLORS[1],
    #     linewidth=1.0,
    #     alpha=0.9,
    #     label=f"{comparison_label} edge, on {reference_label}'s environment",
    # )

    _annotate_characteristic_levels(skew, reference_analysis)

    # Crop the panel to the part of the column the figure is about. On the study's shared
    # limits the buoyant layer occupies maybe a third of the panel width and the whole
    # stratosphere is drawn for nothing, so the few-Kelvin difference band the figure
    # exists to show is thinner than the legend text.
    #
    # The crop runs from the surface up to a little above the higher of the two equilibrium
    # levels, so the LCL and LFC stay in frame and the stratosphere leaves it. Above the EL
    # the parcel is far colder than its environment, and letting that stretch set the
    # window would put the crop right back where it started.
    if zoom_to_cape:
        equilibrium_level_pressures_hPa = [
            analysis["el_pressure_hPa"]
            for analysis in (reference_analysis, comparison_analysis)
            if np.isfinite(analysis["el_pressure_hPa"])
        ]
        crop_top_pressure_hPa = (
            0.92 * min(equilibrium_level_pressures_hPa)
            if equilibrium_level_pressures_hPa
            else float(np.min(shared_pressure_hPa))
        )
        inside_crop = shared_pressure_hPa >= crop_top_pressure_hPa
        masked_to_crop = lambda curve_C: np.where(inside_crop, curve_C, np.nan)

        # Deliberately excludes the dewpoint curves. They sit tens of Kelvin left of the
        # temperature curves near the surface, and including them would widen the window
        # by more than the crop narrows it.
        _zoom_skewt_axes_to_curves(
            skew,
            shared_pressure_hPa,
            [
                masked_to_crop(reference_analysis["environment_temperature_C"]),
                masked_to_crop(reference_analysis["parcel_temperature_C"]),
                masked_to_crop(
                    np.interp(
                        shared_pressure_hPa[::-1],
                        comparison_analysis["pressure_hPa"][::-1],
                        comparison_analysis["environment_temperature_C"][::-1],
                    )[::-1]
                ),
                masked_to_crop(
                    np.interp(
                        shared_pressure_hPa[::-1],
                        comparison_analysis["pressure_hPa"][::-1],
                        comparison_analysis["parcel_temperature_C"][::-1],
                    )[::-1]
                ),
                masked_to_crop(reference_edge_C),
                masked_to_crop(comparison_edge_C),
            ],
            temperature_pad_C=zoom_temperature_pad_C,
        )

    # Recover the CAPE difference by integrating the shaded region, as a check that
    # the area the reader is looking at really is the number in the caption.
    cape_difference_j_kg = (
        comparison_analysis["cape_j_kg"] - reference_analysis["cape_j_kg"]
    )
    cape_difference_from_shaded_area_j_kg = cape_from_temperature_excess(
        shared_pressure_hPa, comparison_edge_C - reference_edge_C
    )
    difference_summary = {
        "reference_label": reference_label,
        "comparison_label": comparison_label,
        "reference_cape_j_kg": reference_analysis["cape_j_kg"],
        "comparison_cape_j_kg": comparison_analysis["cape_j_kg"],
        "cape_difference_j_kg": cape_difference_j_kg,
        "cape_difference_from_shaded_area_j_kg": (
            cape_difference_from_shaded_area_j_kg
        ),
    }
    difference_summary.update(
        decompose_cape_difference(reference_analysis, comparison_analysis)
    )

    relative_change = cape_difference_j_kg / max(
        reference_analysis["cape_j_kg"], 1e-9
    )
    skew.ax.text(
        0.02,
        0.02,
        f"{comparison_label} − {reference_label}\n"
        f"ΔCAPE = {cape_difference_j_kg:+.0f} J/kg ({relative_change:+.1%})\n"
        f"shaded area gives {cape_difference_from_shaded_area_j_kg:+.0f} J/kg",
        transform=skew.ax.transAxes,
        fontsize=_scaled_font_size(0.85),
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "0.7"},
    )
    # Upper right, not upper left. Skewed profiles lean to the upper left, so on a cropped
    # panel that corner is exactly where the top of the buoyant layer ends up, and a legend
    # there covers the part of the shading nearest the equilibrium level.
    skew.ax.legend(
        loc="upper right",
        fontsize=_scaled_font_size(0.68),
        frameon=True,
        framealpha=0.9,
    )
    if title:
        skew.ax.set_title(title, fontsize=_scaled_font_size(1.05))
    fig.tight_layout()
    return fig, skew, difference_summary


def skewt_cape_difference_table(difference_summaries):
    """Collect several ``plot_skewt_cape_difference`` summaries into one table.

    The figures show *where* in the column CAPE moved; this shows *how much* for every
    storm at once, which is the form the fairness question needs.

    Two groups of columns deserve reading rather than skipping. ``from_shaded_area`` is
    the difference recovered by integrating what the figure draws, kept deliberately: if
    it ever drifts away from ``difference``, the shading has stopped meaning what the
    caption claims. ``parcel_term``, ``environment_term`` and ``layer_change_term`` are
    the split described in ``decompose_cape_difference``, and they sum to ``difference``
    exactly. They answer *why* CAPE moved — a colder parcel, a warmer environment, or a
    buoyant layer of a different depth — which the total on its own cannot.
    """
    table_rows = []
    for summary in difference_summaries:
        reference_cape = summary["reference_cape_j_kg"]
        table_rows.append({
            "storm": summary.get("storm_label", ""),
            f"CAPE ({summary['reference_label']})": reference_cape,
            f"CAPE ({summary['comparison_label']})": summary[
                "comparison_cape_j_kg"
            ],
            "difference": summary["cape_difference_j_kg"],
            "percent_change": (
                100.0
                * summary["cape_difference_j_kg"]
                / max(reference_cape, 1e-9)
            ),
            "from_shaded_area": summary[
                "cape_difference_from_shaded_area_j_kg"
            ],
            "parcel_term": summary.get("parcel_term_j_kg", np.nan),
            "environment_term": summary.get("environment_term_j_kg", np.nan),
            "layer_change_term": summary.get("layer_change_term_j_kg", np.nan),
        })
    return pd.DataFrame(table_rows).set_index("storm")


########################################################################
############################## Storm animations ##############################
########################################################################

# Vertical velocity contour levels for the animations, in m/s. Deliberately the
# same asymmetric-around-zero set the manuscript's Figures 3-5 use, so an updraft
# in an animation reads at the same threshold as one in the paper.
ANIMATION_UPDRAFT_LEVELS_M_S = [-10, -5, -2, -1, -0.5, 0.5, 1, 2, 5, 10, 20]

# Fixed colour limits, shared across every frame *and* both arms. This is the
# single most important choice in the whole animation: if each frame autoscaled,
# a storm that weakens over time would look constant, and if each arm autoscaled
# separately the two would be visually incomparable — which would defeat the
# purpose of putting them in the same figure.
ANIMATION_CONDENSATE_LIMITS_KG_KG = (1e-5, 1e-2)
ANIMATION_RAIN_RATE_LIMITS_M_S = (2.7e-4, 1.2e-2)  # ~1 to ~43 mm/hr
ANIMATION_BUOYANCY_LIMIT_M_S2 = 0.08


def _storm_following_cross_section(subset_ds):
    """Slice one y row per output time, chosen to follow the storm.

    Mechanism: a cross section at a fixed y drifts off a moving storm and shows empty air
    for the later frames — which is exactly when two runs differ most. So the row is picked
    per frame from that frame's storm centroid, using the ``storm_position_y`` coordinate
    that ``read_or_compute_development_dataframes`` attaches. The centroid itself is the
    mean y of the gridpoints whose 5 km vertical velocity exceeds 20% of the run's peak
    updraft, so it tracks the updraft rather than the rain.

    All the rows are extracted in one vectorized pointwise indexing operation rather than in
    a loop over time. That keeps the whole thing a single lazy dask expression, so the
    caller can force one pass over the run instead of one pass per frame.

    Early frames can have no storm at all and therefore a NaN centroid. Those fall back to
    the median of the finite centroids, so the indexing cannot fail on a run whose first
    frames are quiescent.

    Args:
        subset_ds: The dataset to slice, already subsampled in space and time.

    Returns:
        ``(cross_section_ds, cross_section_y_values)`` — the sliced dataset with dims
        (time, z, x), and the y coordinate each frame was taken at, in metres.
    """
    if (
        "storm_position_y" in subset_ds.coords
        or "storm_position_y" in subset_ds.data_vars
    ):
        centroid_y_per_time = np.asarray(
            subset_ds["storm_position_y"].values, dtype=float
        )
    else:
        centroid_y_per_time = np.full(subset_ds.sizes["time"], np.nan)

    finite_centroids = centroid_y_per_time[np.isfinite(centroid_y_per_time)]
    fallback_y = (
        float(np.median(finite_centroids))
        if finite_centroids.size
        else float(np.median(subset_ds["y"].values))
    )
    centroid_y_per_time = np.where(
        np.isfinite(centroid_y_per_time), centroid_y_per_time, fallback_y
    )

    # Nearest y index to each frame's centroid, then one pointwise take across time.
    y_values = subset_ds["y"].values
    cross_section_y_indices = np.abs(
        y_values[None, :] - centroid_y_per_time[:, None]
    ).argmin(axis=1)
    cross_section_index = xr.DataArray(cross_section_y_indices, dims="time")

    return (
        subset_ds.isel({"y": cross_section_index}),
        y_values[cross_section_y_indices],
    )


def precompute_storm_animation_fields(
    storm_ds, spatial_stride=2, frame_stride=1
):
    """Reduce a storm's 4-D output to the handful of 2-D fields an animation needs.

    Mechanism: the animation needs six 2-D fields per frame, but reading them
    frame-by-frame would re-stream the whole dataset off disk once per frame. So
    this builds the entire set lazily and forces a *single* dask pass over the
    run, returning eager numpy-backed arrays that the animation then replays from
    memory. That turns the cost from O(frames x dataset) into O(dataset).

    The cross sections follow the storm rather than sitting at a fixed y: each
    frame is taken at that time's storm centroid, using the ``storm_position_y``
    coordinate attached from the development dataframe. A fixed cross section
    would drift off a moving storm and show empty air for the later frames, which
    is exactly when the comparisons matter most.

    Args:
        storm_ds: A postprocessed 5-minute storm dataset, with storm positions
            already attached by ``read_or_compute_development_dataframes``.
        spatial_stride: Take every Nth gridpoint in x and y. 2 quarters the cell
            count for a barely visible loss of detail, and keeps the GIF small.
        frame_stride: Take every Nth output time.

    Returns:
        An eager Dataset with the plan-view fields (time, y, x) and the
        storm-following cross-section fields (time, z, x).
    """
    subset_ds = storm_ds.isel({
        "time": slice(None, None, frame_stride),
        "x": slice(None, None, spatial_stride),
        "y": slice(None, None, spatial_stride),
    })

    cross_section_ds, cross_section_y_values = _storm_following_cross_section(
        subset_ds
    )
    y_values = subset_ds["y"].values

    animation_fields = xr.Dataset({
        # Plan view
        "surface_rain_rate": subset_ds["PCPRR"],
        "column_max_condensate": subset_ds["R_condensate"].max("z"),
        "w_at_5km": subset_ds["WC"].sel({"z": 5000}, method="nearest"),
        # Low-level mean buoyancy, for the cold-pool outline on the plan view
        "low_level_buoyancy": (
            subset_ds["buoyancy"].sel({"z": slice(0, 1000)}).mean("z")
        ),
        # Storm-following cross sections
        "cross_section_condensate": cross_section_ds["R_condensate"],
        "cross_section_w": cross_section_ds["WC"],
        "cross_section_buoyancy": cross_section_ds["buoyancy"],
    })
    # Record where each cross section was taken, so the plan view can mark it.
    animation_fields["cross_section_y"] = xr.DataArray(
        cross_section_y_values, dims="time"
    )

    # Carry the along-flow centroid through as well. The animation does not use it, but a
    # still figure does: it has to pick a window in x for each panel, and the storms in the
    # two arms have propagated different distances by any given time.
    if (
        "storm_position_x" in subset_ds.coords
        or "storm_position_x" in subset_ds.data_vars
    ):
        animation_fields["storm_center_x"] = xr.DataArray(
            np.asarray(subset_ds["storm_position_x"].values, dtype=float),
            dims="time",
        )

    # The single pass over the run.
    return animation_fields.compute()


def animate_storm_comparison(
    animation_fields_by_arm,
    storm_label,
    output_gif_path=None,
    frames_per_second=4,
    figure_width=16.0,
    row_height=3.6,
    dpi=80,
):
    """Animate one storm's evolution, one row of panels per arm.

    Mechanism: rows are arms and columns are fields, so reading straight down a
    column compares the same quantity between the original and harmonized runs at
    the same model time. Every panel uses colour limits fixed across all frames
    and shared between arms, so apparent differences are real differences rather
    than autoscaling artefacts.

    The four columns are chosen to span the storm from the ground up:

    1. Surface rain rate, with the cold-pool boundary outlined — where the storm
       is raining, which is what seeds the rain-sourced tracer in the first place.
    2. Column-maximum condensate with 5 km updraft contours — the storm's
       footprint and where it is actively lifting.
    3. Storm-following cross section of condensate with updraft/downdraft contours
       — the vertical structure, including whether an anvil forms.
    4. The same cross section of buoyancy — the cold pool, which drives the
       low-level convergence that sustains (or undercuts) the updraft.

    Each frame redraws its axes from scratch rather than updating artists in
    place. That is slower than blitting but is the only approach that works
    reliably for filled contours, whose collections cannot be mutated.

    Three rendering details are load-bearing and easy to get wrong:

    * **The figure needs an opaque background.** GIF stores frames as deltas, so
      with a transparent figure patch the previous frame's title shows through the
      next one and the timestamps pile up into an unreadable smear.
    * **Colourbars are built once, outside the frame loop**, from standalone
      ScalarMappables. Adding them inside the loop stacks a new colourbar onto the
      figure on every frame (and once per arm).
    * **dpi is passed explicitly to save().** The project's mplstyle sets a
      publication ``savefig.dpi``, which otherwise overrides the figure's own dpi
      and yields enormous frames.

    Args:
        animation_fields_by_arm: Mapping of arm label -> the Dataset returned by
            ``precompute_storm_animation_fields``.
        storm_label: Human-readable storm name for the title.
        output_gif_path: Where to write the GIF. Skipped when None.
        frames_per_second: Playback rate.
        figure_width: Figure width in inches.
        row_height: Height of one arm's row of panels, in inches.
        dpi: Frame resolution. Bounds the GIF size; 80 gives a legible ~1280 px.

    Returns:
        The matplotlib FuncAnimation.
    """
    arm_labels = list(animation_fields_by_arm)
    n_arms = len(arm_labels)

    fig, axs = plt.subplots(
        nrows=n_arms,
        ncols=4,
        figsize=(figure_width, row_height * n_arms),
        squeeze=False,
        dpi=dpi,
    )
    # Opaque background, or GIF delta-frames let old titles bleed through.
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.05, right=0.94, top=0.88, bottom=0.09, wspace=0.35, hspace=0.35
    )

    # Shared norms, built once from the fixed physical limits above.
    condensate_norm = mpl.colors.LogNorm(*ANIMATION_CONDENSATE_LIMITS_KG_KG)
    rain_norm = mpl.colors.LogNorm(*ANIMATION_RAIN_RATE_LIMITS_M_S)
    buoyancy_norm = mpl.colors.Normalize(
        -ANIMATION_BUOYANCY_LIMIT_M_S2, ANIMATION_BUOYANCY_LIMIT_M_S2
    )
    condensate_cmap = cm.plotting.transparent_under_cmap("viridis")
    rain_cmap = cm.plotting.transparent_under_cmap(cm.plotting.shifted_blues)

    # Cross-section vertical-velocity contours. Deliberately coarser than the
    # plan view's: the full Figure 3-5 level set resolves gravity waves in the
    # stratosphere, which buries the storm itself under contour spaghetti.
    cross_section_updraft_levels = [-5, -2, 2, 5, 10, 20]

    # Colourbars once, from standalone mappables rather than from a frame's artists.
    for column_ix, (cmap, norm, label) in enumerate([
        (rain_cmap, rain_norm, "rain rate (m/s)"),
        (condensate_cmap, condensate_norm, "condensate (kg/kg)"),
        (condensate_cmap, condensate_norm, "condensate (kg/kg)"),
        ("RdBu_r", buoyancy_norm, "buoyancy (m/s$^2$)"),
    ]):
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(
            mappable, ax=axs[:, column_ix].tolist(), fraction=0.04, pad=0.02
        )
        colorbar.set_label(label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    n_frames = min(
        fields.sizes["time"] for fields in animation_fields_by_arm.values()
    )

    def draw_frame(frame_ix):
        t_minutes = None
        for arm_ix, arm_label in enumerate(arm_labels):
            fields = animation_fields_by_arm[arm_label].isel({"time": frame_ix})
            x_km = fields["x"].values / 1000.0
            y_km = fields["y"].values / 1000.0
            z_km = fields["z"].values / 1000.0
            t_minutes = int(fields["t_minutes"].values)

            for ax in axs[arm_ix]:
                ax.clear()

            ########## Column 1: surface rain rate + cold pool ##########
            ax = axs[arm_ix, 0]
            ax.pcolormesh(
                x_km,
                y_km,
                fields["surface_rain_rate"].values,
                cmap=rain_cmap,
                norm=rain_norm,
                shading="auto",
            )
            # Cold-pool boundary, same -0.04 m/s^2 threshold as Figures 3-5
            ax.contour(
                x_km,
                y_km,
                fields["low_level_buoyancy"].values,
                levels=[-0.04],
                colors="black",
                linewidths=1.2,
                linestyles="dashed",
            )
            # Mark where this frame's storm-following cross section was taken
            ax.axhline(
                float(fields["cross_section_y"].values) / 1000.0,
                linestyle="dashdot",
                linewidth=1.2,
                color="gold",
            )
            ax.set_title("Surface rain rate + cold pool", fontsize=9)
            ax.set_ylabel(f"{arm_label}\n\ny (km)", fontsize=9)

            ########## Column 2: column-max condensate + 5 km updraft ##########
            ax = axs[arm_ix, 1]
            ax.pcolormesh(
                x_km,
                y_km,
                fields["column_max_condensate"].values,
                cmap=condensate_cmap,
                norm=condensate_norm,
                shading="auto",
            )
            ax.contour(
                x_km,
                y_km,
                fields["w_at_5km"].values,
                levels=[-2, 2, 10],
                colors=["steelblue", "firebrick", "black"],
                linewidths=0.9,
            )
            ax.set_title("Column-max condensate + W(5 km)", fontsize=9)
            ax.set_ylabel("y (km)", fontsize=9)

            ########## Column 3: cross section condensate + W ##########
            ax = axs[arm_ix, 2]
            ax.pcolormesh(
                x_km,
                z_km,
                fields["cross_section_condensate"].values,
                cmap=condensate_cmap,
                norm=condensate_norm,
                shading="auto",
            )
            ax.contour(
                x_km,
                z_km,
                fields["cross_section_w"].values,
                levels=cross_section_updraft_levels,
                colors="firebrick",
                linewidths=0.6,
            )
            ax.set_ylim(0, 18)
            ax.set_title("Cross section: condensate + W", fontsize=9)
            ax.set_ylabel("z (km)", fontsize=9)

            ########## Column 4: cross section buoyancy ##########
            ax = axs[arm_ix, 3]
            ax.pcolormesh(
                x_km,
                z_km,
                fields["cross_section_buoyancy"].values,
                cmap="RdBu_r",
                norm=buoyancy_norm,
                shading="auto",
            )
            ax.contour(
                x_km,
                z_km,
                fields["cross_section_w"].values,
                levels=[2, 10],
                colors="black",
                linewidths=0.7,
            )
            ax.set_ylim(0, 18)
            ax.set_title("Cross section: buoyancy + W", fontsize=9)
            ax.set_ylabel("z (km)", fontsize=9)

            for ax in axs[arm_ix]:
                ax.set_xlabel("x (km)", fontsize=9)
                ax.tick_params(labelsize=7)

        fig.suptitle(f"{storm_label} — t = {t_minutes} min", fontsize=15)
        return []

    animation = mplanim.FuncAnimation(
        fig=fig, func=draw_frame, frames=n_frames, blit=False
    )

    if output_gif_path is not None:
        output_gif_path = Path(output_gif_path)
        output_gif_path.parent.mkdir(parents=True, exist_ok=True)
        # pillow is the only writer available in this environment (no ffmpeg).
        animation.save(
            str(output_gif_path),
            writer="pillow",
            fps=frames_per_second,
            dpi=dpi,
            savefig_kwargs={"facecolor": "white"},
        )
        print(
            f"Wrote {output_gif_path}"
            f" ({output_gif_path.stat().st_size / 1e6:.1f} MB)"
        )

    plt.close(fig)
    return animation


########################################################################
############################## Still-frame storm comparison ##############################
########################################################################

# Condensate contour used to outline a storm in the overlay row, in kg/kg. Low enough to
# catch the anvil, high enough not to trace every wisp of cirrus the scheme produces.
STORM_OUTLINE_CONDENSATE_KG_KG = 1e-4

# Updraft contour used to mark the storm's core in the overlay row, in m/s.
STORM_OUTLINE_UPDRAFT_M_S = 5.0

# Cold-pool contour, in m/s^2. The same -0.04 threshold the manuscript's Figures 3-5 use,
# so a cold pool here reads at the same strength as one in the paper.
STORM_OUTLINE_BUOYANCY_M_S2 = -0.04

# One colour per arm in the overlay row. Deliberately not the condensate colormap's
# colours, so an outline can never be mistaken for a shaded field.
STORM_OVERLAY_ARM_COLORS = ["tab:blue", "tab:orange", "tab:purple"]


def _select_frame_indices(n_frames, times_minutes, minutes_per_frame, n_times):
    """Choose which frames a still figure should show, by minute or by even spacing.

    Explicit minutes win when given, because the reader usually has a specific moment in
    mind — the hour the squall line has finished organizing, say. Otherwise the frames are
    spread evenly over the *second half* of the run and always include the last one. Early
    frames are nearly identical between arms by construction, since both start from the
    same trigger, so spending panels on them wastes the figure.

    Args:
        n_frames: How many frames are available.
        times_minutes: Explicit model times to show, or None.
        minutes_per_frame: Model minutes between consecutive frames.
        n_times: How many frames to pick when ``times_minutes`` is None.

    Returns:
        A list of frame indices, ascending.
    """
    if times_minutes is not None:
        return [
            int(np.clip(round(minutes / minutes_per_frame), 0, n_frames - 1))
            for minutes in times_minutes
        ]

    # Evenly spaced over the second half, last frame included.
    first_frame_shown = n_frames // 2
    return list(
        np.unique(
            np.linspace(first_frame_shown, n_frames - 1, n_times)
            .round()
            .astype(int)
        )
    )


def plot_storm_structure_comparison(
    animation_fields_by_arm,
    storm_label,
    times_minutes=None,
    n_times=4,
    cross_section_half_width_km=45.0,
    max_height_km=16.0,
    cold_pool_max_height_km=4.0,
    panel_width=3.5,
    panel_height=2.9,
    include_overlay_row=False,
):
    """Condense the storm animation into one still figure of arm-versus-arm structure.

    Mechanism, and why this is laid out the way it is. The animation shows four fields
    over every frame, which is more information than a reader can hold. A still figure has
    to throw most of that away and keep whatever carries the arm-to-arm difference. The
    choice made here is to keep the **storm-following vertical cross section**, because
    every consequence of a CAPE change shows up in it at once: how deep the updraft goes,
    how much condensate reaches the anvil, how far the storm has propagated, and how strong
    the cold pool underneath has become. The plan views are dropped.

    The layout is rows against columns of different kinds, which is the point:

    * **One row per arm**, showing condensate shaded with updraft contours over it. Reading
      down a column compares the two arms at the same model minute, on identical colour
      limits, so a visible difference is a real difference.
    * **Columns are model times**, late in the run by default, because both arms start from
      the same trigger and only separate as the storms mature.

    Every key sits outside the panels, not inside them. Legends placed in an axes cost the
    figure exactly the corner where the anvil spreads, and at small panel sizes they cover
    more of the storm than they explain. The contour key is therefore a single row beneath
    the panels, and the colour scale is one bar down the right-hand side.

    Text sizes are multiples of the active ``font.size`` rather than fixed point values, so
    shrinking the panels and raising ``font.size`` in a ``plt.rc_context`` block does what
    the caller expects instead of leaving the annotations at their original size.

    Panels share one x window per column, centred between the two arms' storm centroids and
    plotted in absolute domain coordinates rather than relative to each storm. That is
    deliberate: a difference in propagation speed is one of the effects being looked for, and
    re-centring each panel on its own storm would subtract exactly that effect out.

    Args:
        animation_fields_by_arm: Mapping of arm label -> Dataset from
            ``precompute_storm_animation_fields``.
        storm_label: Human-readable storm name, used in the figure title.
        times_minutes: Explicit model times to show, in minutes. None spreads
            ``n_times`` panels over the second half of the run.
        n_times: Number of columns when ``times_minutes`` is None.
        cross_section_half_width_km: Half-width of each panel's x window.
        max_height_km: Top of the vertical axis.
        cold_pool_max_height_km: Ceiling on where the buoyancy contour is drawn. The
            -0.04 m/s^2 threshold is met not only by the cold pool but also by the
            negatively buoyant air around the anvil at 12-14 km, and contouring that as
            well scatters closed loops across the top of every panel. A cold pool is a
            near-surface feature, so the contour is cut off above this height.
        panel_width: Width of one panel, in inches.
        panel_height: Height of one panel, in inches.
        include_overlay_row: Add a further row that shades neither arm and instead draws
            both arms' *outlines* — condensate edge, updraft core, cold pool — one colour
            per arm. Off by default. It makes a difference in storm depth a distance on the
            page rather than something the reader estimates from memory, but it costs a
            third of the figure's height, and at small panel sizes the two shaded rows
            already carry the comparison.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (rows, columns).
    """
    arm_labels = list(animation_fields_by_arm)
    n_frames = min(
        fields.sizes["time"] for fields in animation_fields_by_arm.values()
    )

    # Output cadence, read off the data rather than assumed, since the notebooks can
    # subsample frames.
    reference_fields = animation_fields_by_arm[arm_labels[0]]
    frame_minutes = np.asarray(
        reference_fields["t_minutes"].values, dtype=float
    )
    minutes_per_frame = (
        float(frame_minutes[1] - frame_minutes[0])
        if frame_minutes.size > 1
        else 1.0
    )
    frame_indices = _select_frame_indices(
        n_frames, times_minutes, minutes_per_frame, n_times
    )

    n_rows = len(arm_labels) + (1 if include_overlay_row else 0)
    n_columns = len(frame_indices)
    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_columns,
        figsize=(panel_width * n_columns, panel_height * n_rows),
        squeeze=False,
        sharex="col",
        sharey=True,
        layout="constrained",
    )

    # The same fixed limits the animation uses, so a panel here and a frame there are
    # directly comparable.
    condensate_norm = mpl.colors.LogNorm(*ANIMATION_CONDENSATE_LIMITS_KG_KG)
    condensate_cmap = cm.plotting.transparent_under_cmap("viridis")

    for column_ix, frame_ix in enumerate(frame_indices):
        frames_by_arm = {
            arm_label: (
                animation_fields_by_arm[arm_label].isel({"time": frame_ix})
            )
            for arm_label in arm_labels
        }
        this_frame_minutes = int(
            frames_by_arm[arm_labels[0]]["t_minutes"].values
        )

        ########## One x window for the whole column ##########
        # Centre it between the arms' storms so both are in frame even once they have
        # separated, and fall back to the condensate centroid where no storm was tracked.
        panel_center_x_km = _shared_cross_section_center_x_km(frames_by_arm)
        panel_x_limits_km = (
            panel_center_x_km - cross_section_half_width_km,
            panel_center_x_km + cross_section_half_width_km,
        )

        # NaN out the buoyancy above the cold-pool ceiling, so ``contour`` simply has
        # nothing to draw there rather than being masked after the fact.
        def cold_pool_only(buoyancy_field, z_km):
            return np.where(
                z_km[:, None] <= cold_pool_max_height_km, buoyancy_field, np.nan
            )

        ########## One shaded row per arm ##########
        for arm_ix, arm_label in enumerate(arm_labels):
            fields = frames_by_arm[arm_label]
            x_km = fields["x"].values / 1000.0
            z_km = fields["z"].values / 1000.0
            ax = axs[arm_ix, column_ix]

            ax.pcolormesh(
                x_km,
                z_km,
                fields["cross_section_condensate"].values,
                cmap=condensate_cmap,
                norm=condensate_norm,
                shading="auto",
            )
            # Updrafts in one colour and downdrafts in another, both thin: the shading
            # carries the storm's shape and these only mark where it is being driven.
            ax.contour(
                x_km,
                z_km,
                fields["cross_section_w"].values,
                levels=[3, 5, 10, 20],
                colors="firebrick",
                linewidths=0.7,
            )
            ax.contour(
                x_km,
                z_km,
                fields["cross_section_w"].values,
                levels=[-5, -2],
                colors="steelblue",
                linewidths=0.6,
            )
            # The cold pool, which is what decides whether the updraft keeps being fed.
            ax.contour(
                x_km,
                z_km,
                cold_pool_only(fields["cross_section_buoyancy"].values, z_km),
                levels=[STORM_OUTLINE_BUOYANCY_M_S2],
                colors="black",
                linewidths=1.1,
                linestyles="dashed",
            )
            # The arm's name doubles as the row label. Two lines, not three: at small panel
            # sizes a blank spacer line pushes the axis label away from its own axis.
            if column_ix == 0:
                ax.set_ylabel(
                    f"{arm_label}\nz (km)", fontsize=_scaled_font_size(0.85)
                )

        ########## The overlay row: both arms as outlines, no shading ##########
        if include_overlay_row:
            ax = axs[len(arm_labels), column_ix]
            for arm_ix, arm_label in enumerate(arm_labels):
                fields = frames_by_arm[arm_label]
                x_km = fields["x"].values / 1000.0
                z_km = fields["z"].values / 1000.0
                arm_color = STORM_OVERLAY_ARM_COLORS[
                    arm_ix % len(STORM_OVERLAY_ARM_COLORS)
                ]
                # Taper the line width from arm to arm. Line *style* is already spoken for
                # here — it says which quantity a contour is — so it cannot also say which
                # arm. Width can. It matters because a storm this arm did not rerun is
                # identical in both arms, and a thin line drawn exactly over an equally thin
                # one of another colour looks like a single arm rather than two agreeing
                # ones. Drawing the first arm wider leaves it visible as a rim.
                arm_line_width_scale = 1.0 - 0.60 * arm_ix / max(
                    len(arm_labels) - 1, 1
                )

                # Storm outline: solid, and the heaviest of the three, because extent and
                # depth are what the eye should pick up first.
                ax.contour(
                    x_km,
                    z_km,
                    fields["cross_section_condensate"].values,
                    levels=[STORM_OUTLINE_CONDENSATE_KG_KG],
                    colors=[arm_color],
                    linewidths=3.4 * arm_line_width_scale,
                )
                # Updraft core: dotted.
                ax.contour(
                    x_km,
                    z_km,
                    fields["cross_section_w"].values,
                    levels=[STORM_OUTLINE_UPDRAFT_M_S],
                    colors=[arm_color],
                    linewidths=2.0 * arm_line_width_scale,
                    linestyles="dotted",
                )
                # Cold pool: dashed.
                ax.contour(
                    x_km,
                    z_km,
                    cold_pool_only(
                        fields["cross_section_buoyancy"].values, z_km
                    ),
                    levels=[STORM_OUTLINE_BUOYANCY_M_S2],
                    colors=[arm_color],
                    linewidths=2.0 * arm_line_width_scale,
                    linestyles="dashed",
                )
            if column_ix == 0:
                # Kept to one short word. "both arms, outlines only" is longer than the
                # row is tall at small panel sizes, and it runs into the label above it.
                # The legend beneath the figure carries the full explanation.
                ax.set_ylabel(
                    "outlines\nz (km)", fontsize=_scaled_font_size(0.85)
                )

        ########## Shared per-column formatting ##########
        for row_ix in range(n_rows):
            ax = axs[row_ix, column_ix]
            ax.set_xlim(*panel_x_limits_km)
            ax.set_ylim(0, max_height_km)
            ax.tick_params(labelsize=_scaled_font_size(0.7))

            # Cap the tick count rather than letting matplotlib choose. A 45 km window
            # invites a tick every 10 km, and on a panel an inch and a half wide those
            # labels collide into a grey band. ``prune`` drops the first and last, which
            # are the ones that overrun into the neighbouring column.
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3, prune="both"))
            # Fixed height ticks, so every row is read against the same rungs.
            ax.set_yticks(np.arange(0, max_height_km + 1, 5))

            if row_ix == 0:
                ax.set_title(
                    f"t = {this_frame_minutes} min",
                    fontsize=_scaled_font_size(0.9),
                )
            if row_ix == n_rows - 1:
                ax.set_xlabel("x (km)", fontsize=_scaled_font_size(0.85))

    ########## Colourbar and legend, built once for the whole figure ##########
    # The colour scale spans every shaded row, so a colour means the same thing in all of
    # them. When the outline row is present it is excluded, because nothing in it is shaded.
    condensate_mappable = mpl.cm.ScalarMappable(
        norm=condensate_norm, cmap=condensate_cmap
    )
    condensate_colorbar = fig.colorbar(
        condensate_mappable,
        ax=axs[: len(arm_labels), :].ravel().tolist(),
        fraction=0.03,
        pad=0.012,
        aspect=30,
    )
    condensate_colorbar.set_label(
        "condensate (kg/kg)", fontsize=_scaled_font_size(0.8)
    )
    condensate_colorbar.ax.tick_params(labelsize=_scaled_font_size(0.7))

    ########## The key, as rows beneath the panels ##########
    # Everything that needs explaining goes into one legend under the figure, rather than a
    # box inside the first panel. A legend placed in an axes lands on the corner where the
    # anvil spreads, which is the part of the storm the arms differ in most, and on a panel
    # an inch and a half wide it hides more than it explains.
    #
    # It also all goes *below*, never above. An "outside upper" legend and a suptitle both
    # claim the top strip of the figure and overprint each other.
    legend_handles = [
        mpl.lines.Line2D(
            [],
            [],
            color="firebrick",
            linewidth=1.4,
            label="W = +5, +10, +20 m/s",
        ),
        mpl.lines.Line2D(
            [], [], color="steelblue", linewidth=1.4, label="W = −2, −5 m/s"
        ),
        mpl.lines.Line2D(
            [],
            [],
            color="black",
            linewidth=1.4,
            linestyle="dashed",
            label=(
                f"cold pool: buoyancy = {STORM_OUTLINE_BUOYANCY_M_S2} m/s$^2$"
            ),
        ),
    ]

    if include_overlay_row:
        # The outline row encodes two things at once: colour says which arm, and style says
        # which quantity. Both sets are appended here, and the three-column layout below
        # wraps them onto their own lines underneath the shaded rows' key.
        legend_handles += [
            mpl.lines.Line2D(
                [],
                [],
                color=STORM_OVERLAY_ARM_COLORS[
                    arm_ix % len(STORM_OVERLAY_ARM_COLORS)
                ],
                linewidth=3.4
                * (1.0 - 0.60 * arm_ix / max(len(arm_labels) - 1, 1)),
                label=f"outlines: {arm_label}",
            )
            for arm_ix, arm_label in enumerate(arm_labels)
        ] + [
            mpl.lines.Line2D(
                [],
                [],
                color="0.3",
                linewidth=1.8,
                label=(
                    "outline: condensate "
                    f"{STORM_OUTLINE_CONDENSATE_KG_KG:g} kg/kg"
                ),
            ),
            mpl.lines.Line2D(
                [],
                [],
                color="0.3",
                linewidth=1.4,
                linestyle="dotted",
                label=f"outline: W = +{STORM_OUTLINE_UPDRAFT_M_S:g} m/s",
            ),
            mpl.lines.Line2D(
                [],
                [],
                color="0.3",
                linewidth=1.4,
                linestyle="dashed",
                label=f"outline: cold pool",
            ),
        ]

    # Three per line. That is the width the shaded rows' three entries already need, so the
    # optional outline entries wrap underneath them instead of stretching the figure.
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=_scaled_font_size(0.75),
        handlelength=1.6,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    fig.suptitle(
        f"{storm_label} — storm-following cross sections",
        fontsize=_scaled_font_size(1.1),
    )
    return fig, axs


def _shared_cross_section_center_x_km(frames_by_arm):
    """Pick one x centre for a column of panels that keeps every arm's storm in frame.

    The arms' storms propagate at different speeds, so by the end of a run their centroids
    can be tens of kilometres apart. Centring each panel on its own storm would hide that,
    since the whole point of the figure is that a difference in position is visible. So the
    centre is the mean of the arms' centroids, which keeps both storms in a common window
    and leaves their separation on the page as a horizontal offset.

    Where no centroid was tracked — early frames before a storm exists — the condensate
    centre of mass stands in, which is the same quantity computed the crude way.

    Args:
        frames_by_arm: Mapping of arm label -> a single time slice of the animation fields.

    Returns:
        The window centre in kilometres.
    """
    centroid_x_km_per_arm = []
    for fields in frames_by_arm.values():
        if "storm_center_x" in fields:
            centroid_x_m = float(fields["storm_center_x"].values)
            if np.isfinite(centroid_x_m):
                centroid_x_km_per_arm.append(centroid_x_m / 1000.0)
                continue
        # Fallback: condensate centre of mass along x in the cross section.
        x_km = fields["x"].values / 1000.0
        condensate_per_column = np.nansum(
            fields["cross_section_condensate"].values, axis=0
        )
        total_condensate = condensate_per_column.sum()
        centroid_x_km_per_arm.append(
            float((condensate_per_column * x_km).sum() / total_condensate)
            if total_condensate > 0
            else float(np.median(x_km))
        )
    return float(np.mean(centroid_x_km_per_arm))


########################################################################
############## W / condensate / rainfall animation ##############
########################################################################

# Surface accumulation variables RAMS writes, one per hydrometeor species. Summing all of
# them gives total precipitation reaching the ground, which is what "cumulative rainfall"
# means for these storms. Note this is a wider quantity than the ``PCPRR`` rain *rate* the
# other animation shades, which is the rain species alone.
#
# ACCPAERO is deliberately absent: it accumulates aerosol, not precipitation.
ACCUMULATED_PRECIPITATION_VARIABLES = [
    "ACCPR",  # rain
    "ACCPP",  # pristine ice
    "ACCPS",  # snow
    "ACCPA",  # aggregates
    "ACCPG",  # graupel
    "ACCPH",  # hail
]

# Vertical velocity colour limit for the plan views and cross sections, in m/s. Symmetric
# about zero so that an updraft and a downdraft of equal strength read as equally strong,
# which a one-sided scale would hide.
ANIMATION_W_LIMIT_M_S = 15.0

# Accumulated precipitation colour limits, in mm. Logarithmic, because the swath under a
# storm spans three orders of magnitude between its edge and its core, and a linear scale
# renders everything but the core as one flat colour.
ANIMATION_ACCUMULATED_PRECIPITATION_LIMITS_MM = (0.1, 100.0)

# Height the plan views of vertical velocity are taken at, in metres. 5 km is mid-updraft
# for these storms, and it is the level the storm centroid is defined on, so the plan view
# and the cross-section location are consistent with each other.
ANIMATION_PLAN_VIEW_HEIGHT_M = 5000.0


def precompute_w_condensate_rainfall_fields(
    storm_ds,
    spatial_stride=2,
    frame_stride=1,
    plan_view_height_m=ANIMATION_PLAN_VIEW_HEIGHT_M,
):
    """Reduce a run to the five 2-D fields the W/condensate/rainfall animation needs.

    Mechanism: the animation needs five fields per frame, and reading them frame by frame
    would re-stream the whole run off disk once per frame. So the whole set is built lazily
    and forced with a *single* dask pass, returning eager numpy-backed arrays the animation
    then replays from memory. That turns the cost from O(frames x dataset) into O(dataset).

    The five fields split three ways:

    * **Two plan views over the whole domain** — vertical velocity on one level, and
      column-maximum condensate. Both are full-domain, not cropped to the storm, so a
      reader can see where the storm sits relative to the boundaries.
    * **Two storm-following cross sections** of the same two quantities, so each plan view
      has a vertical counterpart directly beside it.
    * **One accumulated-precipitation plan view**, summed over every hydrometeor species.
      It is the only field that is cumulative rather than instantaneous, which is what makes
      it worth a panel: the other four reset every frame and show the storm now, while this
      one only ever grows and shows the whole track the storm has laid down.

    Accumulated precipitation is read from the model's own ``ACCP*`` variables rather than
    integrated from the rain rate here. RAMS accumulates it every model timestep, whereas
    the analysis output is written every five minutes, so integrating the 5-minute rate
    would alias out everything that happened between writes.

    Args:
        storm_ds: A postprocessed 5-minute storm dataset, with storm positions already
            attached by ``read_or_compute_development_dataframes``.
        spatial_stride: Take every Nth gridpoint in x and y. 2 quarters the cell count for
            a barely visible loss of detail and keeps the GIF small.
        frame_stride: Take every Nth output time.
        plan_view_height_m: Height the vertical-velocity plan view is taken at.

    Returns:
        An eager Dataset with the plan-view fields (time, y, x) and the storm-following
        cross-section fields (time, z, x).
    """
    subset_ds = storm_ds.isel({
        "time": slice(None, None, frame_stride),
        "x": slice(None, None, spatial_stride),
        "y": slice(None, None, spatial_stride),
    })

    cross_section_ds, cross_section_y_values = _storm_following_cross_section(
        subset_ds
    )

    # Total surface accumulation, species by species. Skip any the run did not write rather
    # than failing: a warm-rain-only configuration has no ACCPH, and the sum is still
    # meaningful without it.
    present_accumulation_variables = [
        variable
        for variable in ACCUMULATED_PRECIPITATION_VARIABLES
        if variable in subset_ds
    ]
    if not present_accumulation_variables:
        raise KeyError(
            "None of the accumulated-precipitation variables "
            f"{ACCUMULATED_PRECIPITATION_VARIABLES} are in this dataset."
        )
    accumulated_precipitation = sum(
        subset_ds[variable] for variable in present_accumulation_variables
    )

    animation_fields = xr.Dataset({
        # Plan views, full domain
        "plan_view_w": (
            subset_ds["WC"].sel({"z": plan_view_height_m}, method="nearest")
        ),
        "plan_view_column_max_condensate": subset_ds["R_condensate"].max("z"),
        "accumulated_precipitation": accumulated_precipitation,
        # Storm-following cross sections
        "cross_section_w": cross_section_ds["WC"],
        "cross_section_condensate": cross_section_ds["R_condensate"],
    })
    # Where each cross section was taken, so the plan views can mark the slice.
    animation_fields["cross_section_y"] = xr.DataArray(
        cross_section_y_values, dims="time"
    )
    # Record which species went into the sum, so a figure caption cannot misreport it.
    animation_fields.attrs["accumulated_precipitation_variables"] = ", ".join(
        present_accumulation_variables
    )
    animation_fields.attrs["plan_view_height_m"] = float(
        subset_ds["z"].sel({"z": plan_view_height_m}, method="nearest").values
    )

    # The single pass over the run.
    return animation_fields.compute()


def animate_w_condensate_rainfall(
    animation_fields_by_arm,
    storm_label,
    output_gif_path=None,
    frames_per_second=4,
    figure_width=19.0,
    row_height=3.6,
    dpi=80,
    max_cross_section_height_km=18.0,
):
    """Animate vertical velocity, condensate and accumulated rainfall, one row per arm.

    Mechanism: rows are arms and columns are fields, so reading straight down a column
    compares the same quantity between two runs at the same model time. Every panel uses
    colour limits fixed across all frames *and* shared between arms. That is the single most
    important choice here: if each frame autoscaled, a storm that weakened over time would
    look constant, and if each arm autoscaled separately the two rows would not be
    comparable at all, which would defeat the purpose of stacking them.

    The five columns pair each plan view with its vertical counterpart, then end on the
    cumulative panel:

    1. **Plan view of W** at ``plan_view_height_m`` — where the storm is lifting, and how
       the updrafts are organized in the horizontal.
    2. **Cross section of W** — how deep that lifting reaches, and where the compensating
       downdrafts sit.
    3. **Plan view of column-maximum condensate** — the storm's footprint.
    4. **Cross section of condensate** — its vertical structure, including the anvil.
    5. **Plan view of accumulated precipitation** — the whole track laid down so far. This
       is the only cumulative panel, and it is the one to watch for a difference that builds
       slowly: the other four reset every frame, so a small persistent difference in them is
       invisible, while here it integrates into a visible difference in swath.

    A gold line on each plan view marks where that frame's cross section was taken, so
    columns 1 and 3 can be read against columns 2 and 4 without guessing.

    Each frame redraws its axes from scratch rather than updating artists in place. That is
    slower than blitting, but it is the only approach that works for filled contours, whose
    collections cannot be mutated.

    Three rendering details are load-bearing and easy to get wrong:

    * **The figure needs an opaque background.** GIF stores frames as deltas, so with a
      transparent figure patch the previous frame's title shows through the next one and the
      timestamps pile into an unreadable smear.
    * **Colourbars are built once, outside the frame loop**, from standalone
      ScalarMappables. Adding them inside the loop stacks a new colourbar onto the figure on
      every frame, and once per arm.
    * **dpi is passed explicitly to save().** The project's mplstyle sets a publication
      ``savefig.dpi``, which otherwise overrides the figure's own dpi and yields enormous
      frames.

    Args:
        animation_fields_by_arm: Mapping of arm label -> the Dataset returned by
            ``precompute_w_condensate_rainfall_fields``.
        storm_label: Human-readable storm name for the title.
        output_gif_path: Where to write the GIF. Skipped when None.
        frames_per_second: Playback rate.
        figure_width: Figure width in inches.
        row_height: Height of one arm's row of panels, in inches.
        dpi: Frame resolution. Bounds the GIF size; 80 gives a legible ~1520 px.
        max_cross_section_height_km: Top of the cross-section vertical axis.

    Returns:
        The matplotlib FuncAnimation.
    """
    arm_labels = list(animation_fields_by_arm)
    n_arms = len(arm_labels)

    fig, axs = plt.subplots(
        nrows=n_arms,
        ncols=5,
        figsize=(figure_width, row_height * n_arms),
        squeeze=False,
        dpi=dpi,
    )
    # Opaque background, or GIF delta-frames let old titles bleed through.
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.045, right=0.955, top=0.87, bottom=0.09, wspace=0.42, hspace=0.32
    )

    # Shared norms, built once from the fixed physical limits above.
    w_norm = mpl.colors.Normalize(-ANIMATION_W_LIMIT_M_S, ANIMATION_W_LIMIT_M_S)
    condensate_norm = mpl.colors.LogNorm(*ANIMATION_CONDENSATE_LIMITS_KG_KG)
    accumulated_precipitation_norm = mpl.colors.LogNorm(
        *ANIMATION_ACCUMULATED_PRECIPITATION_LIMITS_MM
    )
    condensate_cmap = cm.plotting.transparent_under_cmap("viridis")
    accumulated_precipitation_cmap = cm.plotting.transparent_under_cmap(
        cm.plotting.shifted_blues
    )

    # The height the W plan view was taken at, read off the data rather than assumed, since
    # the caller can move it and the nearest model level is rarely the requested one.
    plan_view_height_km = (
        float(
            animation_fields_by_arm[arm_labels[0]].attrs.get(
                "plan_view_height_m", ANIMATION_PLAN_VIEW_HEIGHT_M
            )
        )
        / 1000.0
    )

    # Colourbars once, from standalone mappables rather than from a frame's artists.
    for column_ix, (cmap, norm, label) in enumerate([
        ("RdBu_r", w_norm, "W (m/s)"),
        ("RdBu_r", w_norm, "W (m/s)"),
        (condensate_cmap, condensate_norm, "condensate (kg/kg)"),
        (condensate_cmap, condensate_norm, "condensate (kg/kg)"),
        (
            accumulated_precipitation_cmap,
            accumulated_precipitation_norm,
            "accumulated precip (mm)",
        ),
    ]):
        mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        colorbar = fig.colorbar(
            mappable, ax=axs[:, column_ix].tolist(), fraction=0.045, pad=0.02
        )
        colorbar.set_label(label, fontsize=8)
        colorbar.ax.tick_params(labelsize=7)

    n_frames = min(
        fields.sizes["time"] for fields in animation_fields_by_arm.values()
    )

    def draw_frame(frame_ix):
        t_minutes = None
        for arm_ix, arm_label in enumerate(arm_labels):
            fields = animation_fields_by_arm[arm_label].isel({"time": frame_ix})
            x_km = fields["x"].values / 1000.0
            y_km = fields["y"].values / 1000.0
            z_km = fields["z"].values / 1000.0
            cross_section_y_km = (
                float(fields["cross_section_y"].values) / 1000.0
            )
            t_minutes = int(fields["t_minutes"].values)

            for ax in axs[arm_ix]:
                ax.clear()

            ########## Column 1: plan view of W ##########
            ax = axs[arm_ix, 0]
            ax.pcolormesh(
                x_km,
                y_km,
                fields["plan_view_w"].values,
                cmap="RdBu_r",
                norm=w_norm,
                shading="auto",
            )
            # Mark this frame's cross-section row, so columns 1 and 2 can be read together.
            ax.axhline(
                cross_section_y_km,
                linestyle="dashdot",
                linewidth=1.2,
                color="gold",
            )
            ax.set_title(
                f"Plan view: W ({plan_view_height_km:.1f} km)", fontsize=9
            )
            ax.set_ylabel(f"{arm_label}\n\ny (km)", fontsize=9)

            ########## Column 2: cross section of W ##########
            ax = axs[arm_ix, 1]
            ax.pcolormesh(
                x_km,
                z_km,
                fields["cross_section_w"].values,
                cmap="RdBu_r",
                norm=w_norm,
                shading="auto",
            )
            ax.set_ylim(0, max_cross_section_height_km)
            ax.set_title("Cross section: W", fontsize=9)
            ax.set_ylabel("z (km)", fontsize=9)

            ########## Column 3: plan view of column-max condensate ##########
            ax = axs[arm_ix, 2]
            ax.pcolormesh(
                x_km,
                y_km,
                fields["plan_view_column_max_condensate"].values,
                cmap=condensate_cmap,
                norm=condensate_norm,
                shading="auto",
            )
            ax.axhline(
                cross_section_y_km,
                linestyle="dashdot",
                linewidth=1.2,
                color="gold",
            )
            ax.set_title("Plan view: column-max condensate", fontsize=9)
            ax.set_ylabel("y (km)", fontsize=9)

            ########## Column 4: cross section of condensate ##########
            ax = axs[arm_ix, 3]
            ax.pcolormesh(
                x_km,
                z_km,
                fields["cross_section_condensate"].values,
                cmap=condensate_cmap,
                norm=condensate_norm,
                shading="auto",
            )
            ax.set_ylim(0, max_cross_section_height_km)
            ax.set_title("Cross section: condensate", fontsize=9)
            ax.set_ylabel("z (km)", fontsize=9)

            ########## Column 5: plan view of accumulated precipitation ##########
            ax = axs[arm_ix, 4]
            ax.pcolormesh(
                x_km,
                y_km,
                fields["accumulated_precipitation"].values,
                cmap=accumulated_precipitation_cmap,
                norm=accumulated_precipitation_norm,
                shading="auto",
            )
            ax.set_title("Plan view: accumulated precip", fontsize=9)
            ax.set_ylabel("y (km)", fontsize=9)

            for ax in axs[arm_ix]:
                ax.set_xlabel("x (km)", fontsize=9)
                ax.tick_params(labelsize=7)

        fig.suptitle(f"{storm_label} — t = {t_minutes} min", fontsize=15)
        return []

    animation = mplanim.FuncAnimation(
        fig=fig, func=draw_frame, frames=n_frames, blit=False
    )

    if output_gif_path is not None:
        output_gif_path = Path(output_gif_path)
        output_gif_path.parent.mkdir(parents=True, exist_ok=True)
        # pillow is the only writer available in this environment (no ffmpeg).
        animation.save(
            str(output_gif_path),
            writer="pillow",
            fps=frames_per_second,
            dpi=dpi,
            savefig_kwargs={"facecolor": "white"},
        )
        print(
            f"Wrote {output_gif_path}"
            f" ({output_gif_path.stat().st_size / 1e6:.1f} MB)"
        )

    plt.close(fig)
    return animation


########################################################################
############## Why a storm's rain-sourced entrainment moved ############
########################################################################

# The rain-sourced tracer these diagnostics lead with. It is emitted wherever the surface
# rain rate exceeds 10 mm/hr -- the middle of the five thresholds, and the one
# `collect_headline_metrics` quotes its single numbers from, so the two agree by
# construction rather than by coincidence.
REPRESENTATIVE_RAIN_SOURCED_TRACER = blt.RS_TRACERS[2]

# The fixed-source tracer total, used throughout this section as the control. Its 36
# component tracers are released from the boundary layer on a fixed schedule that takes no
# account of where it is raining, so it measures how much boundary-layer air of *any* origin
# the updraft is carrying. Comparing the two is what separates "this storm stopped lofting
# rain-cooled air" from "this storm stopped lofting boundary-layer air, and rain-cooled air
# went with it".
BOUNDARY_LAYER_CONTROL_TRACER = "total_fs_tracer"

# Horizontal area of one gridpoint, for turning the development dataframe's rain-area
# gridpoint counts into square kilometres. Every run is 500 m, and `blt` hardcodes the same
# 500 m where it builds `air_mass`.
RAIN_GRID_CELL_AREA_KM2 = 500.0**2 / 1e6

# Height separating "ingested at the updraft base" from "lofted into the storm body".
# Deliberately NOT a round number chosen in advance: it is where the squall line's two arms
# stop agreeing. Below it the harmonized run entrains rain-sourced air about as well as the
# original does; above it, it does not. Move it if another storm's crossover sits elsewhere.
LOW_LEVEL_ENTRAINMENT_TOP_M = 2000.0

# One colour and one dash pattern per arm, shared by every figure in this section, so a
# reader who has learned one figure's convention can read the next without its legend.
ENTRAINMENT_ARM_COLORS = STORM_OVERLAY_ARM_COLORS
ENTRAINMENT_ARM_LINE_STYLES = ["solid", (0, (5, 2)), (0, (1.5, 1.5))]


def rain_sourced_threshold_label(rain_sourced_tracer):
    """Human-readable surface rain threshold for a rain-sourced tracer variable name."""
    threshold_index = blt.RS_TRACERS.index(rain_sourced_tracer)
    return f"{blt.RS_TRACER_THRESHOLDS_MM_HR[threshold_index]} mm/hr"


def _rain_sourced_entrainment_terms(
    tracer_totals_ds,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
):
    """Pull the counts the entrainment identity is built from, one row per output time.

    Mechanism: everything in this section rests on three column-summed numbers that the
    tracer totals dataset already carries, each one summed over the storm's in-analysis
    gridpoints.

    * ``emitted`` (A) -- rain-sourced tracer particles released by the surface rain up to
      this time. This is the supply, and it only ever increases.
    * ``entrained`` (N) -- rain-sourced tracer particles inside the updraft right now.
    * ``updraft_mass`` (M) -- the mass of air the updraft is currently made of.

    A fourth, ``boundary_layer_entrained``, carries the fixed-source tracer total for use as
    a control. It is not part of the identity.

    Times with no updraft at all are dropped, because every ratio below divides by ``M``.
    Times with no rain yet are deliberately *kept*: a storm that has an updraft and no rain
    to put in it is a real state, and dropping it would hide exactly the delayed-onset
    behaviour this section is looking for.

    Args:
        tracer_totals_ds: One storm's column-summed tracer dataset, as returned by
            ``read_or_compute_tracer_totals_datasets``.
        rain_sourced_tracer: Which rain-sourced tracer to use. Defaults to
            ``REPRESENTATIVE_RAIN_SOURCED_TRACER`` (the 10 mm/hr threshold).
        analysis_window_minutes: (start, end) simulation-time window to keep, inclusive.
            A nonzero start is how a caller excludes the spin-up period.

    Returns:
        A DataFrame with columns ``t_minutes``, ``emitted``, ``entrained``,
        ``updraft_mass``, ``boundary_layer_entrained``, and the derived ratios described in
        ``rain_sourced_entrainment_factor_table``.
    """
    rain_sourced_tracer = (
        rain_sourced_tracer or REPRESENTATIVE_RAIN_SOURCED_TRACER
    )

    # The raw counts, straight out of the dataset and into plain numpy.
    simulation_minutes = np.asarray(
        tracer_totals_ds["t_minutes"].values, dtype=float
    )
    terms_df = pd.DataFrame({
        "t_minutes": simulation_minutes,
        "emitted": np.asarray(
            tracer_totals_ds["ACC" + rain_sourced_tracer + "_nc"].values,
            dtype=float,
        ),
        "entrained": np.asarray(
            tracer_totals_ds[rain_sourced_tracer + "_nc"].values, dtype=float
        ),
        "updraft_mass": np.asarray(
            tracer_totals_ds["air_mass"].values, dtype=float
        ),
        "boundary_layer_entrained": np.asarray(
            tracer_totals_ds[BOUNDARY_LAYER_CONTROL_TRACER + "_nc"].values,
            dtype=float,
        ),
        "boundary_layer_emitted": np.asarray(
            tracer_totals_ds[
                "ACC" + BOUNDARY_LAYER_CONTROL_TRACER + "_nc"
            ].values,
            dtype=float,
        ),
    })

    # Apply the caller's time window, then drop the frames with no updraft.
    window_start_minutes, window_end_minutes = analysis_window_minutes
    terms_df = terms_df[
        (terms_df["t_minutes"] >= window_start_minutes)
        & (terms_df["t_minutes"] <= window_end_minutes)
        & (terms_df["updraft_mass"] > 0)
    ].reset_index(drop=True)

    # The three ratios. Each is a quantity the manuscript already names.
    # Rain production efficiency: how much rain-sourced tracer the storm has made per
    # kilogram of updraft it is currently carrying.
    terms_df["rain_production_efficiency"] = (
        terms_df["emitted"] / terms_df["updraft_mass"]
    )
    # Entrainment efficiency: what fraction of everything ever emitted is in the updraft.
    # Undefined before the first rain falls, which is a real state and not an error.
    terms_df["entrainment_efficiency"] = np.where(
        terms_df["emitted"] > 0,
        terms_df["entrained"] / terms_df["emitted"].replace(0.0, np.nan),
        np.nan,
    )
    # The headline metric: in-updraft rain-sourced tracer mixing ratio.
    terms_df["rs_mixing_ratio"] = (
        terms_df["entrained"] / terms_df["updraft_mass"]
    )
    # The control: in-updraft boundary-layer tracer mixing ratio.
    terms_df["boundary_layer_mixing_ratio"] = (
        terms_df["boundary_layer_entrained"] / terms_df["updraft_mass"]
    )
    # What share of everything emitted anywhere is rain-sourced. The fixed-source tracers are
    # released on a schedule that no storm can influence, so this is a property of the rain
    # field alone: how much of the boundary layer has been rained on by now.
    terms_df["emitted_rain_sourced_share"] = terms_df["emitted"] / terms_df[
        "boundary_layer_emitted"
    ].replace(0.0, np.nan)
    # How much more rain-sourced the updraft's air is than the emission mix it draws from.
    # A value of one means the updraft samples the boundary layer indiscriminately.
    terms_df["rain_marking_selectivity"] = (
        terms_df["entrained"]
        / terms_df["boundary_layer_entrained"].replace(0.0, np.nan)
    ) / terms_df["emitted_rain_sourced_share"]
    return terms_df


def rain_sourced_entrainment_factor_table(
    tracer_totals_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
):
    """Split the change in one storm's rain-sourced entrainment into additive factors.

    Mechanism, and why the arithmetic is done this particular way.

    The headline metric is the in-updraft rain-sourced mixing ratio, q_RS = N / M, where N
    is the rain-sourced tracer count inside the updraft and M is the updraft's air mass.
    That quantity factors exactly through the total emitted, A::

        q_RS = N / M = (A / M) * (N / A) = eta_RP * eta_RSE

    with eta_RP the rain production efficiency and eta_RSE the rain-sourced entrainment
    efficiency, both of which the manuscript already defines. Taking logs turns the product
    into a sum, so the change between two arms decomposes into contributions that are
    guaranteed to total the observed change. That is the point of this table. "Entrainment
    fell" restates the question; "the supply fell 31 percent, the updraft shrank 53 percent,
    and the efficiency of getting rain-sourced air into the updraft fell 63 percent" answers
    it.

    Aggregation over the window is by **sum first, then divide** -- never by averaging the
    per-time ratios. This is not a cosmetic choice. The mean of N/A is not the same number
    as (sum N)/(sum A), so averaging the ratios breaks the identity and the contributions
    stop adding up. Summing first keeps the decomposition exact, and the function asserts
    that it closes rather than trusting that it does.

    A consequence worth knowing before reading the output: because the aggregate is a sum
    over the window, a run that reaches the same end state more slowly scores lower here.
    Run the table twice, once over the full simulation and once over a mature-storm window,
    to tell a genuine end-state difference from a delayed onset.

    Args:
        tracer_totals_dss_by_arm: Mapping of arm label -> {storm key -> column-summed
            tracer dataset}.
        storm_key: Which storm to decompose.
        reference_arm_label: The arm changes are measured *from*, normally the original.
        comparison_arm_label: The arm changes are measured *to*. Defaults to the first
            other arm present.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.

    Returns:
        A DataFrame indexed by factor name, with one column per arm plus ``ratio``,
        ``percent_change`` and ``log_ratio``. The ``log_ratio`` entries for emitted, updraft
        mass and entrainment efficiency combine to the one for the mixing ratio.
    """
    arm_labels = list(tracer_totals_dss_by_arm.keys())
    if comparison_arm_label is None:
        comparison_arm_label = next(
            label for label in arm_labels if label != reference_arm_label
        )

    # Aggregate each arm to the window sums the identity needs.
    window_sums_by_arm = {}
    for arm_label in [reference_arm_label, comparison_arm_label]:
        terms_df = _rain_sourced_entrainment_terms(
            tracer_totals_dss_by_arm[arm_label][storm_key],
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        summed_emitted = float(terms_df["emitted"].sum())
        summed_entrained = float(terms_df["entrained"].sum())
        summed_updraft_mass = float(terms_df["updraft_mass"].sum())
        summed_boundary_layer = float(
            terms_df["boundary_layer_entrained"].sum()
        )
        window_sums_by_arm[arm_label] = {
            "emitted (A, # particles)": summed_emitted,
            "updraft mass (M, kg)": summed_updraft_mass,
            "entrained (N, # particles)": summed_entrained,
            "rain production efficiency (A/M, #/kg)": (
                summed_emitted / summed_updraft_mass
            ),
            "entrainment efficiency (N/A)": summed_entrained / summed_emitted,
            "RS mixing ratio (N/M, #/kg)": (
                summed_entrained / summed_updraft_mass
            ),
            # The control, outside the identity: boundary-layer air per kg of updraft.
            "BL mixing ratio (control, #/kg)": (
                summed_boundary_layer / summed_updraft_mass
            ),
        }

    factor_table = pd.DataFrame(window_sums_by_arm)

    # The change, expressed three ways. The log ratio is the one that decomposes.
    factor_table["ratio"] = (
        factor_table[comparison_arm_label] / factor_table[reference_arm_label]
    )
    factor_table["percent_change"] = 100.0 * (factor_table["ratio"] - 1.0)
    factor_table["log_ratio"] = np.log(factor_table["ratio"])

    # Verify the decomposition closes. A mismatch here means the window sums were taken over
    # different sets of times, which would silently produce a plausible-looking table that
    # does not mean what it says.
    summed_contributions = (
        factor_table.loc["emitted (A, # particles)", "log_ratio"]
        - factor_table.loc["updraft mass (M, kg)", "log_ratio"]
        + factor_table.loc["entrainment efficiency (N/A)", "log_ratio"]
    )
    observed_change = factor_table.loc[
        "RS mixing ratio (N/M, #/kg)", "log_ratio"
    ]
    assert np.isclose(summed_contributions, observed_change, atol=1e-9), (
        "Rain-sourced entrainment decomposition does not close: contributions"
        f" sum to {summed_contributions}, observed change is {observed_change}."
    )

    return factor_table


def plot_rain_sourced_entrainment_factors(
    tracer_totals_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    rain_sourced_tracer=None,
    panel_width=3.7,
    panel_height=3.1,
):
    """Plot every factor in the entrainment identity against time, for both arms.

    Mechanism, and what the table cannot show. ``rain_sourced_entrainment_factor_table``
    reduces the whole simulation to one number per factor, which cannot tell "the harmonized
    storm never got there" from "the harmonized storm got there late". Those two have
    completely different implications for the manuscript, so the time series has to be
    looked at directly.

    The final panel carries the decomposition itself. It plots the log ratio between the
    arms of the mixing ratio and of its two factors at every output time, so a reader can
    see which factor is responsible at which stage of the storm's life. The two factor
    curves sum to the mixing-ratio curve at every point, by the same identity the table
    checks over the window.

    Args:
        tracer_totals_dss_by_arm: Mapping of arm label -> {storm key -> tracer dataset}.
        storm_key: Which storm to plot.
        reference_arm_label: The arm the log ratios are measured from.
        comparison_arm_label: The arm the log ratios are measured to. Defaults to the
            first other arm.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        panel_width: Width of one panel, in inches.
        panel_height: Height of one panel, in inches.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (2, 3).
    """
    rain_sourced_tracer = (
        rain_sourced_tracer or REPRESENTATIVE_RAIN_SOURCED_TRACER
    )
    arm_labels = list(tracer_totals_dss_by_arm.keys())
    if comparison_arm_label is None:
        comparison_arm_label = next(
            label for label in arm_labels if label != reference_arm_label
        )

    # One terms table per arm, computed once and reused by every panel.
    terms_by_arm = {
        arm_label: _rain_sourced_entrainment_terms(
            tracer_totals_dss_by_arm[arm_label][storm_key],
            rain_sourced_tracer=rain_sourced_tracer,
        )
        for arm_label in arm_labels
    }

    fig, axs = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(panel_width * 3, panel_height * 2),
        layout="constrained",
        sharex=True,
    )

    # The five absolute quantities, in the order the identity builds them up: supply,
    # updraft, supply per unit updraft, fraction of supply captured, and the product.
    absolute_panels = [
        ("emitted", "Emitted RS tracer\n$A$ (# particles)", True),
        ("updraft_mass", "Updraft mass\n$M$ (kg)", True),
        (
            "rain_production_efficiency",
            "Rain production efficiency\n" r"$\eta_{RP} = A/M$ (# kg$^{-1}$)",
            True,
        ),
        (
            "entrainment_efficiency",
            "RS entrainment efficiency\n" r"$\eta_{RSE} = N/A$",
            False,
        ),
        (
            "rs_mixing_ratio",
            "In-updraft RS mixing ratio\n" r"$q_{RS} = N/M$ (# kg$^{-1}$)",
            False,
        ),
    ]

    for panel_ix, (column_name, panel_title, use_log_scale) in enumerate(
        absolute_panels
    ):
        ax = axs.flat[panel_ix]
        for arm_ix, arm_label in enumerate(arm_labels):
            terms_df = terms_by_arm[arm_label]
            ax.plot(
                terms_df["t_minutes"],
                terms_df[column_name],
                color=ENTRAINMENT_ARM_COLORS[
                    arm_ix % len(ENTRAINMENT_ARM_COLORS)
                ],
                linestyle=ENTRAINMENT_ARM_LINE_STYLES[
                    arm_ix % len(ENTRAINMENT_ARM_LINE_STYLES)
                ],
                linewidth=2.0,
                label=arm_label,
            )
        ax.set_title(
            blt.axes_title(panel_ix, panel_title),
            fontsize=_scaled_font_size(0.8),
        )
        # These quantities climb through several orders of magnitude while the storm spins
        # up, so a linear axis would compress the whole developing phase onto the floor.
        if use_log_scale:
            ax.set_yscale("log")
        blt.label_x_axis(ax, short=True)
        # `sharex` hides the tick labels on the top row, so leaving its axis label in place
        # would print "Sim. time (minutes)" under an axis showing no times.
        if panel_ix < 3:
            ax.set_xlabel("")
        ax.tick_params(labelsize=_scaled_font_size(0.7))
        ax.xaxis.label.set_size(_scaled_font_size(0.75))

    ########## The decomposition panel ##########
    decomposition_ax = axs.flat[5]
    reference_terms_df = terms_by_arm[reference_arm_label].set_index(
        "t_minutes"
    )
    comparison_terms_df = terms_by_arm[comparison_arm_label].set_index(
        "t_minutes"
    )
    # Only times present in both arms can be differenced at all.
    shared_minutes = reference_terms_df.index.intersection(
        comparison_terms_df.index
    )

    for column_name, curve_label, curve_style in [
        (
            "rs_mixing_ratio",
            r"$\Delta \ln q_{RS}$ (total)",
            {"color": "black", "linewidth": 2.6},
        ),
        (
            "rain_production_efficiency",
            r"$\Delta \ln \eta_{RP}$",
            {"color": "tab:green", "linewidth": 2.0, "linestyle": (0, (5, 2))},
        ),
        (
            "entrainment_efficiency",
            r"$\Delta \ln \eta_{RSE}$",
            {
                "color": "tab:red",
                "linewidth": 2.0,
                "linestyle": (0, (1.5, 1.5)),
            },
        ),
    ]:
        # A zero in either arm sends the log to infinity, which would blow up the y scale
        # and hide the finite part of the curve. Suppress the warning the division and the
        # log both raise, then break the line at those times instead of drawing them.
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log(
                comparison_terms_df.loc[shared_minutes, column_name].values
                / reference_terms_df.loc[shared_minutes, column_name].values
            )
        log_ratio = np.where(np.isfinite(log_ratio), log_ratio, np.nan)
        decomposition_ax.plot(
            shared_minutes, log_ratio, label=curve_label, **curve_style
        )

    # Scale the panel from the times when both arms are actually entraining rain-sourced
    # air, defined as at least 1% of that arm's own peak mixing ratio. A bare "greater than
    # zero" test is not enough: a storm one output time into its first rain carries a
    # numerically nonzero but physically meaningless amount, and the log ratio there runs to
    # tens of natural-log units. On a shared scale that single spike flattens the whole
    # mature-storm period onto the zero line, which is the period the panel exists to show.
    # The early part is still drawn; it simply runs off the bottom of the panel.
    reference_mixing_ratio = reference_terms_df.loc[
        shared_minutes, "rs_mixing_ratio"
    ].values
    comparison_mixing_ratio = comparison_terms_df.loc[
        shared_minutes, "rs_mixing_ratio"
    ].values
    is_actively_entraining = (
        reference_mixing_ratio >= 0.01 * np.nanmax(reference_mixing_ratio)
    ) & (comparison_mixing_ratio >= 0.01 * np.nanmax(comparison_mixing_ratio))
    if is_actively_entraining.any():
        active_log_ratios = np.log(
            comparison_mixing_ratio[is_actively_entraining]
            / reference_mixing_ratio[is_actively_entraining]
        )
        log_ratio_margin = max(0.25 * np.ptp(active_log_ratios), 0.5)
        decomposition_ax.set_ylim([
            active_log_ratios.min() - log_ratio_margin,
            max(active_log_ratios.max(), 0.0) + log_ratio_margin,
        ])

    # Zero is "no change between the arms", so it is the line the reader compares against.
    decomposition_ax.axhline(0.0, color="0.6", linewidth=1.0, zorder=0)
    decomposition_ax.set_title(
        blt.axes_title(
            5,
            f"{comparison_arm_label} vs {reference_arm_label}\n"
            r"$\Delta \ln q_{RS} = \Delta \ln \eta_{RP} +"
            r" \Delta \ln \eta_{RSE}$",
        ),
        fontsize=_scaled_font_size(0.8),
    )
    decomposition_ax.set_ylabel("Log ratio", fontsize=_scaled_font_size(0.75))
    decomposition_ax.legend(fontsize=_scaled_font_size(0.65), loc="lower right")
    blt.label_x_axis(decomposition_ax, short=True)
    decomposition_ax.tick_params(labelsize=_scaled_font_size(0.7))
    decomposition_ax.xaxis.label.set_size(_scaled_font_size(0.75))

    # One figure legend for the arms, so five of the six panels need no legend of their own.
    fig.legend(
        *axs.flat[0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(arm_labels),
        frameon=False,
        fontsize=_scaled_font_size(0.8),
    )
    fig.suptitle(
        f"{blt.STORM_LABELS[storm_key]} -- rain-sourced entrainment factors"
        f" ({rain_sourced_threshold_label(rain_sourced_tracer)})",
        fontsize=_scaled_font_size(1.0),
    )
    return fig, axs


def rain_sourced_entrainment_profile_frame(
    tracer_totals_profile_ds,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
):
    """Reduce one run's tracer profile dataset to a time-integrated vertical profile.

    Mechanism: the profile dataset holds, for every output time and every model level, the
    tracer counts and the air mass inside the updraft at that level. Summing over the time
    window rather than averaging is what keeps the level-by-level mixing ratio consistent
    with the column number the factor table reports. The column value is the sum over levels
    of the numerator divided by the sum over levels of the denominator, and that only holds
    if both are accumulated the same way.

    The model's level 0 sits below ground and is dropped, matching what
    ``read_or_compute_tracer_totals_datasets`` does when it builds the column sums.

    Args:
        tracer_totals_profile_ds: One run's (time, z) tracer profile dataset.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.

    Returns:
        A DataFrame indexed by height in metres, with columns ``entrained``,
        ``boundary_layer_entrained``, ``updraft_mass``, ``rs_mixing_ratio`` and
        ``boundary_layer_mixing_ratio``.
    """
    rain_sourced_tracer = (
        rain_sourced_tracer or REPRESENTATIVE_RAIN_SOURCED_TRACER
    )

    # Drop the below-ground level, then keep only the requested times.
    above_ground_ds = tracer_totals_profile_ds.isel(
        {"z": slice(1, tracer_totals_profile_ds.sizes["z"])}
    )
    window_start_minutes, window_end_minutes = analysis_window_minutes
    simulation_minutes = np.asarray(
        above_ground_ds["t_minutes"].values, dtype=float
    )
    in_window = (simulation_minutes >= window_start_minutes) & (
        simulation_minutes <= window_end_minutes
    )
    windowed_ds = above_ground_ds.isel({"time": np.flatnonzero(in_window)})

    # Accumulate over time at each level.
    summed_entrained = (
        windowed_ds[rain_sourced_tracer + "_nc"].sum("time").values
    )
    summed_boundary_layer = (
        windowed_ds[BOUNDARY_LAYER_CONTROL_TRACER + "_nc"].sum("time").values
    )
    summed_updraft_mass = windowed_ds["air_mass"].sum("time").values

    # Mixing ratios per level, undefined where the updraft never reached that level.
    has_updraft = summed_updraft_mass > 0
    profile_df = pd.DataFrame(
        {
            "entrained": summed_entrained,
            "boundary_layer_entrained": summed_boundary_layer,
            "updraft_mass": summed_updraft_mass,
            "rs_mixing_ratio": np.where(
                has_updraft,
                summed_entrained
                / np.where(has_updraft, summed_updraft_mass, 1.0),
                np.nan,
            ),
            "boundary_layer_mixing_ratio": np.where(
                has_updraft,
                summed_boundary_layer
                / np.where(has_updraft, summed_updraft_mass, 1.0),
                np.nan,
            ),
        },
        index=pd.Index(
            np.asarray(windowed_ds["z"].values, dtype=float), name="z_m"
        ),
    )
    return profile_df


def plot_rain_sourced_entrainment_profiles(
    tracer_totals_profile_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
    max_height_km=16.0,
    panel_width=3.1,
    panel_height=5.4,
):
    """Plot where in the column the rain-sourced entrainment changed, against a control.

    Mechanism, and why this is the figure that answers the question. Every other diagnostic
    in this section is a column total, and a column total cannot tell an *ingestion* failure
    from a *lofting* failure. A storm that never gets rain-cooled air into its updraft base,
    and a storm that gets it in and then fails to carry it upward, both show the same drop
    in column-integrated entrainment. They are different storms with different causes.

    Resolving the same numbers by height separates them at once. If the arms agree at low
    levels and diverge aloft, the updraft base is still ingesting rain-sourced air and the
    loss is in the lofting. If they diverge at the surface, the ingestion itself changed.

    The fixed-source tracer is carried through every panel as the control, and it is what
    makes the answer specific. Those tracers are released from the boundary layer on a fixed
    schedule that ignores the rain, so they measure how much boundary-layer air of any
    origin the updraft carries to each height. If the rain-sourced and fixed-source profiles
    fall by the same factor at the same heights, the storm has a general lofting problem and
    the rain-sourced tracer is only along for the ride. If the rain-sourced profile falls
    further, something is specific to rain-cooled air.

    The last panel makes that comparison directly, by dividing the rain-sourced profile
    ratio by the fixed-source profile ratio. A value of one there means the whole
    rain-sourced change is explained by the general boundary-layer change.

    Args:
        tracer_totals_profile_dss_by_arm: Mapping of arm label -> {storm key -> (time, z)
            tracer profile dataset}.
        storm_key: Which storm to plot.
        reference_arm_label: The arm the ratio panels divide by.
        comparison_arm_label: The arm the ratio panels divide into it. Defaults to the
            first other arm.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_window_minutes: (start, end) simulation-time window, inclusive. Restricting
            to the mature storm is usually clearer, because the early frames add almost
            nothing to the integral but do widen the height range that has any data at all.
        max_height_km: Top of the vertical axis.
        panel_width: Width of one panel, in inches.
        panel_height: Height of the panel row, in inches.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (1, 5).
    """
    rain_sourced_tracer = (
        rain_sourced_tracer or REPRESENTATIVE_RAIN_SOURCED_TRACER
    )
    arm_labels = list(tracer_totals_profile_dss_by_arm.keys())
    if comparison_arm_label is None:
        comparison_arm_label = next(
            label for label in arm_labels if label != reference_arm_label
        )

    # One time-integrated profile per arm.
    profiles_by_arm = {
        arm_label: rain_sourced_entrainment_profile_frame(
            tracer_totals_profile_dss_by_arm[arm_label][storm_key],
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        for arm_label in arm_labels
    }
    heights_km = profiles_by_arm[reference_arm_label].index.values / 1000.0

    fig, axs = plt.subplots(
        nrows=1,
        ncols=5,
        figsize=(panel_width * 5, panel_height),
        layout="constrained",
        sharey=True,
        squeeze=False,
    )

    ########## The three absolute profiles ##########
    absolute_panels = [
        ("entrained", "Rain-sourced tracer\n$N_{RS}(z)$ (# particles)"),
        (
            "boundary_layer_entrained",
            "Boundary-layer tracer\n$N_{FS}(z)$ (# particles)",
        ),
        ("updraft_mass", "Updraft mass\n$M(z)$ (kg)"),
    ]
    for panel_ix, (column_name, panel_title) in enumerate(absolute_panels):
        ax = axs[0, panel_ix]
        for arm_ix, arm_label in enumerate(arm_labels):
            profile_df = profiles_by_arm[arm_label]
            # A zero plots as a gap on a log axis rather than as a spike off the left edge.
            plotted_values = profile_df[column_name].replace(0.0, np.nan)
            ax.plot(
                plotted_values.values,
                heights_km,
                color=ENTRAINMENT_ARM_COLORS[
                    arm_ix % len(ENTRAINMENT_ARM_COLORS)
                ],
                linestyle=ENTRAINMENT_ARM_LINE_STYLES[
                    arm_ix % len(ENTRAINMENT_ARM_LINE_STYLES)
                ],
                linewidth=2.0,
                label=arm_label,
            )
        ax.set_xscale("log")
        ax.set_title(
            blt.axes_title(panel_ix, panel_title),
            fontsize=_scaled_font_size(0.78),
        )
        ax.tick_params(labelsize=_scaled_font_size(0.7))

    ########## Panel (d): all three ratios on one axis ##########
    reference_profile_df = profiles_by_arm[reference_arm_label]
    comparison_profile_df = profiles_by_arm[comparison_arm_label]

    def profile_ratio(column_name):
        """Comparison arm over reference arm at each level, NaN where undefined."""
        reference_values = reference_profile_df[column_name].values
        comparison_values = comparison_profile_df[column_name].values
        # Only divide where the reference has something to divide by, and keep only
        # strictly positive results so the log axis has something to draw.
        ratio = np.where(
            reference_values > 0,
            comparison_values
            / np.where(reference_values > 0, reference_values, 1.0),
            np.nan,
        )
        return np.where(ratio > 0, ratio, np.nan)

    ratio_ax = axs[0, 3]
    for column_name, curve_label, curve_style in [
        ("entrained", "$N_{RS}$", {"color": "tab:red", "linewidth": 2.4}),
        (
            "boundary_layer_entrained",
            "$N_{FS}$",
            {"color": "tab:green", "linewidth": 2.0, "linestyle": (0, (5, 2))},
        ),
        (
            "updraft_mass",
            "$M$",
            {
                "color": "tab:blue",
                "linewidth": 2.0,
                "linestyle": (0, (1.5, 1.5)),
            },
        ),
    ]:
        ratio_ax.plot(
            profile_ratio(column_name),
            heights_km,
            label=curve_label,
            **curve_style,
        )
    ratio_ax.set_title(
        blt.axes_title(3, f"{comparison_arm_label}\n/ {reference_arm_label}"),
        fontsize=_scaled_font_size(0.78),
    )
    ratio_ax.legend(fontsize=_scaled_font_size(0.7), loc="upper right")

    ########## Panel (e): how much of the change is rain-specific ##########
    # Dividing the rain-sourced ratio by the boundary-layer ratio removes everything the two
    # tracers have in common -- updraft size, updraft depth, general dilution -- and leaves
    # only what is specific to air that came from under the rain.
    specificity_ax = axs[0, 4]
    with np.errstate(divide="ignore", invalid="ignore"):
        rain_specific_residual = profile_ratio("entrained") / profile_ratio(
            "boundary_layer_entrained"
        )
    specificity_ax.plot(
        rain_specific_residual,
        heights_km,
        color="black",
        linewidth=2.4,
    )
    specificity_ax.set_title(
        blt.axes_title(
            4, "Rain-specific residual\n$(N_{RS}$ ratio$)/(N_{FS}$ ratio$)$"
        ),
        fontsize=_scaled_font_size(0.78),
    )

    # Both ratio panels are read against unity, and both mark the low-level/lofted split.
    for ax in [ratio_ax, specificity_ax]:
        ax.axvline(1.0, color="0.6", linewidth=1.0, zorder=0)
        ax.axhline(
            LOW_LEVEL_ENTRAINMENT_TOP_M / 1000.0,
            color="0.4",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            zorder=0,
        )
        ax.tick_params(labelsize=_scaled_font_size(0.7))

    # Panel (d) spans orders of magnitude and needs a log axis. Panel (e) does not: the
    # residual is a number near one, and a log axis there packs four tick labels into the
    # narrow decade below unity where they overlap each other.
    ratio_ax.set_xscale("log")
    finite_residual = rain_specific_residual[
        np.isfinite(rain_specific_residual)
    ]
    specificity_ax.set_xlim([
        0.0,
        max(1.2, 1.05 * finite_residual.max() if finite_residual.size else 1.2),
    ])

    axs[0, 0].set_ylabel("z (km)", fontsize=_scaled_font_size(0.85))
    axs[0, 0].set_ylim([0, max_height_km])

    window_start_minutes, window_end_minutes = analysis_window_minutes
    window_description = (
        f"t = {window_start_minutes:g}-{min(window_end_minutes, 180.0):g} min"
    )
    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(arm_labels),
        frameon=False,
        fontsize=_scaled_font_size(0.8),
    )
    fig.suptitle(
        f"{blt.STORM_LABELS[storm_key]} -- where the entrained tracer sits"
        f" ({rain_sourced_threshold_label(rain_sourced_tracer)},"
        f" {window_description})",
        fontsize=_scaled_font_size(1.0),
    )
    return fig, axs


def rain_sourced_entrainment_height_partition_table(
    tracer_totals_profile_dss_by_arm,
    storm_key,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
    low_level_top_m=None,
):
    """Quantify the split the profile figure shows, so it can be quoted in text.

    Mechanism: two scalars per quantity say where the material inside the updraft actually
    sits. The first is the fraction of it below ``low_level_top_m``; the second is its
    count-weighted mean height, which is the centre of gravity of the profile.

    Three quantities are reported, and the comparison between them is the point. The
    rain-sourced tracer is what the question is about. The fixed-source tracer is
    boundary-layer air released without regard to the rain, so it says whether any change is
    specific to rain-cooled air. The updraft mass is the storm's own depth, so it says
    whether the storm still has a deep updraft at all. A run whose tracer centre of gravity
    collapses while its updraft mass centre of gravity barely moves still has its depth, and
    is simply not carrying low-level air up it.

    Args:
        tracer_totals_profile_dss_by_arm: Mapping of arm label -> {storm key -> profile ds}.
        storm_key: Which storm to summarize.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.
        low_level_top_m: Height dividing low-level from lofted. Defaults to
            ``LOW_LEVEL_ENTRAINMENT_TOP_M``.

    Returns:
        A DataFrame indexed by arm label.
    """
    low_level_top_m = low_level_top_m or LOW_LEVEL_ENTRAINMENT_TOP_M

    partition_rows = {}
    for arm_label, profile_dss in tracer_totals_profile_dss_by_arm.items():
        profile_df = rain_sourced_entrainment_profile_frame(
            profile_dss[storm_key],
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        heights_m = profile_df.index.values
        is_low_level = heights_m < low_level_top_m

        partition_row = {}
        for column_name, display_name in [
            ("entrained", "RS tracer"),
            ("boundary_layer_entrained", "BL tracer"),
            ("updraft_mass", "updraft mass"),
        ]:
            level_values = profile_df[column_name].values
            column_total = level_values.sum()
            partition_row[
                f"{display_name}: frac below {low_level_top_m / 1000:g} km"
            ] = (level_values[is_low_level].sum() / column_total)
            partition_row[f"{display_name}: mean height (m)"] = (
                level_values * heights_m
            ).sum() / column_total
        partition_rows[arm_label] = partition_row

    return pd.DataFrame(partition_rows).T


def rain_threshold_sensitivity_table(
    tracer_totals_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    analysis_window_minutes=(0.0, np.inf),
):
    """Repeat the factor decomposition at every rain-rate threshold at once.

    Mechanism, and what the threshold sweep discriminates. The five rain-sourced tracers are
    emitted under progressively heavier rain: 1, 5, 10, 20 and 40 mm/hr. That makes the
    sweep a test of *which* property of the rain changed.

    * A roughly uniform drop in the emitted count across all five thresholds means the rain
      covers less ground but is no less intense, which points at storm size.
    * A drop that steepens with threshold means the rain is less intense, which points at
      updraft strength and depth, since heavy surface rain needs a deep vigorous updraft to
      make the condensate in the first place.

    The entrainment efficiency column is read differently. It describes the transport of
    tracer that already exists, so if it moves by about the same factor at every threshold,
    the transport change is a property of the updraft rather than of the rain.

    Args:
        tracer_totals_dss_by_arm: Mapping of arm label -> {storm key -> tracer dataset}.
        storm_key: Which storm to sweep.
        reference_arm_label: The arm ratios are measured from.
        comparison_arm_label: The arm ratios are measured to. Defaults to the first other.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.

    Returns:
        A DataFrame indexed by rain threshold, holding the comparison-over-reference ratio
        of the emitted count, the entrained count, the entrainment efficiency and the
        in-updraft mixing ratio.
    """
    sensitivity_rows = {}
    for tracer_ix, rain_sourced_tracer in enumerate(blt.RS_TRACERS):
        factor_table = rain_sourced_entrainment_factor_table(
            tracer_totals_dss_by_arm,
            storm_key,
            reference_arm_label,
            comparison_arm_label=comparison_arm_label,
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        threshold_label = f"{blt.RS_TRACER_THRESHOLDS_MM_HR[tracer_ix]} mm/hr"
        sensitivity_rows[threshold_label] = {
            "emitted ratio": factor_table.loc[
                "emitted (A, # particles)", "ratio"
            ],
            "entrained ratio": factor_table.loc[
                "entrained (N, # particles)", "ratio"
            ],
            "entrainment efficiency ratio": factor_table.loc[
                "entrainment efficiency (N/A)", "ratio"
            ],
            "RS mixing ratio ratio": factor_table.loc[
                "RS mixing ratio (N/M, #/kg)", "ratio"
            ],
        }

    return pd.DataFrame(sensitivity_rows).T


def plot_rain_area_by_threshold(
    development_dfs_by_arm,
    storm_key,
    thresholds_mm_hr=(1, 10, 40),
    panel_width=3.6,
    panel_height=3.2,
):
    """Plot the surface rain area above each threshold, both arms, against time.

    Mechanism: the rain-sourced tracers are emitted from the surface wherever the rain rate
    exceeds their threshold, so the emitted count in the factor table is an integral over
    exactly these areas. Plotting them closes the loop between "less rain-sourced tracer was
    emitted" and the physical rain field that emitted it. It also does so from the
    development dataframe rather than from the tracers, which makes it an independent path
    to the same conclusion.

    The last panel is the peak updraft, which is the control on the rain panels: it says
    whether a storm making less heavy rain is also running a weaker updraft core, or only a
    smaller one.

    Args:
        development_dfs_by_arm: Mapping of arm label -> {storm key -> development DataFrame}.
        storm_key: Which storm to plot.
        thresholds_mm_hr: Which of the five thresholds to draw panels for.
        panel_width: Width of one panel, in inches.
        panel_height: Height of the panel row, in inches.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (1, len(thresholds_mm_hr) + 1).
    """
    arm_labels = list(development_dfs_by_arm.keys())
    n_panels = len(thresholds_mm_hr) + 1

    fig, axs = plt.subplots(
        nrows=1,
        ncols=n_panels,
        figsize=(panel_width * n_panels, panel_height),
        layout="constrained",
        sharex=True,
        squeeze=False,
    )

    # One panel per requested threshold, showing the area raining that hard.
    for panel_ix, threshold_mm_hr in enumerate(thresholds_mm_hr):
        threshold_index = blt.RS_TRACER_THRESHOLDS_MM_HR.index(threshold_mm_hr)
        ax = axs[0, panel_ix]
        for arm_ix, arm_label in enumerate(arm_labels):
            development_df = development_dfs_by_arm[arm_label][storm_key]
            ax.plot(
                development_df["t_minutes"],
                development_df[f"rain_threshold{threshold_index}"]
                * RAIN_GRID_CELL_AREA_KM2,
                color=ENTRAINMENT_ARM_COLORS[
                    arm_ix % len(ENTRAINMENT_ARM_COLORS)
                ],
                linestyle=ENTRAINMENT_ARM_LINE_STYLES[
                    arm_ix % len(ENTRAINMENT_ARM_LINE_STYLES)
                ],
                linewidth=2.0,
                label=arm_label,
            )
        ax.set_title(
            blt.axes_title(panel_ix, f"Rain area > {threshold_mm_hr} mm/hr"),
            fontsize=_scaled_font_size(0.8),
        )
        if panel_ix == 0:
            ax.set_ylabel("km$^2$", fontsize=_scaled_font_size(0.85))
        blt.label_x_axis(ax, short=True)
        ax.tick_params(labelsize=_scaled_font_size(0.7))
        ax.xaxis.label.set_size(_scaled_font_size(0.75))

    updraft_ax = axs[0, n_panels - 1]
    for arm_ix, arm_label in enumerate(arm_labels):
        development_df = development_dfs_by_arm[arm_label][storm_key]
        updraft_ax.plot(
            development_df["t_minutes"],
            development_df["max_updraft"],
            color=ENTRAINMENT_ARM_COLORS[arm_ix % len(ENTRAINMENT_ARM_COLORS)],
            linestyle=ENTRAINMENT_ARM_LINE_STYLES[
                arm_ix % len(ENTRAINMENT_ARM_LINE_STYLES)
            ],
            linewidth=2.0,
            label=arm_label,
        )
    updraft_ax.set_title(
        blt.axes_title(n_panels - 1, "Peak updraft at 5 km"),
        fontsize=_scaled_font_size(0.8),
    )
    updraft_ax.set_ylabel("m s$^{-1}$", fontsize=_scaled_font_size(0.85))
    blt.label_x_axis(updraft_ax, short=True)
    updraft_ax.tick_params(labelsize=_scaled_font_size(0.7))
    updraft_ax.xaxis.label.set_size(_scaled_font_size(0.75))

    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(arm_labels),
        frameon=False,
        fontsize=_scaled_font_size(0.8),
    )
    fig.suptitle(
        f"{blt.STORM_LABELS[storm_key]} -- the surface rain that emits the"
        " rain-sourced tracer",
        fontsize=_scaled_font_size(1.0),
    )
    return fig, axs


########################################################################
######## Selectivity: which air the updraft draws from #################
########################################################################

# Height layers the shift-share decomposition is reported over. The boundaries are chosen from
# where the mechanisms differ rather than for roundness: below 1.5 km is the inflow layer the
# rain-sourced tracer is emitted into, 1.5-3 km is where an updraft base sits or does not,
# 3-8 km is the storm body, and above 8 km is the anvil and outflow.
ENTRAINMENT_SHIFT_SHARE_LAYERS_M = [
    (0.0, 1500.0),
    (1500.0, 3000.0),
    (3000.0, 8000.0),
    (8000.0, np.inf),
]


def rain_marking_selectivity_table(
    tracer_totals_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
):
    """Split the in-updraft rain-sourced mixing ratio a second, independent way.

    Mechanism, and why a second identity is worth having. ``rain_sourced_entrainment_factor
    _table`` splits q_RS into supply per unit updraft mass and the fraction of that supply
    captured. That split answers "supply or transport", and for the squall line it does. It
    cannot answer "*which* air did the updraft draw from", because both of its factors are
    blind to what the rest of the boundary layer looks like.

    This identity answers that question instead. Write N_RS and N_FS for the rain-sourced and
    fixed-source tracer inside the updraft, A_RS and A_FS for the cumulative emission of each,
    and M for the updraft's air mass. Then::

        q_RS = N_RS/M = (N_FS/M) * (A_RS/A_FS) * [(N_RS/N_FS) / (A_RS/A_FS)]
             = q_BL   *  emitted rain-sourced share  *  selectivity

    exactly, and each factor is separately meaningful.

    * **q_BL**, the boundary-layer tracer per kilogram of updraft, is general lofting. It rises
      when the updraft carries more low-level air of any origin.
    * **The emitted rain-sourced share** is a property of the rain field alone. The fixed-source
      tracers are released on a schedule no storm can influence, so A_FS is the same number in
      both arms and this factor says only how much of the boundary layer has been rained on.
    * **Selectivity** is what is left: how much more rain-sourced the updraft's air is than the
      emission mix it draws from. A value of one means the updraft samples the boundary layer
      indiscriminately. A value above one means it preferentially ingests air that has sat
      under rain, which is what gust-front lifting into a cold pool produces.

    The point of separating these is that they respond to different things. A storm can raise
    q_RS by lofting more air, by raining over more ground, or by drawing more selectively from
    the ground it has already rained on, and only the third is about where the updraft roots.

    Aggregation is by sum over the window then divide, for the same reason as the other
    factor table: averaging the per-time ratios would break the identity.

    Args:
        tracer_totals_dss_by_arm: Mapping of arm label -> {storm key -> tracer dataset}.
        storm_key: Which storm to decompose.
        reference_arm_label: The arm changes are measured from.
        comparison_arm_label: The arm changes are measured to. Defaults to the first other.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.

    Returns:
        A DataFrame indexed by factor name, with one column per arm plus ``ratio``,
        ``percent_change`` and ``log_ratio``. The first three log ratios sum to the fourth.
    """
    arm_labels = list(tracer_totals_dss_by_arm.keys())
    if comparison_arm_label is None:
        comparison_arm_label = next(
            label for label in arm_labels if label != reference_arm_label
        )

    window_sums_by_arm = {}
    for arm_label in [reference_arm_label, comparison_arm_label]:
        terms_df = _rain_sourced_entrainment_terms(
            tracer_totals_dss_by_arm[arm_label][storm_key],
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        summed_updraft_mass = float(terms_df["updraft_mass"].sum())
        summed_rain_sourced_entrained = float(terms_df["entrained"].sum())
        summed_boundary_layer_entrained = float(
            terms_df["boundary_layer_entrained"].sum()
        )
        summed_rain_sourced_emitted = float(terms_df["emitted"].sum())
        summed_boundary_layer_emitted = float(
            terms_df["boundary_layer_emitted"].sum()
        )

        emitted_rain_sourced_share = (
            summed_rain_sourced_emitted / summed_boundary_layer_emitted
        )
        updraft_rain_sourced_fraction = (
            summed_rain_sourced_entrained / summed_boundary_layer_entrained
        )
        window_sums_by_arm[arm_label] = {
            "BL mixing ratio (q_BL, #/kg)": (
                summed_boundary_layer_entrained / summed_updraft_mass
            ),
            "emitted RS share (A_RS/A_FS)": emitted_rain_sourced_share,
            "selectivity (updraft share / emitted share)": (
                updraft_rain_sourced_fraction / emitted_rain_sourced_share
            ),
            "RS mixing ratio (q_RS, #/kg)": (
                summed_rain_sourced_entrained / summed_updraft_mass
            ),
            # Reported alongside because the selectivity row is a ratio of these two, and a
            # reader checking the arithmetic should not have to reconstruct them.
            "updraft RS fraction (N_RS/N_FS)": updraft_rain_sourced_fraction,
        }

    selectivity_table = pd.DataFrame(window_sums_by_arm)
    selectivity_table["ratio"] = (
        selectivity_table[comparison_arm_label]
        / selectivity_table[reference_arm_label]
    )
    selectivity_table["percent_change"] = 100.0 * (
        selectivity_table["ratio"] - 1.0
    )
    selectivity_table["log_ratio"] = np.log(selectivity_table["ratio"])

    # Verify the three factors reproduce the observed change, rather than assuming they do.
    summed_contributions = selectivity_table.loc[
        [
            "BL mixing ratio (q_BL, #/kg)",
            "emitted RS share (A_RS/A_FS)",
            "selectivity (updraft share / emitted share)",
        ],
        "log_ratio",
    ].sum()
    observed_change = selectivity_table.loc[
        "RS mixing ratio (q_RS, #/kg)", "log_ratio"
    ]
    assert np.isclose(summed_contributions, observed_change, atol=1e-9), (
        "Selectivity decomposition does not close: contributions sum to"
        f" {summed_contributions}, observed change is {observed_change}."
    )

    return selectivity_table


def plot_rain_marking_selectivity(
    tracer_totals_dss_by_arm,
    storm_key,
    rain_sourced_tracer=None,
    panel_width=3.7,
    panel_height=3.2,
):
    """Plot the three selectivity factors and their product against time, for both arms.

    Mechanism: the table reduces each factor to one number per window, which cannot show that
    a factor is steady while another one grows. Selectivity in particular is expected to climb
    through a storm's life, because rain has to fall before there is anything rain-marked to
    draw from. Seeing whether the *difference* between arms is steady or growing is what says
    whether the effect belongs to the mature storm or only to its decay.

    Args:
        tracer_totals_dss_by_arm: Mapping of arm label -> {storm key -> tracer dataset}.
        storm_key: Which storm to plot.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        panel_width: Width of one panel, in inches.
        panel_height: Height of the panel row, in inches.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (1, 4).
    """
    rain_sourced_tracer = (
        rain_sourced_tracer or REPRESENTATIVE_RAIN_SOURCED_TRACER
    )
    arm_labels = list(tracer_totals_dss_by_arm.keys())
    terms_by_arm = {
        arm_label: _rain_sourced_entrainment_terms(
            tracer_totals_dss_by_arm[arm_label][storm_key],
            rain_sourced_tracer=rain_sourced_tracer,
        )
        for arm_label in arm_labels
    }

    fig, axs = plt.subplots(
        nrows=1,
        ncols=4,
        figsize=(panel_width * 4, panel_height),
        layout="constrained",
        sharex=True,
        squeeze=False,
    )

    # The three factors, then their product. Every one of them spans orders of magnitude
    # while the storm develops, so all four axes are logarithmic.
    selectivity_panels = [
        (
            "boundary_layer_mixing_ratio",
            "General lofting\n" r"$q_{BL}$ (# kg$^{-1}$)",
        ),
        (
            "emitted_rain_sourced_share",
            "Emitted RS share\n" r"$A_{RS}/A_{FS}$",
        ),
        (
            "rain_marking_selectivity",
            "Selectivity\n" r"$(N_{RS}/N_{FS}) / (A_{RS}/A_{FS})$",
        ),
        (
            "rs_mixing_ratio",
            "Product\n" r"$q_{RS}$ (# kg$^{-1}$)",
        ),
    ]
    # Times when a storm is actually entraining rain-sourced air, taken as at least 1% of its
    # own peak rain-sourced mixing ratio. The first output time after rain begins carries a
    # numerically nonzero but physically meaningless amount, several orders of magnitude below
    # everything that follows. Letting that value into the y scale of a log axis compresses the
    # entire mature storm into the top centimetre of the panel, which is what the panel is for.
    # The early values are still drawn; they run off the bottom.
    is_actively_entraining_by_arm = {
        arm_label: (
            terms_df["rs_mixing_ratio"].values
            >= 0.01 * np.nanmax(terms_df["rs_mixing_ratio"].values)
        )
        for arm_label, terms_df in terms_by_arm.items()
    }

    for panel_ix, (column_name, panel_title) in enumerate(selectivity_panels):
        ax = axs[0, panel_ix]
        active_values = []
        for arm_ix, arm_label in enumerate(arm_labels):
            terms_df = terms_by_arm[arm_label]
            # Zeros plot as gaps on a log axis instead of spikes off the bottom.
            plotted_values = terms_df[column_name].replace(0.0, np.nan)
            ax.plot(
                terms_df["t_minutes"],
                plotted_values,
                color=ENTRAINMENT_ARM_COLORS[
                    arm_ix % len(ENTRAINMENT_ARM_COLORS)
                ],
                linestyle=ENTRAINMENT_ARM_LINE_STYLES[
                    arm_ix % len(ENTRAINMENT_ARM_LINE_STYLES)
                ],
                linewidth=2.0,
                label=arm_label,
            )
            active_values.append(
                plotted_values.values[is_actively_entraining_by_arm[arm_label]]
            )
        ax.set_yscale("log")
        # One decade of headroom either side of the mature-storm range, on a log axis.
        stacked_active_values = np.concatenate(active_values)
        stacked_active_values = stacked_active_values[
            np.isfinite(stacked_active_values) & (stacked_active_values > 0)
        ]
        if stacked_active_values.size:
            ax.set_ylim([
                stacked_active_values.min() / 3.0,
                stacked_active_values.max() * 3.0,
            ])
        ax.set_title(
            blt.axes_title(panel_ix, panel_title),
            fontsize=_scaled_font_size(0.8),
        )
        blt.label_x_axis(ax, short=True)
        ax.tick_params(labelsize=_scaled_font_size(0.7))
        ax.xaxis.label.set_size(_scaled_font_size(0.75))

    # Unity on the selectivity panel is "the updraft samples the boundary layer at random".
    axs[0, 2].axhline(1.0, color="0.6", linewidth=1.0, zorder=0)

    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(arm_labels),
        frameon=False,
        fontsize=_scaled_font_size(0.8),
    )
    fig.suptitle(
        f"{blt.STORM_LABELS[storm_key]} -- which air the updraft draws from"
        f" ({rain_sourced_threshold_label(rain_sourced_tracer)})",
        fontsize=_scaled_font_size(1.0),
    )
    return fig, axs


def entrainment_window_sensitivity_table(
    tracer_totals_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    rain_sourced_tracer=None,
    analysis_windows_minutes=None,
):
    """Recompute every headline ratio over several time windows at once.

    Mechanism, and the confound it is built to catch. The entrainment efficiency N/A divides
    what is in the updraft *now* by everything emitted *ever*. A storm that dies therefore
    scores an efficiency approaching zero, whatever it did while it was alive. Any comparison
    between a short-lived storm and a long-lived one is contaminated by that, and the isolated
    convection pair is exactly such a comparison.

    Running the same ratios over a window when both storms are healthy, and again over the
    decay phase, separates the two. A ratio that is the same in every window belongs to the
    storm's behaviour. A ratio that only appears in the late window belongs to its longevity.

    Args:
        tracer_totals_dss_by_arm: Mapping of arm label -> {storm key -> tracer dataset}.
        storm_key: Which storm to test.
        reference_arm_label: The arm ratios are measured from.
        comparison_arm_label: The arm ratios are measured to. Defaults to the first other.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_windows_minutes: Mapping of window label -> (start, end). Defaults to the
            full run, a both-storms-alive window, and two progressively later ones.

    Returns:
        A DataFrame indexed by window label, one column per ratio.
    """
    if analysis_windows_minutes is None:
        analysis_windows_minutes = {
            "full run": (0.0, np.inf),
            "60-120 min": (60.0, 120.0),
            "t >= 100 min": (100.0, np.inf),
            "t >= 140 min": (140.0, np.inf),
        }

    window_rows = {}
    for (
        window_label,
        analysis_window_minutes,
    ) in analysis_windows_minutes.items():
        factor_table = rain_sourced_entrainment_factor_table(
            tracer_totals_dss_by_arm,
            storm_key,
            reference_arm_label,
            comparison_arm_label=comparison_arm_label,
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        selectivity_table = rain_marking_selectivity_table(
            tracer_totals_dss_by_arm,
            storm_key,
            reference_arm_label,
            comparison_arm_label=comparison_arm_label,
            rain_sourced_tracer=rain_sourced_tracer,
            analysis_window_minutes=analysis_window_minutes,
        )
        window_rows[window_label] = {
            "updraft mass M": factor_table.loc["updraft mass (M, kg)", "ratio"],
            "entrainment efficiency": factor_table.loc[
                "entrainment efficiency (N/A)", "ratio"
            ],
            "general lofting q_BL": selectivity_table.loc[
                "BL mixing ratio (q_BL, #/kg)", "ratio"
            ],
            "emitted RS share": selectivity_table.loc[
                "emitted RS share (A_RS/A_FS)", "ratio"
            ],
            "selectivity": selectivity_table.loc[
                "selectivity (updraft share / emitted share)", "ratio"
            ],
            "RS mixing ratio q_RS": factor_table.loc[
                "RS mixing ratio (N/M, #/kg)", "ratio"
            ],
        }

    return pd.DataFrame(window_rows).T


def updraft_geometry_frame(
    tracer_totals_profile_ds, analysis_window_minutes=(0.0, np.inf)
):
    """Reduce one run's profile dataset to the shape of its updraft.

    Mechanism: the profile dataset counts in-updraft gridpoints at every level as well as
    summing their mass, so it already contains the storm's vertical geometry and no extra
    pass over the raw output is needed. Two views come out of it.

    The first is the time-integrated updraft area at each height, which says where in the
    column the updraft is wide and where it is absent. The second is the updraft base at each
    output time, taken as the lowest above-ground level holding any updraft point. The base
    matters here because ``blt`` defines an updraft point as cloudy *and* rising, so the base
    of the diagnosed updraft is effectively cloud base. A run with a lower cloud base overlaps
    more of the near-surface layer the rain-sourced tracer is emitted into.

    Args:
        tracer_totals_profile_ds: One run's (time, z) tracer profile dataset.
        analysis_window_minutes: (start, end) simulation-time window, inclusive. It applies to
            the area profile only; the base height series always covers the whole run, since
            its own time axis is what makes it readable.

    Returns:
        ``(area_profile_df, base_height_df)``. The first is indexed by height in metres with
        columns ``updraft_area_km2`` and ``updraft_mass``. The second is indexed by simulation
        minutes with column ``updraft_base_m``.
    """
    above_ground_ds = tracer_totals_profile_ds.isel(
        {"z": slice(1, tracer_totals_profile_ds.sizes["z"])}
    )
    heights_m = np.asarray(above_ground_ds["z"].values, dtype=float)
    simulation_minutes = np.asarray(
        above_ground_ds["t_minutes"].values, dtype=float
    )

    # Time-integrated area and mass at each height, over the requested window.
    window_start_minutes, window_end_minutes = analysis_window_minutes
    in_window = (simulation_minutes >= window_start_minutes) & (
        simulation_minutes <= window_end_minutes
    )
    windowed_ds = above_ground_ds.isel({"time": np.flatnonzero(in_window)})
    area_profile_df = pd.DataFrame(
        {
            "updraft_area_km2": (
                windowed_ds["in_updraft"].sum("time").values
                * RAIN_GRID_CELL_AREA_KM2
            ),
            "updraft_mass": windowed_ds["air_mass"].sum("time").values,
        },
        index=pd.Index(heights_m, name="z_m"),
    )

    # Lowest level holding any updraft point, one value per output time. A frame with no
    # updraft anywhere has no base, and NaN is the honest answer rather than a floor value.
    updraft_point_counts = np.asarray(
        above_ground_ds["in_updraft"].values, dtype=float
    )
    updraft_base_m = np.full(updraft_point_counts.shape[0], np.nan)
    for time_ix in range(updraft_point_counts.shape[0]):
        levels_with_updraft = np.flatnonzero(updraft_point_counts[time_ix] > 0)
        if levels_with_updraft.size:
            updraft_base_m[time_ix] = heights_m[levels_with_updraft[0]]
    base_height_df = pd.DataFrame(
        {"updraft_base_m": updraft_base_m},
        index=pd.Index(simulation_minutes, name="t_minutes"),
    )

    return area_profile_df, base_height_df


def updraft_geometry_table(
    tracer_totals_profile_dss_by_arm,
    storm_key,
    analysis_window_minutes=(0.0, np.inf),
    low_level_top_m=None,
):
    """Reduce each arm's updraft geometry to numbers that can be compared.

    Mechanism: four scalars per arm, chosen so that a change in *where the updraft is* can be
    told from a change in *how big it is*. The median updraft base over the window says how
    far down the cloudy updraft reaches. The low-level area says how much of it is in the
    layer the rain-sourced tracer is emitted into. The mass-weighted mean height says whether
    the storm's centre of mass moved. The updraft top says whether it got deeper.

    Args:
        tracer_totals_profile_dss_by_arm: Mapping of arm label -> {storm key -> profile ds}.
        storm_key: Which storm to summarize.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.
        low_level_top_m: Top of the layer counted as low-level. Defaults to
            ``LOW_LEVEL_ENTRAINMENT_TOP_M``.

    Returns:
        A DataFrame indexed by arm label.
    """
    low_level_top_m = low_level_top_m or LOW_LEVEL_ENTRAINMENT_TOP_M
    window_start_minutes, window_end_minutes = analysis_window_minutes

    geometry_rows = {}
    for arm_label, profile_dss in tracer_totals_profile_dss_by_arm.items():
        area_profile_df, base_height_df = updraft_geometry_frame(
            profile_dss[storm_key],
            analysis_window_minutes=analysis_window_minutes,
        )
        heights_m = area_profile_df.index.values
        updraft_areas = area_profile_df["updraft_area_km2"].values
        updraft_masses = area_profile_df["updraft_mass"].values

        # The base series covers the whole run, so window it here to match the area profile.
        in_window = (base_height_df.index.values >= window_start_minutes) & (
            base_height_df.index.values <= window_end_minutes
        )
        windowed_bases = base_height_df["updraft_base_m"].values[in_window]

        levels_with_updraft = np.flatnonzero(updraft_areas > 0)
        geometry_rows[arm_label] = {
            "median updraft base (m)": float(np.nanmedian(windowed_bases)),
            "lowest level reached (m)": (
                float(heights_m[levels_with_updraft[0]])
                if levels_with_updraft.size
                else np.nan
            ),
            "updraft top (m)": (
                float(heights_m[levels_with_updraft[-1]])
                if levels_with_updraft.size
                else np.nan
            ),
            f"area below {low_level_top_m / 1000:g} km (km2-frames)": float(
                updraft_areas[heights_m < low_level_top_m].sum()
            ),
            "mass-weighted mean height (m)": float(
                (updraft_masses * heights_m).sum() / updraft_masses.sum()
            ),
        }

    return pd.DataFrame(geometry_rows).T


def plot_updraft_geometry(
    tracer_totals_profile_dss_by_arm,
    storm_key,
    analysis_window_minutes=(0.0, np.inf),
    max_height_km=16.0,
    panel_width=3.8,
    panel_height=3.6,
):
    """Plot updraft area against height, and updraft base and mass against time.

    Mechanism, and what it discriminates. Three candidate explanations for a change in
    rain-sourced entrainment make different predictions here.

    * A faster updraft predicts little change in area or base height. Speed does not appear in
      this figure at all, which is the point: if these panels barely move, speed is what is
      left.
    * A moister boundary layer lowers the lifting condensation level, which lowers cloud base,
      which lowers the diagnosed updraft base, because ``blt`` requires condensate for a point
      to count as updraft. That shows up as a downward shift in the middle panel.
    * A longer-lived storm keeps updraft mass late in the run, which shows up in the right
      panel as a tail rather than a collapse.

    Args:
        tracer_totals_profile_dss_by_arm: Mapping of arm label -> {storm key -> profile ds}.
        storm_key: Which storm to plot.
        analysis_window_minutes: (start, end) window for the area profile in the left panel.
            The two time-series panels always show the whole run.
        max_height_km: Top of the vertical axis on the left panel.
        panel_width: Width of one panel, in inches.
        panel_height: Height of the panel row, in inches.

    Returns:
        ``(fig, axs)`` with ``axs`` shaped (1, 3).
    """
    arm_labels = list(tracer_totals_profile_dss_by_arm.keys())
    geometry_by_arm = {
        arm_label: updraft_geometry_frame(
            tracer_totals_profile_dss_by_arm[arm_label][storm_key],
            analysis_window_minutes=analysis_window_minutes,
        )
        for arm_label in arm_labels
    }

    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(panel_width * 3, panel_height),
        layout="constrained",
        squeeze=False,
    )

    for arm_ix, arm_label in enumerate(arm_labels):
        area_profile_df, base_height_df = geometry_by_arm[arm_label]
        arm_style = {
            "color": ENTRAINMENT_ARM_COLORS[
                arm_ix % len(ENTRAINMENT_ARM_COLORS)
            ],
            "linestyle": ENTRAINMENT_ARM_LINE_STYLES[
                arm_ix % len(ENTRAINMENT_ARM_LINE_STYLES)
            ],
            "linewidth": 2.0,
            "label": arm_label,
        }

        # Panel (a): how wide the updraft is at each height.
        axs[0, 0].plot(
            area_profile_df["updraft_area_km2"].replace(0.0, np.nan).values,
            area_profile_df.index.values / 1000.0,
            **arm_style,
        )
        # Panel (b): how far down it reaches, frame by frame.
        axs[0, 1].plot(
            base_height_df.index.values,
            base_height_df["updraft_base_m"].values / 1000.0,
            **arm_style,
        )
        # Panel (c): how long it lasts.
        axs[0, 2].plot(
            base_height_df.index.values,
            _column_summed_updraft_mass(
                tracer_totals_profile_dss_by_arm[arm_label][storm_key]
            ),
            **arm_style,
        )

    axs[0, 0].set_xscale("log")
    axs[0, 0].set_ylim([0, max_height_km])
    axs[0, 0].set_ylabel("z (km)", fontsize=_scaled_font_size(0.85))
    axs[0, 0].set_title(
        blt.axes_title(
            0, "Updraft area by height\n(km$^2$, summed over frames)"
        ),
        fontsize=_scaled_font_size(0.8),
    )

    # The base is the lowest cloudy rising level, so a lower curve means the updraft roots
    # closer to where the rain-sourced tracer is emitted.
    axs[0, 1].set_ylabel("z (km)", fontsize=_scaled_font_size(0.85))
    axs[0, 1].set_title(
        blt.axes_title(1, "Updraft base\n(lowest cloudy rising level)"),
        fontsize=_scaled_font_size(0.8),
    )
    axs[0, 2].set_yscale("log")
    axs[0, 2].set_ylabel("kg", fontsize=_scaled_font_size(0.85))
    axs[0, 2].set_title(
        blt.axes_title(2, "Updraft mass\n(storm lifetime)"),
        fontsize=_scaled_font_size(0.8),
    )
    for column_ix in [1, 2]:
        blt.label_x_axis(axs[0, column_ix], short=True)
        axs[0, column_ix].xaxis.label.set_size(_scaled_font_size(0.75))
    for ax in axs.flat:
        ax.tick_params(labelsize=_scaled_font_size(0.7))

    fig.legend(
        *axs[0, 0].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=len(arm_labels),
        frameon=False,
        fontsize=_scaled_font_size(0.8),
    )
    fig.suptitle(
        f"{blt.STORM_LABELS[storm_key]} -- updraft geometry and lifetime",
        fontsize=_scaled_font_size(1.0),
    )
    return fig, axs


def _column_summed_updraft_mass(tracer_totals_profile_ds):
    """Updraft air mass summed over the column, one value per output time."""
    return (
        tracer_totals_profile_ds.isel(
            {"z": slice(1, tracer_totals_profile_ds.sizes["z"])}
        )["air_mass"]
        .sum("z")
        .values
    )


def entrained_tracer_shift_share_table(
    tracer_totals_profile_dss_by_arm,
    storm_key,
    reference_arm_label,
    comparison_arm_label=None,
    rain_sourced_tracer=None,
    analysis_window_minutes=(0.0, np.inf),
    height_layers_m=None,
):
    """Attribute the change in entrained tracer to updraft volume versus air composition.

    Mechanism. The entrained count is a sum over levels of mass times mixing ratio,
    N = sum_z M(z) q(z), so a change in N has exactly two sources: the updraft occupies a
    different amount of mass at height z, or the air it contains at height z has a different
    rain-sourced content. Those are the geometric and the compositional explanation
    respectively, and they are what the candidate mechanisms disagree about.

    The split used is the Bennet decomposition, which is exact::

        N_c - N_r = sum_z [ (M_c - M_r) * (q_c + q_r)/2  +  (M_c + M_r)/2 * (q_c - q_r) ]

    with subscript r for the reference arm and c for the comparison arm. Averaging the two
    arms inside each term is what makes it exact and symmetric. Attributing the mass term at
    the reference arm's mixing ratio instead would leave an interaction remainder, and that
    remainder is precisely the quantity a reader would want attributed.

    Reporting it by height layer is what makes it interpretable. A mass term concentrated in
    the lowest layers means the updraft started reaching into air it previously did not touch.
    A concentration term spread through the column means the same updraft is carrying
    different air.

    Args:
        tracer_totals_profile_dss_by_arm: Mapping of arm label -> {storm key -> profile ds}.
        storm_key: Which storm to decompose.
        reference_arm_label: The arm the change is measured from.
        comparison_arm_label: The arm the change is measured to. Defaults to the first other.
        rain_sourced_tracer: Which rain-sourced tracer. Defaults to the 10 mm/hr one.
        analysis_window_minutes: (start, end) simulation-time window, inclusive.
        height_layers_m: Sequence of (bottom, top) layers. Defaults to
            ``ENTRAINMENT_SHIFT_SHARE_LAYERS_M``.

    Returns:
        A DataFrame indexed by layer label, with the two terms in particles and as a
        percentage of the total change, plus a total row.
    """
    height_layers_m = height_layers_m or ENTRAINMENT_SHIFT_SHARE_LAYERS_M
    arm_labels = list(tracer_totals_profile_dss_by_arm.keys())
    if comparison_arm_label is None:
        comparison_arm_label = next(
            label for label in arm_labels if label != reference_arm_label
        )

    reference_profile_df = rain_sourced_entrainment_profile_frame(
        tracer_totals_profile_dss_by_arm[reference_arm_label][storm_key],
        rain_sourced_tracer=rain_sourced_tracer,
        analysis_window_minutes=analysis_window_minutes,
    )
    comparison_profile_df = rain_sourced_entrainment_profile_frame(
        tracer_totals_profile_dss_by_arm[comparison_arm_label][storm_key],
        rain_sourced_tracer=rain_sourced_tracer,
        analysis_window_minutes=analysis_window_minutes,
    )
    heights_m = reference_profile_df.index.values

    reference_mass = reference_profile_df["updraft_mass"].values
    comparison_mass = comparison_profile_df["updraft_mass"].values
    # A level the updraft never reached contributes no tracer, so its mixing ratio is taken as
    # zero rather than as NaN. Leaving it NaN would poison the sums.
    reference_mixing_ratio = np.nan_to_num(
        reference_profile_df["rs_mixing_ratio"].values
    )
    comparison_mixing_ratio = np.nan_to_num(
        comparison_profile_df["rs_mixing_ratio"].values
    )

    mass_term_per_level = (
        (comparison_mass - reference_mass)
        * (comparison_mixing_ratio + reference_mixing_ratio)
        / 2.0
    )
    composition_term_per_level = (
        (comparison_mass + reference_mass)
        / 2.0
        * (comparison_mixing_ratio - reference_mixing_ratio)
    )
    total_change = (
        comparison_profile_df["entrained"].values.sum()
        - reference_profile_df["entrained"].values.sum()
    )

    # Confirm the two terms reconstruct the change before attributing anything to them.
    assert np.isclose(
        mass_term_per_level.sum() + composition_term_per_level.sum(),
        total_change,
        rtol=1e-8,
    ), "Bennet shift-share does not reconstruct the change in entrained tracer."

    layer_rows = {}
    for layer_bottom_m, layer_top_m in height_layers_m:
        in_layer = (heights_m >= layer_bottom_m) & (heights_m < layer_top_m)
        layer_label = (
            f"{layer_bottom_m / 1000:g}-{layer_top_m / 1000:g} km"
            if np.isfinite(layer_top_m)
            else f">{layer_bottom_m / 1000:g} km"
        )
        layer_rows[layer_label] = {
            "updraft volume term": mass_term_per_level[in_layer].sum(),
            "air composition term": composition_term_per_level[in_layer].sum(),
        }
    layer_rows["all levels"] = {
        "updraft volume term": mass_term_per_level.sum(),
        "air composition term": composition_term_per_level.sum(),
    }

    shift_share_table = pd.DataFrame(layer_rows).T
    for term_name in ["updraft volume term", "air composition term"]:
        shift_share_table[f"{term_name} (% of change)"] = (
            100.0 * shift_share_table[term_name] / total_change
        )
    return shift_share_table
