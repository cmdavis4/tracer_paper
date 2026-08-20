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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import metpy.calc as mpc
from metpy.units import units

# The two in-repo packages live in sibling directories rather than being
# installed, so put them on the path here rather than requiring every notebook to
# do it before its first import of this module.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _in_repo_package_dir in [_PROJECT_ROOT / "cloudy", _PROJECT_ROOT / "replication"]:
    if str(_in_repo_package_dir) not in sys.path:
        sys.path.append(str(_in_repo_package_dir))

# User packages
import xarrams as xrr
import cloudy as cm
import blt_utils_limited as blt


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
    "ic_wk": RAMS_IO_ROOT
    / "isolated_convection_wk/all_tracers"
    / "newtracer_forcing-warmbubble4_deltax-500_frqlite-2_nobulku",
    "sl_wk": RAMS_IO_ROOT
    / "squall_line_wk/all_tracers"
    / "newtracer_blspinup-no_forcing-infinitecoldbubble4_deltax-500_frqlite-2",
    "sc_wk": RAMS_IO_ROOT
    / "supercell_wk/all_tracers"
    / "newtracer_forcing-grant2014_dx-500_frqlite-2_stratwinds",
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
# Both use the ``_qv_fixed`` variants. Note that ``ic_wk/correct_wk_wellmixed``
# and ``ic_wk/correct_wk_wellmixed_qv_fixed`` have byte-identical RAMSINs on disk
# but produce different output (surface q_v of 11.259 vs 11.015 g/kg), so one of
# those two RAMSINs no longer reflects what was actually run. The ``_qv_fixed``
# output is the one that lands on the intended 11 g/kg, so that is what we use;
# do not trust the on-disk RAMSIN for the non-fixed IC directory.
HARMONIZED_ON_SC_PROFILE_RUN_DIRS = {
    "ic_wk": RAMS_IO_ROOT / "from_cumulus/ic_wk/correct_wk_wellmixed_qv_fixed",
    "sl_wk": RAMS_IO_ROOT / "from_cumulus/sl_wk/correct_wk_wellmixed_qv_fixed",
    "sc_wk": PAPER_RUN_DIRS["sc_wk"],
}

# The intended low-level water vapour mixing ratio for each storm, as designed.
# The isolated convection is deliberately drier; the other two are meant to match.
INTENDED_LOW_LEVEL_MIXING_RATIOS_G_KG = {"ic_wk": 11.0, "sl_wk": 13.0, "sc_wk": 13.0}

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
    """Path to a run's precomputed in-updraft tracer vertical-profile dataset.

    This is the expensive derived product behind Figures 7 and 9 and the tracer
    entrainment statistics. It already exists for every run used here (including
    all of the harmonized ones), so those figures need no recomputation.
    """
    return derived_dir_for_run(run_dir, storm_key) / "tracer_totals_profile_ds.nc"


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
    initial_state_ds = xrr.read_rams_output(initial_analysis_filepaths[:1], silent=True)
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
        storm_key: read_initial_sounding_dataframe(run_dirs[storm_key], **kwargs)
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
        figsize=(panel_width * len(z_cutoffs), panel_height * len(variables_to_plot)),
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

    for series_ix, (label, full_sounding_df) in enumerate(labelled_sounding_dfs.items()):
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
                windowed_df = full_sounding_df.loc[full_sounding_df["z"] <= z_cutoff]
                ax.plot(
                    windowed_df[variable],
                    windowed_df[y_variable],
                    label=label,
                    linestyle=line_style,
                    linewidth=line_width,
                )

    # Label once the data are down, so titles reflect the requested windows.
    for variable_ix, (variable, axis_label) in enumerate(variables_to_plot.items()):
        for cutoff_ix, z_cutoff in enumerate(z_cutoffs):
            ax = axs[variable_ix, cutoff_ix]
            window_label = (
                "full column" if not np.isfinite(z_cutoff) else f"z ≤ {z_cutoff/1000:g} km"
            )
            ax.set_title(f"{axis_label} — {window_label}", fontsize=11)
            if cutoff_ix == 0:
                ax.set_ylabel("p (hPa)" if plotting_against_pressure else "z (m)")
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
                comparison_on_grid[variable].values - reference_on_grid[variable].values
            )
            # CAPE/CIN are NaN above the parcel-start cutoff, so ignore those levels.
            finite = np.isfinite(difference)
            if not finite.any():
                continue
            absolute_difference = np.abs(np.where(finite, difference, np.nan))
            height_of_max_difference_ix = int(np.nanargmax(absolute_difference))
            table_rows.append(
                {
                    "comparison": label,
                    "variable": variable,
                    "max_abs_diff": float(np.nanmax(absolute_difference)),
                    "z_of_max_abs_diff_m": float(
                        common_heights[height_of_max_difference_ix]
                    ),
                    "mean_abs_diff": float(np.nanmean(absolute_difference)),
                    "diff_at_lowest_level": float(difference[np.argmax(finite)]),
                }
            )

    return pd.DataFrame(table_rows).set_index(["comparison", "variable"])


def low_level_moisture_table(labelled_sounding_dfs, intended_mixing_ratios_g_kg,
                             sample_height_m=100.0):
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
                sample_height_m, sounding_df["z"].values, sounding_df["q_v"].values
            )
        )
        intended_mixing_ratio = intended_mixing_ratios_g_kg.get(storm_key, np.nan)
        table_rows.append(
            {
                "storm": storm_key,
                "intended_q_v_g_kg": intended_mixing_ratio,
                "actual_q_v_g_kg": actual_mixing_ratio,
                "abs_error_g_kg": actual_mixing_ratio - intended_mixing_ratio,
                "percent_error": 100.0
                * (actual_mixing_ratio - intended_mixing_ratio)
                / intended_mixing_ratio,
            }
        )
    return pd.DataFrame(table_rows).set_index("storm")


def temperature_offset_at_fixed_height_table(labelled_sounding_dfs, reference_label,
                                             sample_heights_m=(100, 1000, 3000, 5000)):
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
                sample_height, reference_df["z"].values, reference_df["TS"].values
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


def read_or_cache_dataframe(cache_path, compute_dataframe, *args, **kwargs):
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

    Args:
        cache_path: Where to read/write the pickled DataFrame.
        compute_dataframe: Callable producing the DataFrame if the cache is cold.
        *args, **kwargs: Passed through to *compute_dataframe*.
    """
    cache_path = Path(cache_path)
    if cache_path.exists():
        print(f"Read cached result from {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"No cache at {cache_path}, computing...")
    result = compute_dataframe(*args, **kwargs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_pickle(cache_path)
    print(f"Wrote result to {cache_path}")
    return result


def read_tracer_totals_datasets(run_dirs):
    """Read each storm's precomputed in-updraft tracer profile dataset and sum in z.

    These files already exist for every run in both arms, so this is pure I/O — no
    recomputation from the raw output is needed for Figures 7 and 9 or the tracer
    entrainment statistics.

    The ``z`` slice from index 1 drops the model's below-ground level, matching the
    manuscript notebook exactly.

    Returns:
        (tracer_totals_profile_dss, tracer_totals_dss) — the vertical profiles and
        their column sums, both keyed by storm.
    """
    tracer_totals_profile_dss = {}
    for storm_key in STORM_KEYS_IN_FIGURE_ORDER:
        if storm_key not in run_dirs:
            continue
        profile_filepath = tracer_totals_profile_filepath(run_dirs[storm_key], storm_key)
        if not profile_filepath.exists():
            raise FileNotFoundError(
                f"Missing precomputed tracer profile dataset at {profile_filepath}"
            )
        tracer_totals_profile_dss[storm_key] = xr.open_dataset(profile_filepath)

    # Sum over the vertical, skipping the below-ground level.
    tracer_totals_dss = {
        storm_key: profile_ds.isel({"z": slice(1, len(profile_ds["z"]))}).sum("z")
        for storm_key, profile_ds in tracer_totals_profile_dss.items()
    }
    return tracer_totals_profile_dss, tracer_totals_dss


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
        dataframe_filepath = development_dataframe_filepath(run_dirs[storm_key], storm_key)

        if dataframe_filepath.exists():
            development_df = pd.read_pickle(dataframe_filepath)
            print(f"Read development dataframe from {dataframe_filepath}")
        else:
            print(f"No development dataframe at {dataframe_filepath}, computing...")
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
    level_ds = storm_ds.sel({"z": upper_level_z_threshold_m}, method="nearest", drop=False)

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
    upper_level_ds = storm_ds.sel({"z": slice(upper_level_z_threshold_m, None, None)}).copy()

    # Number-concentration versions of every current tracer.
    all_current_tracer_vars = blt.FS_TRACERS + blt.RS_TRACERS + ["total_fs_tracer"]
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
            [x for x in upper_level_ds.data_vars if "tracer" in x.lower()] + ["air_mass"]
        ]
        .sum(["x", "y", "z"])
        .to_dataframe()
        .reset_index()
    )


########################################################################
############################## Summary tables ##############################
########################################################################


def collect_headline_metrics(tracer_totals_dss, upper_level_tracer_dfs=None,
                             development_dfs=None):
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
            in_updraft_ds[representative_rs_tracer + "_nc"] / in_updraft_ds["air_mass"]
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
            "mean_rs_mixing_ratio_per_kg": float(rs_mixing_ratio.mean(...).values),
            "max_rs_mixing_ratio_per_kg": float(rs_mixing_ratio.max(...).values),
            "eos_rs_mixing_ratio_per_kg": float(
                rs_mixing_ratio.isel({"time": -1}).values
            ),
            "mean_fs_mixing_ratio_per_kg": float(fs_mixing_ratio.mean(...).values),
            "max_rs_entrainment_efficiency": float(
                rs_entrainment_efficiency.max(...).values
            ),
            "mean_rs_entrainment_efficiency": float(
                rs_entrainment_efficiency.mean(...).values
            ),
            "mean_rs_fraction_of_total": float(rs_fraction_of_total.mean(...).values),
            "max_updraft_mass_kg": float(in_updraft_ds["air_mass"].max(...).values),
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
                    metrics[metric_name] = float(upper_level_df[column].iloc[-1])

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


def compare_arms_table(headline_metrics_by_arm, original_arm_label, harmonized_arm_label):
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
            table_rows.append(
                {
                    "storm": storm_key,
                    "metric": metric_name,
                    original_arm_label: original_value,
                    harmonized_arm_label: harmonized_value,
                    "percent_change": 100.0
                    * (harmonized_value - original_value)
                    / original_value
                    if original_value
                    else np.nan,
                }
            )
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
        shared_metrics = set.intersection(
            *[set(metrics.columns) for metrics in headline_metrics_by_arm.values()]
        )
        for metric_name in sorted(shared_metrics):
            row = {"storm_pair": pair_label, "metric": metric_name}
            for arm_label, metrics in headline_metrics_by_arm.items():
                denominator_value = metrics.loc[denominator_storm, metric_name]
                row[arm_label] = (
                    metrics.loc[numerator_storm, metric_name] / denominator_value
                    if denominator_value
                    else np.nan
                )
            # How far the cross-storm ratio moved between the two arms.
            first_arm_ratio, last_arm_ratio = row[arm_labels[0]], row[arm_labels[-1]]
            row["ratio_percent_change"] = (
                100.0 * (last_arm_ratio - first_arm_ratio) / first_arm_ratio
                if first_arm_ratio
                else np.nan
            )
            table_rows.append(row)

    return pd.DataFrame(table_rows).set_index(["storm_pair", "metric"])


def style_percent_change(table, percent_change_column="percent_change", threshold=10.0):
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
