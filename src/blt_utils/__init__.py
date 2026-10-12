"""Analysis utilities for the boundary-layer tracer transport paper.

This package holds everything the analysis notebook needs besides the notebook
itself:

* **The run registry.** Which RAMS simulation is analysed for each storm, and where
  each run's raw output and derived products live. Every path in the analysis is
  built from ``RUN_DIRS`` and ``DERIVED_DIRS``, so pointing the analysis at a
  different set of runs means editing those two dictionaries and nothing else.
* **Physical thresholds and tracer definitions** shared by every figure.
* **Derived variables.** ``postprocess_storm_ds`` turns raw 5-minute analysis output
  into the dataset the figures read: thermodynamics, in-updraft and in-anvil flags,
  the per-storm analysis mask, and tracer number concentrations.
* **Cached derived products.** The in-updraft tracer profiles, the storm development
  dataframes, and the upper-level tracer totals and mass fluxes. Each is expensive
  to compute, so each is cached on disk and checked against the run before it is
  trusted (see ``read_or_cache_pickle``).
* **Plotting and figure-data helpers.**

Importing it as ``blt`` is the convention throughout the notebooks. The former name
``blt_utils_limited`` still works and refers to this same module.
"""

########################################################################
############################## Imports ##############################
########################################################################

# Native packages
from pathlib import Path

# Non-native packages
import pandas as pd
from metpy.units import units
import metpy.constants as mpconstants
import metpy.calc as mpc
import numpy as np

# Imported for its side effect: it registers the `.pint` accessor on xarray objects,
# which the unit conversions below rely on
import pint_xarray
from tqdm.notebook import tqdm
import xarray as xr
import pickle as pkl

# User packages
import cloudy as cm

########################################################################
############################## Parameters ##############################
########################################################################

# Short names of the three storms, in the order every figure shows them
STORM_NAMES = ["ic_wk", "sl_wk", "sc_wk"]

# Mapping from storm short name to human-readable name
STORM_LABELS = {
    "ic_wk": "Isolated convection",
    "sl_wk": "Squall line",
    "sc_wk": "Supercell",
}
STORM_LABELS_SHORT = {
    "ic_wk": "Iso. conv.",
    "sl_wk": "Squall line",
    "sc_wk": "Supercell",
}

# Random seed
SEED = 325976123

# Condensate loading and vertical velocity thresholds for what constitutes in-updraft gridpoints
IN_UPDRAFT_R_CONDENSATE_THRESHOLD = 1e-4  # kg/kg
IN_UPDRAFT_W_THRESHOLD = 1  # m/s

# Same for anvil (points are defined where the magnitude of vertical motion is
# less than the updraft threshold)
# Following McGee and van den Heever (2014)
ANVIL_R_CONDENSATE_THRESHOLD = 1e-5  # kg/kg
ANVIL_W_THRESHOLD = 1  # m/s (magnitude must be less than this)
ANVIL_Z_THRESHOLD = 10_000  # m

# Address of the dask cluster scheduler
DASK_SCHEDULER_ADDRESS = "127.0.0.1:8786"

# Rain rates for rain-sourced tracer emission
RS_TRACER_THRESHOLD_RAIN_RATES = [
    0.00027777,  # 1 mm/hr
    0.00138885,  # 5 mm/hr
    0.00277771,  # 10 mm/hr
    0.00555542,  # 20 mm/hr
    0.01111083,  # 40 mm/hr
]
# Same but in mm/hr
RS_TRACER_THRESHOLDS_MM_HR = [1, 5, 10, 20, 40]

# Names of the variables for the fixed-source and rain-sourced tracers
RS_TRACERS = [f"TRACERP{str(x).zfill(3)}" for x in range(1, 6)]
FS_TRACERS = [f"TRACERP{str(x).zfill(3)}" for x in range(7, 43)]

# Non-tracer storm variables that we use
STORM_VARIABLES_USED = [
    "THETA",
    "theta_rho",
    "buoyancy",
    "R_condensate",
    "storm_position_x",
    "storm_position_y",
    "UC",
    "VC",
    "WC",
    "PCPRR",
]

# Start time of the simulations
SIMULATION_START_TIME = pd.Timestamp("1991-04-26 21:00:00")
SIMULATION_END_TIME = pd.Timestamp("1991-04-27 00:00:00")

# Grid and output settings shared by all three runs (DELTAX and FRQLITE in each
# run's RAMSIN). The upper-level mass flux integrates over both.
HORIZONTAL_GRID_SPACING_M = 500
LITE_OUTPUT_INTERVAL_SECONDS = 2

# Height thresholds for defining upper levels
# All of the ones we calculated for sensitivity tests
# (the lower-case "l" in the name is historical; notebooks use it as spelled)
ALL_UL_Z_THRESHOlDS = [7, 8, 9, 10]  # km

# The one we actually use
UL_Z_THRESHOLD = 10  # km

# Progressive colormap to use where relevant
PROGRESSIVE_CMAP = "viridis"

########################################################################
############################## Filepaths ##############################
########################################################################

# Root of this repository: src/blt_utils/__init__.py is three levels below it
PROJECT_DIR = Path(__file__).parent.parent.parent
REPLICATION_DIR = PROJECT_DIR / "replication"

# Where the analysis notebook writes its figures and the data behind them.
# Deliberately not `paper_figures/`, which holds the figures of the submitted
# manuscript, made from the original runs
FIGS_DIR = PROJECT_DIR / "analysis" / "paper_figures"
FIGURE_DATA_DIR = FIGS_DIR / "figure_data"

# Matplotlib style to use; this is the format for passing matplotlib a filepath
# rather than a name in your stylelib
MPL_STYLE = f"file://{REPLICATION_DIR / 'cd_bl_slides.mplstyle'}"

# In-repo tree of derived products for the original manuscript runs. Of the runs
# analysed here only the supercell is a manuscript run, and its products live in
# `DERIVED_DIR / "sc_wk"` (see DERIVED_DIRS below)
DATA_DIR = REPLICATION_DIR / "data"
DERIVED_DIR = DATA_DIR / "derived"

# The manuscript runs' upper-level mass flux caches, one dict of all three storms
# per threshold height. Kept for the validation notebooks that compare against the
# manuscript; the analysis here caches per run in UPPER_LEVEL_DERIVED_DIRS instead
UL_FLUXES_DIR = DERIVED_DIR

########################################################################
############################## Run registry ##############################
########################################################################

# Root of the RAMS output tree on moonbow. Every run directory hangs off it, so a
# filesystem move only needs editing here
RAMS_IO_ROOT = Path("/moonbow/cmdavis4/projects/bl_transport/rams_io")

# The simulation analysed for each storm. All three start from the supercell's
# initial thermodynamic profile: the isolated convection and the squall line are
# reruns on that profile, and the supercell is the original manuscript run
RUN_DIRS = {
    "ic_wk": (
        RAMS_IO_ROOT / "ic_wk" / "validation" / "other_forcing_bthp-2.0_radius-2x"
    ),
    "sl_wk": RAMS_IO_ROOT / "sl_wk" / "validation" / "sc_profile",
    "sc_wk": (
        RAMS_IO_ROOT
        / "supercell_wk"
        / "all_tracers"
        / "newtracer_forcing-grant2014_dx-500_frqlite-2_stratwinds"
    ),
}

# Raw RAMS output: the 5-minute analysis files (a-A-*) and the 2-second lite
# files (a-L-*) for each storm
RAMS_OUTPUT_DIRS = {
    storm: run_dir / "output" for storm, run_dir in RUN_DIRS.items()
}

# Where each run's derived products live.
# The two reruns keep theirs beside the run, in `<run_dir>/derived/`.
# The supercell is a manuscript run, and the manuscript runs keep theirs in this
# repo's `replication/data/derived/<storm>/` tree; the run's own `derived/` is
# absent. Getting this wrong would not raise: the supercell would find no cache
# and silently recompute every product from raw output, which takes hours
DERIVED_DIRS = {
    "ic_wk": RUN_DIRS["ic_wk"] / "derived",
    "sl_wk": RUN_DIRS["sl_wk"] / "derived",
    "sc_wk": DERIVED_DIR / "sc_wk",
}

# Suffix on the cache files of the two products that use the air mass below about
# 3 km: the tracer profiles and the development dataframes.
# Caches written before the air mass used the RAMS T-cell thickness (see
# read_t_level_cell_thicknesses) have no suffix. Their air masses are up to 4.9% too
# small below about 3 km. The time-count staleness check cannot detect that, so the
# new file names keep those caches from ever being read again.
# The upper-level caches need no suffix: every level above 7 km is 250 m deep, where
# the old and new air masses agree
AIR_MASS_CACHE_SUFFIX = "_cellthickness"

# In-updraft tracer vertical profiles (Figures 7-9 and the entrainment statistics)
TRACER_TOTALS_PROFILE_FILEPATHS = {
    storm: derived_dir / f"tracer_totals_profile_ds{AIR_MASS_CACHE_SUFFIX}.nc"
    for storm, derived_dir in DERIVED_DIRS.items()
}

# Storm development dataframes (Figures 3-5 and 10)
DEVELOPMENT_DATAFRAME_FILEPATHS = {
    storm: derived_dir / f"development_dataframe{AIR_MASS_CACHE_SUFFIX}.pkl"
    for storm, derived_dir in DERIVED_DIRS.items()
}

# Upper-level tracer totals and mass fluxes (Figures 11-12), one cache file per
# threshold height
UPPER_LEVEL_DERIVED_DIRS = {
    storm: derived_dir / "upper_level"
    for storm, derived_dir in DERIVED_DIRS.items()
}

########################################################################################
############################## Derived variable functions ##############################
########################################################################################


def read_rams_header_field(header_filepath, field_name):
    """Read one numeric field from a RAMS ``*-head.txt`` header file.

    Mechanism: the header lists each field on consecutive lines: its name (such
    as ``__zmn01``), then the number of values, then one value per line. The file
    is scanned for the name line, and the values that follow it are parsed.

    Raises:
        KeyError: if the header has no field of that name.

    Returns:
        The field's values as a 1-D float array.
    """
    with Path(header_filepath).open("r") as header_file:
        for line in header_file:
            if line.strip() == field_name:
                # The line after the name gives how many values follow, one per line
                value_count = int(next(header_file).strip())
                return np.array(
                    [float(next(header_file).strip()) for _ in range(value_count)]
                )
    raise KeyError(f"No field {field_name} in the RAMS header {header_filepath}")


def read_t_level_cell_thicknesses(storm_ds, header_filepath):
    """The thickness (m) of every RAMS T-level grid cell, on a dataset's ``z``.

    Mechanism: the RAMS header stores two sets of vertical levels.

    * ``__ztn01`` holds the T-level centres, which are the dataset's ``z``.
    * ``__zmn01`` holds the W levels, which are the faces between T-level cells.

    Counting from 0, T level k spans W levels k - 1 and k, so its thickness is
    ``zmn[k] - zmn[k - 1]``. T level 0 lies below the ground and has no lower
    face. It gets NaN, which keeps it out of every air-mass sum.

    Why the faces rather than ``z.diff("z")``: on the stretched part of the grid,
    below about 3 km, RAMS does not centre the T levels between their faces. The
    spacing between centres there is 4.7% smaller than the cell thickness (the
    lowest cell is 25.0 m deep, but its centres are 23.8 m apart). Above the
    stretched part every cell is 250 m deep, and the two agree.

    The header's T-level centres are checked against ``storm_ds["z"]``. A header
    from a run with a different vertical grid therefore raises, instead of
    silently giving the wrong thicknesses.

    Args:
        storm_ds: The dataset whose ``z`` coordinate the thicknesses go on.
        header_filepath: Any ``*-head.txt`` file of the run that produced it.

    Returns:
        A DataArray of cell thicknesses (m) on ``storm_ds``'s ``z`` coordinate.
    """
    t_level_centres = read_rams_header_field(header_filepath, "__ztn01")
    w_level_heights = read_rams_header_field(header_filepath, "__zmn01")

    # Refuse a header whose T levels are not this dataset's T levels
    header_matches_dataset = len(t_level_centres) == storm_ds.sizes[
        "z"
    ] and np.allclose(t_level_centres, storm_ds["z"].values)
    if not header_matches_dataset:
        raise ValueError(
            f"The T levels in {header_filepath} do not match the dataset's z"
            " coordinate, so its cell thicknesses do not apply to this dataset."
            " Pass the header file of the run the dataset came from."
        )

    # Each T level above the ground spans the W levels just below and just above it
    t_level_thicknesses = np.full(len(t_level_centres), np.nan)
    t_level_thicknesses[1:] = np.diff(w_level_heights)
    return xr.DataArray(t_level_thicknesses, coords={"z": storm_ds["z"]}, dims="z")


def calculate_thermodynamic_variables(ds, t_level_thicknesses_m=None):
    """Add the derived thermodynamic and physical variables to a RAMS dataset.

    Mechanism: RAMS writes the Exner function (``PI``, already multiplied by
    c_p = 1004 J/kg/K), potential temperature, and mixing ratios. Temperature and
    pressure follow from the Exner function, and the humidity variables follow
    from pressure and vapour mixing ratio through MetPy. Units are attached with
    pint only for each MetPy call and stripped again, so every variable stored on
    the dataset is a plain float array.

    Two variables depend on what the dataset carries:

    * ``air_mass`` needs the base-state density ``DN0``, which parcel output lacks.
      Each gridpoint's air mass is DN0 times the 500 m x 500 m cell area times the
      cell thickness, which ``t_level_thicknesses_m`` must then supply (see
      ``read_t_level_cell_thicknesses``).
    * ``buoyancy`` needs a horizontal layer mean, so it is skipped for datasets
      without x and y dimensions.

    Args:
        ds: RAMS output.
        t_level_thicknesses_m: Thickness (m) of each T-level cell, on ``ds``'s
            ``z``. Required when ``ds`` carries ``DN0``.

    The dataset is modified in place and also returned.
    """
    ds["T"] = ds["PI"] * ds["THETA"] / 1004.0
    ds["R_condensate"] = ds["RTP"] - ds["RV"]
    ds["P"] = 1000.0 * ((ds["PI"] / 1004.0) ** (1004.0 / 287.0))
    vp = mpc.vapor_pressure(ds["P"] * units("hPa"), ds["RV"] * units("kg/kg"))
    ds["dewpoint"] = mpc.dewpoint(vp).pint.to("K").pint.dequantify()
    ds["vapor_pressure"] = vp.pint.to("hPa").pint.dequantify()
    ds["saturation_vapor_pressure"] = (
        mpc.saturation_vapor_pressure(ds["T"] * units("K"))
        .pint.to("hPa")
        .pint.dequantify()
    )
    # Air mass of each gridpoint: base-state density times the cell volume.
    # Parcels don't have DN0 at present; rather than make a separate function we'll just calculate it if we can
    if "DN0" in ds.data_vars:
        # A missing thickness would otherwise fall back to some approximation
        # without saying so, which is how the old centre-spacing bias went unnoticed
        if t_level_thicknesses_m is None:
            raise ValueError(
                "air_mass needs the RAMS T-level cell thicknesses; see"
                " read_t_level_cell_thicknesses"
            )
        ds["air_mass"] = (
            ds["DN0"] * HORIZONTAL_GRID_SPACING_M**2 * t_level_thicknesses_m
        )
    ds["RH"] = mpc.relative_humidity_from_mixing_ratio(
        ds["P"] * units("hPa"), ds["T"] * units("K"), ds["RV"]
    ).pint.dequantify()
    # Make a flag for whether it's supersaturated for convenience
    ds["supersaturated"] = ds["RH"] >= 1

    # Calculate density potential temperature
    ds["theta_rho"] = ds["THETA"] * (1 + 0.608 * ds["RV"] - ds["R_condensate"])

    # And buoyancy
    # Can't calculate this for the parcel dataset without interpolating, so just don't calculate it for now
    if "x" in ds.dims and "y" in ds.dims:
        tr_layer_mean = ds["theta_rho"].mean(["x", "y"])
        ds["buoyancy"] = (
            mpconstants.g * (ds["theta_rho"] - tr_layer_mean) / tr_layer_mean
        ).pint.dequantify()

    # Break the R_condensate term into categories for water and for ice
    ds["R_liquid"] = ds["RCP"] + ds["RRP"]
    ds["R_ice"] = ds["RPP"] + ds["RSP"] + ds["RAP"] + ds["RGP"] + ds["RHP"]

    return ds


def calculate_derived_variables(storm_ds, storm_name, header_filepath=None):
    """Add derived variables, flags, and coordinates to one storm's RAMS output.

    Mechanism: on top of ``calculate_thermodynamic_variables`` this adds

    * the gridpoint classification flags: ``in_updraft`` (cloudy and rising
      faster than 1 m/s) and ``in_anvil`` (cloudy, nearly still, and above
      10 km, following McGee and van den Heever 2014);
    * the per-storm ``in_analysis`` mask from ``ANALYSIS_FLAG_FUNCTIONS``, which
      for the supercell keeps only the right mover;
    * kinematic fields (vertical vorticity, horizontal divergence) and
      base-state-relative winds and potential temperature, the base state being
      the horizontal mean at the first output time;
    * a ``t_minutes`` coordinate, counted from the dataset's first time, and a
      surface rain rate in mm/hr.

    The x and y coordinates are shifted to start at 0, so positions are measured
    from the domain's corner.

    The air mass needs the RAMS T-cell thicknesses, which come from a header
    file. By default that is the first header in ``RAMS_OUTPUT_DIRS[storm_name]``.
    Pass ``header_filepath`` for output from any other run; a header whose
    vertical grid does not match the dataset raises.
    """
    # The T-cell thicknesses that the air mass needs. Only output carrying the
    # base-state density DN0 gets an air mass, so only that output needs a header
    t_level_thicknesses_m = None
    if "DN0" in storm_ds.data_vars:
        if header_filepath is None:
            header_filepath = sorted(
                RAMS_OUTPUT_DIRS[storm_name].glob("a-A-*-head.txt")
            )[0]
        t_level_thicknesses_m = read_t_level_cell_thicknesses(
            storm_ds, header_filepath
        )

    storm_ds = calculate_thermodynamic_variables(storm_ds, t_level_thicknesses_m)

    # Shift the x and y coords so that they start from 0
    storm_ds["x"] = storm_ds["x"] - min(storm_ds["x"])
    storm_ds["y"] = storm_ds["y"] - min(storm_ds["y"])

    # Add an in-updraft flag
    storm_ds["in_updraft"] = (
        storm_ds["R_condensate"] >= IN_UPDRAFT_R_CONDENSATE_THRESHOLD
    ) & (storm_ds["WC"] >= IN_UPDRAFT_W_THRESHOLD)

    # Add an in-anvil flag, following McGee and van den Heever (2014): cloudy, close
    # to neutrally buoyant (so NOT in an active updraft or downdraft), and aloft.
    # calculate_development_dataframe consumes this to build anvil_mass; the flag's
    # definition was missing from this trimmed copy of blt_utils, which made that
    # function raise KeyError on any freshly-read dataset.
    storm_ds["in_anvil"] = (
        (storm_ds["R_condensate"] >= ANVIL_R_CONDENSATE_THRESHOLD)
        & (abs(storm_ds["WC"]) < ANVIL_W_THRESHOLD)
        & (storm_ds["z"] >= ANVIL_Z_THRESHOLD)
    )

    if storm_name == "sc_wk":
        storm_ds["in_right_mover"] = sc_wk_in_right_mover(
            storm_ds["x"], storm_ds["y"]
        )

    # Add a time in minutes coordinate
    storm_ds = storm_ds.assign_coords(
        t_minutes=(
            storm_ds["time"] - storm_ds["time"].values[0]
        ).dt.total_seconds()
        // 60
    )

    # Calculate vertical vorticity
    storm_ds["vertical_vorticity"] = storm_ds["VC"].differentiate(
        "x"
    ) - storm_ds["UC"].differentiate("y")

    # Calculate horizontal divergence
    storm_ds["divergence"] = storm_ds["UC"].differentiate("x") + storm_ds[
        "VC"
    ].differentiate("y")

    # Add base state-relative versions of some variables
    for var in ["UC", "VC", "THETA"]:
        storm_ds[var + "_bsr"] = storm_ds[var] - storm_ds.isel({"time": 0})[
            var
        ].mean(["x", "y"])
    # Add theta deficit
    storm_ds["theta_deficit"] = -storm_ds["THETA_bsr"]

    # Add a flag for whether a gridpoint should be included in analysis
    storm_ds["in_analysis"] = ANALYSIS_FLAG_FUNCTIONS[storm_name](storm_ds)

    # Add a mm/hr version of pcprr
    storm_ds["PCPRR_mm_hr"] = (
        (storm_ds["PCPRR"] * units("kg/m^2/s") / mpconstants.density_water)
        .pint.to("mm/hr")
        .pint.dequantify()
    )

    return storm_ds


######################################################################################
############################## Datapoint flag functions ##############################
######################################################################################


def sc_wk_right_mover_boundary(x):
    """The y position (m) of the line separating the supercell's two movers.

    The supercell splits into a right mover and a left mover. This straight line,
    rising 10 km over every 50 km of x, separates them for the whole simulation.
    Points on or below it belong to the right mover.
    """
    return 78000 + x * (10000 / 50000)


def sc_wk_in_right_mover(x, y):
    """Whether the point (x, y) lies in the supercell's right mover."""
    return y <= sc_wk_right_mover_boundary(x)


def check_storm_ds(ds, lite=False):
    """Raise if a storm's analysis output does not carry the expected tracers.

    Every run carries the fixed-source tracers 7-42 and the rain-sourced tracers
    1-5. Tracer 6, for 80 mm/hr of rain, exists but is unused. Any other tracer
    variable means the output came from a differently configured run. Lite output
    carries no tracers, so it is not checked.
    """
    if not lite:
        # Only tracer not being used should be 6, which was for 80 mm/hr of rain
        assert [
            x
            for x in ds.data_vars
            if x.startswith("TRACERP")
            and x not in RS_TRACERS
            and x not in FS_TRACERS
        ] == ["TRACERP006"]


# Function that returns a flag for whether a gridpoint should be included in analysis.
# The isolated convection and squall line use every gridpoint; the supercell uses
# only its right mover
ANALYSIS_FLAG_FUNCTIONS = {
    "ic_wk": lambda ds: xr.DataArray(
        np.full((len(ds["x"]), len(ds["y"])), True),
        coords={"x": ds["x"], "y": ds["y"]},
    ),
    "sl_wk": lambda ds: xr.DataArray(
        np.full((len(ds["x"]), len(ds["y"])), True),
        coords={"x": ds["x"], "y": ds["y"]},
    ),
    "sc_wk": lambda ds: ds["in_right_mover"],
}


################################################################################
############################## Plotting functions ##############################
################################################################################


def label_x_axis(ax, short=False):
    """Label a time axis in simulation minutes, with ticks every 30 minutes."""
    ax.xaxis.set_ticks(np.arange(0, 181, 30))
    ax.set_xlabel(f"{'Sim.' if short else 'Simulation'} time (minutes)")


def axes_title(letter_ix, title):
    """Prefix a panel title with its letter: index 0 gives "(a) title"."""
    return f"({chr(ord('a') + letter_ix)}) {title}"


################################################################################
############################## Utilities ##############################
################################################################################


def simulation_time_to_minutes(simulation_time):
    """Convert a simulation time (Timestamp or datetime) to minutes since the start."""
    return (simulation_time - SIMULATION_START_TIME) / pd.Timedelta(minutes=1)


def minutes_to_simulation_time(minutes):
    """Convert minutes since the start to a simulation time (numpy datetime64)."""
    return (SIMULATION_START_TIME + pd.Timedelta(minutes=minutes)).to_numpy()


#######################################################################################
############################## Input data postprocessing ##############################
#######################################################################################


def postprocess_storm_ds(ds, storm_name, lite=False, header_filepath=None):
    """Turn one storm's raw RAMS output into the dataset the analysis reads.

    Mechanism: checks the tracer set, adds the derived variables and flags from
    ``calculate_derived_variables``, then (for analysis output) builds the
    tracer quantities:

    * ``total_fs_tracer``, the sum of all fixed-source tracers, and its
      accumulated surface-emission counterpart ``ACCtotal_fs_tracer``;
    * a ``<tracer>_nc`` number-concentration version of every tracer variable,
      the mixing ratio times the gridpoint air mass, so tracer amounts can be
      summed over gridpoints.

    Set ``lite=True`` for lite output, which carries no tracers. Lite output still
    needs every variable ``calculate_derived_variables`` reads, so it does not work
    on lite files that carry only a subset; the upper-level mass flux therefore
    uses ``read_upper_level_lite_dataset`` instead.

    ``header_filepath`` is passed to ``calculate_derived_variables``. Leave it as
    None for the runs in ``RUN_DIRS``; give it for output from any other run.
    """
    # Run basic checks
    check_storm_ds(ds, lite=lite)
    # Calculate derived variables and analysis flag
    ds = calculate_derived_variables(ds, storm_name, header_filepath=header_filepath)
    # Calculate total fixed-source tracer variables
    if not lite:
        ds["total_fs_tracer"] = 0
        ds["ACCtotal_fs_tracer"] = 0
        for tracer_var in FS_TRACERS:
            ds["total_fs_tracer"] = ds["total_fs_tracer"] + ds[tracer_var]
            ds["ACCtotal_fs_tracer"] = (
                ds["ACCtotal_fs_tracer"] + ds["ACC" + tracer_var]
            )
        # Make number concentration versions of all of the tracer variables
        for var in ds.data_vars:
            if "tracer" in var.lower():
                ds[var + "_nc"] = ds[var] * ds["air_mass"]
    return ds


def calculate_tracer_profile_ds(storm_ds):
    """Sum each tracer over the in-updraft gridpoints at every height and time.

    Mechanism: the dataset is masked to in-analysis gridpoints (the supercell's
    right mover), then looped over output times so each step stays small enough
    to compute. At each time it produces three sets of horizontal sums:

    * every tracer and its number concentration over in-updraft gridpoints,
      together with the updraft air mass;
    * the same over in-updraft gridpoints that are also supersaturated, with an
      ``ss_`` prefix;
    * the accumulated surface emission (``ACC*``) as number concentrations.

    The accumulated tracers are surface totals, not 3D fields, so they are
    converted to numbers with the air mass of the lowest model level above
    ground (z index 1). That air mass comes from the base-state density DN0, so
    it does not change in time.

    Returns:
        A dataset with dimensions (time, z).
    """
    tracer_totals_list = []

    # Make a list of all of the tracer variables
    all_current_tracer_vars = FS_TRACERS + RS_TRACERS + ["total_fs_tracer"]
    # Add the accumulated variables
    all_acc_tracer_vars = ["ACC" + tracer for tracer in all_current_tracer_vars]
    # Make number concentration versions of the current tracers
    all_current_nc_tracer_vars = []
    for tracer in all_current_tracer_vars:
        storm_ds[tracer + "_nc"] = storm_ds[tracer] * storm_ds["air_mass"]
        all_current_nc_tracer_vars.append(tracer + "_nc")

    # Filter to in-analysis points
    storm_ds = storm_ds.where(storm_ds["in_analysis"])

    for time_ix in tqdm(range(0, len(storm_ds["time"]))):
        this_time_ds = storm_ds.isel({"time": time_ix})
        # First sum up the 3d tracers
        tracer_totals_list.append(
            this_time_ds.where(this_time_ds["in_updraft"])[
                all_current_tracer_vars
                + all_current_nc_tracer_vars
                + ["in_updraft", "air_mass"]
            ]
            .sum(["x", "y"])
            .compute()
        )
        # Also make a version with supersaturation
        this_time_ss_ds = this_time_ds.where(
            this_time_ds["in_updraft"].astype(bool)
            & this_time_ds["supersaturated"].astype(bool)
        )[
            all_current_tracer_vars
            + all_current_nc_tracer_vars
            + ["supersaturated", "air_mass"]
        ].sum(
            ["x", "y"]
        )
        # Rename all the variables
        this_time_ss_ds = this_time_ss_ds.rename({
            var: "ss_" + var
            for var in this_time_ss_ds.data_vars
            if var != "supersaturated"
        })
        tracer_totals_list.append(this_time_ss_ds.compute())

        # Now sum up the accumulated (2d) tracers
        # Since we're now using DN0, which does not change in time, for calculating the air mass in each gridpoint,
        # we can just pull out the current value of the air mass (which also won't change in time) for converting
        # the accumulated mixing ratio into a number
        acc_nc = (
            this_time_ds[all_acc_tracer_vars]
            * this_time_ds.isel({"z": 1})["air_mass"]
        ).sum(["x", "y"])
        acc_nc = acc_nc.rename({k: k + "_nc" for k in acc_nc.data_vars})
        tracer_totals_list.append(acc_nc.compute())

    # Make it into a dataset
    print("Combining...")
    tracer_totals_profile_ds = xr.combine_by_coords(
        [x.expand_dims("time") for x in tracer_totals_list]
    )
    return tracer_totals_profile_ds


def calculate_development_dataframe(storm_ds):
    """Build the time series that describe one storm's development.

    Mechanism: over in-analysis gridpoints, at every output time, it computes

    * ``max_updraft``: the strongest updraft at 5 km. Each gridpoint first takes
      the weakest w in its 4 x 4 neighbourhood, so a single-gridpoint spike cannot
      set the maximum. The result is then smoothed with a 3-output (15 minute)
      running mean.
    * ``storm_centroid_x`` and ``storm_centroid_y``: the mean position of the 5 km
      gridpoints whose w is at least 20% of the run's peak max_updraft.
    * ``updraft_mass`` and ``anvil_mass``: the air mass of the in-updraft and
      in-anvil gridpoints.
    * ``total_rain`` and ``rain_thresholdN``: the domain-summed surface rain rate,
      and the number of gridpoints raining at least each rain-sourced tracer's
      emission threshold.

    Returns:
        A DataFrame with one row per output time.
    """
    # Filter to in-analysis
    storm_ds = storm_ds.where(storm_ds["in_analysis"])

    max_updrafts = (
        storm_ds.sel({"z": 5000}, method="nearest")["WC"]
        .rolling(x=4, y=4)
        .min()
        .max(["x", "y"])
        # The analysis output is read one timestep per file, so `time` arrives as
        # one dask chunk per timestep. A rolling window cannot span chunks, so
        # rolling(time=3) over that raises "Moving window (=3) must between 1 and 2".
        # Collapse time into a single chunk first; this array is 1-D by now (one
        # value per output time), so it costs nothing.
        .chunk({"time": -1})
        .rolling(time=3)
        .mean()
    ).compute()

    # Calculate centroids
    w20_max = 0.2 * max_updrafts.max()
    storm_centroids = (
        storm_ds["x"]
        .weighted(storm_ds.sel({"z": 5000}, method="nearest")["WC"] >= w20_max)
        .mean(["x", "y"])
        .compute(),
        storm_ds["y"]
        .weighted(storm_ds.sel({"z": 5000}, method="nearest")["WC"] >= w20_max)
        .mean(["x", "y"])
        .compute(),
    )

    updraft_mass = (
        storm_ds.where(storm_ds["in_updraft"])["air_mass"]
        .sum(["x", "y", "z"])
        .compute()
    )

    anvil_mass = (
        storm_ds.where(storm_ds["in_anvil"])["air_mass"]
        .sum(["x", "y", "z"])
        .compute()
    )

    total_rain = storm_ds["PCPRR"].sum(["x", "y"]).compute()

    rain_by_threshold = {
        f"rain_threshold{ix}": (storm_ds["PCPRR"] >= threshold).sum(["x", "y"])
        for ix, threshold in enumerate(RS_TRACER_THRESHOLD_RAIN_RATES)
    }

    development_dict = {
        "t_minutes": storm_ds["t_minutes"],
        "max_updraft": max_updrafts,
        "updraft_mass": updraft_mass,
        "anvil_mass": anvil_mass,
        "total_rain": total_rain,
        "storm_centroid_x": storm_centroids[0],
        "storm_centroid_y": storm_centroids[1],
    }
    development_dict.update(rain_by_threshold)

    development_df = pd.DataFrame(development_dict)
    return development_df


#######################################################################################
############################## Cached derived products ##############################
#######################################################################################


def analysis_output_time_count(storm):
    """How many 5-minute output times a storm's run has on disk.

    Mechanism: RAMS writes one ``a-A-<timestamp>-g1.h5`` file per analysis time,
    so the file count is the time count. Counting files rather than opening the
    dataset makes this cheap enough to guard every cached product against
    staleness, including products read without loading the raw run.

    Returns:
        The number of analysis output times, or None if the output directory is
        absent.
    """
    output_dir = RAMS_OUTPUT_DIRS[storm]
    if not output_dir.is_dir():
        return None
    return len(list(output_dir.glob("a-A-*-g1.h5")))


def _time_count_of_result(result):
    """Number of output times in a cached result: rows of a DataFrame, else `time`."""
    if isinstance(result, pd.DataFrame):
        return len(result)
    return result.sizes["time"]


def read_or_cache_pickle(
    cache_path, compute_result, *args, expected_time_count=None, **kwargs
):
    """Read a pickled result if it is cached and current, else compute and cache it.

    Why this rather than ``cm.utils.read_or_cache_to``: that decorator cannot write
    ``.pkl`` files. ``cm.utils.write_file`` dispatches ``.pkl`` to a ``to_pkl``
    method that neither DataFrames nor Datasets have, and ``read_or_cache_to``
    swallows the resulting error. The cache therefore never populates, and every
    run recomputes while printing that it wrote the file. Writing with
    ``pd.to_pickle`` works for any picklable result.

    Staleness: a cache is keyed only by its path. A cache written while a run was
    still integrating therefore keeps being read after the run finishes with more
    output times, and every consumer then divides a short series by a long one.
    Passing *expected_time_count* makes that self-correcting. A cached result whose
    time count (rows of a DataFrame, length of ``time`` for a Dataset) disagrees is
    discarded and recomputed. A freshly computed result that disagrees raises
    rather than being cached.

    Args:
        cache_path: Where to read and write the pickle.
        compute_result: Callable that produces the result when the cache is cold.
        *args, **kwargs: Passed through to *compute_result*.
        expected_time_count: Number of output times the result must cover,
            normally ``analysis_output_time_count(storm)``. None skips the check.

    Returns:
        The cached or freshly computed result.
    """
    cache_path = Path(cache_path)

    # Use the cache only when it exists AND covers the expected number of times
    if cache_path.exists():
        cached_result = pd.read_pickle(cache_path)
        cache_is_stale = (
            expected_time_count is not None
            and _time_count_of_result(cached_result) != expected_time_count
        )
        if not cache_is_stale:
            print(f"Read cached result from {cache_path}")
            return cached_result
        # Say why it recomputes, because a silent recompute of an expensive
        # reduction looks like a broken cache
        print(
            f"Stale cache at {cache_path}: it holds"
            f" {_time_count_of_result(cached_result)} times but the run has"
            f" {expected_time_count}. Recomputing and overwriting it."
        )
    else:
        print(f"No cache at {cache_path}, computing...")

    result = compute_result(*args, **kwargs)

    # Refuse to cache a result that already fails the caller's expectation, so a bad
    # computation cannot poison later runs the way a stale file would
    if (
        expected_time_count is not None
        and _time_count_of_result(result) != expected_time_count
    ):
        raise ValueError(
            f"{compute_result.__name__} returned"
            f" {_time_count_of_result(result)} times but {expected_time_count}"
            f" were expected for {cache_path}. Not caching it."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(result, cache_path)
    print(f"Wrote result to {cache_path}")
    return result


def read_or_compute_tracer_totals_datasets(storm_dss):
    """Load each storm's in-updraft tracer profiles, computing any that are missing.

    Mechanism: the profiles come from ``calculate_tracer_profile_ds``, which
    collapses every tracer from (time, z, y, x) to (time, z). That touches the
    whole dataset, so the result is cached as netCDF at
    ``TRACER_TOTALS_PROFILE_FILEPATHS[storm]``. A cached file is used only when
    its number of times matches the run's output files on disk. Otherwise it was
    written while the run was still integrating, and it is recomputed and
    overwritten.

    A computed file is written out and then reopened from disk, rather than kept
    in memory. The first run and every later run then take the same code path, so
    any problem with the round trip through disk shows up on the run that created
    the file.

    Args:
        storm_dss: Mapping of storm name -> postprocessed 5-minute dataset. Used
            only for storms whose cache is missing or stale.

    Returns:
        (tracer_totals_profile_dss, tracer_totals_dss): the vertical profiles,
        and their sums over height excluding the below-ground level at z index 0.
        Both are keyed by storm name.
    """
    tracer_totals_profile_dss = {}
    for storm in STORM_NAMES:
        profile_filepath = TRACER_TOTALS_PROFILE_FILEPATHS[storm]

        if profile_filepath.exists():
            # Check the cache's time count against the output files on disk
            cached_profile_ds = xr.open_dataset(profile_filepath)
            expected_time_count = analysis_output_time_count(storm)
            cache_is_stale = (
                expected_time_count is not None
                and cached_profile_ds.sizes.get("time") != expected_time_count
            )
            if not cache_is_stale:
                print(f"Read tracer profile dataset from {profile_filepath}")
                tracer_totals_profile_dss[storm] = cached_profile_ds
                continue
            cached_profile_ds.close()
            print(
                f"Stale tracer profile cache at {profile_filepath}: it holds"
                f" {cached_profile_ds.sizes.get('time')} times but the run has"
                f" {expected_time_count}. Recomputing and overwriting it."
            )

        # Compute it, write it, and reopen it from disk (see docstring)
        print(f"Computing the tracer profile dataset for {storm}...")
        tracer_totals_profile_ds = calculate_tracer_profile_ds(storm_dss[storm])
        profile_filepath.parent.mkdir(parents=True, exist_ok=True)
        tracer_totals_profile_ds.to_netcdf(profile_filepath)
        print(f"Wrote tracer profile dataset to {profile_filepath}")
        tracer_totals_profile_dss[storm] = xr.open_dataset(profile_filepath)

    # Sum over the vertical, skipping the model's below-ground level at index 0
    tracer_totals_dss = {
        storm: profile_ds.isel({"z": slice(1, len(profile_ds["z"]))}).sum("z")
        for storm, profile_ds in tracer_totals_profile_dss.items()
    }
    return tracer_totals_profile_dss, tracer_totals_dss


def read_or_compute_development_dataframes(storm_dss):
    """Load each storm's development dataframe, computing any that are missing.

    Mechanism: the dataframe comes from ``calculate_development_dataframe`` over
    in-analysis gridpoints, which for the supercell means the right mover. It is
    cached at ``DEVELOPMENT_DATAFRAME_FILEPATHS[storm]`` and trusted only when it
    has one row per output time on disk; a cache with a different row count
    predates the run finishing and is recomputed.

    Each storm's centroid track is also attached to its dataset as
    ``storm_position_x`` and ``storm_position_y``, because the storm development
    figures centre their panels on it. This modifies ``storm_dss`` in place.

    Args:
        storm_dss: Mapping of storm name -> postprocessed 5-minute dataset.

    Returns:
        Mapping of storm name -> development DataFrame.
    """
    development_dfs = {}
    for storm in STORM_NAMES:
        this_storm_ds = storm_dss[storm]
        development_dfs[storm] = read_or_cache_pickle(
            DEVELOPMENT_DATAFRAME_FILEPATHS[storm],
            calculate_development_dataframe,
            this_storm_ds.where(this_storm_ds["in_analysis"]),
            expected_time_count=analysis_output_time_count(storm),
        )

        # Attach the storm centroid positions to the storm dataset
        for coordinate in ["x", "y"]:
            storm_dss[storm][f"storm_position_{coordinate}"] = xr.DataArray(
                development_dfs[storm][f"storm_centroid_{coordinate}"].values,
                coords={
                    "time": [
                        minutes_to_simulation_time(minutes)
                        for minutes in development_dfs[storm]["t_minutes"].values
                    ]
                },
            )

    return development_dfs


#######################################################################################
############################## Upper-level transport ##############################
#######################################################################################


def calculate_upper_level_tracer_dataframe(storm_ds, upper_level_z_threshold_m):
    """Total tracer content above a height, as a time series.

    Mechanism: everything at or above the threshold height is summed over all
    three spatial dimensions. The ``in_analysis`` mask is deliberately not applied,
    so the supercell's total includes both its right and left movers, as the
    manuscript describes. No vertical-velocity mask is applied either, so the
    total includes tracer outside the updraft, such as in the anvil.

    The accumulated (``ACC*``) tracer variables are left out. They record the
    tracer emitted at the surface, so their sum over the upper-level volume has no
    meaning. The analysis takes the amount emitted, which it divides the
    upper-level totals by, from the full storm dataset instead.

    Returns:
        A DataFrame with one row per output time. Its columns are the volume sums
        of every current tracer variable (mixing ratio and ``*_nc`` number) and of
        the air mass.
    """
    upper_level_ds = storm_ds.sel(
        {"z": slice(upper_level_z_threshold_m, None, None)}
    ).copy()

    # Number-concentration versions of every current tracer
    all_current_tracer_vars = FS_TRACERS + RS_TRACERS + ["total_fs_tracer"]
    for tracer in all_current_tracer_vars:
        upper_level_ds[tracer + "_nc"] = (
            upper_level_ds[tracer] * upper_level_ds["air_mass"]
        )

    # Every current tracer variable, without the accumulated surface emission
    # (ACC*) variables
    current_tracer_var_names = [
        var_name
        for var_name in upper_level_ds.data_vars
        if "tracer" in var_name.lower() and not var_name.startswith("ACC")
    ]

    return (
        upper_level_ds[current_tracer_var_names + ["air_mass"]]
        .sum(["x", "y", "z"])
        .to_dataframe()
        .reset_index()
    )


def calculate_upward_transport_through_level(storm_ds, level_z_m, horizontal_mask=None):
    """Cumulative air mass and tracer carried upward through one level.

    Mechanism: at each 5-minute output time, every gridpoint of the level nearest
    ``level_z_m`` with w > 0 carries ``rho0 * w * dx^2`` kg/s of air upward, and
    that air carries its tracer mixing ratio q with it. Both are summed over the
    level and accumulated in time with the 5-minute output interval as the time
    step. The ratio of a tracer's cumulative transport to the air's is then the
    flux-weighted mixing ratio of all air that has crossed the level upward, in
    # kg^-1, the same unit as the upper-level tracer per unit upward mass flux.

    Why it exists: it attributes tracer to a region where the air crosses the
    level, inside the updrafts. The supercell's two updrafts are well separated
    there, so masking by the line between the movers splits them cleanly. Masking
    the tracer *above* the level does not, because the two anvils spread out and
    overlap.

    Caveats:
    * Only the 5-minute output carries tracers, so the time integral samples each
      updraft 37 times. That is too coarse for absolute values but adequate for
      comparing two regions of the same run.
    * It counts air as it crosses upward. Air that later sinks back, or that
      leaves the domain, still counts, unlike in the upper-level tracer totals.
    * Density is the base-state density DN0, as in the air mass. It is uniform
      across a level, so it barely affects the ratio.

    Args:
        storm_ds: A storm dataset from ``postprocess_storm_ds``.
        level_z_m: Height of the level, in m. The nearest model level is used.
        horizontal_mask: Optional boolean DataArray on (x, y), such as
            ``storm_ds["in_analysis"]``. None uses every gridpoint.

    Returns:
        A DataFrame with one row per output time: ``t_minutes``,
        ``cumu_upward_air_mass`` in kg, and ``cumu_upward_<tracer>_nc`` in # for
        every rain-sourced tracer and the total fixed-source tracer.
    """
    level_ds = storm_ds.sel({"z": level_z_m}, method="nearest")

    # Output interval, in s, which is the time step of the integral
    output_interval_s = (
        level_ds["time"].values[1] - level_ds["time"].values[0]
    ) / np.timedelta64(1, "s")

    # Upward air mass through each gridpoint over one output interval, in kg. Points
    # with w <= 0, or outside the requested region, contribute nothing
    contributes = level_ds["WC"] > 0
    if horizontal_mask is not None:
        contributes = contributes & horizontal_mask
    upward_air_mass = (
        level_ds["DN0"]
        * level_ds["WC"]
        * HORIZONTAL_GRID_SPACING_M**2
        * output_interval_s
    ).where(contributes, 0.0)

    # Running totals of the air, and of each tracer the air carries with it
    transported_tracers = RS_TRACERS + ["total_fs_tracer"]
    cumulative_transport = {
        "cumu_upward_air_mass": upward_air_mass.sum(["x", "y"]).cumsum("time")
    }
    for tracer in transported_tracers:
        cumulative_transport[f"cumu_upward_{tracer}_nc"] = (
            (upward_air_mass * level_ds[tracer]).sum(["x", "y"]).cumsum("time")
        )

    return (
        xr.Dataset(cumulative_transport)
        .compute()
        .to_dataframe()
        .reset_index()[
            ["time", "t_minutes"] + list(cumulative_transport.keys())
        ]
    )


def read_upper_level_lite_dataset(storm, upper_level_z_thresholds_m):
    """Read one or more model levels of a storm's 2-second lite output, with density.

    Mechanism: lite files exist for every 2 seconds from the simulation start to
    its end, so the file list is built from those times. A missing file raises
    rather than leaving a gap in the time integral. Each file is cut down to the
    level(s) nearest the threshold height(s) as it is read.

    Reading several levels costs little more than reading one. The lite files
    store each variable compressed in chunks that span the full column, so the
    whole column is decompressed to get even a single level.

    Air density is computed from pressure, temperature and vapour mixing ratio,
    because lite files carry no base-state density DN0. The formulas match
    ``calculate_thermodynamic_variables`` exactly. They are applied here directly
    rather than through ``postprocess_storm_ds``, because that also derives
    condensate, vorticity and rain rate. Those need variables that a lite file
    can omit: the squall line's lite files carry only WC, THETA, PI and RV.

    Args:
        storm: Storm name, a key of ``RAMS_OUTPUT_DIRS``.
        upper_level_z_thresholds_m: One height in m, or a list of heights. A single
            height gives a scalar ``z`` coordinate; a list gives a ``z`` dimension
            with one entry per height, in the order given.

    Returns:
        A dataset with WC and air_density on (time, [z,] y, x), chunked so each
        chunk is one whole horizontal level at one time.
    """
    # Every 2-second lite output time, formatted the way RAMS names its files
    lite_output_datetime_strings = (
        pd.date_range(
            start=SIMULATION_START_TIME,
            end=SIMULATION_END_TIME,
            freq=f"{LITE_OUTPUT_INTERVAL_SECONDS}s",
        )
        .strftime(cm.rams.RAMS_DT_STRFTIME_STR)
        .values
    )
    lite_filepaths = [
        RAMS_OUTPUT_DIRS[storm] / f"a-L-{datetime_string}-g1.h5"
        for datetime_string in lite_output_datetime_strings
    ]
    missing_lite_filepaths = [path for path in lite_filepaths if not path.exists()]
    if missing_lite_filepaths:
        raise FileNotFoundError(
            f"{len(missing_lite_filepaths)} of {len(lite_filepaths)} lite files"
            f" are missing for {storm}, starting with {missing_lite_filepaths[0]}"
        )

    # Read only the level(s) nearest the threshold height(s) from each file
    lite_ds = cm.rams.read_rams_output(
        input_filenames=lite_filepaths,
        time_dim_name="time",
        units=False,
        preprocess=lambda ds: ds.sel(
            {"z": upper_level_z_thresholds_m}, method="nearest", drop=False
        ),
    )

    # Temperature and pressure from the Exner function, as in
    # calculate_thermodynamic_variables
    temperature_k = lite_ds["PI"] * lite_ds["THETA"] / 1004.0
    pressure_hpa = 1000.0 * ((lite_ds["PI"] / 1004.0) ** (1004.0 / 287.0))
    vapor_pressure = mpc.vapor_pressure(
        pressure_hpa * units("hPa"), lite_ds["RV"] * units("kg/kg")
    )

    # Moist air density, via the mixing ratio implied by the vapour pressure
    mixing_ratio = mpc.mixing_ratio(
        partial_press=vapor_pressure.pint.to("hPa"),
        total_press=pressure_hpa * units("hPa"),
    ).pint.dequantify()
    lite_ds["air_density"] = (
        mpc.density(
            pressure=pressure_hpa * units("hPa"),
            temperature=temperature_k * units("K"),
            mixing_ratio=mixing_ratio,
        )
        .pint.to("kg/m^3")
        .pint.dequantify()
    )

    # Rechunk to full horizontal layer per chunk (and one level per chunk, when
    # several levels were read)
    full_layer_chunks = {
        "x": len(lite_ds["x"].values),
        "y": len(lite_ds["y"].values),
    }
    if "z" in lite_ds.dims:
        full_layer_chunks["z"] = 1
    return lite_ds[["WC", "air_density"]].chunk(full_layer_chunks).unify_chunks()


def sc_wk_right_mover_mask_for_lite_dataset(lite_ds):
    """The supercell's right-mover flag on a lite dataset's horizontal grid.

    Mechanism: ``sc_wk_right_mover_boundary`` is defined in coordinates that start
    at 0 at the domain's corner, because ``postprocess_storm_ds`` shifts x and y
    that way. ``read_upper_level_lite_dataset`` does not shift them, so the same
    shift is applied here before the boundary is evaluated. Both readers use the
    same model grid, so the shifted coordinates line up gridpoint for gridpoint.
    """
    # Positions measured from the domain's corner, as in postprocess_storm_ds
    x_from_corner = lite_ds["x"] - lite_ds["x"].min()
    y_from_corner = lite_ds["y"] - lite_ds["y"].min()
    return sc_wk_in_right_mover(x_from_corner, y_from_corner)


def calculate_cumulative_mass_flux(lite_level_ds, output_times, horizontal_mask=None):
    """Cumulative upward and downward air mass through a level, from lite output.

    Mechanism: at each 2-second lite output time, the mass through each gridpoint
    of the level is ``w * rho * dx^2 * dt``, with dt the 2-second lite interval.
    Gridpoints with w > 0 make up the upward flux. When the dataset holds
    several levels (a ``z`` dimension), each level is integrated separately. The downward flux is the total
    minus the upward part, so it is negative. Both are summed over the level and
    accumulated in time, then sampled at the 5-minute analysis times so they line
    up with the other upper-level quantities.

    By default the whole domain is used, without the ``in_analysis`` mask, so the
    supercell's flux includes both movers, matching the upper-level tracer totals.
    The right-mover sensitivity test passes a ``horizontal_mask`` instead, and only
    gridpoints where it is True are summed.

    Args:
        lite_level_ds: The output of ``read_upper_level_lite_dataset``.
        output_times: The 5-minute analysis times to sample the running totals at.
        horizontal_mask: Optional boolean DataArray on (y, x). None sums every
            gridpoint.

    Returns:
        A Dataset with ``cumu_upward_mass_flux`` and ``cumu_downward_mass_flux``
        in kg, on the analysis times, with a ``t_minutes`` coordinate.
    """
    # Mass through each gridpoint over one lite interval:
    #   m/s * kg/m^3 * m^2 * s = kg
    per_gridpoint_mass_flux = (
        lite_level_ds["WC"]
        * lite_level_ds["air_density"]
        * HORIZONTAL_GRID_SPACING_M**2
        * LITE_OUTPUT_INTERVAL_SECONDS
    )

    # Keep only the requested region, if one was given
    if horizontal_mask is not None:
        per_gridpoint_mass_flux = per_gridpoint_mass_flux.where(horizontal_mask, 0.0)

    # Split the level-summed flux by the sign of w
    is_upward = lite_level_ds["WC"] > 0
    upward_flux = per_gridpoint_mass_flux.where(is_upward).sum(["x", "y"])
    total_flux = per_gridpoint_mass_flux.sum(["x", "y"])
    downward_flux = total_flux - upward_flux

    # Accumulate in time, then sample at the analysis output times
    cumulative_upward = upward_flux.cumsum("time").sel({"time": output_times})
    cumulative_downward = downward_flux.cumsum("time").sel({"time": output_times})
    cumulative_upward.name = "cumu_upward_mass_flux"
    cumulative_downward.name = "cumu_downward_mass_flux"
    cumulative_flux_ds = xr.merge([cumulative_upward, cumulative_downward])

    # Minutes since the first analysis time, as calculate_derived_variables adds
    return cumulative_flux_ds.assign_coords(
        t_minutes=(
            cumulative_flux_ds["time"] - cumulative_flux_ds["time"].values[0]
        ).dt.total_seconds()
        // 60
    )


def calculate_upper_level_mass_flux(
    storm, upper_level_z_thresholds_m, output_times, right_mover_only=False
):
    """Read a storm's lite output and integrate its upper-level mass flux.

    This chains ``read_upper_level_lite_dataset`` and
    ``calculate_cumulative_mass_flux`` and computes the result. Pass one height
    for a single level, or a list of heights to integrate them all in one pass
    over the files. Expect it to take tens of minutes per storm, because it opens
    all 5401 lite files.

    ``right_mover_only`` restricts the supercell's flux to its right mover, for the
    sensitivity test. The line separating the movers is applied at the flux level
    itself, where the two updrafts are still separate.
    """
    if right_mover_only and storm != "sc_wk":
        raise ValueError("Only the supercell has a right mover")
    lite_level_ds = read_upper_level_lite_dataset(storm, upper_level_z_thresholds_m)
    horizontal_mask = (
        sc_wk_right_mover_mask_for_lite_dataset(lite_level_ds)
        if right_mover_only
        else None
    )
    return calculate_cumulative_mass_flux(
        lite_level_ds, output_times, horizontal_mask=horizontal_mask
    ).compute()


def upper_level_mass_flux_cache_path(storm, z_km, right_mover_only=False):
    """Cache file for one storm's upper-level mass flux at one height (in km).

    The right-mover-only flux, which only the supercell has, gets its own file.
    """
    region_suffix = "_right_mover" if right_mover_only else ""
    return (
        UPPER_LEVEL_DERIVED_DIRS[storm]
        / f"ul_mass_fluxes{region_suffix}_z-{z_km * 1000}.pkl"
    )


def read_or_compute_upper_level_mass_fluxes(
    storm, z_kms, output_times, right_mover_only=False
):
    """Load a storm's upper-level mass flux at several heights, computing any missing.

    Mechanism: each height has its own cache file, from
    ``upper_level_mass_flux_cache_path``. A cache is used when it exists and covers
    every output time. All heights without a usable cache are then computed
    together, in a single pass over the lite files, and each is written to its own
    file. Computing four heights therefore takes about as long as computing one.

    Each returned Dataset has a scalar ``z`` coordinate, the model level actually
    used. That is the same form a single-height computation produces, so caches
    written either way are interchangeable.

    Args:
        storm: Storm name.
        z_kms: Threshold heights, in km.
        output_times: The 5-minute analysis times to sample the running totals at.
        right_mover_only: Restrict the supercell's flux to its right mover. These
            results are cached in separate files.

    Returns:
        Mapping of height in km -> Dataset with ``cumu_upward_mass_flux`` and
        ``cumu_downward_mass_flux``.
    """
    expected_time_count = len(output_times)
    flux_dss_by_height = {}

    # Use every cache that exists and covers every output time
    for z_km in z_kms:
        cache_path = upper_level_mass_flux_cache_path(storm, z_km, right_mover_only)
        if not cache_path.exists():
            continue
        cached_flux_ds = pd.read_pickle(cache_path)
        if _time_count_of_result(cached_flux_ds) == expected_time_count:
            print(f"Read cached result from {cache_path}")
            flux_dss_by_height[z_km] = cached_flux_ds
        else:
            print(
                f"Stale cache at {cache_path}: it holds"
                f" {_time_count_of_result(cached_flux_ds)} times but the run has"
                f" {expected_time_count}. Recomputing and overwriting it."
            )

    # Compute every remaining height in one pass over the lite files
    heights_to_compute_km = [z_km for z_km in z_kms if z_km not in flux_dss_by_height]
    if heights_to_compute_km:
        print(
            f"Computing the {storm} upper-level mass flux at"
            f" {heights_to_compute_km} km from the lite output..."
        )
        all_heights_flux_ds = calculate_upper_level_mass_flux(
            storm,
            [z_km * 1000 for z_km in heights_to_compute_km],
            output_times,
            right_mover_only=right_mover_only,
        )

        # Split by height, and write each height to its own cache file
        for level_ix, z_km in enumerate(heights_to_compute_km):
            flux_ds = all_heights_flux_ds.isel({"z": level_ix})
            if _time_count_of_result(flux_ds) != expected_time_count:
                raise ValueError(
                    f"The {storm} flux at {z_km} km has"
                    f" {_time_count_of_result(flux_ds)} times but"
                    f" {expected_time_count} were expected. Not caching it."
                )
            cache_path = upper_level_mass_flux_cache_path(
                storm, z_km, right_mover_only
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pd.to_pickle(flux_ds, cache_path)
            print(f"Wrote result to {cache_path}")
            flux_dss_by_height[z_km] = flux_ds

    return {z_km: flux_dss_by_height[z_km] for z_km in z_kms}


################################################################################
############################## Figure data ##############################
################################################################################


def save_storm_data(ds, name, verbose=True):
    """Save the storm variables a figure uses to ``FIGURE_DATA_DIR/<name>.nc``."""
    if verbose:
        print(f"Saving storm data {name}...")
    ds[STORM_VARIABLES_USED].compute().to_netcdf(
        FIGURE_DATA_DIR.joinpath(name).with_suffix(".nc"), engine="h5netcdf"
    )


def save_figure_data(figure_data, data_name):
    """Pickle a figure's plotted data to ``FIGURE_DATA_DIR/<data_name>.pkl``."""
    with Path(FIGURE_DATA_DIR.joinpath(data_name).with_suffix(".pkl")).open(
        "wb"
    ) as f:
        pkl.dump(figure_data, f)


def read_figure_data(figure_name):
    """Read back the data that ``save_figure_data`` saved for a figure."""
    with Path(FIGURE_DATA_DIR.joinpath(figure_name).with_suffix(".pkl")).open(
        "rb"
    ) as f:
        figure_data = pkl.load(f)
    return figure_data


def plot_and_save(
    figure_data, figure_name, fn, *args, convert_xarray_to_numpy=True, **kwargs
):
    """Call a plotting function and record its positional arguments.

    Mechanism: the arguments are stored in ``figure_data[figure_name]`` before
    ``fn(*args, **kwargs)`` is called, so the data behind every plotted element
    can be saved alongside the figure. xarray arguments are converted to numpy
    arrays first, so the saved data does not depend on xarray.
    """
    # First save the data
    # Convert anything in xarray to numpy
    if convert_xarray_to_numpy:
        args = [
            (
                x.values
                if (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset))
                else x
            )
            for x in args
        ]
    figure_data[figure_name] = args
    return fn(*args, **kwargs)
