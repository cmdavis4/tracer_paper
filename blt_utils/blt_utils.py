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
import pint_xarray
from tqdm.notebook import tqdm
import xarray as xr
import pickle as pkl
from scipy import interpolate
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# User packages
import common as cm

########################################################################
############################## Parameters ##############################
########################################################################

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

# Matplotlib style to use
MPL_STYLE = "cd_bl_slides"

# Random seed
SEED = 325976123

# Base directories to which we'll save figures and from which to read derived data
BASE_FIGS_DIR = Path(f"/moonbow/cmdavis4/projects/bl_transport/figures/")
BASE_DERIVED_DIR = Path(f"/moonbow/cmdavis4/projects/bl_transport/rams_io/derived")
# Path to which to save figures for the "storyboard"
STORYBOARD_FIGS_DIR = BASE_FIGS_DIR.joinpath("storyboard")
TRACER_PAPER_FIGS_DIR = Path(
    f"/home/cmdavis4/projects/bl_transport/tracer_paper/figures"
)
TRAJECTORY_PAPER_FIGS_DIR = Path(f"/home/cmdavis4/projects/trajectory_paper/figures")
# Path to which we'll save entrainment figures
ENTRAINMENT_FIGS_DIR = Path(
    "/moonbow/cmdavis4/projects/bl_transport/figures/masters_thesis/entrainment_mechanisms"
)

# Condensate loading and vertical velocity thresholds for what constitutes in-updraft gridpoints
IN_UPDRAFT_R_CONDENSATE_THRESHOLD = 1e-4  # kg/kg
IN_UPDRAFT_W_THRESHOLD = 1  # m/s

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

# Start time of the simulations
SIMULATION_START_TIME = pd.Timestamp("1991-04-26 21:00:00")
SIMULATION_END_TIME = pd.Timestamp("1991-04-27 00:00:00")

# Progressive colormap to use where relevant
PROGRESSIVE_CMAP = "viridis"

# Color for trajectory arrows
TRAJECTORY_ARROW_COLOR = "grey"

# Background color for 3D figures/animations
TRAJECTORY_BACKGROUND_COLOR = "#f8f6f1"

# Variables to track in the scalar trajectories
TRAJECTORY_TRACKED_SCALARS = [
    "UC",
    "VC",
    "WC",
    "PCPRR",
    "PI",
    "RTP",
    "RV",
    "THETA",
    "RCP",
    "RRP",  # Liquid hydrometeor MRs
    "RPP",
    "RSP",
    "RAP",
    "RGP",
    "RHP",  # Ice hydrometeor MRs
]

TRAJECTORY_ANIMATION_SLICE_STEP = 5


########################################################################
############################## Filepaths ##############################
########################################################################

# Paths to 5-minute rams output
RAMS_OUTPUT_DIRS = {
    "ic_wk": Path(
        "/moonbow/cmdavis4/projects/bl_transport/rams_io/isolated_convection_wk/all_tracers/newtracer_forcing-warmbubble4_deltax-500_frqlite-2_nobulku/output"
    ),
    "sl_wk": Path(
        "/moonbow/cmdavis4/projects/bl_transport/rams_io/squall_line_wk/all_tracers/newtracer_blspinup-no_forcing-infinitecoldbubble4_deltax-500_frqlite-2/output"
    ),
    "sc_wk": Path(
        f"/moonbow/cmdavis4/projects/bl_transport/rams_io/supercell_wk/all_tracers/newtracer_forcing-grant2014_dx-500_frqlite-2_stratwinds/output"
    ),
}

# Paths to precalculated tracer vertical profile datasets
TRACER_TOTALS_PROFILE_FILEPATHS = {
    "ic_wk": BASE_DERIVED_DIR.joinpath("ic_wk").joinpath("tracer_totals_profile_ds.nc"),
    "sl_wk": BASE_DERIVED_DIR.joinpath("sl_wk").joinpath("tracer_totals_profile_ds.nc"),
    "sc_wk": BASE_DERIVED_DIR.joinpath("sc_wk").joinpath(
        f"tracer_totals_profile_ds_stratwinds.nc"
    ),
}

# Paths to storm development dataframes
DEVELOPMENT_DATAFRAME_FILEPATHS = {
    "ic_wk": BASE_DERIVED_DIR.joinpath("ic_wk").joinpath("development_dataframe.pkl"),
    "sl_wk": BASE_DERIVED_DIR.joinpath("sl_wk").joinpath("development_dataframe.pkl"),
    "sc_wk": BASE_DERIVED_DIR.joinpath("sc_wk").joinpath(
        f"development_dataframe_stratwinds.pkl"
    ),
}

# Paths to parcel trajectories
TRAJECTORY_PATHS = {
    "ic_wk": Path(
        "/moonbow/cmdavis4/projects/bl_transport/rams_io/isolated_convection_wk/all_tracers/newtracer_forcing-warmbubble4_deltax-500_frqlite-2_nobulku/trajectories/trajectory-allforwardnobulku_start-19910426210000_end-19910427000000_dt-2.nc"
    ),
    "sl_wk": Path(
        "/moonbow/cmdavis4/projects/bl_transport/rams_io/squall_line_wk/all_tracers/newtracer_blspinup-no_forcing-infinitecoldbubble4_deltax-500_frqlite-2/trajectories/trajectory-allforward_start-19910426210000_end-19910427000000_dt-2.nc"
    ),
    "sc_wk": Path(
        "/moonbow/cmdavis4/projects/bl_transport/rams_io/supercell_wk/all_tracers/newtracer_forcing-grant2014_dx-500_frqlite-2/trajectories/trajectory-allforward_start-19910426210000_end-19910427000000_dt-2.nc"
    ),
}

# Paths to parcel initialization location dicts
PI_DICT_PATHS = {
    "ic_wk": BASE_DERIVED_DIR.joinpath("ic_wk").joinpath(
        "parcel_initialization_dicts.pkl"
    ),
    "sl_wk": BASE_DERIVED_DIR.joinpath("sl_wk").joinpath(
        "parcel_initialization_dicts.pkl"
    ),
    "sc_wk": BASE_DERIVED_DIR.joinpath("sc_wk").joinpath(
        "parcel_initialization_dicts.pkl"
    ),
}

########################################################################################
############################## Derived variable functions ##############################
########################################################################################


def calculate_thermodynamic_variables(ds, fail_if_vars_missing=False):
    """
    Function to calculate basic derived thermodynamic/physical variables.
    """
    needed_vars = [
        "PI",
        "THETA",
        "RTP",
        "RV",
        "DN0",
        "RCP",
        "RRP",
        "RPP",
        "RSP",
        "RAP",
        "RGP",
        "RHP",
    ]

    # Write a little helper function
    # In hindsight I wish I had done this by writing a function that takes a function
    # that creates the new variable and then silently continuing if that function
    # throws an error, rather than all of these manual checks; a project for another day
    def vars_are_present(vars):
        return all([x in ds.data_vars for x in vars])

    if not vars_are_present(needed_vars) and fail_if_vars_missing:
        raise ValueError(
            "Not all variables needed for thermodynamic variable calculations"
            "are present in dataset and fail_if_vars_missing was passed as True; required"
            f"variables are {needed_vars}"
        )
    if vars_are_present(["PI", "THETA"]):
        ds["T"] = ds["PI"] * ds["THETA"] / 1004.0

    if vars_are_present(["RTP", "RV"]):
        ds["R_condensate"] = ds["RTP"] - ds["RV"]

    if vars_are_present(["PI"]):
        ds["P"] = 1000.0 * ((ds["PI"] / 1004.0) ** (1004.0 / 287.0))

    if vars_are_present(["P", "RV"]):
        vp = mpc.vapor_pressure(ds["P"] * units("hPa"), ds["RV"] * units("kg/kg"))
        ds["dewpoint"] = mpc.dewpoint(vp).pint.to("K").pint.dequantify()
        ds["vapor_pressure"] = vp.pint.to("hPa").pint.dequantify()

    if vars_are_present(["P", "RV"]):
        ds["theta_e"] = (
            mpc.equivalent_potential_temperature(
                pressure=ds["P"] * units("hPa"),
                temperature=ds["T"] * units("K"),
                dewpoint=ds["dewpoint"] * units("K"),
            )
            .pint.to("K")
            .pint.dequantify()
        )

    if vars_are_present(["T"]):
        ds["saturation_vapor_pressure"] = (
            mpc.saturation_vapor_pressure(ds["T"] * units("K"))
            .pint.to("hPa")
            .pint.dequantify()
        )

    if vars_are_present(["DN0"]):
        ds["air_mass"] = ds["DN0"] * 500**2 * ds["z"].diff(dim="z")

    if vars_are_present(["P", "T", "RV"]):
        ds["RH"] = mpc.relative_humidity_from_mixing_ratio(
            ds["P"] * units("hPa"), ds["T"] * units("K"), ds["RV"]
        ).pint.dequantify()
        ds["supersaturated"] = ds["RH"] >= 1

    if vars_are_present(["THETA", "RV", "R_condensate"]):
        ds["theta_rho"] = ds["THETA"] * (1 + 0.608 * ds["RV"] - ds["R_condensate"])

    if vars_are_present(["theta_rho"]) and "x" in ds.dims and "y" in ds.dims:
        tr_layer_mean = ds["theta_rho"].mean(["x", "y"])
        ds["buoyancy"] = (
            mpconstants.g * (ds["theta_rho"] - tr_layer_mean) / tr_layer_mean
        ).pint.dequantify()

    if vars_are_present(["RCP", "RRP"]):
        ds["R_liquid"] = ds["RCP"] + ds["RRP"]

    if vars_are_present(["RPP", "RSP", "RAP", "RGP", "RHP"]):
        ds["R_ice"] = ds["RPP"] + ds["RSP"] + ds["RAP"] + ds["RGP"] + ds["RHP"]

    return ds


def with_t_minutes_coord(ds):
    return ds.assign_coords(
        t_minutes=(ds["time"] - SIMULATION_START_TIME.to_numpy()).dt.total_seconds()
        // 60
    )


def calculate_derived_variables(storm_ds, storm_name):
    """
    Function to calculate derived thermodynamic variables as well as do basic preprocessing on the rams output
    """
    storm_ds = calculate_thermodynamic_variables(storm_ds)

    # Shift the x and y coords so that they start from 0
    storm_ds["x"] = storm_ds["x"] - min(storm_ds["x"])
    storm_ds["y"] = storm_ds["y"] - min(storm_ds["y"])

    # Add an in-updraft flag
    storm_ds["in_updraft"] = (
        storm_ds["R_condensate"] >= IN_UPDRAFT_R_CONDENSATE_THRESHOLD
    ) & (storm_ds["WC"] >= IN_UPDRAFT_W_THRESHOLD)

    if storm_name == "sc_wk":
        storm_ds["in_right_mover"] = sc_wk_in_right_mover(storm_ds["x"], storm_ds["y"])

    # Add a time in minutes coordinate
    storm_ds = with_t_minutes_coord(storm_ds)

    # Calculate vertical vorticity
    storm_ds["vertical_vorticity"] = storm_ds["VC"].differentiate("x") - storm_ds[
        "UC"
    ].differentiate("y")

    # Calculate horizontal divergence
    storm_ds["divergence"] = storm_ds["UC"].differentiate("x") + storm_ds[
        "VC"
    ].differentiate("y")

    # Add base state-relative versions of some variables
    for var in ["UC", "VC", "THETA"]:
        storm_ds[var + "_bsr"] = storm_ds[var] - storm_ds.isel({"time": 0})[var].mean(
            ["x", "y"]
        )
    # Add theta deficit
    storm_ds["theta_deficit"] = -storm_ds["THETA_bsr"]

    # Add a flag for whether a gridpoint should be included in analysis
    storm_ds["in_analysis"] = ANALYSIS_FLAG_FUNCTIONS[storm_name](storm_ds)

    return storm_ds


######################################################################################
############################## Datapoint flag functions ##############################
######################################################################################


SC_WK_RIGHT_MOVER_BOUNDARY_Y_INTERCEPT = 78000
SC_WK_RIGHT_MOVER_BOUNDARY_SLOPE = 10000 / 50000


# Define if a supercell point is in the right mover
def sc_wk_right_mover_boundary(x):
    return SC_WK_RIGHT_MOVER_BOUNDARY_Y_INTERCEPT + x * SC_WK_RIGHT_MOVER_BOUNDARY_SLOPE


def sc_wk_in_right_mover(x, y):
    return y <= sc_wk_right_mover_boundary(x)


# Basic check on RAMS output data quality that raises an error if it fails
def check_storm_ds(ds, lite=False):
    if not lite:
        # Only tracer not being used should be 6, which was for 80 mm/hr of rain
        assert [
            x
            for x in ds.data_vars
            if x.startswith("TRACERP") and x not in RS_TRACERS and x not in FS_TRACERS
        ] == ["TRACERP006"]


# Function for excluding left mover parcels in the supercell dataset
def sc_parcel_exclusion_flag(trajectory_ds):
    # Handle left mover supercell trajectories
    # Get parcels' location when they become entrained
    # Limit to lofted
    lofted_trajectory_ds = trajectory_ds.sel({"parcel_ix": trajectory_ds["lofted"]})
    # Find when they became entrained and make it a dataframe
    crossing_times = (
        parcels_are_currently_entrained(lofted_trajectory_ds)
        .idxmax("time")
        .to_dataframe()
    )

    crossing_locations = {}
    for parcel_ix, row in crossing_times.iterrows():
        crossing_locations[parcel_ix] = lofted_trajectory_ds.sel(
            {"parcel_ix": parcel_ix, "time": row["time"]}
        )[["x", "y", "z"]].to_pandas()

    crossing_locations = pd.DataFrame.from_dict(crossing_locations, orient="index")
    crossing_locations["in_left_mover"] = ~sc_wk_in_right_mover(
        crossing_locations["x"], crossing_locations["y"]
    )
    print("Left mover value counts:")
    print(crossing_locations["in_left_mover"].value_counts())

    # Set lofted to false for the LM ones

    lm_idxs = crossing_locations[crossing_locations["in_left_mover"]].index.values

    return xr.DataArray(
        trajectory_ds["parcel_ix"].isin(lm_idxs),
        coords={"parcel_ix": trajectory_ds["parcel_ix"]},
    )


# Define our entrainment criterion
# Don't apply the .any('time') yet so we can use this for defining parcels entrained into the left mover
def parcels_are_currently_entrained(parcel_ds):
    return parcel_ds["z"] >= 6000


# Function that returns a flag for whether a gridpoint should be included in analysis
ANALYSIS_FLAG_FUNCTIONS = {
    "ic_wk": lambda ds: xr.DataArray(
        np.full((len(ds["x"]), len(ds["y"])), True), coords={"x": ds["x"], "y": ds["y"]}
    ),
    "sl_wk": lambda ds: xr.DataArray(
        np.full((len(ds["x"]), len(ds["y"])), True), coords={"x": ds["x"], "y": ds["y"]}
    ),
    "sc_wk": lambda ds: ds["in_right_mover"],
}
# Function that returns a flag for whether a trajectory should be included in analysis
TRAJECTORY_FLAG_FUNCTIONS = {
    "ic_wk": lambda ds: xr.DataArray(
        np.full(len(ds["parcel_ix"]), True), coords={"parcel_ix": ds["parcel_ix"]}
    ),
    "sl_wk": lambda ds: xr.DataArray(
        np.full(len(ds["parcel_ix"]), True), coords={"parcel_ix": ds["parcel_ix"]}
    ),
    "sc_wk": lambda ds: ~sc_parcel_exclusion_flag(ds),
}


################################################################################
############################## Plotting functions ##############################
################################################################################


# Function to label time on the x axis with consistent tick mark values
def label_x_axis(ax, short=False):
    ax.xaxis.set_ticks(np.arange(0, 181, 30))
    ax.set_xlabel(f"{'Sim.' if short else 'Simulation'} time (minutes)")


# Make figure panel titles with (a), (b), (c), etc
def axes_title(letter_ix, title):
    return f"({chr(ord('a') + letter_ix)}) {title}"


# Make one of the entrainment figures
def entrainment_figure(storm_name, figure_times, **subplots_adjust_kwargs):

    scale = 1.2
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(7 * scale, 3.5 * scale))
    base_subplots_kwargs = {}
    # base_subplots_kwargs = dict(wspace=-0.1, hspace=0.2)
    base_subplots_kwargs.update(subplots_adjust_kwargs)
    if base_subplots_kwargs:
        fig.subplots_adjust(**base_subplots_kwargs)

    for row, fs_rs in enumerate(["fs", "rs"]):
        for col in range(3):
            try:
                axs[row, col].imshow(
                    mpimg.imread(
                        ENTRAINMENT_FIGS_DIR.joinpath(
                            f"{storm_name}_{fs_rs}_entrainment_mechanism_t{col+1}.png"
                        )
                    )
                )
            except FileNotFoundError:
                pass
            axs[row, col].set_axis_off()
            axs[row, col].set_title(
                axes_title(
                    row * 3 + col,
                    f"t={simulation_time_to_minutes(figure_times[fs_rs][col]):.0f} minutes",
                )
            )
        cm.plotting.add_row_header(
            axs[row, 0], {"fs": "Fixed source", "rs": "Rain-sourced"}[fs_rs], pad=-10
        )

    return fig


def entrainment_figure_subfigs(storm_name, figure_times, **subplots_adjust_kwargs):

    outer_fig = plt.figure(figsize=(6, 11))
    subfigs = outer_fig.subfigures(ncols=2)
    # fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(6, 11))
    # subfigs =
    # fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(11, 7))
    # base_subplots_kwargs = dict(wspace=-0.1, hspace=0.2)
    # base_subplots_kwargs.update(subplots_adjust_kwargs)
    # fig.subplots_adjust(**base_subplots_kwargs)

    for col, fs_rs in enumerate(["fs", "rs"]):
        axs = subfigs[col].subplots(nrows=3)
        axs = axs.flatten()
        for row in range(3):
            try:
                axs[row].imshow(
                    mpimg.imread(
                        ENTRAINMENT_FIGS_DIR.joinpath(
                            f"{storm_name}_{fs_rs}_entrainment_mechanism_t{row+1}.png"
                        )
                    )
                )
            except FileNotFoundError:
                pass
            axs[row].set_axis_off()
            axs[row].set_title(
                axes_title(
                    row * 2 + col,
                    f"t={simulation_time_to_minutes(figure_times[fs_rs][row]):.0f} minutes",
                )
            )
        # cm.plotting.add_row_header(axs[row, 0], {"fs": "Fixed source", "rs": "Rain-sourced"}[fs_rs], pad=-20)

    return outer_fig


################################################################################
############################## Utilities ##############################
################################################################################


# Apply xarray.where in a much faster way, for some reason
def filter_parcel_ixs(ds, condition):
    if callable(condition):
        condition = condition(ds)
    # Otherwise assume its a DataArray of bools with only a parcel_ix dimension
    return ds["parcel_ix"].values[np.nonzero(condition.values)[0]]


# Convert a simulation time (i.e. Timestamp or datetime) to minutes from the simulation start
def to_minutes(simulation_time):
    return (simulation_time - SIMULATION_START_TIME) / pd.Timedelta(minutes=1)


# Alias for backwards compatibility
simulation_time_to_minutes = to_minutes


# Reverse of the above
def to_sim_time(minutes):
    return (SIMULATION_START_TIME + pd.Timedelta(minutes=minutes)).to_numpy()


# Alias for backward compatibility
minutes_to_simulation_time = to_sim_time


# Parcel initialization helper function for taking the Cartesian product
def parcel_init_cartesian_product(
    parcel_x_ixs, parcel_y_ixs, parcel_z_ixs, parcel_ts, group
):
    x_ixs, y_ixs, z_ixs, ts = np.meshgrid(
        parcel_x_ixs, parcel_y_ixs, parcel_z_ixs, parcel_ts
    )
    x_ixs = x_ixs.ravel()
    y_ixs = y_ixs.ravel()
    z_ixs = z_ixs.ravel()
    ts = ts.ravel()
    return [
        {
            "x_ix": x_ixs[i],
            "y_ix": y_ixs[i],
            "z_ix": z_ixs[i],
            "initialization_time": ts[i],
            "group": group,
        }
        for i in range(len(x_ixs))
    ]


# Randomly choose parcel indexes, handling the case in which the sample size is larger than the number of parcels
def maybe_random_choice(parcel_ixs, size):
    if size >= len(parcel_ixs):
        return parcel_ixs
    else:
        return np.random.default_rng(seed=SEED).choice(
            parcel_ixs, size=size, replace=False
        )


#######################################################################################
############################## Input data postprocessing ##############################
#######################################################################################


# Postprocess the 5-minute RAMS output
def postprocess_storm_ds(ds, storm_name, lite=False, check=True):
    # Run basic checks
    if check:
        check_storm_ds(ds, lite=lite)
    # Calculate derived variables and analysis flag
    ds = calculate_derived_variables(ds, storm_name)
    # Calculate total fixed-source tracer variables
    if not lite:
        ds["total_fs_tracer"] = 0
        ds["ACCtotal_fs_tracer"] = 0
        for tracer_var in FS_TRACERS:
            ds["total_fs_tracer"] = ds["total_fs_tracer"] + ds[tracer_var]
            ds["ACCtotal_fs_tracer"] = ds["ACCtotal_fs_tracer"] + ds["ACC" + tracer_var]
        # Join on storm centroids
        development_df = pd.read_pickle(DEVELOPMENT_DATAFRAME_FILEPATHS[storm_name])
        for coord in ["x", "y"]:
            ds[f"storm_position_{coord}"] = xr.DataArray(
                development_df[f"storm_centroid_{coord}"].values,
                coords={
                    "time": [
                        minutes_to_simulation_time(x)
                        for x in development_df["t_minutes"].values
                    ]
                },
            )
    return ds


# Postprocess the precalculated tracer profile datasets
def calculate_tracer_profile_ds(storm_ds):
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
        this_time_ss_ds = this_time_ss_ds.rename(
            {
                var: "ss_" + var
                for var in this_time_ss_ds.data_vars
                if var != "supersaturated"
            }
        )
        tracer_totals_list.append(this_time_ss_ds.compute())

        # Now sum up the accumulated (2d) tracers
        # Since we're now using DN0, which does not change in time, for calculating the air mass in each gridpoint,
        # we can just pull out the current value of the air mass (which also won't change in time) for converting
        # the accumulated mixing ratio into a number
        acc_nc = (
            this_time_ds[all_acc_tracer_vars] * this_time_ds.isel({"z": 1})["air_mass"]
        ).sum(["x", "y"])
        acc_nc = acc_nc.rename({k: k + "_nc" for k in acc_nc.data_vars})
        tracer_totals_list.append(acc_nc.compute())

    # Make it into a dataset
    print("Combining...")
    tracer_totals_profile_ds = xr.combine_by_coords(
        [x.expand_dims("time") for x in tracer_totals_list]
    )
    return tracer_totals_profile_ds


# Postprocess the dataframe containing the information on each storm's development
def calculate_development_dataframe(storm_ds):
    # Filter to in-analysis
    storm_ds = storm_ds.where(storm_ds["in_analysis"])

    max_updrafts = (
        storm_ds.sel({"z": 5000}, method="nearest")["WC"]
        .rolling({"x": 4, "y": 4})
        .min()
        .max(["x", "y"])
        .rolling({"time": 3})
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

    total_rain = storm_ds["PCPRR"].sum(["x", "y"]).compute()

    rain_by_threshold = {
        f"rain_threshold{ix}": (storm_ds["PCPRR"] >= threshold).sum(["x", "y"])
        for ix, threshold in enumerate(RS_TRACER_THRESHOLD_RAIN_RATES)
    }

    development_dict = {
        "t_minutes": storm_ds["t_minutes"],
        "max_updraft": max_updrafts,
        "updraft_mass": updraft_mass,
        "total_rain": total_rain,
        "storm_centroid_x": storm_centroids[0],
        "storm_centroid_y": storm_centroids[1],
    }
    development_dict.update(rain_by_threshold)

    development_df = pd.DataFrame(development_dict)
    return development_df


# Postprocess the trajectory output
def postprocess_trajectories_ds(
    trajectory_ds, storm_name, thermodynamic_variables=True, entrainment_variables=True
):
    # TODO: Refactor the time sorting and t_minutes logic into the actual
    # trajectory calculations, so that we don't have to call this function
    # if the thermodynamic and entrainment variables aren't needed

    # Sort by time, for backtrajectories
    trajectory_ds = trajectory_ds.sortby("time")
    # Calculate thermodynamic variables
    if thermodynamic_variables:
        trajectory_ds = calculate_thermodynamic_variables(trajectory_ds)

    # Make a t_minutes coordinate like we do with the RAMS output
    trajectory_ds = with_t_minutes_coord(trajectory_ds)
    trajectory_ds = trajectory_ds.assign_coords(
        t_minutes=trajectory_ds["t_minutes"],
    )

    if entrainment_variables:
        # Create a flag for whether a parcel originated in a rainy region
        trajectory_ds["rs"] = (
            trajectory_ds.sel({"time": trajectory_ds["initialization_time"]})["PCPRR"]
            >= RS_TRACER_THRESHOLD_RAIN_RATES[0]
        )
        trajectory_ds = trajectory_ds.assign_coords(rs=trajectory_ds["rs"])

        # Calculate whether it's entrained
        trajectory_ds["lofted"] = parcels_are_currently_entrained(trajectory_ds).any(
            "time"
        )

        # Create flag for whether to include a parcel in analysis
        trajectory_ds["in_analysis"] = xr.DataArray(
            TRAJECTORY_FLAG_FUNCTIONS[storm_name](trajectory_ds),
            coords={"parcel_ix": trajectory_ds["parcel_ix"]},
        )
        # Also want to use this to turn off the lofted flag where parcels should be excluded
        trajectory_ds["lofted"] = trajectory_ds["lofted"] & trajectory_ds["in_analysis"]

        # Add the first time at which the parcel is entrained
        entrainment_bools = parcels_are_currently_entrained(trajectory_ds)
        trajectory_ds["time_of_entrainment"] = entrainment_bools.idxmax("time").where(
            entrainment_bools.any("time")
        )

    return trajectory_ds
