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
STORM_NAMES = ["ic_wk", "sl_wk", "sc_wk"]
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

# Matplotlib style to use; this is the format for passing matplotlib a filepath
# rather than a name in your stylelib
MPL_STYLE = f"file://{Path(__file__).parent.joinpath('cd_bl_slides.mplstyle')}"

# Random seed
SEED = 325976123

# # Path to this file
this_dir = Path(__file__).parent
# Set data and figure relative paths
FIGS_DIR = this_dir.joinpath("figures")
DATA_DIR = this_dir.joinpath("data")
DERIVED_DIR = DATA_DIR.joinpath("derived")
FIGURE_DATA_DIR = DATA_DIR.joinpath("figure_data")


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

# Progressive colormap to use where relevant
PROGRESSIVE_CMAP = "viridis"


########################################################################
############################## Filepaths ##############################
########################################################################

# # Paths to 5-minute rams output
# RAMS_OUTPUT_DIRS = {storm: DATA_DIR.joinpath(storm) for storm in STORM_NAMES}

# Paths to precalculated tracer vertical profile datasets
TRACER_TOTALS_PROFILE_FILEPATHS = {
    storm: DERIVED_DIR.joinpath(storm).joinpath("tracer_totals_profile_ds.nc")
    for storm in STORM_NAMES
}

# Paths to storm development dataframes
DEVELOPMENT_DATAFRAME_FILEPATHS = {
    storm: DERIVED_DIR.joinpath(storm).joinpath("development_dataframe.pkl")
    for storm in STORM_NAMES
}

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
# TRACER_TOTALS_PROFILE_FILEPATHS = {
#     "ic_wk": BASE_DERIVED_DIR.joinpath("ic_wk").joinpath("tracer_totals_profile_ds.nc"),
#     "sl_wk": BASE_DERIVED_DIR.joinpath("sl_wk").joinpath("tracer_totals_profile_ds.nc"),
#     "sc_wk": BASE_DERIVED_DIR.joinpath("sc_wk").joinpath(
#         f"tracer_totals_profile_ds_stratwinds.nc"
#     ),
# }

# Paths to storm development dataframes
# DEVELOPMENT_DATAFRAME_FILEPATHS = {
#     "ic_wk": BASE_DERIVED_DIR.joinpath("ic_wk").joinpath("development_dataframe.pkl"),
#     "sl_wk": BASE_DERIVED_DIR.joinpath("sl_wk").joinpath("development_dataframe.pkl"),
#     "sc_wk": BASE_DERIVED_DIR.joinpath("sc_wk").joinpath(
#         f"development_dataframe_stratwinds.pkl"
#     ),
# }

########################################################################################
############################## Derived variable functions ##############################
########################################################################################


def calculate_thermodynamic_variables(ds):
    """
    Function to calculate basic derived thermodynamic/physical variables.
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
    # Parcels don't have DN0 at present; rather than make a separate function we'll just calculate it if we can
    if "DN0" in ds.data_vars:
        ds["air_mass"] = ds["DN0"] * 500**2 * ds["z"].diff(dim="z")
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
    storm_ds = storm_ds.assign_coords(
        t_minutes=(storm_ds["time"] - storm_ds["time"].values[0]).dt.total_seconds()
        // 60
    )

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


# Define if a supercell point is in the right mover
def sc_wk_right_mover_boundary(x):
    return 78000 + x * (10000 / 50000)


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


################################################################################
############################## Plotting functions ##############################
################################################################################


# Function to label time on the x axis with consistent tick mark values
def label_x_axis(ax):
    ax.xaxis.set_ticks(np.arange(0, 181, 30))
    ax.set_xlabel("Simulation time (minutes)")


# Make figure panel titles with (a), (b), (c), etc
def axes_title(letter_ix, title):
    return f"({chr(ord('a') + letter_ix)}) {title}"


################################################################################
############################## Utilities ##############################
################################################################################


# Convert a simulation time (i.e. Timestamp or datetime) to minutes from the simulation start
def simulation_time_to_minutes(simulation_time):
    return (simulation_time - SIMULATION_START_TIME) / pd.Timedelta(minutes=1)


# Reverse of the above
def minutes_to_simulation_time(minutes):
    return (SIMULATION_START_TIME + pd.Timedelta(minutes=minutes)).to_numpy()


#######################################################################################
############################## Input data postprocessing ##############################
#######################################################################################


# Postprocess the 5-minute RAMS output
def postprocess_storm_ds(ds, storm_name, lite=False):
    # Run basic checks
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


def save_storm_data(ds, name, verbose=True):
    if verbose:
        print(f"Saving storm data {name}...")
    ds[STORM_VARIABLES_USED].compute().to_netcdf(
        FIGURE_DATA_DIR.joinpath(name).with_suffix(".nc"), engine="h5netcdf"
    )


def save_figure_data(figure_data, data_name):
    with Path(FIGURE_DATA_DIR.joinpath(data_name).with_suffix(".pkl")).open("wb") as f:
        pkl.dump(figure_data, f)


def read_figure_data(figure_name):
    with Path(FIGURE_DATA_DIR.joinpath(figure_name).with_suffix(".pkl")).open(
        "rb"
    ) as f:
        figure_data = pkl.load(f)
    return figure_data


def plot_and_save(
    figure_data, figure_name, fn, *args, convert_xarray_to_numpy=True, **kwargs
):
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
