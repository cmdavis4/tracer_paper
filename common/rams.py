import re
from pathlib import Path
import datetime as dt

import xarray as xr
from tqdm.notebook import tqdm
import subprocess
import numpy as np
import matplotlib.pyplot as plt
import pint_xarray
from pint import UnitRegistry
import pandas as pd
import metpy.calc as mpc
from metpy.units import units
import metpy.constants as mpconstants

from .utils import current_dt_str

# Load our custom units
from pint import UnitRegistry

# First we create the registry.
ureg = UnitRegistry()
# Then we append the new definitions
ureg.load_definitions(str(Path(__file__).parent.joinpath("rams_pint_units.txt")))
pint_xarray.setup_registry(ureg)

# strftime format string that produces dates in the same format as in RAMS output filenames; miscellaneously useful
RAMS_DT_STRFTIME_STR = r"%Y-%m-%d-%H%M%S"

# Define a regex to pull out the datetime from RAMS filenames
RAMS_FILENAME_DATETIME_REGEX = r"[0-9]{4}\-[0-9]{2}\-[0-9]{2}\-[0-9]{6}"

# Mapping from the phony dim number assigned to each dimension to what it actually is. This appears to be programmatic,
# at least for full output files, but is not for lite files.
RAMS_ANALYSIS_FILE_DIMENSIONS_DICT = {
    "phony_dim_0": "y",
    "phony_dim_1": "x",
    "phony_dim_2": "z",
    "phony_dim_3": "p",
    "phony_dim_4": "kppz",
    "phony_dim_5": "zs",
    "phony_dim_6": "zg",
}

# Templates for the commands that are issued to the subprocesses that actually run rams
RAMS_SERIAL_COMMAND_TEMPLATE = "{rams_executable_path} -f {ramsin_path}"
RAMS_MPIEXEC_COMMAND_TEMPLATE = (
    "/home/C837213679/software/mpich-3.3.2_new/bin/mpiexec -machinefile {machsfile_path} -np "
    "{n_cores} {rams_executable_path} -f {ramsin_path}"
)

# Mapping from the names of dimensions as they are given in header files to the conventional names
HEADER_NAME_DIMENSION_DICT = {
    "__ztn{grid_number}": "z",
    "__ytn{grid_number}": "y",
    "__xtn{grid_number}": "x",
}

# List of variables that are considered part of the initial sounding for a RAMS run
SOUNDING_NAMELIST_VARIABLES = ["PS", "TS", "RTS", "US", "VS"]


# Read in RAMS output variable names, units, dimensions, descriptions
RAMS_VARIABLES_DF = pd.read_csv(Path(__file__).parent.joinpath("rams_variables.csv"))

# Need to do a little cleaning
RAMS_VARIABLES_DF["units"] = RAMS_VARIABLES_DF["units"].str.replace("#", "1")
RAMS_VARIABLES_DF = RAMS_VARIABLES_DF.drop(RAMS_VARIABLES_DF.columns[0], axis=1)


# Human-readable names for each hydrometeor species
HYDROMETEOR_SPECIES_FULL_NAMES = {
    "PP": "pristine ice",
    "SP": "snow",
    "AP": "aggregates",
    "HP": "hail",
    "GP": "graupel",
    "CP": "cloud",
    "DP": "drizzle",
    "RP": "rain",
}


def generate_ramsin(
    ramsin_name,
    parameters,
    rams_input_dir,
    rams_output_dir,
    ramsin_dir,
    ramsin_template_path,
    mkdirs=False,
):
    """Generates a RAMSIN file, given a template RAMSIN and a set of parameters to change relative to the template.

    Args:
        ramsin_name (str): Name of this RAMSIN; this will be used in the filename, which will be `RAMSIN.{ramsin_name}`
        parameters (dict): Dict of parameters to change relative to those contained in the template RAMSIN. Keys should
            be strs corresponding to the names of the parameters, and values should be strs, to avoid any ambiguity
            with how they are written. Values will be written exactly as given, meaning that any quotes must be included
            within the str.
        rams_input_dir (str or pathlib.Path): Directory from which to read input files in the RAMS run; this sets the
            prefix of the TOPFILES, SFCFILES, SSTFPFX, and NDVIFPFX parameters in the RAMSIN. This behavior can be
            overridden on a per-parameter basis by passing any of the four aforementioned parameters explicitly in
            `parameters`.
        rams_output_dir (str or pathlib.Path): Directory to which to write the output of the RAMS run; this sets the
            prefix of the AFILEPREF parameter in the RAMSIN. This behavior can be overridden by passing AFILEPREF
            explicitly in `parameters`.
        ramsin_dir (str or pathlib.Path): Directory to which the RAMSIN will be written.
        ramsin_template_path (str or pathlib.Path): Path to a template RAMSIN file. The generated RAMSIN will be exactly
            the contents of the template RAMSIN, with only the values of the parameters given in `parameters` changed.

    Returns:
        str: Text of the generated RAMSIN
    """
    # Make a copy of parameters since we're going to modify it
    parameters = {k: v for k, v in parameters.items()}

    # First make the 4 paths into pathlib.Paths if they're not already
    rams_input_dir = (
        Path(rams_input_dir) if rams_input_dir is not None else rams_input_dir
    )
    rams_output_dir = (
        Path(rams_output_dir) if rams_output_dir is not None else rams_output_dir
    )
    ramsin_dir = Path(ramsin_dir)
    ramsin_template_path = Path(ramsin_template_path)

    # Make these directories if need be
    if mkdirs:
        for this_dir in [rams_input_dir, rams_output_dir, ramsin_dir]:
            if this_dir is not None:
                this_dir.mkdir(exist_ok=True, parents=False)

    # First replace the IO paths
    with ramsin_template_path.open("r") as f:
        ramsin = f.read()

    # Add the input and output directories to the parameter dict so that we can replace them like
    # normal parameters
    input_dir_sub_suffixes = {
        "TOPFILES": "toph",
        "SFCFILES": "sfch",
        "SSTFPFX": "ssth",
        "NDVIFPFX": "ndh",
    }
    output_dir_sub_suffixes = {"AFILEPREF": "a"}

    # Replace these in the RAMSIN, if they're not explicitly being passed in the parameters
    for param_name, suffix in input_dir_sub_suffixes.items():
        if param_name not in parameters and rams_input_dir is not None:
            parameters[param_name] = f"'{str(rams_input_dir.joinpath(suffix))}'"
    for param_name, suffix in output_dir_sub_suffixes.items():
        if param_name not in parameters and rams_output_dir is not None:
            parameters[param_name] = f"'{str(rams_output_dir.joinpath(suffix))}'"

    # Now replace the parameters given in the RAMSIN with the rendered paths
    # This isn't super efficient but it doesn't matter for a text file of this size
    for parameter_name, parameter_value in parameters.items():
        parameter_regex = r"(^\s*{}\s*\=\s*).*?(\n[^\n\!]*[\=\$])".format(
            parameter_name
        )
        replacement_regex = r"\g<1>{},\g<2>".format(parameter_value)
        ramsin, n_subs = re.subn(
            parameter_regex,
            replacement_regex,
            ramsin,
            count=1,
            flags=re.MULTILINE | re.DOTALL,
        )
        if n_subs == 0:
            raise ValueError(
                "Field {} not found in template RAMSIN".format(parameter_name)
            )

    # Write the rendered RAMSIN to disk
    with ramsin_dir.joinpath("RAMSIN.{}".format(ramsin_name)).open("w") as f:
        f.write(ramsin)

    # Return the rendered RAMSIN in case we wanna look at it
    return ramsin


def run_rams_for_ramsin(
    ramsin_path,
    stdout_path,
    rams_executable_path,
    machsfile_path=None,
    log_command=True,
    log_ramsin=True,
    dry_run=False,
    asynchronous=True,
    verbose=True,
):
    # First check if the ramsin path is more than 256 characters, which RAMS can't handle
    if len(str(Path(ramsin_path).resolve())) > 256:
        raise ValueError("RAMS cannot handle ramsin paths longer than 256 characters")
    # Convert the rams executable path to an absolute path
    rams_executable_path = str(Path(rams_executable_path).resolve())
    if not machsfile_path:
        # Running serially
        command = RAMS_SERIAL_COMMAND_TEMPLATE.format(
            rams_executable_path=rams_executable_path,
            ramsin_path=str(Path(ramsin_path).resolve()),
        )
    else:
        # Running in parallel, so need mpiexec
        # Total up the number of cores from the nodelist
        with Path(machsfile_path).open("r") as f:
            nodelist = f.readlines()
        n_cores = sum([int(s.split(":")[1]) for s in nodelist])
        # Write the nodelist to a machs file

        nodes_str = ",".join(nodelist)
        command = RAMS_MPIEXEC_COMMAND_TEMPLATE.format(
            machsfile_path=str(Path(machsfile_path).resolve()),
            n_cores=n_cores,
            rams_executable_path=rams_executable_path,
            ramsin_path=str(Path(ramsin_path).resolve()),
        )

    write_mode = "w"

    if log_command:
        import hashlib

        with Path(rams_executable_path).open("rb") as rams_exe_f:
            rams_checksum = hashlib.md5(rams_exe_f.read()).hexdigest()
        with Path(stdout_path).open(write_mode) as stdout_f:
            hashes = "#" * 47
            stdout_f.write(f"{hashes}\nRAMS CHECKSUM: {rams_checksum}\n{hashes}\n\n")
            stdout_f.write(
                "##############################\n         BEGIN COMMAND\n##############################\n"
            )
            stdout_f.write(f"{command} > {stdout_path}")
            stdout_f.write(
                "\n##############################\n         END COMMAND\n##############################\n\n"
            )
        write_mode = "a"

    if log_ramsin:
        # If we're logging the whole ramsin, we need to open the file in write mode first and write the ramsin,
        # then open it in append mode and write the stdout from rams that way
        with Path(stdout_path).open(write_mode) as stdout_f:
            with Path(ramsin_path).open("r") as ramsin_f:
                stdout_f.write(
                    "##############################\n         BEGIN RAMSIN\n##############################\n"
                )
                stdout_f.write(ramsin_f.read())
                stdout_f.write(
                    "\n##############################\n         END RAMSIN\n##############################\n\n"
                )
        write_mode = "a"
    # Print the command we'll run, plus a pipe indicating where the stdout will be written (in reality this is
    # handled by the `stdout` argument to `subprocess.run` below, but this is convenient)
    if verbose:
        print(f"{command} > {stdout_path}")
    if dry_run:
        completed = True
    else:
        with Path(stdout_path).open(write_mode) as stdout_f:
            if asynchronous:
                completed = subprocess.Popen(
                    command.split(" "), stdout=stdout_f, start_new_session=True
                )

            else:
                completed = subprocess.run(
                    command.split(" "),
                    stdout=stdout_f,
                )

    return completed


def run_rams(
    parameter_sets_dict,
    run_dir,
    rams_executable_path,
    ramsin_template_path,
    nodelist=None,
    log_command=True,
    log_ramsin=True,
    dry_run=False,
    parallel=True,
    block=True,
    date_filenames=False,
    verbose=True,
):
    # Make the run_dir a Path if it's not already
    run_dir = Path(run_dir)

    # Make directories for the IO and ramsins
    # General file structure this creates will be:
    # - {run_name}/
    #   - {run_name}_machsfile.master (if one nodelist passed)
    #   - {parameter_set_name}/
    #     - input
    #     - output
    #     - RAMSIN.{parameter_set_name}
    #     - {parameter_set_name}.stdout
    run_dir.mkdir(parents=False, exist_ok=True)

    # Create the date suffix if we're using it
    fname_suffix = ("_dt-" + current_dt_str()) if date_filenames else ""
    # Create the file structure for each parameter set
    parameter_set_dirs = {}
    for parameter_set_name, parameters in parameter_sets_dict.items():
        this_parameter_set_dir = run_dir.joinpath(parameter_set_name + fname_suffix)
        this_input_dir = this_parameter_set_dir.joinpath("input")
        this_output_dir = this_parameter_set_dir.joinpath("output")
        # this_parameter_set_dir.mkdir(parents=False, exist_ok=True)
        # this_input_dir.mkdir(parents=False, exist_ok=True)
        # this_output_dir.mkdir(parents=False, exist_ok=True)
        parameter_set_dirs[parameter_set_name] = {"top": this_parameter_set_dir}

        generate_ramsin(
            parameter_set_name + fname_suffix,
            parameters,
            rams_input_dir=this_input_dir,
            rams_output_dir=this_output_dir,
            ramsin_dir=this_parameter_set_dir,
            ramsin_template_path=ramsin_template_path,
            mkdirs=True,
        )

    # Create machsfile(s) if we were passed a nodelist/nodelists
    if nodelist:
        # nodelist can be a single list/array, or a dictionary of list/arrays with one for each parameter_set
        if isinstance(nodelist, dict):
            # Check that the keys in the nodelist dict are the same as those of the parameter sets dict
            if not nodelist.keys() == parameter_sets_dict.keys():
                raise ValueError(
                    "If nodelist is a dict, its keys must match those of parameter_sets_dict exactly"
                )
        else:
            # If it's not actually a dict, make one with the same value for all parameter sets
            nodelist = {
                parameter_set_name: nodelist
                for parameter_set_name in parameter_set_dirs.keys()
            }
        for parameter_set_name, parameter_set_dir in parameter_set_dirs.items():
            # Figure out the machsfile path for this parameter set
            this_machsfile_path = parameter_set_dir["top"].joinpath(
                "{}_machsfile.master".format(parameter_set_name)
            )
            with this_machsfile_path.open("w") as f:
                # Write the nodelist for this parameter set to it
                f.write("\n".join(nodelist[parameter_set_name]))
            # Save the path in parameter_set_dirs
            parameter_set_dir["machsfile"] = this_machsfile_path
    else:
        for parameter_set_dir in parameter_set_dirs.values():
            parameter_set_dir.update({"machsfile": None})

    # Now run rams
    run_results = []
    for parameter_set_name in parameter_sets_dict.keys():
        this_parameter_set_dirs = parameter_set_dirs[parameter_set_name]
        run_results.append(
            run_rams_for_ramsin(
                ramsin_path=str(
                    this_parameter_set_dirs["top"].joinpath(
                        "RAMSIN.{}".format(parameter_set_name + fname_suffix)
                    )
                ),
                stdout_path=str(
                    this_parameter_set_dirs["top"].joinpath(
                        "{}.stdout".format(parameter_set_name + fname_suffix)
                    )
                ),
                rams_executable_path=(
                    rams_executable_path[parameter_set_name]
                    if isinstance(rams_executable_path, dict)
                    else rams_executable_path
                ),
                machsfile_path=this_parameter_set_dirs["machsfile"],
                log_command=log_command,
                log_ramsin=log_ramsin,
                dry_run=dry_run,
                asynchronous=parallel,
                verbose=verbose,
            )
        )
    if parallel and block and not dry_run:
        # We want to block until all of the subprocesses we spawned are finished
        # Call the wait method of each subprocess; it will return if the process is finished
        try:
            [sp.wait() for sp in run_results]
        finally:
            # If we interrupt this it won't kill the processes, so we implement that manually
            [sp.kill() for sp in run_results]
    return run_results


def fill_rams_output_dimensions(ds, header_filepath, dim_names, grid_number=1):
    # Rename dimensions and assign coordinates
    # Analysis files should always have the same variables and therefore the same dimensions, but we may have
    # dropped out all of the variables that contain a given dimension, so first see which dimensions are actually
    # present in this subsetted dataset
    # This will also be ok even if we are dropping out variables bc phony_dims='sort' evidently checks
    # the dimensions before dropping the variables, so they'll be labeled consistenly even if e.g.
    # you don't read in any variables that use the z dimension (phony_dim_2)
    dims_to_rename = {k: v for k, v in dim_names.items() if k in list(ds.dims)}
    ds = ds.rename_dims(dims_to_rename)

    # Read in the actual values of the coordinates in this dataset from the header file
    # Just use the first filepath given; they should all be the same for the same RAMS run
    dimension_vals = {}
    with Path(header_filepath).open("r") as f:
        # Make a copy of the header name dimension dict since we'll be popping from it
        header_name_dimension_dict = {
            k.format(**{"grid_number": str(grid_number).zfill(2)}): v
            for k, v in HEADER_NAME_DIMENSION_DICT.items()
            if v in dims_to_rename.values()
        }
        while header_name_dimension_dict:
            for line in f:
                line = line.strip()
                if line in header_name_dimension_dict.keys():
                    this_header_name = line
                    break
            n_levels = int(
                next(f).strip()
            )  # The line after the fieldname lists the number of levels
            levels = [
                float(next(f).strip()) for _ in range(n_levels)
            ]  # Get the levels themselves
            dimension_vals[header_name_dimension_dict[this_header_name]] = levels
            header_name_dimension_dict.pop(this_header_name)
    try:
        ds = ds.assign_coords(dimension_vals)
    except ValueError:
        raise ValueError(
            "Mismatch between dimension lengths in dataset and header;\n"
            f"Passed dimension dict: {dim_names}\n"
            f"Dimension sizes in dataset: {ds.dims}\n"
            f"Dimension lengths from header: { {k: len(v) for k, v in dimension_vals.items()} }"
        )
    return ds


def read_rams_output(
    input_filenames,
    dim_names=RAMS_ANALYSIS_FILE_DIMENSIONS_DICT,
    keep_vars=[],
    preprocess=None,
    time_dim_name="time",
    parallel=True,
    chunks="auto",
    concatenate=True,
    silent=False,
    open_dataset_kwargs={},
    units=False,
):
    """
    Full docstring to come; in the meantime, note that if dask is installed and the parallel=True option still causes
    the function to fail with a strange error, running `conda upgrade xarray dask` can often resolve this issue.

    Use of the `parallel` option is significantly faster and uses half the maximum memory as compared to serial reading
    (since the intermediate step of concatenation in serial reading requires the entire dataset to temporarily have two
    copies stored in memory); it is highly recommended to use it.
    """

    # Define function for printing if we're not running silently
    def maybe_print(x):
        if not silent:
            print(x)

    # If trying to read in parallel, first check if dask is installed
    if parallel:
        try:
            import dask
        except ImportError:
            print(
                "dask must be installed to use the `parallel` option; falling back to reading serially"
            )
            parallel = False

    # Convert input filenames to paths if they're not already
    input_filenames = [Path(x) for x in input_filenames]
    # Pull the times out from these
    input_datetimes = []
    for fpath in input_filenames:
        time = pd.to_datetime(
            re.search(RAMS_FILENAME_DATETIME_REGEX, fpath.name).group(0)
        )
        if not time:
            raise ValueError(
                f"File {str(fpath.name)} does not contain a valid timestamp in the filename"
            )
        input_datetimes.append(time)

    # Figure out the variables we want to drop, since that's what xarray needs for open_dataset
    drop_vars = (
        []
        if not keep_vars
        else [x for x in RAMS_VARIABLES_DF["name"].values if x not in keep_vars]
    )

    # The main difference between the serial and parallel processing is that serial calls xr.open_dataset and
    # parallel calls xr.open_mfdataset; the former requires manual concatenation while the latter handles it
    # within open_mfdataset. Note that there is actually a parallel argument to open_mfdataset, i.e. it is possible
    # to use this function serially, but the only reason not to use open_mfdataset is dask/xarray installation/version
    # issues and if these are resolved then the parallel option should work, so we don't currently have an option
    # for using open_mfdataset with `parallel=False` since I'm not sure why that would be useful.
    if parallel:
        maybe_print(
            f"Reading and concatenating {len(input_filenames)} RAMS individual timestep outputs..."
        )
        from dask.diagnostics import ProgressBar
        from contextlib import nullcontext

        open_ds_context_manager = nullcontext if silent else ProgressBar
        # Now actually call open_mfdataset
        with open_ds_context_manager():
            ds = xr.open_mfdataset(
                input_filenames,
                concat_dim=time_dim_name,
                combine="nested",
                preprocess=preprocess,
                phony_dims="sort",
                engine="h5netcdf",
                drop_variables=drop_vars,
                parallel=True,
                chunks=chunks,
                **open_dataset_kwargs,
            )
    # Serial reading logic
    else:
        maybe_print(
            f"Reading {len(input_filenames)} RAMS individual timestep outputs..."
        )
        # Create a list to store the datasets we'll co=ncat
        to_concat = []
        wrapped_to_read = tqdm(input_filenames) if not silent else input_filenames
        for ds_path in wrapped_to_read:
            # Read in this dataset
            ds = xr.open_dataset(
                ds_path,
                drop_variables=drop_vars,
                engine="h5netcdf",
                phony_dims="sort",
                **open_dataset_kwargs,
            )
            # Append this to our list
            to_concat.append(ds)
        # Now concatenate along the time dimension
        if len(to_concat) > 1:
            if concatenate:
                maybe_print("Concatenating along time...")
                ds = xr.concat(to_concat, dim=time_dim_name)
            else:
                ds = to_concat
        else:
            ds = to_concat[0]

    # Now handle the RAMS dimensions if we're supposed to
    if dim_names:
        # Get the grid number we're looking at and the path to the header file
        grid_number = int(re.search(r"g([1-9]+).h5", input_filenames[0].name).group(1))
        header_filename = re.sub(r"g[1-9].h5", r"head.txt", input_filenames[0].name)
        ds = fill_rams_output_dimensions(
            ds=ds,
            header_filepath=input_filenames[0].parent.joinpath(header_filename),
            dim_names=dim_names,
            grid_number=grid_number,
        )

    # Assign file datetimes
    ds = ds.assign_coords(**{time_dim_name: input_datetimes})

    # Sort this across time
    ds = ds.sortby(time_dim_name)

    # Align the chunks if we used dask
    if parallel:
        ds = ds.unify_chunks()

    # Give the dataset units if we should
    if units:
        ds = ds.pint.quantify(
            RAMS_VARIABLES_DF.set_index("name")["units"].to_dict(), unit_registry=ureg
        )
    # Give the variables attributes either way
    rams_attrs_dicts = RAMS_VARIABLES_DF.set_index("name").to_dict(orient="index")
    for var in ds.data_vars:
        ds[var] = ds[var].assign_attrs(rams_attrs_dicts.get(var, {}))
    return ds


def write_rams_formatted_sounding(df, output_path, second_copy=None):
    # First do a bunch of checks on the input data
    # Make sure we have all of the required columns
    if not all([x in df.columns for x in SOUNDING_NAMELIST_VARIABLES]):
        raise ValueError(
            f"Sounding dataframes must contain columns {SOUNDING_NAMELIST_VARIABLES}"
        )
    # Make sure pressure is monotonically decreasing and has no duplicate values
    if not (df["PS"].is_monotonic_decreasing and df["PS"].nunique() == len(df)):
        raise ValueError(
            "'PS' field must be monotonically decreasing with no duplicate values"
        )
    # Should be all good, write it out
    # Always write to output_path, and to second_copy if it's passed
    output_paths = [output_path]
    if second_copy:
        output_paths.append(second_copy)
    for this_output_path in output_paths:
        df[SOUNDING_NAMELIST_VARIABLES].to_csv(
            str(this_output_path),
            sep=",",
            header=False,
            index=False,
            float_format="%.4f",
            lineterminator=",\n",
        )


def get_z_levels(deltaz, dzrat, dzmax, nnzp=None, max_height=None):
    """
    Can pass either of nnzp or max_height, depending on if you know the number of levels or the maximum
    height that you want to reach. You must pass one.
    """
    if not nnzp and not max_height:
        raise ValueError("Must pass one of nnzp or max_height")
    # Could maybe do this analytically but we'll just brute force it
    # First there's a sub-ground layer and immediately above ground layer, each of height deltaz / 2
    heights = [-1 * deltaz / 2, deltaz / 2]

    def need_more_heights():
        if nnzp:
            return len(heights) <= nnzp
        elif max_height:
            return heights[-1] < max_height

    while need_more_heights():
        deltaz = min(deltaz * dzrat, dzmax)
        heights.append(heights[-1] + deltaz)
    return np.array(heights)


def format_sounding_field_ramsin_str(values):
    """
    Given an array of values corresponding to one of the sounding fields in a RAMSIN, return a string of these values
    for hardcoding into a RAMSIN that can be passed to `generate_ramsin`
    """
    values = np.array(values)

    return ",\n          ".join(
        [
            np.array2string(
                values[ix : ix + 5],
                formatter={"float_kind": lambda x: "%.4f" % x},
                separator=",    ",
            )[1:-1]
            for ix in range(0, len(values), 5)
        ]
    )


def split_snowfall_nodelists(parameter_sets):
    node_assignments = {
        k: int(ix // (len(parameter_sets) / 3)) + 1
        for ix, k in enumerate(parameter_sets.keys())
    }
    n_jobs_per_node = {1: 0, 2: 0, 3: 0}
    for v in node_assignments.values():
        n_jobs_per_node[v] += 1
    n_cores_per_job = {k: 64 // v for k, v in n_jobs_per_node.items()}
    return {
        k: [f"snowfall{node_assignments[k]}:{n_cores_per_job[node_assignments[k]]}"]
        for ix, (k, v) in enumerate(parameter_sets.items())
    }


def parse_rams_stdout_walltimes(rams_stdout_path, plot=True):
    """
    Read in the stdout from a RAMS run and parse (and optionally plot) the walltime per simulation timestep. I find
    this helpful for empirically checking a run that seems slower than it should be against prior runs.
    """
    sim_times = []
    walltimes = []
    with Path(rams_stdout_path).open("r") as f:
        for line in f.readlines():
            maybe_match = re.search(
                r"Timestep.*Sim time\(sec\)=\s*([0-9\.]+).*Wall time\(sec\)=\s*([0-9\.]+)",
                line,
            )
            if maybe_match:
                sim_times.append(float(maybe_match.group(1)))
                walltimes.append(float(maybe_match.group(2)))
    # Drop the first entry from each, since it's always way longer
    sim_times = sim_times[1:]
    walltimes = walltimes[1:]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(sim_times, walltimes)
        ax.set_xlabel("Simulation time (s)")
        ax.set_ylabel("Walltime per timestep (s)")
    return (sim_times, walltimes)


def calculate_thermodynamic_variables(ds):
    """
    Function to calculate basic derived thermodynamic/physical variables.
    """
    # Temperature
    ds["T"] = ds["PI"] * ds["THETA"] / 1004.0
    # Condensate mixing ratio
    ds["R_condensate"] = ds["RTP"] - ds["RV"]
    # Pressure
    ds["P"] = 1000.0 * ((ds["PI"] / 1004.0) ** (1004.0 / 287.0))
    vp = mpc.vapor_pressure(ds["P"] * units("hPa"), ds["RV"] * units("kg/kg"))
    # Dewpoint
    ds["dewpoint"] = mpc.dewpoint(vp).pint.to("K").pint.dequantify()
    # Vapor pressure
    ds["vapor_pressure"] = vp.pint.to("hPa").pint.dequantify()
    # Saturation vapor pressure
    ds["saturation_vapor_pressure"] = (
        mpc.saturation_vapor_pressure(ds["T"] * units("K"))
        .pint.to("hPa")
        .pint.dequantify()
    )
    # Relative humidity
    ds["RH"] = mpc.relative_humidity_from_mixing_ratio(
        ds["P"] * units("hPa"), ds["T"] * units("K"), ds["RV"]
    )
    # Flag for supersaturation
    ds["supersaturated"] = ds["RH"] >= 1
    # Density potential temperature
    ds["theta_rho"] = ds["THETA"] * (1 + 0.608 * ds["RV"] - ds["R_condensate"])
    # Buoyancy
    # From Tompkins 2001
    tr_layer_mean = ds["theta_rho"].mean(["x", "y"])
    ds["buoyancy"] = (
        mpconstants.g * (ds["theta_rho"] - tr_layer_mean) / tr_layer_mean
    ).pint.dequantify()
    # Total liquid mixing ratio
    ds["R_liquid"] = ds["RCP"] + ds["RRP"]
    # Total ice mixing ratio
    ds["R_ice"] = ds["RPP"] + ds["RSP"] + ds["RAP"] + ds["RGP"] + ds["RHP"]
    # Vertical vorticity
    ds["vertical_vorticity"] = ds["VC"].differentiate("x") - ds["UC"].differentiate("y")
    # Horizontal divergence
    ds["divergence"] = ds["UC"].differentiate("x") + ds["VC"].differentiate("y")
    return ds


def calculate_derived_variables(storm_ds, storm_name):
    """
    Function to calculate derived thermodynamic variables as well as do basic preprocessing on the rams output
    """
    storm_ds = calculate_thermodynamic_variables(storm_ds)

    # Shift the x and y coords so that they start from 0
    storm_ds["x"] = storm_ds["x"] - min(storm_ds["x"])
    storm_ds["y"] = storm_ds["y"] - min(storm_ds["y"])

    # Add a time in minutes coordinate
    storm_ds = storm_ds.assign_coords(
        t_minutes=(storm_ds["time"] - storm_ds["time"].values[0]).dt.total_seconds()
        // 60
    )

    # Add a flag for whether a gridpoint should be included in analysis
    storm_ds["in_analysis"] = ANALYSIS_FLAG_FUNCTIONS[storm_name](storm_ds)

    return storm_ds
