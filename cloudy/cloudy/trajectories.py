"""Trajectory analysis and calculation utilities.

This module provides functions for initializing, calculating, and managing
atmospheric trajectories using Julia-based computation engines.

IMPORTANT: This module requires Julia and Julia packages. The first time you
use this module, it will automatically install the required Julia packages.
You can also manually run: python setup_julia.py
"""

import pandas as pd
import xarray as xr
from pathlib import Path, PosixPath
from jinja2 import Template
import subprocess
import datetime as dt
from typing import Any, Dict, List, Optional, Union
import warnings
import json

from .types_core import PathLike

from .utils import raise_if_exists
from .rams import RAMS_DT_STRFTIME_STR

# Global flag to track if Julia environment has been checked
_JULIA_ENV_CHECKED = False


def load_model_adapter(adapter: Union[str, PathLike, Dict]) -> Dict:
    """Load a model adapter configuration.

    Args:
        adapter: Can be:
            - A string name of a preset adapter (e.g., "rams", "cm1")
            - A path to a custom JSON adapter file
            - A dictionary containing adapter configuration

    Returns:
        Dictionary containing the adapter configuration

    Raises:
        FileNotFoundError: If preset or custom adapter file not found
        ValueError: If adapter configuration is invalid
    """
    # If already a dict, validate and return
    if isinstance(adapter, dict):
        _validate_adapter_config(adapter)
        return adapter

    # Try as preset name first
    if isinstance(adapter, str) and not adapter.endswith('.json'):
        preset_path = Path(__file__).parent.parent / "model_adapters" / f"{adapter}.json"
        if preset_path.exists():
            with open(preset_path, 'r') as f:
                config = json.load(f)
            _validate_adapter_config(config)
            return config
        # If not a preset, treat as a path
        adapter_path = Path(adapter)
    else:
        adapter_path = Path(adapter)

    # Load from custom path
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Model adapter not found: {adapter_path}\n"
            f"Available presets: rams, cm1\n"
            f"Or provide a path to a custom adapter JSON file"
        )

    with open(adapter_path, 'r') as f:
        config = json.load(f)

    _validate_adapter_config(config)
    return config


def _validate_adapter_config(config: Dict) -> None:
    """Validate a model adapter configuration.

    Args:
        config: Adapter configuration dictionary

    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = {'name', 'variables', 'dimensions', 'file_naming', 'grid_stagger', 'reader_type'}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Model adapter missing required keys: {missing_keys}")

    # Validate variables section
    required_vars = {'u_wind', 'v_wind', 'w_wind'}
    missing_vars = required_vars - set(config['variables'].keys())
    if missing_vars:
        raise ValueError(f"Model adapter 'variables' section missing: {missing_vars}")

    # Validate dimensions section
    required_dims = {'x', 'y', 'z', 'time'}
    missing_dims = required_dims - set(config['dimensions'].keys())
    if missing_dims:
        raise ValueError(f"Model adapter 'dimensions' section missing: {missing_dims}")

    # Validate file_naming section
    if 'pattern' not in config['file_naming']:
        raise ValueError("Model adapter 'file_naming' must contain 'pattern'")

    # Validate reader_type
    valid_readers = {'rams', 'netcdf'}
    if config['reader_type'] not in valid_readers:
        raise ValueError(f"Model adapter 'reader_type' must be one of: {valid_readers}")


def _ensure_julia_environment():
    """Ensure Julia environment is set up with required packages.

    This function checks if the Julia Project.toml has been instantiated
    and does so if needed. It only runs once per Python session.
    """
    global _JULIA_ENV_CHECKED

    if _JULIA_ENV_CHECKED:
        return

    cloudy_dir = Path(__file__).parent.parent
    project_toml = cloudy_dir / "Project.toml"
    manifest_toml = cloudy_dir / "Manifest.toml"

    # If Manifest.toml exists, assume environment is set up
    if manifest_toml.exists():
        _JULIA_ENV_CHECKED = True
        return

    print("First-time setup: Installing Julia packages for trajectories...")
    print(f"This may take a few minutes but only happens once.")
    print(f"Using Project.toml from: {project_toml}")

    try:
        result = subprocess.run(
            [
                "julia",
                "--project=" + str(cloudy_dir),
                "-e",
                "using Pkg; Pkg.instantiate()",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
        )
        print("✓ Julia packages installed successfully!")
        _JULIA_ENV_CHECKED = True

    except subprocess.CalledProcessError as e:
        warnings.warn(
            "Failed to automatically install Julia packages. "
            f"Please run 'python setup_julia.py' manually.\nError: {e.stderr}"
        )
    except FileNotFoundError:
        warnings.warn(
            "Julia not found. Please install Julia:\n"
            "  - Using pixi: pixi add julia\n"
            "  - Or download from: https://julialang.org/downloads/\n"
            "Then run: python setup_julia.py"
        )
    except subprocess.TimeoutExpired:
        warnings.warn(
            "Julia package installation timed out. "
            "Please run 'python setup_julia.py' manually."
        )


def validate_parcel_initializations(pi_dicts: List[Dict[str, Any]]) -> None:
    """
    Validate parcel initialization dictionaries.

    Args:
        pi_dicts: List of parcel initialization dictionaries

    Raises:
        ValueError: If validation fails
    """
    required_keys = {"x", "y", "z", "initialization_time", "parcel_ix"}
    # Check if they all have the same keys
    first_keys = set(pi_dicts[0].keys())
    if not all(set(d.keys()) == first_keys for d in pi_dicts):
        raise ValueError(
            "All parcel initialization dictionaries must have the same keys"
        )
    if not all([x in first_keys for x in required_keys]):
        raise ValueError(f"Parcel initializations require keys {required_keys}")
    # Check parcel ixs aren't repeated
    parcel_ixs = [d["parcel_ix"] for d in pi_dicts]
    if not len(parcel_ixs) == len(set(parcel_ixs)):
        raise ValueError("parcel_ix values are not unique")
    extra_keys = [x for x in first_keys if x not in required_keys]
    print(f"Received extra keys {extra_keys}, propagating through")


def write_parcel_initializations(
    pi_dicts: List[Dict[str, Any]],
    output_fpath: PathLike,
    exist_ok: bool = False,
    mkdir: bool = False,
    assign_parcel_ixs: bool = False,
) -> pd.DataFrame:
    """
    Write parcel initializations to CSV file.

    Args:
        pi_dicts: List of parcel initialization dictionaries
        output_fpath: Output file path
        exist_ok: Whether to overwrite existing files
        mkdir: Whether to create parent directories
        assign_parcel_ixs: Whether to assign parcel indices automatically

    Returns:
        DataFrame containing the parcel initializations
    """
    validate_parcel_initializations(pi_dicts)
    if not exist_ok:
        raise_if_exists(output_fpath)
    if mkdir:
        output_fpath.parent.mkdir(parents=False, exist_ok=True)
    # Need to assign these parcel_ixs if they don't have them
    if assign_parcel_ixs:
        for parcel_ix, pi_dict in enumerate(pi_dicts):
            pi_dict["parcel_ix"] = parcel_ix
    # Now write to disk
    # Convert to a pandas dataframe for better datetime handling
    pi_df = pd.DataFrame(pi_dicts)
    pi_df.to_csv(output_fpath, index=False)
    return pi_df


def calculate_trajectories_julia(
    parcel_initialization_csv_path: PathLike,
    simulation_output_dir: PathLike,
    output_path: PathLike,
    trajectory_timestep: Union[int, float],
    trajectory_end_time: dt.datetime,
    back_trajectories: bool,
    model_adapter: Union[str, PathLike, Dict] = "rams",
    restart_ds_path: Optional[PathLike] = None,
    tracked_scalars: List[str] = [],
    parallel: bool = True,
    grid_stagger: Optional[float] = None,
    initial_file_timestep: Optional[int] = None,
    file_timestep_interval: Optional[Union[int, float]] = None,
    write_interval: int = 1,
    flush_interval: Optional[int] = None,
    max_iterations: int = 5,
    raise_on_nonconvergence: bool = False,
    stop_on_early_convergence: bool = True,
    use_subdomain_limiting: bool = True,
    raise_on_oob: bool = True,
    exist_ok: bool = False,
    test_write: bool = True,
    progress_bar: bool = True,
    verbose: bool = True,
    pipe_stdout: bool = True,
    block: bool = True,
    dry_run: bool = False,
    save_on_error: bool = True,
) -> Optional[Union[xr.Dataset, subprocess.Popen]]:
    """Calculate atmospheric trajectories using Julia-based computation engine.

    Args:
        parcel_initialization_csv_path: Path to CSV file containing parcel initial conditions
        simulation_output_dir: Directory containing model simulation output files
        output_path: Path where trajectory results will be saved
        trajectory_timestep: Time step for trajectory calculation (seconds)
        trajectory_end_time: End time for trajectory calculation
        back_trajectories: Whether to calculate backward trajectories
        model_adapter: Model adapter configuration. Can be:
            - A preset name string (e.g., "rams", "cm1")
            - Path to a custom adapter JSON file
            - Dictionary with adapter configuration
        restart_ds_path: Path to restart dataset for continuing calculations
        tracked_scalars: List of scalar variables to track along trajectories
        parallel: Whether to use parallel computation
        grid_stagger: Grid staggering parameter for interpolation (overrides adapter default if provided)
        initial_file_timestep: Timestep number of first file (for timestep-based naming like CM1).
            Required when using timestep-based file naming.
        file_timestep_interval: Time interval (seconds) between output files.
            Required when using timestep-based file naming.
        write_interval: Interval for writing output (in timesteps)
        flush_interval: Interval for flushing output buffers
        max_iterations: Maximum iterations for convergence
        raise_on_nonconvergence: Whether to raise error on convergence failure
        stop_on_early_convergence: Whether to stop when converged early
        use_subdomain_limiting: Whether to use subdomain limiting for efficiency
        raise_on_oob: Whether to raise error when parcels go out of bounds
        exist_ok: Whether to overwrite existing output files
        test_write: Whether to test write permissions before computation
        progress_bar: Whether to show progress bar
        verbose: Whether to print verbose output
        pipe_stdout: Whether to pipe stdout to file
        block: Whether to wait for computation to complete
        dry_run: If True, generate script but don't execute
        save_on_error: Whether to save partial results on error

    Returns:
        Dataset containing trajectory results (if block=True and not dry_run),
        subprocess.Popen object (if block=False), or None (if dry_run or file doesn't exist)
    """
    # Ensure Julia environment is set up
    _ensure_julia_environment()

    # Load model adapter configuration
    adapter_config = load_model_adapter(model_adapter)

    # Validate timestep parameters for timestep-based file naming
    uses_timestep_naming = '{timestep' in adapter_config['file_naming']['pattern']
    if uses_timestep_naming:
        if initial_file_timestep is None or file_timestep_interval is None:
            raise ValueError(
                f"Model adapter '{adapter_config['name']}' uses timestep-based file naming. "
                "You must provide both 'initial_file_timestep' and 'file_timestep_interval' parameters."
            )

    # Use adapter grid_stagger if not explicitly provided
    if grid_stagger is None:
        # Get default stagger from adapter (0.0 if not specified)
        grid_stagger = adapter_config['grid_stagger'].get('default', {}).get('x', 0.0)

    # Read the template file
    template_path = Path(__file__).parent / "run_trajectories_template.jl"
    with open(template_path, "r") as template_file:
        template_content = template_file.read()

    # Create a Jinja2 template object
    template = Template(template_content)

    # Prepare the arguments for templating
    template_args = {
        "trajectories_jl_path": str(Path(__file__).parent / "trajectories.jl"),
        "parcel_initialization_csv_path": f'"{str(parcel_initialization_csv_path)}"',
        "simulation_output_dir": f'"{str(simulation_output_dir)}"',
        "output_path": f'"{str(output_path)}"',
        "trajectory_timestep": trajectory_timestep,
        "trajectory_end_time": (
            f'"{str(trajectory_end_time.strftime(RAMS_DT_STRFTIME_STR))}"'
        ),
        "back_trajectories": str(back_trajectories).lower(),
        "restart_ds_path": (
            f'"{str(restart_ds_path)}"' if restart_ds_path else "nothing"
        ),
        "tracked_scalars": str(tracked_scalars).replace("'", '"'),
        "parallel": str(parallel).lower(),
        "grid_stagger": grid_stagger,
        "write_interval": write_interval,
        "flush_interval": flush_interval or 0,
        "max_iterations": max_iterations,
        "stop_on_early_convergence": str(stop_on_early_convergence).lower(),
        "raise_on_nonconvergence": str(raise_on_nonconvergence).lower(),
        "use_subdomain_limiting": str(use_subdomain_limiting).lower(),
        "raise_on_oob": str(raise_on_oob).lower(),
        "exist_ok": str(exist_ok).lower(),
        "test_write": str(test_write).lower(),
        "progress_bar": str(progress_bar).lower(),
        "verbose": str(verbose).lower(),
        "save_on_error": str(save_on_error).lower(),
        # Model adapter configuration
        "adapter_config_json": json.dumps(adapter_config),
        # Timestep numbering (for timestep-based file naming)
        "initial_file_timestep": initial_file_timestep if initial_file_timestep is not None else "nothing",
        "file_timestep_interval": file_timestep_interval if file_timestep_interval is not None else "nothing",
    }

    # Render the template with the arguments
    rendered_script = template.render(template_args)

    # Write the rendered script to a temporary file
    print(rendered_script)
    temp_script_path = Path(output_path).with_suffix(".jl")
    print(f"\nWriting julia script to {temp_script_path}")
    with open(temp_script_path, "w") as temp_script_file:
        temp_script_file.write(rendered_script)

    # Execute the Julia script using subprocess

    if not dry_run:
        # Get the cloudy directory to use its Project.toml
        cloudy_dir = Path(__file__).parent.parent

        stdout_path = Path(output_path).with_suffix(".stdout")
        print(f"Writing stdout to {stdout_path}")
        with stdout_path.open("w") as stdout_f:
            sp = subprocess.Popen(
                [
                    "julia",
                    "--project=" + str(cloudy_dir),
                    "--threads",
                    "64",
                    temp_script_path,
                ],
                start_new_session=True,
                stdout=stdout_f if pipe_stdout else None,
                stderr=stdout_f if pipe_stdout else None,
            )
        if block:
            try:
                sp.wait()
            finally:
                # If we interrupt this it won't kill the processes, so we implement that manually
                sp.kill()
                return sp
        else:
            return sp

    if Path(output_path).exists():
        return xr.open_dataset(output_path)
    else:
        print("output_path does not exist, returning None")
        return None
