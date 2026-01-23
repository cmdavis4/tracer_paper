# Model Adapters

This directory contains configuration files that allow the trajectory calculation code to work with different atmospheric model outputs.

## Overview

Model adapters define how to read and interpret data from different atmospheric models. Each adapter is a JSON file that specifies:

- Variable name mappings (e.g., which variable contains the U wind component)
- Dimension name mappings (e.g., what the x/y/z coordinates are called)
- File naming conventions
- Grid stagger configuration
- Which reader function to use

## Available Presets

### RAMS (Regional Atmospheric Modeling System)
**File:** `rams.json`

- Uses separate header and data files
- Wind variables: UC, VC, WC
- Dimensions: x, y, z
- File pattern: `a-L-{datetime}-g{grid}.h5`
- Requires RAMS-specific header file parsing

### CM1 (Cloud Model 1)
**File:** `cm1.json`

- Single NetCDF file per timestep
- Wind variables: u, v, w
- Dimensions: xh, yh, zh (scalar grid)
- File pattern: `cm1out_{timestep:06d}.nc`
- Standard NetCDF reader

## Using Model Adapters

### Python API

```python
from cloudy.trajectories import calculate_trajectories_julia

# Use a preset adapter by name
calculate_trajectories_julia(
    ...,
    model_adapter="cm1",  # or "rams"
    ...
)

# Use a custom adapter file
calculate_trajectories_julia(
    ...,
    model_adapter="/path/to/custom_adapter.json",
    ...
)

# Use an adapter dictionary
custom_adapter = {
    "name": "MyModel",
    "variables": {"u_wind": "U", "v_wind": "V", "w_wind": "W"},
    ...
}
calculate_trajectories_julia(
    ...,
    model_adapter=custom_adapter,
    ...
)
```

## Creating a Custom Adapter

To create an adapter for a new model, copy one of the existing adapters and modify it:

```json
{
  "name": "MyModel",
  "description": "Description of the model",

  "variables": {
    "u_wind": "U",      // Name of U wind component in your files
    "v_wind": "V",      // Name of V wind component
    "w_wind": "W"       // Name of W wind component
  },

  "dimensions": {
    "x": "x",           // Name of x dimension in your files
    "y": "y",           // Name of y dimension
    "z": "z",           // Name of z dimension
    "time": "time"      // Name of time dimension
  },

  "file_naming": {
    // Use {datetime} for datetime-based naming
    "pattern": "output_{datetime}.nc",
    "datetime_format": "yyyy-mm-dd_HHMMSS"

    // OR use {timestep} for timestep-based naming
    // "pattern": "output_{timestep:06d}.nc",
    // "datetime_format": null
  },

  "grid_stagger": {
    // Stagger offsets for wind variables (Arakawa C-grid)
    "u_wind": {"x": 0.5, "y": 0.0, "z": 0.0},
    "v_wind": {"x": 0.0, "y": 0.5, "z": 0.0},
    "w_wind": {"x": 0.0, "y": 0.0, "z": 0.5}
  },

  // "rams" for RAMS-specific reader, "netcdf" for generic NetCDF
  "reader_type": "netcdf"
}
```

### Required Fields

- `name`: String identifying the adapter
- `variables`: Object with `u_wind`, `v_wind`, `w_wind` keys
- `dimensions`: Object with `x`, `y`, `z`, `time` keys
- `file_naming`: Object with `pattern` and `datetime_format` keys
- `grid_stagger`: Object defining stagger for each wind variable
- `reader_type`: Either `"rams"` or `"netcdf"`

### File Naming Patterns

**Datetime-based naming** (like RAMS):
```json
{
  "pattern": "output_{datetime}.nc",
  "datetime_format": "yyyy-mm-dd-HHMMSS"
}
```

**Timestep-based naming** (like CM1):
```json
{
  "pattern": "cm1out_{timestep:06d}.nc",
  "datetime_format": null
}
```

Pattern placeholders:
- `{datetime}`: Replaced with formatted datetime
- `{timestep}` or `{timestep:06d}`: Replaced with timestep number (with optional padding)
- `{grid}` or `{grid:02d}`: Replaced with grid number (with optional padding)

### Grid Stagger

The `grid_stagger` section defines how wind variables are staggered relative to scalar variables. For an Arakawa C-grid (most common):

- U wind is staggered +0.5 in x
- V wind is staggered +0.5 in y
- W wind is staggered +0.5 in z
- Scalars have no stagger (0.0 in all directions)

## Troubleshooting

### Variable not found
If you get errors about missing variables, check:
1. The variable names in your adapter match those in your NetCDF files
2. Use `ncdump -h <file>` to see available variables

### Dimension not found
If you get errors about missing dimensions, check:
1. The dimension names in your adapter match those in your NetCDF files
2. Your files actually contain coordinate arrays for x, y, z

### File not found
If the trajectory code can't find your files, check:
1. The `file_naming.pattern` matches your actual files
2. The datetime format (if used) matches your file naming
3. Files exist in the `simulation_output_dir` you specified

## Notes

- The adapter system only affects how data is read; trajectory calculations are model-independent
- Scalar variables (tracked_scalars) are always assumed to be on the unstaggered grid
- For models other than RAMS, use `reader_type: "netcdf"` and ensure coordinates are stored as variables in your NetCDF files
