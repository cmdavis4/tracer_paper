"""
Matplotlib function hooks for capturing plotting data.

This module provides a system to hook into matplotlib functions and store
the data passed to them in a global dictionary.
"""

import matplotlib.pyplot as plt
import matplotlib.axes
import xarray as xr
import functools
import inspect
from typing import Dict, Any, Callable, List, Tuple

# Global dictionary to store captured data
CAPTURED_DATA: Dict[str, Tuple[Any]] = {}

HOOKED_PYPLOT_FUNCTIONS = [
    "plot",
    "scatter",
    "bar",
    "hist",
    "imshow",
    "contour",
    "contourf",
    "pcolormesh",
    "fill_between",
    "errorbar",
    "boxplot",
    "violin",
    "pie",
    "polar",
    "loglog",
    "semilogx",
    "semilogy",
]

HOOKED_AXES_METHODS = [
    "plot",
    "scatter",
    "bar",
    "hist",
    "imshow",
    "contour",
    "contourf",
    "pcolormesh",
    "fill_between",
    "errorbar",
    "boxplot",
    "violin",
]


def store_call_data(
    *args, artist_name: str, convert_xarray_to_numpy: bool = True
) -> None:
    """Store function call data in the global dictionary."""
    if artist_name in CAPTURED_DATA:
        raise ValueError(
            f"Artist name {artist_name} is already present in this figure's data"
        )
    if convert_xarray_to_numpy:
        args = [
            (
                x.values
                if (isinstance(x, xr.DataArray) or isinstance(x, xr.Dataset))
                else x
            )
            for x in args
        ]
    CAPTURED_DATA[artist_name] = tuple(args)


def create_hooked_fn(original_func: Callable) -> Callable:
    """Create a hook wrapper around a matplotlib function."""

    # Get the names of keyword arguments to store_call_data programatically
    # so we can remove them
    hook_kwargs = [
        x for x in inspect.signature(store_call_data).parameters.keys() if x != "args"
    ]

    @functools.wraps(original_func)
    def wrapper(*args, **kwargs):
        original_func_kwargs = {k: v for k, v in kwargs.items() if k not in hook_kwargs}
        store_call_data_kwargs = {k: v for k, v in kwargs.items() if k in hook_kwargs}
        if "artist_name" in store_call_data_kwargs:
            # If they don't pass artist_name then we don't save the data
            store_call_data(*args, **store_call_data_kwargs)

        # Call the original function
        return original_func(*args, **original_func_kwargs)

    return wrapper


def install_pyplot_hooks() -> None:
    """
    Install hooks on specified pyplot functions.

    Args:
        functions: List of function names to hook. If None, hooks common plotting functions.
    """
    for func_name in HOOKED_PYPLOT_FUNCTIONS:
        if hasattr(plt, func_name):
            original_func = getattr(plt, func_name)
            hooked_func = create_hooked_fn(original_func)
            setattr(plt, func_name, hooked_func)
            print(f"Hooked plt.{func_name}")


def install_axes_hooks() -> None:
    """
    Install hooks on Axes methods.

    Args:
        functions: List of method names to hook. If None, hooks common plotting methods.
    """
    for func_name in HOOKED_AXES_METHODS:
        if hasattr(matplotlib.axes.Axes, func_name):
            original_func = getattr(matplotlib.axes.Axes, func_name)
            hooked_func = create_hooked_fn(original_func)
            setattr(matplotlib.axes.Axes, func_name, hooked_func)
            print(f"Hooked ax.{func_name}")


def get_captured_data(func_name: str = None) -> Dict[str, Any]:
    """
    Get captured data for a specific function or all functions.

    Args:
        func_name: Specific function name to get data for. If None, returns all data.

    Returns:
        Dictionary containing captured data.
    """
    if func_name:
        return CAPTURED_DATA.get(func_name, [])
    return CAPTURED_DATA.copy()


def clear_captured_data(func_name: str = None) -> None:
    """
    Clear captured data for a specific function or all functions.

    Args:
        func_name: Specific function name to clear data for. If None, clears all data.
    """
    if func_name:
        CAPTURED_DATA.pop(func_name, None)
    else:
        CAPTURED_DATA.clear()


def print_captured_summary() -> None:
    """Print a summary of captured data."""
    print("=== Captured Data Summary ===")
    if not CAPTURED_DATA:
        print("No data captured yet.")
        return

    for func_name, calls in CAPTURED_DATA.items():
        print(f"{func_name}: {len(calls)} calls")
        for i, call in enumerate(calls):
            print(
                f"  Call {i+1}: {len(call['args'])} args, {len(call['kwargs'])} kwargs"
            )


def install_all_hooks() -> None:
    """Install hooks on both pyplot and axes functions."""
    print("Installing matplotlib hooks...")
    install_pyplot_hooks()
    install_axes_hooks()
    print("All hooks installed successfully!")


# Example usage functions
def example_usage():
    """Demonstrate how to use the hooks."""
    # Install hooks
    install_all_hooks()

    # Create some plots
    import numpy as np

    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    plt.figure()
    plt.plot(x, y, label="sin(x)")
    plt.scatter([1, 2, 3], [0.5, 0.8, 1.2])

    # Check captured data
    print_captured_summary()

    # Get specific data
    plot_data = get_captured_data("plt.plot")
    print(f"Plot calls: {len(plot_data)}")
    if plot_data:
        print(f"First plot call args: {plot_data[0]['args']}")
        print(f"First plot call kwargs: {plot_data[0]['kwargs']}")


if __name__ == "__main__":
    example_usage()
