#!/usr/bin/env python3
"""Setup script for installing Julia dependencies for the trajectories submodule."""

import subprocess
import sys
from pathlib import Path


def setup_julia_packages():
    """Install Julia packages required for the trajectories submodule."""
    cloudy_dir = Path(__file__).parent
    project_toml = cloudy_dir / "Project.toml"

    if not project_toml.exists():
        print(f"Error: Project.toml not found at {project_toml}")
        sys.exit(1)

    print("Installing Julia packages for cloudy.trajectories...")
    print(f"Using Project.toml from: {cloudy_dir}")

    try:
        # Activate the project and instantiate (install) all dependencies
        result = subprocess.run(
            [
                "julia",
                "--project=" + str(cloudy_dir),
                "-e",
                "using Pkg; Pkg.instantiate()"
            ],
            check=True,
            capture_output=True,
            text=True
        )

        print("✓ Julia packages installed successfully!")
        print(result.stdout)

    except subprocess.CalledProcessError as e:
        print("✗ Failed to install Julia packages:")
        print(e.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("✗ Julia not found. Please install Julia first:")
        print("  - Using pixi: pixi add julia")
        print("  - Or download from: https://julialang.org/downloads/")
        sys.exit(1)


if __name__ == "__main__":
    setup_julia_packages()
