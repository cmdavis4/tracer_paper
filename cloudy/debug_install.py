#!/usr/bin/env python3
"""Diagnostic script to debug package installation issues."""

import sys
import os
import site

print("=" * 70)
print("PYTHON ENVIRONMENT DIAGNOSTICS")
print("=" * 70)

print("\n1. Python executable:")
print(f"   {sys.executable}")

print("\n2. Python version:")
print(f"   {sys.version}")

print("\n3. sys.path (where Python looks for modules):")
for i, path in enumerate(sys.path):
    print(f"   [{i}] {path}")

print("\n4. Site packages directories:")
for pkg_dir in site.getsitepackages():
    print(f"   {pkg_dir}")
    if os.path.exists(pkg_dir):
        # Check if cloudy is installed there
        cloudy_path = os.path.join(pkg_dir, "cloudy")
        cloudy_dist = os.path.join(pkg_dir, "cloudy-1.0.0.dist-info")
        if os.path.exists(cloudy_path):
            print(f"   ✓ Found cloudy/ directory")
        if os.path.exists(cloudy_dist):
            print(f"   ✓ Found cloudy-1.0.0.dist-info/")
            top_level = os.path.join(cloudy_dist, "top_level.txt")
            if os.path.exists(top_level):
                with open(top_level) as f:
                    print(f"   ✓ top_level.txt contains: {f.read().strip()}")

print("\n5. User site packages:")
user_site = site.getusersitepackages()
print(f"   {user_site}")
if os.path.exists(user_site):
    cloudy_path = os.path.join(user_site, "cloudy")
    if os.path.exists(cloudy_path):
        print(f"   ✓ Found cloudy/ in user site")

print("\n6. Current working directory:")
print(f"   {os.getcwd()}")
# Check if we're in the source directory
if os.path.exists("./cloudy") and os.path.exists("./pyproject.toml"):
    print(
        "   ⚠ WARNING: You are in the source directory! This might cause import issues."
    )

print("\n7. Try importing cloudy:")
try:
    import cloudy

    print(f"   ✓ SUCCESS: import cloudy")
    print(f"   Location: {cloudy.__file__}")
    print(f"   Available modules: {cloudy.__all__}")
except ImportError as e:
    print(f"   ✗ FAILED: {e}")
except Exception as e:
    print(f"   ✗ ERROR: {e}")

print("\n8. Check for 'cloudy' in sys.modules:")
if "cloudy" in sys.modules:
    print(f"   ✓ 'cloudy' is in sys.modules")
    print(f"   Module object: {sys.modules['cloudy']}")
else:
    print(f"   ✗ 'cloudy' is NOT in sys.modules")

print("\n9. List all installed packages with 'cloudy' in the name:")
try:
    import importlib.metadata

    for dist in importlib.metadata.distributions():
        if "cloudy" in dist.name.lower():
            print(f"   - {dist.name} {dist.version}")
            print(f"     Location: {dist._path}")
except Exception as e:
    print(f"   Could not list packages: {e}")

print("\n" + "=" * 70)
print("DIAGNOSTICS COMPLETE")
print("=" * 70)
