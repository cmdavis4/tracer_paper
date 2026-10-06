"""Compare the initial base states of the two squall-line runs.

fixed-False = old "hacked" sounding (theta forced constant below 900 hPa).
fixed-True  = corrected Weisman-Klemp sounding.

We read the ACTUAL t=0 state from each run's output (domain-mean profile),
cross-check it against the RAMSIN read-back, then compute surface-based and
most-unstable CAPE/CIN/LCL/LFC with metpy so the storm-vs-no-storm question can
be tied to a concrete stability difference rather than eyeballed.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import metpy.calc as mpc
from metpy.units import units

import xarrams as xrr

BASE = Path("/moonbow/cmdavis4/projects/bl_transport/rams_io/"
            "squall_line_wk/validation")
RUNS = {"fixed-False (hacked)": "fixed-False", "fixed-True (corrected)": "fixed-True"}


def surface_and_mu_parcel_metrics(df):
    """Surface-based and most-unstable CAPE/CIN/LCL/LFC from a sounding df.

    df columns: z (m), PS (hPa), TS (degC), RTS (% RH). Assumes ascending z.
    """
    df = df[df["z"] >= 0].sort_values("z").reset_index(drop=True)
    p = df["PS"].values * units("hPa")
    T = df["TS"].values * units("degC")
    rh = df["RTS"].values * units("percent")
    Td = mpc.dewpoint_from_relative_humidity(T, rh)

    # Surface-based parcel.
    sbcape, sbcin = mpc.surface_based_cape_cin(p, T, Td)
    lcl_p, _ = mpc.lcl(p[0], T[0], Td[0])
    try:
        lfc_p, _ = mpc.lfc(p, T, Td)
    except Exception:
        lfc_p = np.nan * units("hPa")

    # Most-unstable parcel.
    mucape, mucin = mpc.most_unstable_cape_cin(p, T, Td)

    # Map LCL / LFC pressures to heights for intuition.
    z = df["z"].values
    def p_to_z(pp):
        if not np.isfinite(getattr(pp, "magnitude", np.nan)):
            return np.nan
        return float(np.interp(pp.to("hPa").magnitude, p.to("hPa").magnitude[::-1], z[::-1]))

    return {
        "SBCAPE": sbcape.to("J/kg").magnitude,
        "SBCIN": sbcin.to("J/kg").magnitude,
        "MUCAPE": mucape.to("J/kg").magnitude,
        "MUCIN": mucin.to("J/kg").magnitude,
        "LCL_z": p_to_z(lcl_p),
        "LFC_z": p_to_z(lfc_p),
    }


rows = {}
bl_profiles = {}
for label, sub in RUNS.items():
    # --- actual t=0 state from output (domain mean) ---
    t0_file = sorted((BASE / sub / "output").glob("a-A-*-210000-g1.h5"))
    ds = xrr.read_rams_output(t0_file)
    out_df = xrr.to_sounding_df(ds.squeeze())

    # --- RAMSIN read-back (intended sounding) ---
    rin_df = xrr.sounding_df_from_ramsin(str(BASE / sub / "RAMSIN"))

    # Cross-check: surface theta/qv from each path (should agree closely).
    out_dv = xrr.soundings.calculate_sounding_derived_vars.__wrapped__ \
        if hasattr(xrr.soundings.calculate_sounding_derived_vars, "__wrapped__") \
        else None
    # surface T comparison (cheap, no CAPE loop):
    print(f"[{label}] surface TS  out={out_df['TS'].iloc[0]:.3f}C  "
          f"ramsin={rin_df['TS'].iloc[0]:.3f}C  |  "
          f"surface RTS out={out_df['RTS'].iloc[0]:.2f}%  ramsin={rin_df['RTS'].iloc[0]:.2f}%")

    # Use the output (actual) profile for the parcel metrics.
    rows[label] = surface_and_mu_parcel_metrics(out_df)

    # Low-level theta / theta_e / qv profile for the stability story.
    bl = out_df[out_df["z"] <= 3000].copy()
    p = bl["PS"].values * units("hPa")
    T = bl["TS"].values * units("degC")
    rh = bl["RTS"].values * units("percent")
    Td = mpc.dewpoint_from_relative_humidity(T, rh)
    bl["theta"] = mpc.potential_temperature(p, T).to("K").magnitude
    bl["qv"] = mpc.mixing_ratio_from_relative_humidity(p, T, rh).to("g/kg").magnitude
    bl["theta_e"] = mpc.equivalent_potential_temperature(p, T, Td).to("K").magnitude
    bl_profiles[label] = bl

print()
print("=== Surface-based / most-unstable parcel metrics (from actual t=0 output) ===")
print(pd.DataFrame(rows).T.round(1).to_string())

print()
print("=== Boundary-layer structure (actual t=0 output, every ~250 m) ===")
for label, bl in bl_profiles.items():
    print(f"\n--- {label} ---")
    show = bl.iloc[::max(1, len(bl)//10)][["z", "theta", "theta_e", "qv", "RTS", "TS"]]
    print(show.round(3).to_string(index=False))
    # BL theta lapse rate over the lowest 1 km.
    near = bl[bl["z"] <= 1000]
    dtheta = near["theta"].iloc[-1] - near["theta"].iloc[0]
    dz = near["z"].iloc[-1] - near["z"].iloc[0]
    print(f"  theta lapse 0-1km: {dtheta/dz*1000:+.2f} K/km   "
          f"(surface theta={bl['theta'].iloc[0]:.2f} K, theta_e={bl['theta_e'].iloc[0]:.1f} K)")
