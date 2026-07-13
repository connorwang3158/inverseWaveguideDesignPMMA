"""A3 verification of the SRG-arm winner: Fourier-order convergence (the
slant staircase converges slower than smooth freeform patterns) + meent
cross-check + staircase-resolution (NL) sensitivity. Reads the best SRG row
from metagrating_results.csv. Usage: python3 verify_srg_winner.py"""

import csv
import json

import numpy as np

from metagrating_model import solve_orders, H_BUDGET, NX
from crosscheck_meent import meent_orders


def occ_srg_nl(slant_deg, duty, depth, period, nl):
    """Slanted-SRG occupancy at arbitrary staircase resolution nl."""
    occ = np.zeros((nl, NX))
    xs = np.arange(NX) / NX
    for k in range(nl):
        z_top = (k + 0.5) * H_BUDGET / nl
        if z_top < H_BUDGET - depth:
            continue
        z_in = z_top - (H_BUDGET - depth)
        shift = np.tan(np.deg2rad(slant_deg)) * z_in / period
        occ[k] = (((xs - shift) % 1.0) < duty).astype(float)
    return occ


def best_srg_params():
    with open("metagrating_results.csv") as f:
        rows = [r for r in csv.DictReader(f) if r["arm"] == "SRG"]
    best = max(rows, key=lambda r: float(r["eta_TE_bin"]))
    return json.loads(best["params_json"])


def main():
    p = best_srg_params()
    print("SRG winner:", {k: round(v, 3) for k, v in p.items()})
    per = p["period_nm"]

    print("\n[1] Fourier-order convergence (NL=6 staircase, grcwa TE):")
    occ6 = occ_srg_nl(p["slant_deg"], p["duty"], p["depth_nm"], per, 6)
    for g in (41, 61, 81, 121, 161):
        r = solve_orders(occ6, per, "TE", nG=g)
        eta = r["orders"].get(-1, 0.0)
        print(f"  nG={g:4d}  eta(-1)={float(eta):.5f}  "
              f"|R+T-1|={float(r['energy_residual']):.1e}")

    print("\n[2] meent cross-check (fto=60, NL=6):")
    m, tot = meent_orders(occ6, per, fto=60, pol=0, ucell_mode="n")
    print(f"  meent eta(-1)={m[-1]:.5f}   R+T={tot:.8f}")

    print("\n[3] staircase-resolution sensitivity (nG=121, grcwa TE):")
    print("    (does the 92% survive a smoother slant discretization?)")
    for nl in (6, 12, 24, 48):
        occ = occ_srg_nl(p["slant_deg"], p["duty"], p["depth_nm"], per, nl)
        # NL sublayers of thickness H/NL — rebuild thicknesses accordingly
        import metagrating_model as mm
        old = mm.NL
        mm.NL = nl                     # solve_orders reads thickness from NL
        try:
            r = solve_orders(occ, per, "TE", nG=121)
        finally:
            mm.NL = old
        print(f"  NL={nl:3d}  eta(-1)={float(r['orders'].get(-1,0.0)):.5f}  "
              f"|R+T-1|={float(r['energy_residual']):.1e}")


if __name__ == "__main__":
    main()
