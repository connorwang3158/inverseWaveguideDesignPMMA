"""Reciprocity verification for the out-coupler (2026-07-17 audit, §2.5).

The transmission cascade applies the in-coupler's eta_1 to the out-coupler.
The out-coupler is a DIFFERENT electromagnetic problem: PMMA superstrate,
guided-angle incidence, air substrate. For a lossless reciprocal grating with
correct n*cos(theta) power normalization the two efficiencies must be equal:
    eta(air, 0 deg -> PMMA, theta_d; m=+1) == eta(PMMA, theta_d -> air, 0; m=-1)
This script checks that claim numerically at the record geometry instead of
leaving it to a reader's charity. Also repeats the nG convergence check AT THE
RECORD GEOMETRY (v3 checked it only at the scalar-optimal design, §2.13).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grcwa

from physics.rigorous_solver import grating_orders_rcwa, PERIOD_Y_NM, profile_binary

# v4 record geometry (results/best_design_ever_v4.csv)
N_G = 1.5
PERIOD, DEPTH, DUTY = 445.5, 195.1, 0.4482
LAM = 532.0


def outcoupler_rcwa(period_nm, depth_nm, duty, wavelength_nm, n_sup,
                    theta_deg, pol, nG=61, Nx=256):
    """Substrate-side solve: plane wave inside PMMA (superstrate) at the
    guided angle -> corrugated surface -> air substrate. Returns order-(-1)
    transmitted power efficiency (the out-coupled beam, exiting at 0 deg for
    a matched period)."""
    lam, eps_sup = float(wavelength_nm), n_sup ** 2
    obj = grcwa.obj(nG, [period_nm, 0.0], [0.0, PERIOD_Y_NM],
                    1.0 / lam, theta_deg * np.pi / 180.0, 0.0, verbose=0)
    obj.Add_LayerUniform(100.0, eps_sup)                   # PMMA superstrate
    occ = profile_binary(Nx=Nx, n_sub=1, duty=duty)
    Ny = 8
    obj.Add_LayerGrid(depth_nm, Nx, Ny)                    # binary: z-symmetric
    obj.Add_LayerUniform(100.0, 1.0)                       # air substrate
    obj.Init_Setup()
    ep = np.tile((1.0 + (eps_sup - 1.0) * occ[0])[:, None], (1, Ny)).flatten()
    obj.GridLayer_geteps(ep)
    if pol == "s":
        obj.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    else:
        obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    out = {}
    for i, (gx, gy) in enumerate(obj.G):
        if gy == 0 and abs(gx) <= 2:
            out[int(gx)] = out.get(int(gx), 0.0) + float(np.real(Ti[i]))
    return out


def main():
    x = LAM / PERIOD
    theta_d = np.rad2deg(np.arcsin(x / N_G))
    print(f"record geometry: period={PERIOD} depth={DEPTH} duty={DUTY} "
          f"lam={LAM} n={N_G}\n"
          f"guided angle theta_d = {theta_d:.3f} deg (x = {x:.5f})\n")

    print("== reciprocity: in-coupler eta(+1) vs out-coupler eta(-1) ==")
    for pol in ("s", "p"):
        fwd = grating_orders_rcwa(PERIOD, DEPTH, duty=DUTY, wavelength_nm=LAM,
                                  n_sub=N_G, pol=pol, theta_deg=0.0)
        rev = outcoupler_rcwa(PERIOD, DEPTH, DUTY, LAM, N_G, theta_d, pol)
        e_f, e_r = fwd["T1"], rev.get(-1, 0.0)
        lab = "TE" if pol == "s" else "TM"
        print(f"  {lab}:  eta_in(+1) = {e_f:.6f}   eta_out(-1) = {e_r:.6f}"
              f"   |diff| = {abs(e_f - e_r):.2e}")

    print("\n== nG convergence AT THE RECORD (v3 only checked the scalar-"
          "optimal design) ==")
    for pol in ("s", "p"):
        lab = "TE" if pol == "s" else "TM"
        vals = []
        for nG in (41, 61, 81, 101):
            r = grating_orders_rcwa(PERIOD, DEPTH, duty=DUTY,
                                    wavelength_nm=LAM, n_sub=N_G, pol=pol,
                                    theta_deg=0.0, nG=nG)
            vals.append(r["T1"])
        drift = max(vals) - min(vals)
        print(f"  {lab}: T(+1) @ nG 41/61/81/101 = "
              + " ".join(f"{v:.6f}" for v in vals) + f"   spread {drift:.1e}")


if __name__ == "__main__":
    main()
