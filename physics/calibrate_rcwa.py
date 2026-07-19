"""
v3 physics calibration: rigorous RCWA grid behind the engine's coupling term.

Builds physics/rcwa_eta_grid.npz — the lookup table behind the v3 engine's
RCWA-calibrated grating-coupling term (waveguide_physics.eta_rcwa): the
first-order transmitted efficiency of the binary PMMA in-coupler, solved
VECTORIALLY with grcwa (Moharam & Gaylord RCWA) on a dense grid over the PMMA
design window:

    n (1.48..1.50) x period (430..449 nm) x depth (20..400 nm)
    x duty (0.2..0.8) x wavelength (450/532/635 nm) x polarization (TE, TM)

Why (2026-07-13 audit, results/design_rcwa_check_na.csv): at the TIR-mandated
PMMA periods (430-449 nm, BELOW the visible wavelengths), scalar diffraction
theory overestimates first-order coupling ~5x at 532 nm and ~15x at 635 nm,
and puts the depth optimum at the wrong place (scalar keeps climbing to the
400 nm bound; rigorous efficiency peaks near 200 nm). Pommet et al., JOSA A
11, 1827 (1994) quantify exactly this scalar-theory failure at ~lambda-scale
features. Fitting the engine's coupling term to rigorous solves removes that
error class at the source, and makes the coupling polarization-resolved
(scalar theory is pol-blind).

The refractive index gets its own (3-point) axis rather than being pinned at
the 1.49 midpoint: measurement showed eta_TM moves ~20% across the PMMA
bounds n in [1.48, 1.50] at the record geometry (TE is nearly insensitive),
so treating n as a residual would have rebuilt a smaller version of the very
error class this calibration removes.

After building, the script verifies the engine's trilinear interpolant
against FRESH grcwa solves at random OFF-GRID points and records the
comparison in results/rcwa_calibration_check.csv — that file is the L2
evidence that the interpolation error is negligible next to the ~5-15x
scalar error it replaces.

Usage (from the project root):
    python3 physics/calibrate_rcwa.py             # full grid + verification
    python3 physics/calibrate_rcwa.py --quick     # coarse smoke grid (do NOT
                                                  # commit a quick grid)
Outputs:
    physics/rcwa_eta_grid.npz            the calibration table (committed)
    results/rcwa_calibration_check.csv   off-grid interpolant-vs-RCWA audit
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import res_path
from physics.rigorous_solver import NG, grating_orders_rcwa

# Grid axes. Depth is the structured dimension (Kogelnik-like oscillation of
# eta with phase depth) and gets the densest sampling; eta varies slowly over
# the narrow TIR-mandated period window and smoothly with duty; the n axis
# captures the ~20% TM sensitivity across the PMMA index bounds.
NS = np.linspace(1.48, 1.50, 3)
PERIODS = np.linspace(430.0, 449.0, 5)
DEPTHS = np.linspace(20.0, 400.0, 77)          # 5 nm steps
DUTIES = np.linspace(0.2, 0.8, 13)             # 0.05 steps
WLS = np.array([450.0, 532.0, 635.0])          # RGB primaries, match WL
POLS = ("s", "p")                              # s=TE, p=TM (engine order)
# NG (Fourier orders) comes from rigorous_solver; its convergence_check shows
# T(+1) stable to 5 decimals already at nG=41 for this window
GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "rcwa_eta_grid.npz")
N_OFFGRID = 48   # random off-grid verification solves


def _init_worker(ns, periods, depths, duties):
    """Install the parent's grid axes in the worker. Required for --quick
    correctness on macOS/Windows: their spawn start method re-imports this
    module in each worker, which would silently restore the FULL axes while
    the parent hands out quick-grid indices — every solve would then run at
    the wrong coordinates. (Linux fork inherits the rebinding and never
    noticed.)"""
    global NS, PERIODS, DEPTHS, DUTIES
    NS, PERIODS, DEPTHS, DUTIES = ns, periods, depths, duties


def _solve_one(task):
    """One rigorous solve: (pol, wl, n, period, depth, duty) indices -> T(+1)."""
    ipol, iwl, ien, iper, idep, idut = task
    r = grating_orders_rcwa(PERIODS[iper], DEPTHS[idep], DUTIES[idut],
                            WLS[iwl], n_sub=NS[ien], pol=POLS[ipol], nG=NG)
    return task, r["T1"]


def _solve_point(n, period, depth, duty, wl, pol):
    return grating_orders_rcwa(period, depth, duty, wl, n_sub=n, pol=pol,
                               nG=NG)["T1"]


def build_grid(n_procs=None):
    tasks = [(ipol, iwl, ien, iper, idep, idut)
             for ipol in range(len(POLS))
             for iwl in range(len(WLS))
             for ien in range(len(NS))
             for iper in range(len(PERIODS))
             for idep in range(len(DEPTHS))
             for idut in range(len(DUTIES))]
    eta = np.zeros((len(POLS), len(WLS), len(NS), len(PERIODS), len(DEPTHS),
                    len(DUTIES)))
    n_procs = n_procs or max(os.cpu_count() - 1, 1)
    print(f"solving {len(tasks)} rigorous RCWA problems (nG={NG}) "
          f"on {n_procs} processes ...")
    t0, done = time.time(), 0
    with Pool(n_procs, initializer=_init_worker,
              initargs=(NS, PERIODS, DEPTHS, DUTIES)) as pool:
        for task, t1 in pool.imap_unordered(_solve_one, tasks, chunksize=8):
            eta[task] = t1
            done += 1
            if done % 200 == 0 or done == len(tasks):
                rate = done / (time.time() - t0)
                print(f"  {done}/{len(tasks)}  ({rate:.1f} solves/s, "
                      f"~{(len(tasks)-done)/rate/60:.1f} min left)")
    return eta


def verify_interpolant(eta, rng):
    """Compare the ENGINE's trilinear interpolant (the exact code every
    training label and design search will run through) against fresh rigorous
    solves at random off-grid points."""
    import torch
    from physics.waveguide_physics import _interp_eta

    grid = {"ns": torch.tensor(NS), "periods": torch.tensor(PERIODS),
            "depths": torch.tensor(DEPTHS), "duties": torch.tensor(DUTIES),
            "eta": torch.tensor(eta)}
    grid["eta_unpol"] = grid["eta"].mean(dim=0)

    rows, errs = [], []
    print(f"\nverifying interpolant at {N_OFFGRID} random off-grid points ...")
    for _ in range(N_OFFGRID):
        en = rng.uniform(NS[0], NS[-1])
        per = rng.uniform(PERIODS[0], PERIODS[-1])
        dep = rng.uniform(DEPTHS[0], DEPTHS[-1])
        dut = rng.uniform(DUTIES[0], DUTIES[-1])
        iwl = rng.integers(0, len(WLS))
        ipol = rng.integers(0, len(POLS))
        exact = _solve_point(en, per, dep, dut, WLS[iwl], POLS[ipol])
        approx = _interp_eta(grid, torch.tensor([en]), torch.tensor([per]),
                             torch.tensor([dep]), torch.tensor([dut]),
                             wl_idx=int(iwl), pol=("TE", "TM")[ipol]).item()
        errs.append(abs(approx - exact))
        rows.append([f"{en:.4f}", f"{per:.2f}", f"{dep:.2f}", f"{dut:.4f}",
                     f"{WLS[iwl]:.0f}", ("TE", "TM")[ipol],
                     f"{exact:.6f}", f"{approx:.6f}", f"{errs[-1]:.6f}"])
    errs = np.array(errs)
    print(f"interpolant vs rigorous RCWA: mean |err| {errs.mean():.5f}, "
          f"max |err| {errs.max():.5f}  (efficiencies span 0..{eta.max():.3f})")

    out = res_path("rcwa_calibration_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "period_nm", "depth_nm", "duty", "wavelength_nm",
                    "pol", "rcwa_eta1", "interpolated_eta1", "abs_error"])
        w.writerows(rows)
        w.writerow([])
        w.writerow(["mean_abs_error", f"{errs.mean():.6f}"])
        w.writerow(["max_abs_error", f"{errs.max():.6f}"])
    print(f"saved off-grid audit -> {out}")
    return float(errs.mean()), float(errs.max())


def main():
    global DEPTHS, DUTIES, PERIODS, N_OFFGRID
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="coarse smoke grid — never commit this")
    ap.add_argument("--procs", type=int, default=None)
    args = ap.parse_args()
    if args.quick:
        PERIODS, DEPTHS, DUTIES = (np.linspace(430, 449, 3),
                                   np.linspace(20, 400, 6),
                                   np.linspace(0.2, 0.8, 3))
        N_OFFGRID = 6

    t0 = time.time()
    eta = build_grid(args.procs)
    mean_err, max_err = verify_interpolant(eta, np.random.default_rng(20260717))

    meta = json.dumps({
        "built": datetime.date.today().isoformat(), "nG": NG,
        "n_axis": list(NS), "profile": "binary", "incidence_deg": 0.0,
        "quick": args.quick, "offgrid_mean_abs_err": round(mean_err, 6),
        "offgrid_max_abs_err": round(max_err, 6)})
    np.savez_compressed(GRID_PATH, ns=NS, periods=PERIODS, depths=DEPTHS,
                        duties=DUTIES, wls=WLS, eta=eta, meta=np.array(meta))
    print(f"\nsaved calibration grid -> {GRID_PATH}")
    print(f"meta: {meta}")
    print(f"total {((time.time()-t0)/60):.1f} min")
    if args.quick:
        print("WARNING: --quick grid written. Rebuild the full grid before "
              "trusting any training run or record.")


if __name__ == "__main__":
    main()
