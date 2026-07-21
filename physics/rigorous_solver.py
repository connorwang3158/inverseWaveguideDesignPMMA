"""
Rigorous electromagnetic solver layer (L2) for AR waveguide gratings.

Wraps grcwa — an established open-source Rigorous Coupled-Wave Analysis (RCWA)
implementation (W. Jin et al., "Inverse design of lightweight broadband
reflector...", and grcwa docs: grcwa.readthedocs.io) — to solve Maxwell's
equations VECTORIALLY for the in-coupler grating. This replaces the scalar
approximation in waveguide_physics.py wherever paper claims are made (claims
ladder level L2, see architecture_framework.md).

What this adds over the scalar model:
  * true polarization physics: independent TE (s) and TM (p) efficiencies
  * exact treatment of deep/high-contrast gratings where scalar theory fails
  * arbitrary grating PROFILES via staircase discretization:
      binary, blazed (sawtooth), sinusoidal, slanted, trapezoidal
  * efficiency per diffraction order, any incidence angle, conical mounts

Solver landscape (what exists beyond this file — documented so nothing is
hidden): RCWA/FMM is the standard for PERIODIC structures (this file; also
torcwa [Kim & Lee, Comp. Phys. Comm. 282:108552, 2023] for GPU, meent, S4).
For NON-periodic / arbitrary 3D shapes use FDTD (Meep [Oskooi et al., Comp.
Phys. Comm. 181:687, 2010], Lumerical) or FEM (COMSOL, JCMsuite). For the
mm-scale light transport (TIR bounces, pupil expansion) use ray tracing
(Zemax OpticStudio) — no single tool spans both scales; the ladder in
architecture_framework.md is the standard industrial workflow.

Speed: seconds per wavelength/polarization at nG=81 — fine for verification,
NOT for training loops (train on the analytic engine, verify here).

Usage:
    python3 rigorous_solver.py          # runs the scalar-vs-RCWA validation study
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import res_path

try:
    import grcwa
except ImportError as e:
    raise SystemExit("pip3 install grcwa   (required for rigorous solving)") from e

N_PMMA = 1.49           # PMMA refractive index (Nilsen et al. 2025 range midpoint)
NG = 61                 # Fourier orders retained; convergence_check() shows
                        # T(+1) stable to 5 decimals already at nG=41

# Representative period for the scalar-vs-RCWA validation sweep. This MUST sit
# inside the TIR-guided PMMA window (~430-449 nm, waveguide_physics FIX-1); the
# old default of 500 nm is OUTSIDE that window and does not guide RGB, so a
# "validation" there does not describe the regime any headline design uses.
# (The large-period approach to scalar theory is separately covered by
# validate.py V4.) Kept equal to the record period.
GUIDED_PERIOD_NM = 438.2
PERIOD_Y_NM = 50.0      # dummy sub-wavelength y-period (1D grating in 2D solver)


# ---------------------------------------------------------------------------
# Grating profile builders: fraction of PMMA (vs air) at each lateral x and
# each staircase sub-layer z. All return [n_sublayers, Nx] occupancy in {0,1}.
# ---------------------------------------------------------------------------

def profile_binary(Nx=256, n_sub=1, duty=0.5, **_):
    g = np.zeros((n_sub, Nx)); g[:, : int(round(duty * Nx))] = 1.0
    return g

def profile_blazed(Nx=256, n_sub=16, **_):
    """Sawtooth: tooth height rises linearly across the period (staircase)."""
    g = np.zeros((n_sub, Nx)); x = np.linspace(0, 1, Nx, endpoint=False)
    for i in range(n_sub):                      # i=0 is the TOP sub-layer
        g[i, x >= (i + 0.5) / n_sub] = 1.0      # material fills where tooth is tall
    return g

def profile_sinusoidal(Nx=256, n_sub=16, **_):
    g = np.zeros((n_sub, Nx)); x = np.linspace(0, 1, Nx, endpoint=False)
    h = 0.5 * (1 + np.sin(2 * np.pi * x))       # normalized height 0..1
    for i in range(n_sub):
        g[i, h >= (i + 0.5) / n_sub] = 1.0
    return g

def profile_slanted(Nx=256, n_sub=16, duty=0.5, slant_deg=30.0, depth_nm=300.0,
                    period_nm=500.0, **_):
    """Parallelogram teeth tilted by slant_deg (staircase of shifted binaries)."""
    g = np.zeros((n_sub, Nx))
    shift_frac = np.tan(np.deg2rad(slant_deg)) * depth_nm / period_nm
    for i in range(n_sub):
        s = ((i + 0.5) / n_sub) * shift_frac
        idx = (np.arange(Nx) / Nx - s) % 1.0 < duty
        g[i, idx] = 1.0
    return g

PROFILES = {"binary": profile_binary, "blazed": profile_blazed,
            "sinusoidal": profile_sinusoidal, "slanted": profile_slanted}


# ---------------------------------------------------------------------------
# Core RCWA call
# ---------------------------------------------------------------------------

def grating_orders_rcwa(period_nm, depth_nm, duty=0.5, wavelength_nm=532.0,
                        n_sub=N_PMMA, pol="s", theta_deg=0.0, profile="binary",
                        nG=NG, Nx=256, n_sublayers=None, **prof_kw):
    """Solve the full vectorial diffraction problem for one grating.

    Geometry (in-coupling): plane wave from AIR above -> corrugated PMMA
    surface (grating layer, staircase of n_sublayers) -> semi-infinite PMMA
    substrate (the waveguide slab). Transmitted order m=+1 propagating inside
    PMMA is the coupled beam.

    Returns dict: {'T_orders': {m: eff}, 'R_total', 'T_total', 'T1'} —
    efficiencies normalized to incident power (grcwa RT_Solve(normalize=1)).
    """
    if n_sublayers is None:
        n_sublayers = 1 if profile == "binary" else 16
    lam, eps_sub = float(wavelength_nm), n_sub ** 2

    obj = grcwa.obj(nG, [period_nm, 0.0], [0.0, PERIOD_Y_NM],
                    1.0 / lam, theta_deg * np.pi / 180.0, 0.0, verbose=0)
    obj.Add_LayerUniform(100.0, 1.0)                       # air superstrate
    occ = PROFILES[profile](Nx=Nx, n_sub=n_sublayers, duty=duty,
                            depth_nm=depth_nm, period_nm=period_nm, **prof_kw)
    Ny = 8  # y is uniform (1D grating); >1 needed by grcwa's 2D FFT machinery
    for _ in range(n_sublayers):                           # grating staircase
        obj.Add_LayerGrid(depth_nm / n_sublayers, Nx, Ny)
    obj.Add_LayerUniform(100.0, eps_sub)                   # PMMA substrate
    obj.Init_Setup()

    ep_all = np.concatenate([
        np.tile((1.0 + (eps_sub - 1.0) * occ[i])[:, None], (1, Ny)).flatten()
        for i in range(n_sublayers)])
    obj.GridLayer_geteps(ep_all)

    if pol == "s":   amps = dict(p_amp=0.0, s_amp=1.0)
    elif pol == "p": amps = dict(p_amp=1.0, s_amp=0.0)
    else:            amps = dict(p_amp=np.sqrt(0.5), s_amp=np.sqrt(0.5))
    obj.MakeExcitationPlanewave(amps["p_amp"], 0.0, amps["s_amp"], 0.0, order=0)

    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    orders = {}
    for i, (gx, gy) in enumerate(obj.G):
        if gy == 0 and abs(gx) <= 3:
            orders[int(gx)] = orders.get(int(gx), 0.0) + float(np.real(Ti[i]))
    return {"T_orders": orders, "R_total": float(np.real(np.sum(Ri))),
            "T_total": float(np.real(np.sum(Ti))),
            "T1": orders.get(1, 0.0) + 0.0}


def eta_unpolarized(period_nm, depth_nm, duty=0.5, wavelength_nm=532.0, **kw):
    """First-order coupling efficiency, unpolarized = mean of TE and TM."""
    te = grating_orders_rcwa(period_nm, depth_nm, duty, wavelength_nm, pol="s", **kw)
    tm = grating_orders_rcwa(period_nm, depth_nm, duty, wavelength_nm, pol="p", **kw)
    return 0.5 * (te["T1"] + tm["T1"]), te["T1"], tm["T1"]


# ---------------------------------------------------------------------------
# Validation study: scalar model vs rigorous RCWA  (gate G4)
# ---------------------------------------------------------------------------

def scalar_eta(depth_nm, duty, n=N_PMMA, lam=532.0):
    phi = 2 * np.pi * depth_nm * (n - 1.0) / lam
    return 4.0 * (np.sin(np.pi * duty) / np.pi) ** 2 * np.sin(phi / 2) ** 2


def convergence_check(nGs=(41, 61, 81, 101)):
    print("convergence (eta_TE at 500nm period, 300nm depth):")
    for g in nGs:
        r = grating_orders_rcwa(500, 300, 0.5, 532.0, pol="s", nG=g)
        print(f"  nG={g:4d}  T(+1)={r['T1']:.5f}")


def verify_designs(csv_in=None, csv_out=None,
                   wavelengths=(450.0, 532.0, 635.0)):
    """Rigorous (vector, per-polarization) check of the analytic engine's top
    designs: for each design row, solve the in-coupler with RCWA at normal
    incidence for RGB and record TE/TM first-order efficiencies next to the
    scalar prediction. This is the L2 evidence backing every headline design."""
    import csv
    csv_in = csv_in or res_path("optimal_designs.csv")
    csv_out = csv_out or res_path("design_rcwa_check.csv")
    if not os.path.exists(csv_in):
        print(f"{csv_in} not found — run baselines/optimize_pmma.py first"); return
    with open(csv_in) as f:
        rows = list(csv.DictReader(f))
    out = []
    print(f"RCWA verification of {len(rows)} designs x {len(wavelengths)} wavelengths")
    print(f"{'rank':>4} {'lam':>5} {'scalar':>8} {'TE':>8} {'TM':>8} {'unpol':>8}")
    for r in rows:
        n = float(r["n"]); per = float(r["period(nm)"])
        dep = float(r["depth(nm)"]); dut = float(r["duty"])
        for lam in wavelengths:
            s = scalar_eta(dep, dut, n=n, lam=lam)
            unp, te, tm = eta_unpolarized(per, dep, dut, lam, n_sub=n)
            out.append([r["rank"], lam, per, dep, dut, s, te, tm, unp])
            print(f"{r['rank']:>4} {lam:5.0f} {s:8.4f} {te:8.4f} {tm:8.4f} {unp:8.4f}")
    with open(csv_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "wavelength_nm", "period_nm", "depth_nm", "duty",
                    "scalar_eta1", "rcwa_TE", "rcwa_TM", "rcwa_unpol"])
        w.writerows(out)
    print(f"saved -> {csv_out}")


def main():
    import csv
    convergence_check()
    per = GUIDED_PERIOD_NM
    print(f"\nScalar vs RCWA (binary PMMA grating, {per:.1f}nm period [guided window], "
          f"532nm, normal incidence)")
    print(f"{'depth':>6} {'scalar':>8} {'RCWA-TE':>8} {'RCWA-TM':>8} {'unpol':>8}")
    rows = []
    for depth in (50, 100, 150, 200, 250, 300, 350, 400):
        s = scalar_eta(depth, 0.5)
        unp, te, tm = eta_unpolarized(per, depth, 0.5)
        rows.append([per, depth, s, te, tm, unp])
        print(f"{depth:6d} {s:8.4f} {te:8.4f} {tm:8.4f} {unp:8.4f}")
    with open(res_path("rcwa_validation.csv"), "w", newline="") as f:
        w = csv.writer(f)
        # period_nm is now an explicit column so the geometry is self-documenting
        # and this file can never again be mistaken for the record regime.
        w.writerow(["period_nm", "depth_nm", "scalar_eta1",
                    "rcwa_TE", "rcwa_TM", "rcwa_unpol"])
        w.writerows(rows)
    print("saved -> results/rcwa_validation.csv")

    print(f"\nProfile comparison at period={per:.1f}nm, depth=300nm "
          f"(unpolarized first-order):")
    for prof in PROFILES:
        unp, te, tm = eta_unpolarized(per, 300, 0.5, profile=prof)
        print(f"  {prof:11s} eta1={unp:.4f}  (TE {te:.4f} / TM {tm:.4f})")


if __name__ == "__main__":
    import sys
    # outputs are keyed on the physics ENGINE_VERSION so L2 evidence files
    # from different engines never overwrite each other (the v2-era files
    # design_rcwa_check_na.csv / design_rcwa_check.csv are kept as-is)
    from waveguide_physics import ENGINE_VERSION
    if "--designs-na" in sys.argv:
        # L2 check of the neural-adjoint winners
        verify_designs(csv_in=res_path("optimal_designs_na.csv"),
                       csv_out=res_path(
                           f"design_rcwa_check_na_{ENGINE_VERSION}.csv"))
    elif "--designs" in sys.argv:
        # L2 check of optimize_pmma.py winners
        verify_designs(csv_out=res_path(
            f"design_rcwa_check_{ENGINE_VERSION}.csv"))
    else:
        main()
