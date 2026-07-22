"""
Differentiable RCWA forward model for the PMMA metagrating in-coupler study.

Implements the locked spec (SPEC_DECISIONS_METAGRATING.md):
  D1  Freeform arm = multilayer density rho(x,z) over NL sublayers within the
      shared depth budget H_BUDGET, with an EXACT no-overhang constraint via a
      monotone cumulative parameterization (material can only grow downward,
      so any material voxel has material beneath it -> nanoimprint-releasable).
  D2  TE is the primary objective; TM and unpolarized are always reported.
  D3  Period bounds LAM_MIN..LAM_MAX shared by both arms (532 nm guided
      window: lambda/n < Lambda < lambda). The freeform arm scans a discrete
      period grid; the SRG arm optimizes Lambda continuously in the same range.
  D4  Robust formulation: eroded/nominal/dilated projection thresholds
      (0.7/0.5/0.3); robust objective = smooth-min over the triple.

Engine: grcwa with the autograd backend (differentiable; energy conservation
verified to ~1e-14 in validate.py). meent is the independent cross-check
(crosscheck_meent.py), the spec's engine-swap clause applies (grcwa primary).

Design-field pipeline (all steps autograd-differentiable and monotone-
preserving, so the no-overhang guarantee survives filtering & projection):

    psi[NL,Nx] (raw logits)
      -> sigmoid                         s_k(x) in (0,1)      "growth events"
      -> circulant cone filter along x   (min feature size W_MIN)
      -> cumulative o_k = 1 - prod_{j<=k}(1 - s_j)   monotone in k (D1)
      -> Wang-Lazarov-Sigmund projection (beta, eta)  -> occupancy in (0,1)
      -> eps(x) = 1 + (n^2-1)*occupancy  -> grcwa GridLayer -> eta_{-1}

Physics anchors: RCWA, Moharam & Gaylord (1981), Moharam et al. (1995);
factorization, Li (1996); topology optimization with filter+projection, 
Wang, Lazarov & Sigmund, Struct. Multidisc. Optim. 43, 767 (2011); robust
eroded/nominal/dilated, Wang et al., and standard photonic TO practice
(Jensen & Sigmund, Laser Photonics Rev. 5, 308 (2011)).

Units: nm for lengths; wavelength 532 nm.
"""

import autograd.numpy as np
from autograd import grad
import grcwa

grcwa.set_backend('autograd')

# ---------------------------- shared platform --------------------------------
LAM = 532.0                 # design wavelength (nm), D-spec
N_PMMA = 1.4936             # PMMA at 532 nm (dispersion-corrected; cite primary
                            # Sellmeier source at manuscript time)  # TODO-cite
EPS_PMMA = N_PMMA ** 2
LAM_MIN, LAM_MAX = 370.0, 500.0   # matched period bounds (D3)
H_BUDGET = 400.0            # relief depth budget (nm), imprint-realizable
                            # ceiling  # TODO-cite NIL literature
W_MIN = 40.0                # minimum feature size (nm)  # TODO-cite NIL
NL = 6                      # sublayers of the relief region (rho(x,z), D1)
NX = 128                    # design pixels per period
NY = 8                      # dummy y-resolution (1D grating in 2D solver)
PERIOD_Y = 50.0             # sub-wavelength dummy y-period
NG_DEFAULT = 81             # Fourier orders (freeform needs more than binary;
                            # confirmed by convergence_study())
ROBUST_ETAS = (0.7, 0.5, 0.3)   # eroded / nominal / dilated thresholds (D4)


# ------------------------- design-field pipeline -----------------------------

def filter_matrix(nx=NX, period=440.0, w_min=W_MIN):
    """Circulant normalized cone (hat) filter along x, radius w_min/2."""
    dx = period / nx
    r = max(1, int(round(w_min / 2.0 / dx)))
    base = np.zeros(nx)
    for j in range(-r, r + 1):
        base[j % nx] = max(0.0, 1.0 - abs(j) / (r + 1))
    F = np.stack([np.roll(base, i) for i in range(nx)])
    return F / F.sum(axis=1, keepdims=True)


def project(x, beta, eta):
    """Wang-Lazarov-Sigmund threshold projection (monotone, differentiable)."""
    return (np.tanh(beta * eta) + np.tanh(beta * (x - eta))) / (
        np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta)))


def occupancy(psi, Fmat, beta, eta):
    """psi[NL,Nx] -> occupancy[NL,Nx] in (0,1), monotone increasing with
    depth (EXACT no-overhang, D1). Row 0 = top of relief."""
    s = 1.0 / (1.0 + np.exp(-psi))            # (0,1)
    s = np.dot(s, Fmat.T)                     # length-scale filter along x
    # cumulative growth: o_k = 1 - prod_{j<=k} (1 - s_j)  (monotone in k)
    log1m = np.log(1.0 - s * (1.0 - 1e-9))
    o = 1.0 - np.exp(np.cumsum(log1m, axis=0))
    return project(o, beta, eta)


def binarize(psi, Fmat, eta=0.5):
    """Hard-threshold occupancy (beta -> inf limit) for final re-simulation."""
    return (occupancy(psi, Fmat, beta=512.0, eta=eta) > 0.5).astype(float)


# ------------------------------- RCWA solve ----------------------------------

def _build(period, nG, thicknesses):
    obj = grcwa.obj(nG, [period, 0.0], [0.0, PERIOD_Y], 1.0 / LAM, 0.0, 0.0,
                    verbose=0)
    obj.Add_LayerUniform(100.0, 1.0)               # air superstrate
    for th in thicknesses:
        obj.Add_LayerGrid(th, NX, NY)
    obj.Add_LayerUniform(100.0, EPS_PMMA)          # PMMA substrate
    obj.Init_Setup()
    return obj


def solve_orders(occ, period, pol="TE", nG=NG_DEFAULT):
    """occ[NL,Nx] occupancy -> dict with per-order transmitted efficiencies
    into PMMA, R/T totals, and the energy-conservation residual.
    Differentiable w.r.t. occ (autograd flows through GridLayer_geteps)."""
    obj = _build(period, nG, [H_BUDGET / NL] * NL)
    eps_rows = 1.0 + (EPS_PMMA - 1.0) * occ                     # [NL,Nx]
    ep_all = np.concatenate([np.tile(eps_rows[k][:, None], (1, NY)).flatten()
                             for k in range(NL)])
    obj.GridLayer_geteps(ep_all)
    if pol == "TE":
        obj.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
    else:                                                        # TM
        obj.MakeExcitationPlanewave(1.0, 0.0, 0.0, 0.0, order=0)
    Ri, Ti = obj.RT_Solve(normalize=1, byorder=1)
    orders = {}
    for i, (gx, gy) in enumerate(obj.G):
        if gy == 0 and abs(gx) <= 3:
            orders[int(gx)] = orders.get(int(gx), 0.0) + Ti[i]
    Rt, Tt = np.sum(Ri), np.sum(Ti)
    return {"orders": orders, "R": Rt, "T": Tt,
            "energy_residual": np.abs(Rt + Tt - 1.0)}


def eta_m1(occ, period, pol="TE", nG=NG_DEFAULT):
    """Transmitted -1-order efficiency into the guided mode (the objective)."""
    return solve_orders(occ, period, pol, nG)["orders"].get(-1, 0.0)


# --------------------------- robust objective (D4) ---------------------------

def smooth_min(vals, tau=50.0):
    v = np.array(vals)
    return -np.log(np.sum(np.exp(-tau * v))) / tau


def objective_robust(psi, Fmat, period, beta, pol="TE", nG=NG_DEFAULT,
                     etas=ROBUST_ETAS):
    """Robust objective: smooth-min of eta_{-1} over the eroded/nominal/
    dilated triple. Maximize this."""
    return smooth_min([eta_m1(occupancy(psi, Fmat, beta, e), period, pol, nG)
                       for e in etas])


def objective_nominal(psi, Fmat, period, beta, pol="TE", nG=NG_DEFAULT):
    return eta_m1(occupancy(psi, Fmat, beta, 0.5), period, pol, nG)


# ------------------------------ rigor checks ---------------------------------

def fd_gradient_check(period=440.0, n_checks=4, eps=1e-4, seed=0, nG=41):
    """A3: autograd gradient vs central finite differences at a random psi.
    Returns worst relative error over sampled coordinates."""
    rng = np.random.RandomState(seed)
    psi = rng.randn(NL, NX) * 0.5
    Fmat = filter_matrix(NX, period)
    f = lambda p: objective_nominal(p, Fmat, period, beta=8.0, nG=nG)
    g = grad(f)(psi)
    worst = 0.0
    idx = [(rng.randint(NL), rng.randint(NX)) for _ in range(n_checks)]
    for (i, j) in idx:
        e = np.zeros((NL, NX)); e[i, j] = eps
        fd = (f(psi + e) - f(psi - e)) / (2 * eps)
        rel = abs(fd - g[i, j]) / (abs(fd) + 1e-12)
        worst = max(worst, rel)
        print(f"  dJ/dpsi[{i},{j}]  autograd {g[i,j]:+.6e}  FD {fd:+.6e}  "
              f"rel {rel:.2e}")
    return worst


def convergence_study(period=440.0, seed=1, nGs=(41, 61, 81, 101, 121)):
    """A3: Fourier-order convergence ON A FREEFORM PATTERN at W_MIN scale."""
    rng = np.random.RandomState(seed)
    Fmat = filter_matrix(NX, period)
    occ = occupancy(rng.randn(NL, NX) * 2.0, Fmat, beta=64.0, eta=0.5)
    print(f"freeform convergence (period {period} nm, W_MIN {W_MIN} nm):")
    vals = []
    for g in nGs:
        r = solve_orders(occ, period, "TE", nG=g)
        vals.append(r["orders"].get(-1, 0.0))
        print(f"  nG={g:4d}  eta(-1)={vals[-1]:.6f}  |R+T-1|={r['energy_residual']:.1e}")
    return dict(zip(nGs, vals))


if __name__ == "__main__":
    print("=== FD gradient check (A3) ===")
    worst = fd_gradient_check()
    print(f"worst relative error: {worst:.2e}  "
          f"({'PASS' if worst < 1e-3 else 'FAIL'} @ 1e-3)")
    print("\n=== Freeform Fourier-order convergence (A3) ===")
    convergence_study()
