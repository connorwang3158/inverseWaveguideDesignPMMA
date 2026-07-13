"""
Two-arm optimization for the PMMA metagrating in-coupler study
(SPEC_DECISIONS_METAGRATING.md — all four locked decisions implemented).

  Arm 1 (baseline): parametric slanted SRG {slant, duty, depth, period},
         differential evolution + Nelder-Mead polish (derivative-free polish
         because the staircase parameterization is piecewise-constant).
  Arm 2 (frontier): freeform rho(x,z) topology optimization, Adam on autograd
         gradients, beta-ramped WLS projection, robust eroded/nominal/dilated
         smooth-min objective (D4), multi-start, discrete period grid (D3).

Every headline number is the BINARIZED re-simulation (A3), never the relaxed
grayscale field. Live progress -> metagrating_live.html (auto-refreshing);
per-metric hall of fame -> metagrating_hof.json; results table ->
metagrating_results.csv; best freeform design -> metagrating_best_topo.json.

Usage:
  python3 optimize_metagrating.py --smoke            # reduced budgets, ~10 min
  python3 optimize_metagrating.py                    # production (overnight)
  python3 optimize_metagrating.py --arm srg|topo     # one arm only
"""

import argparse
import csv
import json
import os
import time

import autograd.numpy as np
from autograd import grad
from scipy import optimize as sopt

from metagrating_model import (
    NL, NX, H_BUDGET, W_MIN, LAM_MIN, LAM_MAX, NG_DEFAULT, ROBUST_ETAS,
    filter_matrix, occupancy, binarize, solve_orders, eta_m1,
    objective_robust,
)
import metagrating_dashboard as dash

RESULTS_CSV = "metagrating_results.csv"


# --------------------------- shared reporting --------------------------------

def report_design(occ_bin, period, nG):
    """Binarized re-simulation (A3): TE/TM/unpol eta_{-1} + energy residual."""
    te = solve_orders(occ_bin, period, "TE", nG)
    tm = solve_orders(occ_bin, period, "TM", nG)
    return {
        "eta_TE": float(te["orders"].get(-1, 0.0)),
        "eta_TM": float(tm["orders"].get(-1, 0.0)),
        "eta_unpol": float(0.5 * (te["orders"].get(-1, 0.0)
                                  + tm["orders"].get(-1, 0.0))),
        "eta_p1_TE": float(te["orders"].get(1, 0.0)),   # +1 order (asymmetry)
        "energy_residual": float(max(te["energy_residual"],
                                     tm["energy_residual"])),
    }


def append_result(row, header):
    new = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)


HEADER = ["arm", "period_nm", "seed", "start", "eta_TE_bin", "eta_TM_bin",
          "eta_unpol_bin", "eta_TE_robust_worst", "eta_p1_TE",
          "asym_ratio_m1_p1", "energy_residual", "params_json", "when"]


# ------------------------------ SRG baseline ---------------------------------

def occ_srg(slant_deg, duty, depth, period):
    """Slanted-SRG occupancy on the shared [NL,NX] grid: teeth of height
    `depth` growing up from the substrate (no-overhang by construction),
    lateral shift tan(slant)*z across the tooth height."""
    occ = np.zeros((NL, NX))
    xs = np.arange(NX) / NX
    for k in range(NL):                       # k=0 top of relief region
        z_from_top = (k + 0.5) * H_BUDGET / NL
        if z_from_top < H_BUDGET - depth:     # above the teeth: air
            continue
        z_in_tooth = z_from_top - (H_BUDGET - depth)   # 0 at tooth top
        shift = np.tan(np.deg2rad(slant_deg)) * z_in_tooth / period
        occ[k] = (((xs - shift) % 1.0) < duty).astype(float)
    return occ


def run_srg(seed=0, nG=NG_DEFAULT, maxiter=25, popsize=12, live=True):
    """Differential evolution over {slant, duty, depth, period} (D3 bounds),
    then Nelder-Mead polish; returns the binarized report."""
    bounds = [(-50.0, 50.0), (0.2, 0.8), (60.0, H_BUDGET), (LAM_MIN, LAM_MAX)]
    hist, it = [], [0]

    def f(p):
        eta = eta_m1(occ_srg(*p), p[3], "TE", nG)
        it[0] += 1
        if live and it[0] % 5 == 0:
            hist.append(float(eta) if not hist else max(hist[-1], float(eta)))
            dash.update({"arm": "SRG (DE)", "period": f"{p[3]:.0f}",
                         "seed": seed, "start": "-", "iteration": it[0],
                         "total_iters": "?", "beta": "-", "history": hist,
                         "occ": occ_srg(*p).tolist(), "energy_residual": 0.0,
                         "note": "baseline arm: differential evolution"})
        return -eta

    de = sopt.differential_evolution(f, bounds, seed=seed, maxiter=maxiter,
                                     popsize=popsize, polish=False, tol=1e-6)
    nm = sopt.minimize(f, de.x, method="Nelder-Mead",
                       options={"maxfev": 120, "xatol": 1e-3, "fatol": 1e-6})
    p = nm.x if nm.fun < de.fun else de.x
    slant, duty, depth, period = [float(v) for v in p]
    occ_b = (occ_srg(slant, duty, depth, period) > 0.5).astype(float)
    rep = report_design(occ_b, period, nG)
    params = {"slant_deg": slant, "duty": duty, "depth_nm": depth,
              "period_nm": period}
    print(f"[SRG] eta_TE={100*rep['eta_TE']:.3f}%  eta_TM={100*rep['eta_TM']:.3f}%  "
          f"asym={rep['eta_TE']/(rep['eta_p1_TE']+1e-12):.2f}  {params}")
    for m in ("eta_TE", "eta_TM", "eta_unpol"):
        dash.record(m, rep[m], "SRG", params)
    append_result(["SRG", f"{period:.2f}", seed, 0,
                   f"{rep['eta_TE']:.6f}", f"{rep['eta_TM']:.6f}",
                   f"{rep['eta_unpol']:.6f}", "-",
                   f"{rep['eta_p1_TE']:.6f}",
                   f"{rep['eta_TE']/(rep['eta_p1_TE']+1e-12):.3f}",
                   f"{rep['energy_residual']:.2e}",
                   json.dumps(params), time.strftime("%F %T")], HEADER)
    return rep, params


# -------------------------- topology optimization ----------------------------

def adam_step(x, g, m, v, t, lr, b1=0.9, b2=0.999, eps=1e-8):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mh, vh = m / (1 - b1 ** t), v / (1 - b2 ** t)
    return x + lr * mh / (np.sqrt(vh) + eps), m, v   # ASCENT (maximize)


def run_topo(period, seed=0, start=0, n_iters=300, nG=NG_DEFAULT, lr=0.05,
             live=True):
    """One topology-optimization run at fixed period (D3 grid point)."""
    rng = np.random.RandomState(seed * 100 + start)
    Fmat = filter_matrix(NX, period)
    psi = rng.randn(NL, NX) * 0.3
    gradf = grad(lambda p, b: objective_robust(p, Fmat, period, b, "TE", nG))
    m = np.zeros_like(psi); v = np.zeros_like(psi)
    betas = [4.0, 8.0, 16.0, 32.0, 64.0]
    hist = []
    for i in range(n_iters):
        beta = betas[min(i * len(betas) // n_iters, len(betas) - 1)]
        g = gradf(psi, beta)
        psi, m, v = adam_step(psi, g, m, v, i + 1, lr)
        if live and (i % 5 == 0 or i == n_iters - 1):
            occ_now = occupancy(psi, Fmat, beta, 0.5)
            r = solve_orders(occ_now, period, "TE", nG)
            hist.append(float(r["orders"].get(-1, 0.0)))
            dash.update({"arm": "topology (freeform)", "period": period,
                         "seed": seed, "start": start, "iteration": i + 1,
                         "total_iters": n_iters, "beta": beta,
                         "history": hist, "occ": occ_now.tolist(),
                         "energy_residual": float(r["energy_residual"]),
                         "note": "robust smooth-min objective (eroded/nominal/dilated)"})
    # A3: binarized re-simulation is the headline; also worst-of-triple robust
    occ_b = binarize(psi, Fmat, 0.5)
    rep = report_design(occ_b, period, nG)
    robust_worst = min(
        float(eta_m1(binarize(psi, Fmat, e), period, "TE", nG))
        for e in ROBUST_ETAS)
    params = {"period_nm": period, "n_iters": n_iters, "nG": nG,
              "W_MIN": W_MIN, "H_BUDGET": H_BUDGET, "NL": NL}
    print(f"[topo Λ={period:.0f} s{start}] bin eta_TE={100*rep['eta_TE']:.3f}%  "
          f"robust-worst={100*robust_worst:.3f}%  eta_TM={100*rep['eta_TM']:.3f}%")
    hit = False
    for mname in ("eta_TE", "eta_TM", "eta_unpol"):
        hit |= dash.record(mname, rep[mname], "topology", params)
    if hit:
        with open("metagrating_best_topo.json", "w") as f:
            json.dump({"occupancy": occ_b.tolist(), "period_nm": period,
                       "report": rep, "params": params}, f)
    append_result(["topology", f"{period:.2f}", seed, start,
                   f"{rep['eta_TE']:.6f}", f"{rep['eta_TM']:.6f}",
                   f"{rep['eta_unpol']:.6f}", f"{robust_worst:.6f}",
                   f"{rep['eta_p1_TE']:.6f}",
                   f"{rep['eta_TE']/(rep['eta_p1_TE']+1e-12):.3f}",
                   f"{rep['energy_residual']:.2e}",
                   json.dumps(params), time.strftime("%F %T")], HEADER)
    return rep, psi


# ---------------------------------- main -------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="reduced budgets")
    ap.add_argument("--arm", choices=["both", "srg", "topo"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        nG, iters, starts = 41, 60, 1
        periods = (410.0, 440.0)
        de_iter, de_pop = 6, 6
    else:
        nG, iters, starts = NG_DEFAULT, 300, 3
        periods = (380.0, 410.0, 440.0, 470.0, 500.0)
        de_iter, de_pop = 25, 12

    print(f"config: nG={nG} topo iters={iters} starts={starts} "
          f"periods={periods} seed={args.seed}")
    if args.arm in ("both", "srg"):
        run_srg(seed=args.seed, nG=nG, maxiter=de_iter, popsize=de_pop)
    if args.arm in ("both", "topo"):
        for period in periods:
            for s in range(starts):
                run_topo(period, seed=args.seed, start=s, n_iters=iters, nG=nG)
    print(f"\nresults -> {RESULTS_CSV} | hall of fame -> metagrating_hof.json"
          f" | live view -> metagrating_live.html")


if __name__ == "__main__":
    main()
