"""
Cross-solver validation (spec 5.6.2): the same structure solved in grcwa
(primary) and meent (independent), agreeing within tolerance.

Stage 0 auto-detects meent's ucell convention (refractive index vs
permittivity) on a bare Fresnel interface where the analytic answer is known,
then Stage 1 compares transmitted-order efficiencies for a binary PMMA
grating and Stage 2 for a freeform-style multilayer pattern.

Usage: python3 crosscheck_meent.py
"""

import numpy as np

from metagrating_model import (N_PMMA, EPS_PMMA, NL, NX, H_BUDGET,
                               solve_orders, filter_matrix, occupancy)

TOL = 5e-3      # absolute efficiency agreement tolerance per order
LAM = 532.0


def meent_orders(occ, period, fto=40, pol=0, ucell_mode="n"):
    """Solve with meent: air top, PMMA bottom, NL grating sublayers.
    Returns dict order -> transmitted efficiency."""
    import meent
    val_hi = N_PMMA if ucell_mode == "n" else EPS_PMMA
    ucell = 1.0 + (val_hi - 1.0) * np.asarray(occ)[:, None, :]   # [NL,1,NX]
    mee = meent.call_mee(
        backend=0, pol=pol, n_top=1.0, n_bot=N_PMMA,
        theta=1e-10, phi=0.0, wavelength=LAM,
        period=[period], fto=[fto, 0],
        thickness=[H_BUDGET / NL] * NL, ucell=ucell)
    res = mee.conv_solve()                    # ResultNumpy object
    de_ri, de_ti = res.de_ri, res.de_ti
    ti = np.asarray(de_ti).flatten().real
    c = len(ti) // 2
    return ({m: float(ti[c + m]) for m in range(-3, 4)},
            float(np.sum(de_ri) + np.sum(de_ti)))


def detect_ucell_mode(period=440.0):
    """Bare interface (occ all zeros -> uniform air layer on PMMA):
    T(0th) must equal the Fresnel transmittance 4n/(n+1)^2."""
    T_analytic = 4 * N_PMMA / (1 + N_PMMA) ** 2
    occ0 = np.zeros((NL, NX))
    best, best_err = None, 1e9
    for mode in ("n", "eps"):
        try:
            orders, tot = meent_orders(occ0, period, ucell_mode=mode)
            err = abs(orders[0] - T_analytic)
            print(f"  ucell as {mode:3s}: T0={orders[0]:.5f} "
                  f"(analytic {T_analytic:.5f}, err {err:.1e}, R+T={tot:.6f})")
            if err < best_err:
                best, best_err = mode, err
        except Exception as e:
            print(f"  ucell as {mode}: FAILED ({e})")
    assert best_err < 1e-3, "meent Fresnel sanity failed for both conventions"
    return best


def compare(occ, period, mode, label):
    g_te = solve_orders(occ, period, "TE", nG=81)
    m_te, m_tot = meent_orders(occ, period, fto=40, pol=0, ucell_mode=mode)
    print(f"\n{label} (TE, period {period:.0f} nm)")
    print(f"{'order':>6} {'grcwa':>10} {'meent':>10} {'|diff|':>9}")
    worst = 0.0
    for m in (-2, -1, 0, 1, 2):
        gv = float(g_te["orders"].get(m, 0.0)); mv = m_te.get(m, 0.0)
        worst = max(worst, abs(gv - mv))
        print(f"{m:6d} {gv:10.5f} {mv:10.5f} {abs(gv-mv):9.1e}")
    ok = worst < TOL
    print(f"worst |diff| = {worst:.2e}  -> {'PASS' if ok else 'FAIL'} @ {TOL}")
    return ok


def main():
    print("Stage 0: meent ucell convention on a bare Fresnel interface")
    mode = detect_ucell_mode()
    print(f"  -> using ucell as {mode!r}")

    ok = True
    # Stage 1: binary grating (duty 0.5, depth = full budget)
    occ_bin = np.zeros((NL, NX)); occ_bin[:, : NX // 2] = 1.0
    ok &= compare(occ_bin, 440.0, mode, "Stage 1: binary grating")

    # Stage 2: freeform-style multilayer pattern (seeded, filtered, projected)
    rng = np.random.RandomState(3)
    Fmat = filter_matrix(NX, 440.0)
    occ_ff = np.asarray(occupancy(rng.randn(NL, NX) * 2.0, Fmat, 64.0, 0.5))
    occ_ff = (occ_ff > 0.5).astype(float)
    ok &= compare(occ_ff, 440.0, mode, "Stage 2: freeform multilayer")

    print(f"\nCROSS-SOLVER: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
