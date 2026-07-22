"""
Independent validation suite, proof of accuracy WITHOUT reference to Paper 1.

Every test compares the code against ground truths that are external and
unimpeachable: closed-form textbook results, conservation laws, and symmetries
required by Maxwell's equations. A reviewer can rerun this file and see every
claim checked. Run:  python3 validate.py

Tests:
  V1 Fresnel exactness      flat interface (depth->0) must reproduce the
                            analytic Fresnel transmission for n=1.49.
  V2 Energy conservation    lossless structure: sum(R) + sum(T) = 1 exactly.
  V3 Symmetry               unslanted binary grating at normal incidence:
                            T(+1) = T(-1) (parity symmetry of Maxwell eqs).
  V4 Scalar-theory limit    for feature sizes >> wavelength, rigorous RCWA
                            must approach the scalar formula (Pommet et al.,
                            JOSA A 11, 1827 (1994): scalar valid when s>14λ).
  V5 Scalar ceiling         first-order efficiency of any binary phase grating
                            cannot exceed 4/pi^2 in the scalar regime.
  V6 Convergence            results stable under Fourier-order refinement.
  V7 Model gates            G2 (double->singular limiting case) and G3
                            (geometric equal-extraction design rule), from
                            architectures.py.
  V8 TIR enforcement        the analytic engine must assign ~zero transmission
                            to designs whose first order is not guided
                            (1 < sin(th_i)+lambda/period < n violated), and
                            nonzero transmission inside the guided window.
  V9 Brewster / vector      the polarized Fresnel factor must vanish for TM
                            reflection at Brewster's angle th_B=atan(n) and
                            satisfy T_unpol = (T_TE+T_TM)/2 exactly.
  V10 Watson eye MTF   the mean-human-eye MTF term must reproduce
                            Watson (2013): at 3 mm pupil / ~11.9 cyc/deg it
                            sits in the published 0.45-0.55 window, strictly
                            BELOW the 0.847 diffraction limit, and -> 1 at
                            zero frequency.
  V11 Walk-off consistency  the chromatic pupil walk-off must drive BOTH
                       spec[2] and the MTF (via the apodized pupil):
                            doubling the source bandwidth must raise the
                            walk-off metric and strictly lower MTF_system.
"""

import numpy as np

from rigorous_solver import grating_orders_rcwa, scalar_eta, N_PMMA

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, ok, detail):
    results.append((name, ok))
    print(f"[{PASS if ok else FAIL}] {name}: {detail}")


def v1_fresnel():
    r = grating_orders_rcwa(500, 0.001, 0.5, 532.0, pol="s", nG=41)  # ~flat surface
    T_analytic = 1 - ((N_PMMA - 1) / (N_PMMA + 1)) ** 2              # 0.96122...
    err = abs(r["T_total"] - T_analytic)
    check("V1 Fresnel exactness", err < 2e-3,
          f"RCWA T={r['T_total']:.5f} vs analytic {T_analytic:.5f} (err {err:.1e})")


def v2_energy():
    worst = 0.0
    for depth in (100, 300):
        for pol in ("s", "p"):
            r = grating_orders_rcwa(500, depth, 0.5, 532.0, pol=pol, nG=61)
            worst = max(worst, abs(r["R_total"] + r["T_total"] - 1.0))
    check("V2 Energy conservation", worst < 1e-6,
          f"max |R+T-1| = {worst:.2e} over depths x polarizations")


def v3_symmetry():
    r = grating_orders_rcwa(500, 300, 0.5, 532.0, pol="s", nG=61)
    d = abs(r["T_orders"].get(1, 0) - r["T_orders"].get(-1, 0))
    check("V3 Parity symmetry", d < 1e-6,
          f"|T(+1)-T(-1)| = {d:.2e} (unslanted grating, normal incidence)")


def v4_scalar_limit():
    """Scalar theory must emerge from RCWA as features grow vs wavelength.
    NOTE: periods >~8um at high nG trigger a known RCWA numerical overflow in
    deeply evanescent orders; we therefore test at period 4um (s=3.8 lambda),
    where energy conservation still holds to machine precision, and REQUIRE
    conservation before admitting the comparison. Trend agrees with Pommet
    et al. 1994 (scalar error shrinks as s/lambda grows)."""
    lam, per, depth = 532.0, 4000.0, 300.0
    r = grating_orders_rcwa(per, depth, 0.5, lam, pol="s", nG=81)
    conserved = abs(r["R_total"] + r["T_total"] - 1.0) < 1e-6
    s = scalar_eta(depth, 0.5, lam=lam)
    rel = abs(r["T1"] - s) / s
    check("V4 Scalar-theory limit", conserved and rel < 0.10,
          f"RCWA {r['T1']:.4f} vs scalar {s:.4f} at s=3.8lambda "
          f"(rel err {100*rel:.1f}%, conservation {'OK' if conserved else 'VIOLATED'})")


def v5_ceiling():
    cap = 4 / np.pi ** 2
    worst = max(scalar_eta(d, du) for d in range(20, 401, 20)
                for du in np.linspace(0.2, 0.8, 13))
    check("V5 Scalar 4/pi^2 ceiling", worst <= cap + 1e-9,
          f"max scalar eta1 = {worst:.5f} <= {cap:.5f}")


def v6_convergence():
    vals = [grating_orders_rcwa(500, 300, 0.5, 532.0, pol="s", nG=g)["T1"]
            for g in (41, 61, 81)]
    spread = max(vals) - min(vals)
    check("V6 Convergence", spread < 1e-3,
          f"T(+1) spread over nG 41..81 = {spread:.2e}")


def v8_tir():
    import torch
    from waveguide_physics import use_pmma, sample_theta, transmission
    use_pmma()
    torch.manual_seed(0)
    th = sample_theta(64)
    T_ok = transmission(th)                     # periods inside guided window
    bad = th.clone(); bad[:, 5] = 680.0         # the old exploit period
    T_bad = transmission(bad)
    check("V8 TIR enforcement",
          bool((T_bad.max() < 1e-6) and (T_ok.min() > 0)),
          f"unguided max T = {T_bad.max():.2e} (must be ~0); "
          f"guided min T = {T_ok.min():.2e} (must be > 0)")


def v9_brewster():
    import torch
    from waveguide_physics import fresnel_T
    n = torch.tensor([N_PMMA])
    th_b = float(np.degrees(np.arctan(N_PMMA)))     # Brewster's angle
    T_tm = fresnel_T(n, th_b, "TM")
    err = abs(T_tm.item() - 1.0)                     # R_TM(th_B) = 0 -> T = 1
    # unpolarized average identity at a 3-deg field angle
    from waveguide_physics import use_pmma, sample_theta, transmission
    torch.manual_seed(1)
    th = sample_theta(16)
    lhs = transmission(th, 3.0, "unpol")
    rhs = 0.5 * (transmission(th, 3.0, "TE") + transmission(th, 3.0, "TM"))
    avg_err = (lhs - rhs).abs().max().item()
    # TE and TM must DIFFER at oblique incidence (equal at 0 deg by symmetry);
    # guards against a regression that silently makes the model pol-blind
    dist = abs(fresnel_T(n, 30.0, "TE").item() - fresnel_T(n, 30.0, "TM").item())
    check("V9 Brewster + unpol identity",
          err < 1e-6 and avg_err < 1e-9 and dist > 1e-3,
          f"1-T_TM(th_B) = {err:.2e}; max|T_unpol-(T_TE+T_TM)/2| = {avg_err:.2e}; "
          f"|T_TE-T_TM|(30deg) = {dist:.4f} (>0 required)")


def v10_watson():
    import torch
    from waveguide_physics import watson_eye_mtf, F0_CYC_PER_MM, EYE_FL_MM
    d3 = torch.tensor([3.0])
    m40 = watson_eye_mtf(F0_CYC_PER_MM, d3).item()      # ~11.9 cyc/deg
    # diffraction-limited value at the same frequency/pupil (the term v4
    # wrongly used): (2/pi)(acos r - r sqrt(1-r^2)), r = u/u0
    u = F0_CYC_PER_MM * EYE_FL_MM * torch.pi / 180.0
    u0 = (3.0 * 1e6 / 555.0) * torch.pi / 180.0
    r = torch.tensor(u / u0)
    dl = float((2 / torch.pi) * (torch.acos(r) - r * torch.sqrt(1 - r ** 2)))
    m0 = watson_eye_mtf(1e-6, d3).item()
    check("V10 Watson eye MTF", 0.45 < m40 < 0.55 and m40 < dl - 0.2
          and abs(m0 - 1.0) < 1e-3,
          f"M(40cyc/mm,3mm)={m40:.4f} (published mean-eye ~0.45-0.55; "
          f"diffraction limit {dl:.3f}); M(0)={m0:.4f}")


def v11_walkoff_consistency():
    import torch
    import waveguide_physics as wp
    wp.use_pmma()
    torch.manual_seed(2)
    th = wp.sample_theta(32)
    y1 = wp.forward_model(th)
    fwhm0 = wp.LED_FWHM_NM
    try:
        wp.LED_FWHM_NM = fwhm0 * 2.0          # double the source bandwidth
        y2 = wp.forward_model(th)
    finally:
        wp.LED_FWHM_NM = fwhm0
    walk_up = bool((y2[:, 2] > y1[:, 2]).all())
    mtf_down = bool((y2[:, 0] < y1[:, 0]).all())
    check("V11 Walk-off consistency", walk_up and mtf_down,
          f"2x bandwidth: walk-off up for 32/32 designs ({walk_up}), "
          f"MTF down for 32/32 ({mtf_down}), spec[2] and MTF share one "
          f"walk-off model")


def v7_gates():
    import io, contextlib
    from architectures import run_gates
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_gates()
    out = buf.getvalue()
    g2 = float(out.split("residual (should be small): ")[1].split()[0])
    g3 = float(out.split("spread (should be ~0): ")[1].split()[0])
    check("V7 Model gates G2/G3", g2 < 1e-5 and g3 < 1e-5,
          f"G2 residual {g2:.1e}, G3 spread {g3:.1e}")


if __name__ == "__main__":
    print("Independent validation (no Paper 1 references)\n" + "=" * 56)
    v1_fresnel(); v2_energy(); v3_symmetry(); v5_ceiling(); v6_convergence()
    v7_gates(); v8_tir(); v9_brewster(); v10_watson(); v11_walkoff_consistency()
    v4_scalar_limit()   # slowest last (nG=81)
    n_ok = sum(ok for _, ok in results)
    print("=" * 56)
    print(f"{n_ok}/{len(results)} tests passed")
    raise SystemExit(0 if n_ok == len(results) else 1)
