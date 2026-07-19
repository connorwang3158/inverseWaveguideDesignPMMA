# Referee-Report Execution — Stage 0 + Stage 1 Results

**Repo:** `connorwang3158/inverseWaveguideDesignPMMA` · **Executed:** 2026-07-17
**Scope:** Stage 0 diagnostics (all six), Stage 1 physics corrections (engine → **v4**), reciprocity solve, second-solver cross-check, gate re-optimization. Grid rebuild (Stage 2) and retraining (Stage 3) are compute-bound and deliberately not run — see "What remains."

---

## 1. Stage 0 verdicts — the report was mostly right, wrong twice

| Referee claim | Verdict | Measured |
|---|---|---|
| §2.1 chrom metric = internal guided-angle spread | ✅ **Confirmed** | θ_R−θ_B = 28.826° vs reported 28.827° |
| §2.3 T_FOV ≡ 0.94·T | ✅ **Confirmed** | ratio 0.9401 ± 0.0006 over 1000 designs |
| §2.2 record parked on both sigmoid edges | ✅ **Confirmed** | B at +0.83 widths above TIR edge (θᵢ=0); R at +0.83 widths **past evanescence** (θᵢ=5°) |
| §2.9 N_b and ℓ are 2× short | ✅ **Confirmed** (by inspection: `L/(2t·tanθ)` in code) |
| §2.10 effective dimensionality ≪ 8 | ✅ **Confirmed, stronger than claimed** | Sobol total-effect: period 0.78, n 0.65, depth 0.18, duty 0.08; **σ, α, t, Lc all 0.000**. Effective dim = 4. The 0.90–1.96 mm thickness "agreement" is a flat direction. |
| §2.4 MTF_sys = 0.784 is MTF_chrom's random-phase mean | ❌ **Refuted** | At the record: MTF_diff 0.847 × rough 1.000 × **chrom 0.973** × grat 0.954 × coup 0.997 = 0.784. The 0.7836 coincidence was with the *product*, not the chromatic term. (The 0.543 floor and fringe-lottery structure are real, but weren't biting at the record because `RESID_DISP=0.10` shrank the phases.) |
| §2.10 S(L_c) ≈ 1 always | ❌ **Refuted (wrong range)** | PMMA mode samples Lc = 200–400 µm, so S = 0.43–0.60, not ≈1. The *conclusion* still holds — Lc is inert — but because σ·S·N_b is negligible, not because S≈1. |
| §2.2 hard mask collapses the score | ⚠️ **Did not materialize** | Hard-mask re-score of top-10 finalists is bit-identical, because the transmission cascade is green-only and green sits mid-window. The exploit is real but expresses through the *chromatic metric and field edge*, not through T. |

**Gate (per §5):** claim (2) confirmed → spec vector rebuilt before any retraining. ✔

## 2. Stage 1 — physics engine v4 (all changes in `physics/waveguide_physics.py`, diff attached)

- **FIX-7 (§2.9):** `N_b = L/(t·tanθ_d)`, path `= L/sinθ_d`. Both factor-of-2 errors removed.
- **FIX-8 (§2.1):** `chromatic_spread_deg` is now the **non-cancelling LED-bandwidth blur**, photopic-weighted `rad2deg(Δλ_FWHM/(n_k·Λ·cosθ_d,k))` with Δλ = 25 nm. The old internal spread survives as `internal_angle_spread_deg()` (diagnostic only).
- **FIX-9 (§2.4):** MTF_chrom uses **finite-bandwidth primaries** (Gaussian per-line envelopes) — floor and fringe lottery removed. This exposed a second bug: `RESID_DISP=0.10` is physically indefensible (matched couplers cancel dispersion *exactly*, including within-band, for collimated input; a 10% residual would give MTF ≈ 10⁻⁶). Re-derived as the in/out **period-mismatch tolerance** (NIL ≈ 0.1% → 0.001), documented in-code.
- **FIX-10 (§2.2):** `tir_penalty` now evaluates at **0 and ±FOV_DEG** with a 0.01 guard band (v3 checked normal incidence only — that's how red got parked past evanescence at the field edge). New `hard_guided_ok()` audit helper, no relaxation.
- **FIX-11 (§2.8):** **n is a material property.** Design n = n(532 nm); B/R indices follow PMMA Sellmeier offsets (Sultanova 2009: 1.5006/1.4937/1.4886) in PMMA mode. The optimizer can no longer give red an index PMMA doesn't have — exactly where x<n binds.
- `ENGINE_VERSION → "v4"`; the physics-probe system will (correctly) invalidate every existing checkpoint.

## 3. Verification runs

**Reciprocity (§2.5), `physics/verify_reciprocity.py`:** substrate-side solve (PMMA superstrate, θ_d = 52.76°, air substrate) at the record geometry:
TE η_in(+1) = η_out(−1) = 0.156165 (|Δ| = 5×10⁻⁷); TM 0.030813 vs 0.030786 (|Δ| = 3×10⁻⁵). **The out-coupler = in-coupler assumption is now verified, not assumed.** Convergence re-checked *at the record*: TE spread 10⁻⁶ over nG 41–101; TM 6×10⁻⁵ and still creeping at nG=101 (the "slow plateau" warning is mildly real; fine at 3 decimals).

**Second solver (§2.13), `physics/crosscheck_solver_pmma.py`:** grcwa vs meent, 60 solves (10 geometries × RGB × TE/TM) concentrated at the record, window edges, and worst-case corners. **PASS**: worst |Δ| = 2.8×10⁻³ (deep/high-duty TM corner); ~10⁻⁴ near the record; TE ≤ 3×10⁻⁵ everywhere.

## 4. Gate result — the paper's headline finding survives

Multi-start gradient search (300×400) on the exact v4 physics:

| | v3 record | v4 record |
|---|---|---|
| depth | 199.5 nm | **194.9 nm** ✅ interior optimum survives |
| period | 448.13 nm (sigmoid edge) | 445.56 nm (off the edge — exploit closed) |
| T (unpol) | 1.25 % | 1.17 % |
| chrom metric | 28.8° (cancelled quantity) | 4.21° (LED-bandwidth blur) |
| MTF | 0.784 | 0.664 |
| TE/TM at 5° | — | T_TE 2.11 % vs T_TM 0.08 %, diattenuation 0.93 → the **TE-only result (§2.6) reproduces and strengthens** |
| top-5 diversity | 2.2× thickness spread | single basin (norm. std 0.019); thickness remains a flat direction |

Two new quantitative findings for the revision:
1. **The ±5° spec is physically unsatisfiable in PMMA.** With real dispersion the widest simultaneous-RGB window is ≈4.7° *total*; the optimizer carries an irreducible TIR penalty. The honest field half-angle is ≈±2.3°, exactly the report's §2.2 point 3, made worse by FIX-11 (red's index is 1.4886, not 1.500).
2. Sobol table (Stage 0, D6) is ready to publish: 4 live parameters, 4 inert — the "8-parameter tangled space" framing must go.

## 5. What remains (not runnable in this session)

- **Stage 2:** θᵢ-axis calibration grid (kills the invented `A(θ)` Gaussian and makes T_FOV independent — until then it stays 0.94·T and the "four-metric" claim must be dropped), period axis 5→15, depth 77→20, optional slant axis. Hours of RCWA compute.
- **Stage 3:** retrain surrogate + tandem + neural-adjoint on v4 (all checkpoints are now stale by design), 5 seeds × 2 arms, honest baselines, wall-clock speedup number — or take reframing path (A).
- **Manuscript edits** (not in this repo): the 25× → 8.9× claim, §IV-A vs Table II, `luo2025` → Dehghani et al., abstract/figures/scope box, why-PMMA answer, new references.

## Files

- `stage0_diagnostics.py` — all six diagnostics, re-runnable
- `physics/waveguide_physics.py` — v4 engine (see `v4_physics_changes.diff`)
- `physics/verify_reciprocity.py`, `physics/crosscheck_solver_pmma.py`
- `results/best_design_ever_v4.csv`, `results/optimal_designs.csv` — v4 record + top-5
