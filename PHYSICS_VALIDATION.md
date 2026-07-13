# Physics Validation & Revision Record (v2, 2026-07-12)

Rigorous, equation-level justification of the forward model in
`waveguide_physics.py` (v2), the audit findings that forced the revision, and
the vector-electromagnetic verification of the resulting designs. Every claim
below is reproducible: `python3 validate.py` (9 independent tests) and
`python3 rigorous_solver.py --designs` (exact Maxwell solutions).

---

## 1. Audit findings (what was wrong in v1)

### F1 — CRITICAL: the TIR guiding condition was never enforced

A diffractive waveguide only works if the first-order beam is trapped by total
internal reflection. From the grating equation and Snell's law, the guided
window in dimensionless tangential-wavevector form is

```
1  <  sin(θᵢ) + λ/Λ  <  n                                            (1)
```

with the left inequality being TIR at the slab faces (n·sinθ_d > 1) and the
right being propagation of the diffracted order (sinθ_d < 1). The v1 engine
imposed neither. Consequence: the optimizer drove the period to Λ ≈ 682 nm,
where **all three wavelengths sit 3–16° below the critical angle
(41.8° for n = 1.49)** — the "designs" in the previous `pareto_results.csv`
did not guide light at all. Numerically (n = 1.4999, Λ = 681.8 nm):

| λ (nm) | sinθ_d | θ_d | θ_c | guided? |
|---|---|---|---|---|
| 450 | 0.440 | 26.1° | 41.8° | no |
| 532 | 0.520 | 31.4° | 41.8° | no |
| 635 | 0.621 | 38.4° | 41.8° | no |

**Fix:** smooth sigmoid mask on (1) multiplying the transmission; a
differentiable `tir_penalty()` for optimizers; PMMA period bounds narrowed to
the full-RGB window **λ_red/n < Λ < λ_blue → 429–450 nm**. Validation test V8
confirms unguided designs now transmit < 10⁻²⁰.

Applying (1) across field angles gives the *index-limited FOV*: the common
RGB window for Λ = 438 nm, n = 1.499 is **[−1.5°, +2.9°] ≈ 4.4° wide** — the
well-known FOV penalty of low-index substrates (Kress & Chatterjee 2021).
The v1 metric "T at 20° field" was physically unreachable for every full-RGB
PMMA design; the PMMA-mode FOV metric now evaluates at 5°.

### F2 — Field-angle grating equation missing

v1 evaluated the grating at normal incidence for every field angle. v2 uses
the exact planar-mount momentum conservation (k-space vector form):

```
k_x,out = k_x,in + mK,  K = 2π/Λ   ⟹   n·sinθ_d = sinθᵢ + mλ/Λ,  m = +1   (2)
```

### F3 — Polarization was averaged away (now vector-resolved)

v1 collapsed s/p immediately via `(1 − (Rs+Rp)/2)²`. v2 computes the exact
Fresnel amplitude coefficients per polarization (Hecht ch. 4; Born & Wolf):

```
r_TE = (cosθᵢ − n·cosθ_t)/(cosθᵢ + n·cosθ_t)
r_TM = (n·cosθᵢ − cosθ_t)/(n·cosθᵢ + cosθ_t)                         (3)
T_pol = 1 − r_pol²  (per interface);  cascade carries TE and TM separately
T_unpol = (T_TE + T_TM)/2      — average of POWER transmissions
```

Verified by test V9: R_TM(θ_B) = 0 at Brewster's angle θ_B = atan(n) to
machine precision, and the unpolarized identity holds exactly. Note that at
normal incidence T_TE = T_TM by symmetry — the *grating* polarization
splitting is a vector-diffraction effect beyond any scalar model, and is
quantified by RCWA below (§3).

### F4 — Bounce count was a constant 10

The number of TIR bounces over coupler separation L is geometric:

```
N_b = L / (2·t·tanθ_d)                                               (4)
```

so thin waveguides bounce (and scatter) more. v1's constant made ∂loss/∂t
wrong in sign over most of the design space. v2 uses (4) with L = 20 mm,
clamped to [1, 60]. The v2 optimizer correspondingly discovers t → 2 mm
(fewest bounces → least roughness/bulk loss), a physically interpretable
result.

### F5 — Ad-hoc floors removed

The v1 efficiency floor (0.02 + 0.98η) let a zero-depth grating couple 2% of
light, and the angular acceptance floor (0.3 + 0.7·detune) kept 30% coupling
at arbitrary detuning. Both removed; the acceptance Gaussian now applies to
the in-coupler only (the out-coupler sees the guided angle by construction).

### F6 — Chromatic MTF upgraded from ad-hoc Gaussian to exact 3-line PSF

```
MTF_chrom(f) = | Σ_k w_k · exp(i·2πf·x_k) |,   x_k = f_eye·κ·Δθ_k     (5)
```

with photopic weights w = V(450):V(532):V(635) = 0.034:0.771:0.194 and κ the
residual-dispersion fraction (0.10, documented calibration constant; matched
in/out couplers cancel most in-guide dispersion). Equation (5) is the exact
modulus of the optical transfer function of a three-delta PSF (cf. Thibos
1987) and reduces to 1 as the spread → 0.

---

## 2. The transmission cascade (v2, per polarization)

```
T_pol(θᵢ) = M_TIR · T_F,pol² · exp(−α·ℓ) · exp(−N_b·(4πσn·cosθ_d/λ)²·S(L_c))
            · [η₁·A(θᵢ)] · η₁                                          (6)

ℓ = N_b·t/cosθ_d            (zig-zag path length)
η₁ = 4(sin πD/π)²·sin²(φ/2),  φ = 2πd(n−1)/λ    (scalar binary phase grating,
                                                  ceiling 4/π² ≈ 40.5%)
A(θᵢ) = exp(−(sinθᵢ/0.35)²)                      (in-coupler acceptance)
S(L_c) = 1/(1 + L_c/3·10⁵)                       (Payne–Lacey-motivated weight)
```

Anchors: Fresnel — Hecht/Born & Wolf; Beer–Lambert bulk; per-bounce roughness
loss — Tien, *Appl. Opt.* **10**, 2395 (1971); correlation-length weighting —
Payne & Lacey, *Opt. Quantum Electron.* **26**, 977 (1994); scalar grating
efficiency — Goodman, *Fourier Optics*. Remaining L1 heuristics are flagged
`# SYNC` in code (S(L_c), acceptance width, roughness/grating/coupler MTF
coefficients, κ) and must be reconciled against Paper 1 before publication
claims.

---

## 3. Vector electromagnetic verification (RCWA, exact Maxwell)

`rigorous_solver.py` wraps grcwa (RCWA per Moharam & Gaylord 1981; Moharam et
al. 1995) and passes: Fresnel exactness to 3.5·10⁻¹², energy conservation
R+T = 1 to 2.5·10⁻¹⁴, ±1-order parity symmetry to 1.8·10⁻¹⁵, Fourier-order
convergence to 3.8·10⁻⁶, and approach to scalar theory at large Λ/λ (3.5% at
s = 3.8λ, consistent with Pommet et al. 1994).

**Where scalar theory breaks (and why it matters here).** The guided-window
constraint forces Λ ≈ 438 nm ≈ 0.8λ_green — deep in the regime where Pommet
et al. (1994) prove scalar theory fails (valid only for features ≳ 14λ).
RCWA at the optimizer's winning design (Λ = 438.2 nm, D = 0.5, n = 1.4992,
λ = 532 nm):

| depth (nm) | scalar η₁ | RCWA TE | RCWA TM | RCWA unpol |
|---|---|---|---|---|
| 100 | 0.034 | 0.076 | 0.022 | 0.049 |
| **200** | 0.125 | **0.138** | 0.033 | **0.085** |
| 300 | 0.243 | 0.092 | 0.035 | 0.063 |
| 400 | 0.346 | 0.048 | 0.088 | 0.068 |
| 450 | 0.382 | 0.055 | **0.094** | 0.075 |

Three rigorous conclusions:

1. **The scalar-optimal depth (400 nm, φ→π) is NOT the vector optimum.** For
   unpolarized light the true optimum is ≈ 200 nm (η = 8.5% vs 6.8%); the
   scalar model overestimates η at 400 nm by ~5×.
2. **Polarization splitting is large: TE/TM ≈ 4.2 at 200 nm depth.** With a
   TE-polarized source (standard for LED/LCoS projectors), the in-coupler
   reaches η_TE = 13.8% — polarization management is a first-order design
   lever, consistent with Zhao et al. (2024).
3. **L2-corrected headline design:** Λ = 438.2 nm, d ≈ 200 nm, D = 0.5,
   t ≈ 2 mm, TE-polarized input. The tandem network still trains on the L1
   engine (differentiable, milliseconds); headline numbers must quote the
   RCWA column (claims-ladder L2, per `architecture_framework.md`).

Full tables: `design_rcwa_check.csv` (5 designs × RGB × TE/TM),
`rcwa_depth_sweep.csv` (depth scan at the winning period).

---

## 4. Revised results snapshot (v2 physics, PMMA mode)

| quantity | v1 (broken) | v2 (physical) |
|---|---|---|
| optimal period | 681.8 nm (not guided!) | 438.2 nm (RGB guided) |
| transmission (L1 scalar) | 37.6% | 11.0% |
| transmission (L2 RCWA-corrected, unpol/TE) | — | ~0.7% / ~1.9% (η² basis) |
| chromatic spread | 8.9° (of unguided rays) | 31.9° (real, in-guide) |
| full-RGB FOV window | unconstrained (fictitious) | 4.4° (index-limited) |
| TIR feasibility | violated at all λ | satisfied at all λ (V8) |

The v2 numbers are less flattering and correct: single-layer full-RGB PMMA
waveguides are severely index-limited — which is precisely the quantitative
story (and the honest contribution) the paper can now defend. Validation:
**9/9 tests pass** (`validate.py`): V1 Fresnel, V2 energy conservation,
V3 parity, V4 scalar limit, V5 4/π² ceiling, V6 convergence, V7 architecture
gates, V8 TIR enforcement, V9 Brewster + unpolarized identity.
