# Physics Validation & Revision Record (current engine: v5, 2026-07-18)

Rigorous, equation-level justification of the forward model in
`waveguide_physics.py`, the audit findings that forced each revision, and
the vector-electromagnetic verification of the resulting designs. Sections
1–4 are the v2-era record (2026-07-12), kept for provenance; **Section 5 is
the v5 revision** (2026-07-18 independent physics-accuracy review) and
describes the engine as it now runs. Every claim below is reproducible:
`python3 validate.py` (11 independent tests) and
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
constraint forces Λ ≈ 438 nm ≈ 0.8λ_green. Pommet et al. (1994) report >±5%
scalar error below ~14λ features as the *worst case* (the error is smallest
near 50% fill factor, where our gratings sit), so the a-priori argument alone
would be soft — but we do not rely on it: we **measured** the scalar error at
our geometry with RCWA, and it is 5–15×.
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
story (and the honest contribution) the paper can now defend.

---

## 5. v5 revision (2026-07-18) — response to the independent accuracy review

An independent physics-accuracy review (2026-07-18) re-derived every
load-bearing equation from first principles. It confirmed the hard physics
(TIR window + edge enforcement, grating equation, polarized Fresnel, RCWA
calibration, Tien roughness, bounce geometry, Sellmeier dispersion) and found
the remaining problems concentrated in the system-metric layer. v5 fixes all
of them (FIX-12…FIX-17 in `waveguide_physics.py`):

### 5.1 Eye MTF is now actually Watson (2013)  — FIX-12, review §2.1

The v4 "Watson" term was the aberration-free diffraction-limited pupil MTF
(0.847 at 40 cyc/mm). v5 implements Watson's real mean-human-eye formula:

```
M(u,d) = √D(u,d,555nm) · [1 + (u/u₁(d))²]^(−0.62)
u₁(d)  = 21.95 − 5.512·d + 0.3922·d²      (u in cyc/deg, d in mm, valid 2–6)
```

At 3 mm / 11.9 cyc/deg this gives **0.4884** (test V10 pins it inside the
published 0.45–0.55 mean-eye window and strictly below the diffraction
limit). The whole MTF product drops accordingly — the v4 headline 0.78 was
inflated by ~0.35 of pure mislabeling; the v5 ceiling of the PMMA space is
≈ 0.47.

### 5.2 One consistent chromatic model: pupil walk-off — FIX-13, review §2.7

v4's spec[2] (in-guide angular fan, ~3.7°, "non-cancelling") and mtf_chrom
(the same fan × 0.001, "99.9% cancelled") contradicted each other ~1000×.
The consistent physics: matched in/out periods cancel the output *angle*
exactly for every wavelength (sinθ_out = sinθᵢ); the surviving within-band
effect is **lateral pupil walk-off**

```
σ_x = L·σ_λ / (n·Λ·sinθ_d·cos²θ_d)        [mm, per primary]
```

which vignettes the eye pupil rather than blurring the retinal image. v5
uses ONE walk-off model in both channels: spec[2] := photopic-weighted σ_x
(mm), and the eye-MTF term is evaluated at the walk-off-reduced effective
pupil d_eff = D − σ_x (smooth clamp ≥ 1 mm) through Watson's formula.
mtf_chrom retains only the fabrication period-mismatch residual
(RESID_DISP ≈ 0.1%), which is the one genuinely surviving *angular* error.
Test V11 verifies end-to-end consistency: doubling the LED bandwidth raises
spec[2] and lowers MTF for 32/32 random designs.

### 5.3 Fresnel double-count removed — FIX-14, review §2.3 (pessimistic-side)

grcwa launches the wave from air through the corrugated interface and
normalizes T(+1) to incident power, so the RCWA η already contains the
coupler-face reflection (and by reciprocity the exit face). The extra flat
`fresnel_T²` factor (~0.92) double-counted it and is removed; topology is
now stated (couplers on the entry/exit faces). This *raises* throughput —
the review's point that not all v4 errors were optimistic.

### 5.4 Throughput is a relative FOM + re-interaction de-rating — FIX-15, §2.2

Spec[1] is now explicitly a **relative figure of merit** (single eyebox
position, no exit-pupil expansion), not a device efficiency. A first-order
re-interaction survival term is added:

```
T_reint = (1 − η)^m,   m = relu(W_IN/(2·t·tanθ_d) − 1)
```

— the guided beam re-crosses the in-coupler while the bounce advance is
shorter than the coupler aperture, and each encounter re-diffracts ~η back
out (the mechanism behind the thickness-dependent input-efficiency ceiling
of Zhao et al., *Opt. Express* 32(7), 12340–12357 (2024)). Absolute
percentages must be framed against the ~10% @20° / ~3% @30° system
efficiencies reported for conventional diffractive combiners (*Light Sci.
Appl.* 13, 2024).

### 5.5 Small fixes — FIX-16/17, review §2.4–§2.6, §2.8

* V(532) corrected 0.862 → 0.885 (0.862 is V(530)).
* n(532) pinned to the Sellmeier value 1.49369 in PMMA mode — a specified
  material's index is not a design lever; ±0.01 lot scatter is UQ, not a
  bound to optimize inside.
* σ ∈ [0.7, 1.1] nm documented as the **spin-coated** PMMA range (Nilsen et
  al. 2025); untreated PMMA is rougher and out of scope.
* Normal-incidence RCWA grid + reciprocity reuse for the out-coupler now
  stated explicitly in the cascade docstring.

### 5.6 Validation status

**11/11 tests pass** (`validate.py`): V1 Fresnel, V2 energy conservation,
V3 parity, V4 scalar limit, V5 4/π² ceiling, V6 convergence, V7 architecture
gates, V8 TIR enforcement, V9 Brewster + unpolarized identity, **V10 Watson
eye MTF (external anchor)**, **V11 walk-off spec/MTF consistency**.

Consequences for quoted numbers: all v4-era records, surrogates, and paper
numbers are stale (`ENGINE_VERSION = "v5"` invalidates them via the
physics-probe system). Gradient-probed v5 ceilings of the PMMA space:
MTF ≈ 0.466, T_FOM ≈ 1.35%, walk-off floor ≈ 1.52 mm. Regenerate all
headline numbers with `bash overnight.sh` before quoting anything.

Remaining L1 heuristics, honestly flagged `# SYNC` (review §2.9): the
roughness/grating/coupler MTF coefficients, the acceptance width, the S(L_c)
weighting, the W_IN aperture, and the chord pupil-overlap model — these are
uncalibrated heuristics and the paper must not quote MTF as a rigorous
number until they are pinned or bounded.
