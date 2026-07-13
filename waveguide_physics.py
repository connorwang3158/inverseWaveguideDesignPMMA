"""
Differentiable forward physics engine for a singular flat diffractive AR waveguide.

PyTorch port of the analytic model from:
  Wang, "Modeling Diffractive Singular Flat AR Waveguide Optical Performance"
  (github.com/connorwang3158/ModelingSingularFlatDiffractiveWaveguidesWithinARGlasses)

Structure mirrors the paper:
  Transmission = Fresnel (TE/TM-resolved) x Beer-Lambert bulk x roughness
                 scattering (Tien 1971 per-bounce) x in-coupler grating
                 x out-coupler grating x TIR guiding mask
  MTF_system   = MTF_diffraction x MTF_roughness x MTF_chromatic x MTF_grating
                 x MTF_coupler
  Chromatic    = grating-equation angular spread across RGB
  FOV          = transmission evaluated at field angle (guiding + Fresnel +
                 coupler acceptance all angle-resolved)

============================ v2 PHYSICS REVISION ==============================
This revision fixes flaws found in the 2026-07 audit:

  FIX-1  TIR GUIDING CONSTRAINT (critical). A first-order beam is guided only if
             1 < sin(theta_i) + lambda/period < n            (dimensionless)
         Left inequality  = total internal reflection at the slab faces
         (n*sin(theta_d) > 1); right inequality = the diffracted order
         propagates (sin(theta_d) < 1). The previous engine never enforced
         this, so optimizers exploited large periods (~680 nm) whose "designs"
         were not guided at all. A smooth sigmoid mask now multiplies the
         transmission; `tir_penalty()` gives optimizers a restoring gradient;
         PMMA period bounds are narrowed to the full-RGB guided window
         (lambda_red/n < period < lambda_blue  ->  ~429..450 nm for n=1.49).
         Ref: Kress & Chatterjee, Nanophotonics 10(1), 41-74 (2021), Sec. on
         waveguide FOV/index limits.

  FIX-2  FIELD-ANGLE GRATING EQUATION. In-coupling at field angle theta_i obeys
             n sin(theta_d) = sin(theta_i) + m*lambda/period   (m = +1)
         The previous engine evaluated the grating at normal incidence for all
         field angles. The FOV metric is now the transmission with the exact
         angle-shifted diffraction + guiding window (this IS the physical FOV
         limiter for low-index waveguides).

  FIX-3  POLARIZATION-RESOLVED (VECTOR) TRANSMISSION. Fresnel is computed per
         polarization (TE=s, TM=p) from the exact amplitude coefficients and
         carried separately through the cascade; "unpolarized" is the average
         of the two POWER transmissions, (T_TE + T_TM)/2 -- not the average of
         reflectances squared (Hecht, Optics, ch. 4). The scalar grating
         efficiency remains polarization-blind (documented limitation of
         scalar theory; per-polarization efficiencies are verified with the
         rigorous RCWA layer in rigorous_solver.py -- Moharam & Gaylord 1981,
         Moharam et al. 1995).

  FIX-4  GEOMETRIC BOUNCE COUNT. The number of TIR bounces between couplers a
         distance L_PROP_MM apart is
             N_b = L_PROP_MM / (2 t tan(theta_d)),
         not a constant 10. Thin waveguides bounce MORE, so roughness/bulk
         loss now scales correctly with thickness (gradient direction wrt t
         was previously wrong).

  FIX-5  AD-HOC FLOORS REMOVED. The 0.02 efficiency floor and the 0.3 angular
         acceptance floor distorted the physics (a zero-depth grating coupled
         light; detuned fields kept 30% coupling). Removed. The angular
         acceptance Gaussian applies to the in-coupler only.

  FIX-6  CHROMATIC MTF is the exact modulus of the complex sum of the
         photopically-weighted displaced line spread contributions,
             MTF_chrom(f) = | sum_k w_k exp(i 2 pi f x_k) |,
         (three-primary lateral chromatic displacement; cf. Thibos, JOSA A
         4(8), 1673 (1987)) instead of an ad-hoc Gaussian. x_k uses a residual
         dispersion fraction RESID_DISP (documented calibration constant).

Remaining L1 heuristics (documented, flagged, verified at L2 by RCWA where
possible): roughness-MTF coefficient, grating/coupler MTF coefficients,
Payne-Lacey correlation weighting S(Lc), coupler acceptance width. Constants
marked `# SYNC` must still be reconciled with the Paper-1 repo values.
===============================================================================

Design vector theta (8, physical units):
  [0] n      refractive index            (1.45 .. 2.20)
  [1] alpha  absorption coeff, 1/mm      (1e-5 .. 1e-2)
  [2] sigma  RMS roughness, nm           (0.3 .. 6.0)
  [3] Lc     correlation length, nm      (300 .. 1.2e6)   (log-sampled)
  [4] t      waveguide thickness, mm     (0.3 .. 2.0)
  [5] period grating period, nm          (300 .. 700; PMMA mode 430 .. 449)
  [6] depth  grating depth, nm           (20 .. 400)
  [7] duty   grating duty cycle          (0.2 .. 0.8)

Spec vector y (4):
  [0] MTF_system at 40 cyc/mm (green-weighted)
  [1] total double-coupler transmission, unpolarized (0..1)
  [2] lateral chromatic spread, degrees (blue-to-red first-order angle diff)
  [3] transmission at design field angle FOV_DEG (guiding-aware)
"""

import torch

# RGB design wavelengths (nm), matching Paper 1
WL = torch.tensor([450.0, 532.0, 635.0])
# CIE 1924 photopic luminous efficiency V(lambda) at the RGB primaries,
# normalized to sum 1 (0.038, 0.862, 0.217 raw)
V_PHOTOPIC = torch.tensor([0.038, 0.862, 0.217])
V_PHOTOPIC = V_PHOTOPIC / V_PHOTOPIC.sum()

F0_CYC_PER_MM = 40.0          # evaluation spatial frequency (industry benchmark)
PUPIL_MM = 3.0                # eye pupil diameter (Watson 2013 diffraction limit)
EYE_FL_MM = 17.0              # reduced-eye focal length
L_PROP_MM = 20.0              # in-coupler -> out-coupler propagation distance
N_BOUNCE_MAX = 60.0           # numerical ceiling on bounce count
RESID_DISP = 0.10             # residual chromatic dispersion fraction reaching
                              # the retina (matched couplers cancel most of the
                              # in-guide dispersion)  # SYNC calibration
ACCEPT_SIN = 0.35             # in-coupler angular acceptance width (sin space)
                              # cf. Zhao et al., Opt. Express 32, 12340 (2024)  # SYNC
FOV_DEG = 20.0                # design half-field angle for the FOV metric
                              # (use_pmma() lowers it -- see FIX-1/FIX-2 note)
TIR_SOFTNESS = 0.005          # sigmoid width of the guiding mask (sin space)

# Parameter bounds (min, max) — from Paper 1's literature-reported material ranges
BOUNDS = torch.tensor([
    [1.45, 2.20],      # n
    [1e-5, 1e-2],      # alpha (1/mm)
    [0.3, 6.0],        # sigma (nm)
    [300.0, 1.2e6],    # Lc (nm)
    [0.3, 2.0],        # t (mm)
    [300.0, 700.0],    # period (nm)
    [20.0, 400.0],     # depth (nm)
    [0.2, 0.8],        # duty
])
LOG_DIMS = (1, 3)  # alpha and Lc are sampled/normalized in log-space

# PMMA-only mode: material params pinned to PMMA's literature range (Nilsen et
# al., Opt. Express 33, 20051 (2025): n~1.49, RMS roughness ~0.87 nm, long
# correlation length, low bulk absorption). Geometry free EXCEPT the period,
# which is restricted to the full-RGB guided window (FIX-1):
#   lambda_red/n_min < period < lambda_blue  ->  635/1.48 = 429.1 .. 450 nm.
PMMA_BOUNDS = torch.tensor([
    [1.48, 1.50],       # n
    [5e-5, 5e-4],       # alpha (1/mm)
    [0.7, 1.1],         # sigma (nm)
    [2e5, 4e5],         # Lc (nm)
    [0.3, 2.0],         # t (mm)        — free geometry
    [430.0, 449.0],     # period (nm)   — full-RGB TIR window (FIX-1)
    [20.0, 400.0],      # depth (nm)    — free geometry
    [0.2, 0.8],         # duty          — free geometry
])


def use_pmma():
    """Switch the engine to the PMMA-only design space (call before anything
    else). Also lowers the FOV evaluation angle: the full-RGB guided field
    window of a n=1.49 single-layer waveguide is only a few degrees wide
    (a direct consequence of FIX-1's guiding inequality), so evaluating at
    20 deg would return an honest-but-uninformative 0 for every design."""
    global FOV_DEG
    BOUNDS.copy_(PMMA_BOUNDS)
    FOV_DEG = 5.0
    print("[physics] PMMA-only mode: material pinned, geometry free, "
          f"period restricted to RGB-guided window, FOV metric at {FOV_DEG:.0f} deg")


def sample_theta(n_samples: int, generator=None) -> torch.Tensor:
    """Uniform (log-uniform for LOG_DIMS) sampling within physical bounds."""
    u = torch.rand(n_samples, 8, generator=generator)
    lo, hi = BOUNDS[:, 0].clone(), BOUNDS[:, 1].clone()
    theta = lo + u * (hi - lo)
    for d in LOG_DIMS:
        theta[:, d] = torch.exp(
            torch.log(lo[d]) + u[:, d] * (torch.log(hi[d]) - torch.log(lo[d]))
        )
    return theta


def normalize_theta(theta: torch.Tensor) -> torch.Tensor:
    """Map physical theta -> [0,1]^8 (log for LOG_DIMS)."""
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    z = (theta - lo) / (hi - lo)
    for d in LOG_DIMS:
        z[:, d] = (torch.log(theta[:, d]) - torch.log(lo[d])) / (
            torch.log(hi[d]) - torch.log(lo[d])
        )
    return z


def denormalize_theta(z: torch.Tensor) -> torch.Tensor:
    """Map [0,1]^8 -> physical theta (inverse of normalize_theta)."""
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    theta = lo + z * (hi - lo)
    cols = []
    for d in range(8):
        if d in LOG_DIMS:
            cols.append(torch.exp(
                torch.log(lo[d]) + z[:, d] * (torch.log(hi[d]) - torch.log(lo[d]))
            ))
        else:
            cols.append(theta[:, d])
    return torch.stack(cols, dim=1)


# ----------------------------------------------------------------------------
# Core wave/ray geometry (vector k-space form of the grating equation)
# ----------------------------------------------------------------------------
# Planar (non-conical) mount. The tangential wavevector is conserved modulo
# grating vectors:  k_x,out = k_x,in + m*K,  K = 2*pi/period. Dividing by
# k0 = 2*pi/lambda gives the dimensionless form used below:
#     x := sin(theta_i) + m*lambda/period      (m = +1, in-coupling from air)
#     n sin(theta_d) = x                        (diffracted beam inside medium)
# Guided propagation requires 1 < x < n (FIX-1).

_SAFE_SIN = 0.9995


def grating_x(period: torch.Tensor, wl, field_deg: float = 0.0) -> torch.Tensor:
    """Dimensionless tangential wavevector x = sin(theta_i) + lambda/period.
    period [B]; wl scalar or [K] -> returns [B] or [B,K]."""
    wl_t = torch.as_tensor(wl, dtype=period.dtype, device=period.device)
    s_i = torch.sin(torch.deg2rad(torch.tensor(float(field_deg),
                                               device=period.device)))
    if wl_t.dim() == 0:
        return s_i + wl_t / period
    return s_i + wl_t.unsqueeze(0) / period.unsqueeze(1)


def guided_mask(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Smooth indicator of the guiding window 1 < x < n (FIX-1). Sigmoid width
    TIR_SOFTNESS keeps gradients alive at the window edges; at 3 widths from
    an edge the mask is >0.95."""
    return (torch.sigmoid((x - 1.0) / TIR_SOFTNESS)
            * torch.sigmoid((n - x) / TIR_SOFTNESS))


def diffraction_angle(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """In-guide propagation angle theta_d = asin(x/n), numerically safeguarded.
    Only meaningful where guided_mask > 0; the clamp never activates inside
    the guided window because x < n there by construction."""
    s = (x / n).clamp(min=1e-4, max=_SAFE_SIN)
    return torch.asin(s)


def _diffraction_angles_rgb(n: torch.Tensor, period: torch.Tensor,
                            field_deg: float = 0.0) -> torch.Tensor:
    """First-order in-guide angles for the RGB design wavelengths, [B,3] rad."""
    x = grating_x(period, WL, field_deg)          # [B,3]
    return diffraction_angle(x, n.unsqueeze(1))


def chromatic_spread_deg(theta: torch.Tensor) -> torch.Tensor:
    """Lateral chromatic spread: red-blue first-order angle difference (deg)."""
    ang = _diffraction_angles_rgb(theta[:, 0], theta[:, 5])
    return torch.rad2deg(ang[:, 2] - ang[:, 0])


def n_bounces(t: torch.Tensor, theta_d: torch.Tensor) -> torch.Tensor:
    """TIR bounce count over the coupler separation (FIX-4):
    N_b = L_PROP / (2 t tan(theta_d)), clamped to [1, N_BOUNCE_MAX]."""
    return (L_PROP_MM / (2.0 * t * torch.tan(theta_d))).clamp(1.0, N_BOUNCE_MAX)


def fresnel_T(n: torch.Tensor, field_deg: float, pol: str) -> torch.Tensor:
    """Exact single-interface air->medium power transmittance at incidence
    field_deg for polarization 'TE' (s) or 'TM' (p). (Hecht, Optics, ch. 4;
    Born & Wolf ch. 1.) By reciprocity the medium->air transmittance at the
    conjugate internal angle is identical, so the two-interface factor used in
    the cascade is fresnel_T**2."""
    th_i = torch.deg2rad(torch.tensor(float(field_deg), device=n.device))
    cos_i = torch.cos(th_i)
    sin_t = (torch.sin(th_i) / n).clamp(max=_SAFE_SIN)
    cos_t = torch.sqrt(1.0 - sin_t ** 2)
    if pol == "TE":
        r = (cos_i - n * cos_t) / (cos_i + n * cos_t)
    elif pol == "TM":
        r = (n * cos_i - cos_t) / (n * cos_i + cos_t)
    else:
        raise ValueError("pol must be 'TE' or 'TM'")
    return 1.0 - r ** 2


# ----------------------------------------------------------------------------
# Transmission cascade (batched: theta [B,8] -> [B])
# ----------------------------------------------------------------------------

def _transmission_pol(theta: torch.Tensor, field_deg: float,
                      pol: str) -> torch.Tensor:
    """Single-polarization total double-coupler transmission (green-weighted).

    Cascade (all terms dimensionless power fractions):
      guided-mask x Fresnel_in x Fresnel_out x Beer-Lambert x Tien-roughness
      x eta_in(angle) x eta_out
    """
    n, alpha, sigma, Lc, t, period, depth, duty = theta.unbind(dim=1)
    lam_g = 532.0  # green carries the transmission benchmark, per Paper 1

    # --- grating geometry at this field angle (FIX-2)
    x_g = grating_x(period, lam_g, field_deg)      # [B]
    mask = guided_mask(x_g, n)                     # FIX-1
    ang = diffraction_angle(x_g, n)                # in-guide angle [B]

    # --- Fresnel, polarization-resolved (FIX-3), two interfaces
    T_fres = fresnel_T(n, field_deg, pol) ** 2

    # --- Bulk Beer-Lambert along the zig-zag propagation path
    NB = n_bounces(t, ang)                         # FIX-4
    path_mm = NB * t / torch.cos(ang)
    T_bulk = torch.exp(-alpha * path_mm)

    # --- Roughness scattering per TIR bounce: Tien (1971, Appl. Opt. 10, 2395)
    #     specular loss per bounce exp[-(4 pi sigma n cos(theta)/lambda)^2],
    #     scaled by a Payne & Lacey (1994)-motivated correlation weighting
    #     S(Lc) in (0,1]: longer correlation length -> more of the scattered
    #     lobe stays within the guided beam.  # SYNC S(Lc) against Paper 1
    lam_mm = lam_g * 1e-6
    per_bounce = (4 * torch.pi * (sigma * 1e-6) * n * torch.cos(ang) / lam_mm) ** 2
    S_corr = 1.0 / (1.0 + (Lc / 3e5))
    T_scatter = torch.exp(-per_bounce * S_corr * NB)

    # --- Grating coupling: scalar-diffraction first-order efficiency of a
    #     binary phase grating (Goodman, Fourier Optics; O'Shea et al.):
    #         eta_1 = 4 (sin(pi*duty)/pi)^2 sin^2(phi/2),
    #         phi   = 2 pi depth (n-1)/lambda
    #     Ceiling 4/pi^2 ~ 40.5% at duty=0.5, phi=pi. Scalar theory is
    #     polarization-blind — per-pol efficiencies are verified by RCWA
    #     (rigorous_solver.py; Pommet et al. 1994 quantify the scalar error at
    #     ~1-lambda features). Angular acceptance applies to the in-coupler
    #     only (FIX-5); the out-coupler sees the guided angle by construction.
    phi = 2 * torch.pi * depth * (n - 1.0) / lam_g
    eta = 4.0 * (torch.sin(torch.pi * duty) / torch.pi) ** 2 * torch.sin(phi / 2) ** 2
    s_i = torch.sin(torch.deg2rad(torch.tensor(float(field_deg))))
    accept = torch.exp(-(s_i / ACCEPT_SIN) ** 2)
    T_grating = (eta * accept) * eta               # in-coupler x out-coupler

    return mask * T_fres * T_bulk * T_scatter * T_grating


def transmission(theta: torch.Tensor, field_deg: float = 0.0,
                 pol: str = "unpol") -> torch.Tensor:
    """Total double-coupler transmission. pol in {'TE','TM','unpol'}.
    Unpolarized = (T_TE + T_TM)/2 — average of POWER transmissions (FIX-3)."""
    if pol == "unpol":
        return 0.5 * (_transmission_pol(theta, field_deg, "TE")
                      + _transmission_pol(theta, field_deg, "TM"))
    return _transmission_pol(theta, field_deg, pol)


def transmission_polarized(theta: torch.Tensor, field_deg: float = 0.0):
    """Convenience: dict with TE, TM, unpol, and the TE/TM diattenuation."""
    te = _transmission_pol(theta, field_deg, "TE")
    tm = _transmission_pol(theta, field_deg, "TM")
    return {"TE": te, "TM": tm, "unpol": 0.5 * (te + tm),
            "diattenuation": (te - tm) / (te + tm + 1e-12)}


# ----------------------------------------------------------------------------
# Feasibility / FOV analysis (FIX-1, FIX-2)
# ----------------------------------------------------------------------------

def tir_penalty(theta: torch.Tensor, field_deg: float = 0.0,
                margin: float = 0.01) -> torch.Tensor:
    """Differentiable penalty > 0 when any RGB wavelength leaves the guiding
    window 1+margin < x < n-margin at the given field angle. Add to any
    minimization objective to steer optimizers into physical designs."""
    n, period = theta[:, 0], theta[:, 5]
    x = grating_x(period, WL, field_deg)                       # [B,3]
    lo = torch.relu((1.0 + margin) - x)
    hi = torch.relu(x - (n.unsqueeze(1) - margin))
    return (lo + hi).sum(dim=1)


def fov_window_deg(theta: torch.Tensor):
    """Exact guided field window per design (degrees, in air), i.e. the range
    of incidence angles theta_i with 1 < sin(theta_i)+lambda/period < n for
    ALL RGB wavelengths simultaneously. Returns (lo_deg, hi_deg, width_deg);
    width 0 means no common full-RGB window. This is the fundamental
    index-limited FOV of a diffractive waveguide (Kress & Chatterjee 2021)."""
    n, period = theta[:, 0], theta[:, 5]
    lam = WL.to(period.device)                                 # [3]
    lo_sin = (1.0 - lam.unsqueeze(0) / period.unsqueeze(1))    # [B,3]
    hi_sin = (n.unsqueeze(1) - lam.unsqueeze(0) / period.unsqueeze(1))
    lo = lo_sin.max(dim=1).values.clamp(-_SAFE_SIN, _SAFE_SIN)
    hi = hi_sin.min(dim=1).values.clamp(-_SAFE_SIN, _SAFE_SIN)
    lo_deg = torch.rad2deg(torch.asin(lo))
    hi_deg = torch.rad2deg(torch.asin(hi))
    return lo_deg, hi_deg, (hi_deg - lo_deg).clamp(min=0.0)


# ----------------------------------------------------------------------------
# MTF cascade (batched: theta [B,8] -> [B])
# ----------------------------------------------------------------------------

def mtf_system(theta: torch.Tensor) -> torch.Tensor:
    """Five-component multiplicative MTF cascade at F0_CYC_PER_MM (green)."""
    n, alpha, sigma, Lc, t, period, depth, duty = theta.unbind(dim=1)
    f = F0_CYC_PER_MM  # cyc/mm at the retina-conjugate image plane

    # 1) Diffraction-limited eye MTF (Watson 2013 form), green
    lam_mm = 532e-6
    fc = PUPIL_MM / (lam_mm * EYE_FL_MM)   # diffraction cutoff, cyc/mm
    xx = torch.tensor(min(f / fc, 0.999))
    mtf_diff = (2 / torch.pi) * (torch.acos(xx) - xx * torch.sqrt(1 - xx ** 2))
    mtf_diff = mtf_diff * torch.ones_like(n)

    # 2) Roughness MTF: Gaussian attenuation from scatter-induced angular blur
    #    (heuristic L1 coefficient)  # SYNC
    blur_rough = (sigma / 6.0) ** 2 * (1.0 / (1.0 + Lc / 3e5)) * 8e-3
    mtf_rough = torch.exp(-2 * (torch.pi * blur_rough * f) ** 2)

    # 3) Chromatic MTF (FIX-6): exact modulus of the photopically-weighted
    #    three-line complex sum, |sum_k w_k exp(i 2 pi f x_k)|, with retinal
    #    displacements x_k = EYE_FL * RESID_DISP * (theta_k - theta_green).
    #    Reduces to 1 as the spread -> 0; equals the true MTF of a 3-delta
    #    PSF (cf. Thibos 1987 lateral-chromatic treatment).
    ang = _diffraction_angles_rgb(n, period)                     # [B,3] rad
    dtheta = ang - ang[:, 1:2]
    x_k = EYE_FL_MM * RESID_DISP * dtheta                        # [B,3] mm
    w = V_PHOTOPIC.to(theta.device).unsqueeze(0)                 # [1,3]
    re = (w * torch.cos(2 * torch.pi * f * x_k)).sum(dim=1)
    im = (w * torch.sin(2 * torch.pi * f * x_k)).sum(dim=1)
    mtf_chrom = torch.sqrt(re ** 2 + im ** 2 + 1e-12)

    # 4) Grating MTF: contrast loss from periodic wavefront modulation — scales
    #    with phase depth (heuristic L1 coefficient 0.15)  # SYNC
    phi = 2 * torch.pi * depth * (n - 1.0) / 532.0
    mtf_grat = 1.0 - 0.15 * torch.sin(phi / 2) ** 2

    # 5) Coupler MTF: contrast degradation from finite diffraction efficiency
    #    across two coupling events (Goodsell et al. 2024 framing); eta
    #    normalized by the 4/pi^2 scalar ceiling (heuristic L1)  # SYNC
    eta = 4.0 * (torch.sin(torch.pi * duty) / torch.pi) ** 2 * torch.sin(phi / 2) ** 2
    mtf_coup = 0.80 + 0.20 * (eta / 0.4053)

    return mtf_diff * mtf_rough * mtf_chrom * mtf_grat * mtf_coup


def forward_model(theta: torch.Tensor) -> torch.Tensor:
    """theta [B,8] -> spec y [B,4]: [MTF, T_total, chrom_spread_deg, T_at_FOV].
    Transmissions are unpolarized; use transmission_polarized() for TE/TM."""
    return torch.stack([
        mtf_system(theta),
        transmission(theta, field_deg=0.0),
        chromatic_spread_deg(theta),
        transmission(theta, field_deg=FOV_DEG),
    ], dim=1)


# Spec normalization scales (keep losses balanced across metrics)
SPEC_SCALE = torch.tensor([1.0, 0.10, 30.0, 0.10])


def normalize_spec(y: torch.Tensor) -> torch.Tensor:
    return y / SPEC_SCALE.to(y.device)


if __name__ == "__main__":
    torch.manual_seed(0)
    use_pmma()
    th = sample_theta(5)
    y = forward_model(th)
    print("theta sample:\n", th)
    print("spec [MTF, T, chrom_deg, T_fov]:\n", y)
    pol = transmission_polarized(th)
    print("T_TE:", pol["TE"].tolist())
    print("T_TM:", pol["TM"].tolist())
    lo, hi, wdt = fov_window_deg(th)
    print("full-RGB guided FOV window (deg):",
          [f"[{a:.1f},{b:.1f}] w={c:.1f}" for a, b, c in
           zip(lo.tolist(), hi.tolist(), wdt.tolist())])
    # gradient check
    th.requires_grad_(True)
    forward_model(th).sum().backward()
    print("grad finite:", torch.isfinite(th.grad).all().item())
    # TIR sanity: an unguided design must transmit ~0
    bad = th.detach().clone()
    bad[:, 5] = 680.0  # old exploit period — outside the guided window
    print("unguided-design T (must be ~0):", transmission(bad).max().item())
