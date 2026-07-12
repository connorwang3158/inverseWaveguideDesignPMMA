"""
Differentiable forward physics engine for a singular flat diffractive AR waveguide.

PyTorch port of the analytic model from:
  Wang, "Modeling Diffractive Singular Flat AR Waveguide Optical Performance"
  (github.com/connorwang3158/ModelingSingularFlatDiffractiveWaveguidesWithinARGlasses)

Structure mirrors the paper:
  Transmission = Fresnel x Beer-Lambert bulk x roughness scattering (Payne-Lacey form)
                 x in-coupler grating x out-coupler grating
  MTF_system   = MTF_diffraction x MTF_roughness x MTF_chromatic x MTF_grating x MTF_coupler
  Chromatic    = grating-equation angular spread across RGB
  FOV          = transmission evaluated at field angle (Fresnel + coupling detuning)

All components are closed-form and autograd-differentiable. NOTE: constants marked
`# SYNC` are simplified placeholders — sync them with the exact values/formulas in
the Paper-1 repo before running publication experiments (unit-test targets:
Si3N4 loss 93.37-93.39%, PMMA system MTF 0.6426-0.6430).

Design vector theta (8, physical units):
  [0] n      refractive index            (1.45 .. 2.20)
  [1] alpha  absorption coeff, 1/mm      (1e-5 .. 1e-2)
  [2] sigma  RMS roughness, nm           (0.3 .. 6.0)
  [3] Lc     correlation length, nm      (300 .. 1.2e6)   (log-sampled)
  [4] t      waveguide thickness, mm     (0.3 .. 2.0)
  [5] period grating period, nm          (300 .. 700)
  [6] depth  grating depth, nm           (20 .. 400)
  [7] duty   grating duty cycle          (0.2 .. 0.8)

Spec vector y (4):
  [0] MTF_system at 40 cyc/mm (green-weighted)
  [1] total double-coupler transmission (0..1, paper reports ~0.06)
  [2] lateral chromatic spread, degrees (blue-to-red first-order angle diff)
  [3] transmission at design field angle (FOV robustness metric)
"""

import torch

# RGB design wavelengths (nm), matching Paper 1
WL = torch.tensor([450.0, 532.0, 635.0])
F0_CYC_PER_MM = 40.0          # evaluation spatial frequency (industry benchmark)
PUPIL_MM = 3.0                # eye pupil diameter (Watson 2013 diffraction limit)
FOV_DEG = 20.0                # design half-field angle for the FOV metric

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
# Physics components (batched: theta [B,8] -> each metric [B])
# ----------------------------------------------------------------------------

def _smooth_sin(s_raw: torch.Tensor, cap: float = 0.98) -> torch.Tensor:
    """Smoothly clamp sin(angle) into (0, cap) — keeps gradients alive where a
    hard clamp would saturate (evanescent / beyond-TIR regimes)."""
    return cap * torch.tanh(s_raw.clamp(min=0.02) / cap)


def _diffraction_angles(n: torch.Tensor, period: torch.Tensor) -> torch.Tensor:
    """First-order in-guide grating equation at normal incidence:
    n sin(th) = lambda/period  ->  sin(th) = lambda/(n*period).
    Returns angles [B,3] in radians."""
    s_raw = WL.to(period.device) / (n.unsqueeze(1) * period.unsqueeze(1))
    return torch.asin(_smooth_sin(s_raw))


def chromatic_spread_deg(theta: torch.Tensor) -> torch.Tensor:
    """Lateral chromatic spread: red-blue first-order angle difference (degrees)."""
    ang = _diffraction_angles(theta[:, 0], theta[:, 5])
    return torch.rad2deg(ang[:, 2] - ang[:, 0])


def transmission(theta: torch.Tensor, field_deg: float = 0.0) -> torch.Tensor:
    """Total double-coupler transmission at a given field angle (green-weighted)."""
    n, alpha, sigma, Lc, t, period, depth, duty = theta.unbind(dim=1)
    lam_g = 532.0  # green carries the transmission benchmark, per Paper 1
    th_i = torch.deg2rad(torch.tensor(field_deg, device=theta.device))

    # --- Fresnel (two air/material interfaces), angle-dependent (unpolarized avg)
    sin_t = (torch.sin(th_i) / n).clamp(max=0.999)
    cos_i, cos_t = torch.cos(th_i), torch.sqrt(1 - sin_t ** 2)
    rs = ((cos_i - n * cos_t) / (cos_i + n * cos_t)) ** 2
    rp = ((n * cos_i - cos_t) / (n * cos_i + cos_t)) ** 2
    T_fresnel = (1 - 0.5 * (rs + rp)) ** 2

    # --- Bulk Beer-Lambert along the zig-zag propagation path
    ang = torch.asin(_smooth_sin(lam_g / (n * period)))            # in-guide angle
    n_bounces = 10.0                                                # SYNC: pupil-expansion bounce count
    path_mm = n_bounces * t / torch.cos(ang).clamp(min=0.2)
    T_bulk = torch.exp(-alpha * path_mm)

    # --- Roughness scattering, Payne-Lacey-form per TIR bounce  # SYNC constants
    lam_mm = lam_g * 1e-6
    per_bounce = (4 * torch.pi * (sigma * 1e-6) * n * torch.cos(ang) / lam_mm) ** 2
    corr = 1.0 / (1.0 + (Lc / 3e5))       # longer Lc -> more forward (less lost) scatter
    T_scatter = torch.exp(-per_bounce * corr * n_bounces * 0.5)

    # --- Grating coupling efficiency (in + out), phase-depth + duty + detuning
    phase = torch.pi * depth * (n - 1.0) / lam_g
    eta = torch.sin(phase) ** 2 * torch.exp(-((duty - 0.5) ** 2) / 0.045)
    detune = torch.exp(-(torch.sin(th_i) / 0.35) ** 2)             # angular acceptance
    eta = (0.05 + 0.95 * eta) * (0.3 + 0.7 * detune)               # floor avoids dead gradients
    T_grating = eta ** 2                                            # two couplers

    return T_fresnel * T_bulk * T_scatter * T_grating * 0.55       # SYNC global scale -> ~6%


def mtf_system(theta: torch.Tensor) -> torch.Tensor:
    """Five-component multiplicative MTF cascade at F0_CYC_PER_MM (green)."""
    n, alpha, sigma, Lc, t, period, depth, duty = theta.unbind(dim=1)
    f = F0_CYC_PER_MM  # cyc/mm at the retina-conjugate image plane

    # 1) Diffraction-limited eye MTF (Watson 2013 form), green
    lam_mm = 532e-6
    fc = PUPIL_MM / (lam_mm * 17.0)  # diffraction cutoff at retina-conjugate plane, cyc/mm (eye f=17mm)  # SYNC
    x = torch.tensor(min(f / fc, 0.999))
    mtf_diff = (2 / torch.pi) * (torch.acos(x) - x * torch.sqrt(1 - x ** 2))
    mtf_diff = mtf_diff * torch.ones_like(n)

    # 2) Roughness MTF: Gaussian attenuation from scatter-induced angular blur
    blur_rough = (sigma / 6.0) ** 2 * (1.0 / (1.0 + Lc / 3e5)) * 8e-3  # SYNC
    mtf_rough = torch.exp(-2 * (torch.pi * blur_rough * f) ** 2)

    # 3) Chromatic MTF: Gaussian PSF from RGB angular displacement (Paper 1 Eq. 3)
    spread = torch.deg2rad(chromatic_spread_deg(theta))
    blur_chrom = spread * 17.0 * 1e-3 * 0.10    # SYNC: retinal mm per rad, weight
    mtf_chrom = torch.exp(-2 * (torch.pi * blur_chrom * f) ** 2)

    # 4) Grating MTF: periodic wavefront modulation contrast loss
    mtf_grat = 1.0 - 0.12 * torch.sin(torch.pi * depth * (n - 1) / 532.0) ** 2  # SYNC

    # 5) Coupler MTF: finite-efficiency contrast degradation across coupling events
    phase = torch.pi * depth * (n - 1.0) / 532.0
    eta = (0.05 + 0.95 * torch.sin(phase) ** 2
           * torch.exp(-((duty - 0.5) ** 2) / 0.045))
    mtf_coup = 0.80 + 0.20 * eta                                                 # SYNC

    return mtf_diff * mtf_rough * mtf_chrom * mtf_grat * mtf_coup


def forward_model(theta: torch.Tensor) -> torch.Tensor:
    """theta [B,8] -> spec y [B,4]: [MTF, T_total, chrom_spread_deg, T_at_FOV]."""
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
    th = sample_theta(5)
    y = forward_model(th)
    print("theta sample:\n", th)
    print("spec [MTF, T, chrom_deg, T_fov]:\n", y)
    # gradient check
    th.requires_grad_(True)
    forward_model(th).sum().backward()
    print("grad finite:", torch.isfinite(th.grad).all().item())
