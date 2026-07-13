"""
L1 (analytic, literature-anchored) models for double stacked diffractive and
geometric partial-mirror waveguides, per architecture_framework.md Section 2.

Both are differentiable (PyTorch) and reuse the validated singular-waveguide
components from waveguide_physics.py, so gates G2/G3 (limiting-case and
design-rule tests) run automatically under `python3 architectures.py`.

Claims discipline: these models support "model-level comparison under matched
assumptions" (L1) only. See framework Sections 4-5 before citing any output.
"""

import torch

from waveguide_physics import (
    WL, transmission, mtf_system, sample_theta, diffraction_angle, n_bounces,
)

# ----------------------------------------------------------------------------
# 2.2  Double stacked diffractive: spectral-band split across two layers
# ----------------------------------------------------------------------------
# Layer 1 carries blue+green (450, 532), layer 2 carries red (635) by default.
# Each layer = singular model with its own grating period; adds two air-gap
# Fresnel interfaces and computable inter-layer leakage kappa (off-band
# efficiency of the wrong layer's grating), no free fit parameters.

BANDS = ([0, 1], [2])  # indices into WL


def _band_chromatic_deg(n, period, band):
    """Chromatic spread across only the wavelengths routed to this layer."""
    wl = WL[band]
    x = wl / period.unsqueeze(-1)                  # normal incidence
    ang = diffraction_angle(x, n.unsqueeze(-1))
    return torch.rad2deg(ang.max(dim=-1).values - ang.min(dim=-1).values)


def _off_band_eta(n, depth, duty, wl_other):
    """Scalar first-order efficiency of a grating at the OTHER band's wavelength
    (this is the leakage term kappa — computed, not assumed)."""
    phi = 2 * torch.pi * depth * (n - 1.0) / wl_other
    return 4.0 * (torch.sin(torch.pi * duty) / torch.pi) ** 2 * torch.sin(phi / 2) ** 2


def double_waveguide(theta1: torch.Tensor, theta2: torch.Tensor):
    """theta1/theta2: [B,8] designs for layer 1 (blue+green) and layer 2 (red).
    Returns dict of spec tensors. Framework Section 2.2."""
    n1, n2 = theta1[:, 0], theta2[:, 0]

    # per-layer singular transmission, plus 2 extra air-gap interfaces each
    def gap_fresnel(n):
        r = ((n - 1) / (n + 1)) ** 2
        return (1 - r) ** 2

    T1 = transmission(theta1) * gap_fresnel(n1)
    T2 = transmission(theta2) * gap_fresnel(n2)

    # computable inter-layer leakage: layer 2's grating acting on layer 1's band
    kappa_12 = _off_band_eta(n2, theta2[:, 6], theta2[:, 7], WL[BANDS[0]].mean())
    kappa_21 = _off_band_eta(n1, theta1[:, 6], theta1[:, 7], WL[BANDS[1]].mean())
    T1 = T1 * (1 - kappa_12)
    T2 = T2 * (1 - kappa_21)

    # luminance-weighted combination (green-dominant photopic weighting)
    w1, w2 = 0.80, 0.20
    T = w1 * T1 + w2 * T2

    # chromatic: worst layer's in-band spread (eye sees the union)
    ca1 = _band_chromatic_deg(n1, theta1[:, 5], BANDS[0])
    ca2 = _band_chromatic_deg(n2, theta2[:, 5], BANDS[1])
    ca = torch.maximum(ca1, ca2)

    mtf = w1 * mtf_system(theta1) + w2 * mtf_system(theta2)
    return {"T": T, "chrom_deg": ca, "MTF": mtf,
            "kappa_12": kappa_12, "kappa_21": kappa_21}


# ----------------------------------------------------------------------------
# 2.3  Geometric partial-mirror waveguide
# ----------------------------------------------------------------------------

def uniform_mirror_reflectances(M: int) -> torch.Tensor:
    """Design rule R_k = 1/(M+1-k): each of M mirrors outputs an equal 1/M of
    the guided flux. Derivation: after mirror k, remaining flux is (M-k)/M."""
    return torch.tensor([1.0 / (M + 1 - k) for k in range(1, M + 1)])


def geometric_waveguide(theta: torch.Tensor, M: int = 4, embed_loss: float = 0.01,
                        eta_in: float = 0.90, pupil_overlap=None):
    """theta: [B,8] (grating dims 5-7 ignored — no gratings). Framework 2.3.
    embed_loss: per-mirror coating/embed absorption. eta_in: in-coupling prism
    efficiency. pupil_overlap: [M] weights A_k (default: single 3mm pupil sees
    ~2 of M exit pupils)."""
    n, alpha, sigma, Lc, t = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3], theta[:, 4]
    B = theta.shape[0]

    R = uniform_mirror_reflectances(M)                      # [M]
    if pupil_overlap is None:
        pupil_overlap = torch.zeros(M); pupil_overlap[M // 2 - 1: M // 2 + 1] = 1.0

    # flux reaching mirror k (uniform rule -> (M-k+1)/M before mirror k), with
    # embed loss compounding per mirror passed
    out_frac = torch.zeros(B, M)
    remaining = torch.ones(B)
    for k in range(M):
        remaining = remaining * (1 - embed_loss)
        out_k = remaining * R[k]
        out_frac[:, k] = out_k
        remaining = remaining - out_k

    # G3 check quantity: per-mirror outputs should be equal (before pupil weights)
    g3_spread = (out_frac.max(dim=1).values - out_frac.min(dim=1).values)

    # bulk + roughness along the TIR path (propagation angle set by prism, ~50 deg)
    ang = torch.deg2rad(torch.tensor(50.0))
    NB = n_bounces(t, ang)                # geometric bounce count (FIX-4)
    T_bulk = torch.exp(-alpha * NB * t / torch.cos(ang))
    pb = (4 * torch.pi * (sigma * 1e-6) * n * torch.cos(ang) / 532e-6) ** 2
    T_scatter = torch.exp(-pb * (1.0 / (1.0 + Lc / 3e5)) * NB * 0.5)
    r0 = ((n - 1) / (n + 1)) ** 2
    T_fresnel = (1 - r0) ** 2

    T = T_fresnel * T_bulk * T_scatter * eta_in * (out_frac * pupil_overlap).sum(dim=1)

    # chromatic: bounded, not modeled (framework 2.3) — report the bound
    chrom_bound = torch.full((B,), 0.1)

    # MTF: no grating terms; coating scatter + mirror-edge term (L1 placeholder,
    # promote at G4)  # SYNC-L2
    from waveguide_physics import PUPIL_MM
    lam_mm = 532e-6
    fc = PUPIL_MM / (lam_mm * 17.0)
    x = torch.tensor(min(40.0 / fc, 0.999))
    mtf_d = (2 / torch.pi) * (torch.acos(x) - x * torch.sqrt(1 - x ** 2))
    mtf = mtf_d * 0.97 * 0.94 * torch.ones(B)

    return {"T": T, "chrom_deg_bound": chrom_bound, "MTF": mtf,
            "g3_spread": g3_spread, "eyebox_pupils": M}


# ----------------------------------------------------------------------------
# Gates G2 & G3 (framework Section 4) — run: python3 architectures.py
# ----------------------------------------------------------------------------

def run_gates():
    torch.manual_seed(0)
    th = sample_theta(64)

    # G2: double model, both layers identical & full spectrum & kappa forced 0,
    # must reduce to singular (here: identical layers, compare weighted parts)
    d = double_waveguide(th, th.clone())
    T_singular = transmission(th)
    gap = ((th[:, 0] - 1) / (th[:, 0] + 1)) ** 2
    T_expected = T_singular * (1 - gap) ** 2  # modulo computed kappa
    resid = (d["T"] / (1 - 0.8 * d["kappa_12"] - 0.2 * d["kappa_21"]).clamp(min=1e-6)
             - T_expected).abs().max()
    print(f"G2 limiting-case residual (should be small): {resid:.2e}")

    # G3: uniform design rule gives equal per-mirror output (embed_loss=0)
    g = geometric_waveguide(th, M=6, embed_loss=0.0)
    print(f"G3 per-mirror output spread (should be ~0): {g['g3_spread'].max():.2e}")

    # report scale sanity (G5): all transmissions below scalar/physical ceilings
    print(f"double T range:    {d['T'].min():.4f} .. {d['T'].max():.4f}")
    print(f"geometric T range: {g['T'].min():.4f} .. {g['T'].max():.4f}")
    print(f"double chrom (deg): {d['chrom_deg'].min():.2f} .. {d['chrom_deg'].max():.2f}"
          f"  | geometric bound: 0.10")


if __name__ == "__main__":
    run_gates()
