"""
Differentiable forward model for a single-layer flat diffractive AR waveguide.

Everything here is written in PyTorch so the design parameters can be pushed
around by gradient descent and the same code can label the training data for
the surrogate network. It follows the analytic model in
  Wang, "Modeling Diffractive Singular Flat AR Waveguide Optical Performance"
  (github.com/connorwang3158/ModelingSingularFlatDiffractiveWaveguidesWithinARGlasses)

The cascade is wavelength-resolved. Every term is evaluated at each primary's
own wavelength and its own material index: the grating equation, the guiding
window, the in-guide angle, the bounce count, the Beer-Lambert path, the
Debye-Waller roughness exponent, the Fresnel coefficients, and the RCWA
coupling efficiency. Running the cascade at green and calling the answer
full-colour hides the effect that dominates a low-index guide, since red sits
closest to the edge of the guiding window and couples several times more
weakly than blue. transmission_rgb() returns all three primaries;
transmission() returns the white-balanced number, which is the worst primary,
because a display that has to hit a white point can only use as much light as
its weakest channel carries.

Transmission is a product of dimensionless power fractions:
  a total-internal-reflection guiding mask, the in-coupler grating, bulk
  Beer-Lambert absorption over the zig-zag path, per-bounce roughness scatter
  (Tien 1971), a survival factor for the beam re-hitting the in-coupler, and
  the out-coupler grating. In PMMA mode both grating efficiencies come from a
  rigorous RCWA table (see below); the interface reflection at the corrugated
  face is already inside that efficiency, so there is no separate flat-Fresnel
  factor there.

System MTF is a product of five contrast terms: the human eye (Watson 2013,
evaluated at a pupil shrunk by the chromatic walk-off), roughness blur, a
residual chromatic term, and grating- and coupler-contrast terms.

The chromatic spec is the within-band lateral walk-off across the eye pupil
in mm. With matched in/out grating periods the output angle is the same for
every wavelength, so the surviving chromatic effect is a sideways shift of
the exit beam, not an angular blur.

The FOV metric is the transmission evaluated at a field angle, with the
grating equation, guiding window, and coupler acceptance all angle-resolved.

Modeling notes worth keeping in mind:
  * Guiding. A first diffraction order is trapped only when
        1 < sin(theta_i) + lambda/period < n.
    The left side is TIR at the slab faces, the right side is the order still
    propagating inside the glass. A smooth mask enforces it and tir_penalty()
    gives the optimizer a gradient back into the window. For a low-index
    substrate like PMMA this window is only a few degrees wide, which is the
    real FOV limit (Kress & Chatterjee, Nanophotonics 10, 41 (2021)).
  * Fresnel is done per polarization (TE=s, TM=p) from the exact amplitude
    coefficients; unpolarized is the mean of the two power transmissions, not
    an average of reflectances (Hecht ch. 4).
  * Grating coupling. The guiding window forces PMMA periods of 430 to 449 nm,
    below the visible wavelengths, where the scalar phase-grating formula
    overshoots the true first-order efficiency by 5 to 15 times (measured
    against grcwa; consistent with Pommet et al., JOSA A 11, 1827 (1994)).
    PMMA mode therefore interpolates the efficiency from a rigorous RCWA grid
    (rcwa_eta_grid.npz, built by calibrate_rcwa.py), polarization-resolved and
    carrying its own refractive-index axis. Full-material mode keeps the
    scalar formula.
  * The index n is a per-color material property, not a single free number:
    the blue and red indices follow PMMA's Sellmeier dispersion, and in PMMA
    mode n at green is pinned to the material value rather than optimized.
  * The throughput number is a relative figure of merit for ranking designs
    at one eye position. It does not model exit-pupil expansion or eyebox
    uniformity, so it is not a device efficiency; compare it against the
    roughly 10% (20 deg FOV) and 3% (30 deg FOV) system efficiencies reported
    for real diffractive combiners.

A few contrast coefficients in the MTF cascade are still fitted by hand and
are marked with a SYNC comment; they are honest placeholders, not calibrated
constants.

Design vector theta (8, physical units):
  [0] n      refractive index at 532 nm  (1.45 to 2.20; pinned to the PMMA
             value 1.49369 in PMMA mode)
  [1] alpha  absorption coeff, 1/mm      (1e-5 to 1e-2)
  [2] sigma  RMS roughness, nm           (0.3 to 6.0; PMMA mode 0.7 to 1.1,
             which is the spin-coated surface, not bare PMMA)
  [3] Lc     correlation length, nm      (300 to 1.2e6, log-sampled)
  [4] t      waveguide thickness, mm     (0.3 to 2.0)
  [5] period grating period, nm          (300 to 700; PMMA mode 430 to 449)
  [6] depth  grating depth, nm           (20 to 400)
  [7] duty   grating duty cycle          (0.2 to 0.8)

Spec vector y (4):
  [0] system MTF at 40 cyc/mm (green-weighted, anchored to the Watson eye)
  [1] white-balanced double-coupler throughput, unpolarized (0 to 1), meaning
      the worst of the three primaries; a relative ranking metric, not a
      device efficiency
  [2] within-band chromatic pupil walk-off, mm, photopically weighted
  [3] white-balanced throughput at the design field angle FOV_DEG
"""

import os

import torch

# Stamped onto every checkpoint, record, and run table so results from
# different versions of the physics never get pooled together. Bump it when a
# change makes old numbers non-comparable; the surrogate checkpoints store a
# probe of the current physics and refuse to load once it no longer matches.
ENGINE_VERSION = "v6"

# Source spectral width per display primary (LED FWHM, nm). Sets the
# within-band chromatic walk-off and the finite-bandwidth chromatic MTF.
# Typical micro-LED or LED-illuminated LCoS primaries land at 20 to 25 nm.
LED_FWHM_NM = 25.0

# The refractive index is a material property, not three independent numbers.
# theta[0] is the index at green (532 nm); the blue and red indices follow
# PMMA's Sellmeier dispersion (Sultanova, Kasarova & Nikolov, Acta Phys. Pol.
# A 116, 585 (2009)): n(450)=1.5006, n(532)=1.4937, n(635)=1.4886, i.e. fixed
# offsets from green. Red has the lowest index, which is exactly where the
# x < n guiding constraint binds, so it matters that we use the real red index
# rather than a single number. The offsets are zero in full-material mode,
# where the material is unknown.
PMMA_DISP_OFFSET = torch.tensor([+0.00692, 0.0, -0.00506])  # B, G, R vs green
_DISP_ACTIVE = False  # toggled by use_pmma() / use_full()

# In PMMA mode n(532) is pinned to the Sellmeier value: a specified material's
# index is not something the fabricator can dial in, so it is not a design
# variable. Grade and lot scatter of about +/-0.01 is an uncertainty to
# propagate, not a lever to optimize over.
PMMA_N_532 = 1.49369          # Sultanova et al. 2009 single-term Sellmeier
PMMA_N_TOL = 0.01             # grade/lot uncertainty band (not sampled)

def n_rgb(n: torch.Tensor) -> torch.Tensor:
    """Per-primary refractive index [B,3]: design n (green) + material
    dispersion offsets (PMMA mode only)."""
    off = PMMA_DISP_OFFSET.to(n.device) if _DISP_ACTIVE \
        else torch.zeros(3, device=n.device)
    return n.unsqueeze(1) + off.unsqueeze(0)

# RGB design wavelengths (nm)
WL = torch.tensor([450.0, 532.0, 635.0])
# CIE 1924 photopic luminous efficiency V(lambda) at the RGB primaries,
# normalized to sum 1 (raw 0.038, 0.885, 0.217). The green value is V(532);
# interpolating the CIE table there gives 0.885, not the 0.862 that sits at
# 530 nm.
V_PHOTOPIC = torch.tensor([0.038, 0.885, 0.217])
V_PHOTOPIC = V_PHOTOPIC / V_PHOTOPIC.sum()

F0_CYC_PER_MM = 40.0          # evaluation spatial frequency (industry benchmark)
PUPIL_MM = 3.0                # eye pupil diameter (Watson 2013 valid range 2-6)
EYE_FL_MM = 17.0              # reduced-eye focal length
L_PROP_MM = 20.0              # in-coupler to out-coupler propagation distance
W_IN_MM = 3.0                 # in-coupler aperture width, sized to the eye
                              # pupil. Sets the re-interaction count
                              # m = W_IN/(2 t tan th_d) - 1; compare the
                              # pupil-size term of Zhao et al. 2024.  # SYNC
N_BOUNCE_MAX = 60.0           # numerical ceiling on bounce count
RESID_DISP = 0.001            # residual dispersion fraction reaching the
                              # retina. For matched in/out coupler periods the
                              # out-coupler cancels the in-guide dispersion for
                              # a collimated input, since
                              #   sin(th_out) = n sin(th_d) - lambda/period
                              #              = sin(th_i)
                              # is wavelength-independent, within each primary's
                              # band included. What is left is the in/out period
                              # mismatch from fabrication,
                              #   d(sin th_out) = (lambda/period)(dP/P);
                              # nanoimprint period reproducibility of about
                              # 0.1% gives 0.001.
ACCEPT_SIN = 0.35             # in-coupler angular acceptance width (sin space).
                              # No longer applied in the transmission cascade:
                              # it was an uncalibrated Gaussian standing in for
                              # a field-resolved coupling efficiency, and under
                              # the per-primary cascade the guiding window
                              # carries the field dependence with real physics
                              # instead. Kept for the acceptance diagnostic
                              # below and for comparison with the angular
                              # acceptance of Zhao et al. 2024.
FOV_DEG = 20.0                # design half-field angle for the FOV metric
                              # (use_pmma() lowers it, see the note there)
TIR_SOFTNESS = 0.005          # sigmoid width of the guiding mask (sin space)

# Parameter bounds (min, max). BOUNDS is mutated in place by use_pmma() and
# use_full(), so FULL_BOUNDS keeps a pristine copy to restore the full design
# space from.
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
FULL_BOUNDS = BOUNDS.clone()   # pristine copy (BOUNDS is mutated by mode switches)
LOG_DIMS = (1, 3)  # alpha and Lc are sampled/normalized in log-space
_FULL_FOV_DEG = 20.0           # default FOV metric angle for the full space

# PMMA-only mode: the material parameters are pinned to PMMA's literature
# range (Nilsen et al., Opt. Express 33(9), 20051-20062 (2025)). The geometry
# is free except the period, which is held to the full-RGB guided window,
#   lambda_red/n_red < period < lambda_blue  ->  635/1.4886 = 426.6 to 450 nm
# with a 430 nm lower bound for a guard band. The 0.7 to 1.1 nm roughness
# window is the spin-coated value: Nilsen et al. show untreated plastic
# waveguides are much rougher than glass and only reach sub-nm quality after
# an acrylic-resin spin-coat, so every result at this sigma assumes the coated
# surface and untreated PMMA is out of scope.
PMMA_BOUNDS = torch.tensor([
    [PMMA_N_532, PMMA_N_532],  # n, pinned material constant
    [5e-5, 5e-4],       # alpha (1/mm)
    [0.7, 1.1],         # sigma (nm), spin-coated PMMA only (see note above)
    [2e5, 4e5],         # Lc (nm)
    [0.3, 2.0],         # t (mm), free geometry
    [430.0, 449.0],     # period (nm), full-RGB guiding window
    [20.0, 400.0],      # depth (nm), free geometry
    [0.2, 0.8],         # duty, free geometry
])


def use_pmma():
    """Switch the engine to the PMMA-only design space (call before anything
    else). Also lowers the FOV evaluation angle: the full-RGB guided field
    window of a n=1.49 single-layer waveguide is only a few degrees wide
    (a direct consequence of the guiding inequality), so evaluating at 20 deg
    would return an honest but uninformative 0 for every design."""
    global FOV_DEG, PMMA_MODE, _DISP_ACTIVE
    BOUNDS.copy_(PMMA_BOUNDS)
    # The full-colour guided window of an n = 1.49 single-layer guide is only
    # about 4 deg wide and sits roughly in [-0.5, +3.8] deg. Evaluating the
    # field metric at 5 deg puts red outside the window, so the metric would
    # report a green-only number at an angle where the image is no longer full
    # colour, and under the per-primary cascade it would collapse to zero for
    # every design and carry no gradient. 3 deg sits inside the window.
    FOV_DEG = 3.0
    PMMA_MODE = True
    _DISP_ACTIVE = True   # PMMA Sellmeier dispersion on
    _rcwa_grid()   # fail loudly now if the calibration grid is missing; a
    # silent scalar fallback would steer every search with the wrong physics,
    # which is the exact failure the checkpoint probe system exists to prevent
    print("[physics] PMMA-only mode: material pinned, geometry free, "
          f"period restricted to RGB-guided window, FOV metric at "
          f"{FOV_DEG:.0f} deg, RCWA-calibrated coupling ({ENGINE_VERSION})")


PMMA_MODE = False


def use_full():
    """Restore the full material design space (inverse of use_pmma()),
    including the full-space FOV metric angle. BOUNDS is mutated in place, so
    without this a process that once called use_pmma() could never get the
    full space or its 20-deg FOV metric back."""
    global FOV_DEG, PMMA_MODE, _DISP_ACTIVE
    BOUNDS.copy_(FULL_BOUNDS)
    FOV_DEG = _FULL_FOV_DEG
    PMMA_MODE = False
    _DISP_ACTIVE = False  # unknown material, so no dispersion offsets


# ----------------------------------------------------------------------------
# RCWA-calibrated grating coupling
# ----------------------------------------------------------------------------

_RCWA_GRID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "rcwa_eta_grid.npz")
_RCWA_GRID = None


def _rcwa_grid():
    """Lazy-load the rigorous calibration grid built by calibrate_rcwa.py.
    eta has shape [pol(TE,TM), wl(450,532,635), n, period, depth, duty]."""
    global _RCWA_GRID
    if _RCWA_GRID is None:
        import numpy as np
        if not os.path.exists(_RCWA_GRID_PATH):
            raise SystemExit(
                f"{_RCWA_GRID_PATH} not found. The {ENGINE_VERSION} engine's "
                "RCWA-calibrated coupling term needs it. Build it once with:\n"
                "    python3 physics/calibrate_rcwa.py")
        d = np.load(_RCWA_GRID_PATH)
        g = {k: torch.as_tensor(d[k], dtype=torch.get_default_dtype())
             for k in ("ns", "periods", "depths", "duties", "eta")}
        g["eta_unpol"] = g["eta"].mean(dim=0)
        # normalizer for the coupler-MTF heuristic: the grid's own unpolarized
        # ceiling at 532 nm (the scalar 4/pi^2 ceiling no longer applies)
        g["coup_ceil_532"] = g["eta_unpol"][1].max()
        _RCWA_GRID = g
    return _RCWA_GRID


def _axis_lerp(axis: torch.Tensor, q: torch.Tensor):
    """Left indices + fractional weights for linear interpolation of q on a
    sorted 1-D axis. q is clamped to the axis range, which queries only leave
    by numerical noise since BOUNDS and the grid cover the same window;
    gradients flow through the fractional weight."""
    qc = q.clamp(axis[0].item(), axis[-1].item())
    idx = torch.searchsorted(axis, qc.detach().contiguous(), right=True)
    idx = idx.clamp(1, len(axis) - 1)
    x0, x1 = axis[idx - 1], axis[idx]
    return idx - 1, (qc - x0) / (x1 - x0)


def _interp_eta_rgb(grid, n, period, depth, duty,
                    pol: str = "unpol") -> torch.Tensor:
    """All three primaries at once, [B,3].

    The interpolation weights over (n, period, depth, duty) do not depend on
    wavelength, so they are computed once and applied to each of the three
    wavelength slices. That makes the per-color cascade cost about the same as
    the old green-only one instead of three times as much."""
    tab = {"TE": grid["eta"][0], "TM": grid["eta"][1],
           "unpol": grid["eta_unpol"]}[pol]                     # [W,N,P,D,U]
    hn, fn = _axis_lerp(grid["ns"], n)
    ip, fp = _axis_lerp(grid["periods"], period)
    jd, fd = _axis_lerp(grid["depths"], depth)
    ku, fu = _axis_lerp(grid["duties"], duty)
    out = []
    for w in range(3):
        t_w = tab[w]
        eta = torch.zeros_like(fp)
        for dn, wn in ((0, 1 - fn), (1, fn)):
            for dp, wp in ((0, 1 - fp), (1, fp)):
                for dd, wd in ((0, 1 - fd), (1, fd)):
                    for du, wu in ((0, 1 - fu), (1, fu)):
                        eta = eta + (wn * wp * wd * wu
                                     * t_w[hn + dn, ip + dp, jd + dd, ku + du])
        out.append(eta)
    return torch.stack(out, dim=1)                              # [B,3]


def eta_rgb(n, period, depth, duty, pol: str = "unpol") -> torch.Tensor:
    """RCWA-calibrated first-order coupling efficiency at all three primaries,
    [B,3]. This is the vector-electromagnetic quantity the cascade runs on."""
    return _interp_eta_rgb(_rcwa_grid(), n, period, depth, duty, pol)


def _interp_eta(grid, n, period, depth, duty, wl_idx: int = 1,
                pol: str = "unpol") -> torch.Tensor:
    """Differentiable multilinear interpolation of the rigorous first-order
    coupling efficiency over (n, period, depth, duty) at one RGB wavelength
    (wl_idx 0/1/2 = 450/532/635 nm) and polarization ('TE'|'TM'|'unpol').
    Piecewise-linear, so exact at grid nodes; the off-grid error is audited
    in results/rcwa_calibration_check.csv. The n axis is real physics, not
    padding: eta_TM moves ~20% across the PMMA bounds [1.48, 1.50]."""
    tab = {"TE": grid["eta"][0], "TM": grid["eta"][1],
           "unpol": grid["eta_unpol"]}[pol][wl_idx]           # [N,P,D,U]
    hn, fn = _axis_lerp(grid["ns"], n)
    ip, fp = _axis_lerp(grid["periods"], period)
    jd, fd = _axis_lerp(grid["depths"], depth)
    ku, fu = _axis_lerp(grid["duties"], duty)
    eta = torch.zeros_like(fp)
    for dn, wn in ((0, 1 - fn), (1, fn)):
        for dp, wp in ((0, 1 - fp), (1, fp)):
            for dd, wd in ((0, 1 - fd), (1, fd)):
                for du, wu in ((0, 1 - fu), (1, fu)):
                    eta = eta + (wn * wp * wd * wu
                                 * tab[hn + dn, ip + dp, jd + dd, ku + du])
    return eta


def eta_rcwa(n, period, depth, duty, wl_idx: int = 1,
             pol: str = "unpol") -> torch.Tensor:
    """RCWA-calibrated first-order coupling efficiency (PMMA window)."""
    return _interp_eta(_rcwa_grid(), n, period, depth, duty, wl_idx, pol)


def eta_first_order(n, period, depth, duty, wl_idx: int = 1,
                    pol: str = "unpol") -> torch.Tensor:
    """The engine's single source of truth for grating coupling efficiency.

    In PMMA mode this is the rigorous RCWA-calibrated interpolation,
    polarization-resolved. In full-material mode it is the scalar binary-
    phase-grating formula (Goodman, Fourier Optics),
    eta_1 = 4 (sin(pi*duty)/pi)^2 sin^2(phi/2) with phi = 2 pi depth (n-1)/
    lambda. The calibration grid only covers the PMMA window, and scalar
    theory is polarization-blind, so `pol` is ignored in the full-material
    branch (Pommet et al. 1994 quantify how far scalar theory drifts at
    roughly one-wavelength features)."""
    if PMMA_MODE:
        return eta_rcwa(n, period, depth, duty, wl_idx=wl_idx, pol=pol)
    lam = float(WL[wl_idx])
    phi = 2 * torch.pi * depth * (n - 1.0) / lam
    return (4.0 * (torch.sin(torch.pi * duty) / torch.pi) ** 2
            * torch.sin(phi / 2) ** 2)


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
    """Map physical theta -> [0,1]^8 (log for LOG_DIMS). Pinned dimensions
    with a zero-width bound, such as n in PMMA mode, map to the constant 0.5
    so the downstream logit/sigmoid parameterizations stay finite."""
    lo, hi = BOUNDS[:, 0], BOUNDS[:, 1]
    span = hi - lo
    pinned = span <= 0
    z = (theta - lo) / torch.where(pinned, torch.ones_like(span), span)
    z = torch.where(pinned.unsqueeze(0), torch.full_like(z, 0.5), z)
    for d in LOG_DIMS:
        if pinned[d]:        # a pinned log dim maps to 0.5 like any other
            z[:, d] = 0.5
            continue
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
# Guided propagation requires 1 < x < n.

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
    """Smooth indicator of the guiding window 1 < x < n. Sigmoid width
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
    """First-order in-guide angles for the RGB design wavelengths, [B,3] rad.
    Uses the per-primary material index n_rgb(n)."""
    x = grating_x(period, WL, field_deg)          # [B,3]
    return diffraction_angle(x, n_rgb(n))


def internal_angle_spread_deg(theta: torch.Tensor) -> torch.Tensor:
    """Diagnostic only: the red-blue first-order in-guide angle difference in
    degrees. In a combiner with matched in/out-coupler periods the out-coupler
    exactly cancels this dispersion for a collimated input, since
    sin(theta_out) = n sin(theta_d) - lambda/period = sin(theta_i), so it is
    not an output chromatic aberration. It is kept here for geometry checks
    but is not part of the spec vector."""
    ang = _diffraction_angles_rgb(theta[:, 0], theta[:, 5])
    return torch.rad2deg(ang[:, 2] - ang[:, 0])


def dispersion_rad_per_nm(n: torch.Tensor, period: torch.Tensor,
                          ang: torch.Tensor) -> torch.Tensor:
    """Single source of truth for the grating angular dispersion
    d(theta_d)/d(lambda) = 1/(n * period * cos(theta_d))  [rad/nm], [B,3].
    Both the walk-off spec metric and the chromatic MTF call this, so the two
    channels stay consistent instead of drifting apart. n is the per-primary
    index n_rgb(n) [B,3]; period [B]; ang [B,3] rad."""
    return 1.0 / (n * period.unsqueeze(1) * torch.cos(ang).clamp(min=0.05))


def pupil_walkoff_mm(theta: torch.Tensor) -> torch.Tensor:
    """Per-primary RMS lateral pupil walk-off, [B,3] mm.

    With matched in/out-coupler periods the output angle is wavelength-
    independent (sin th_out = sin th_i for every lambda, within-band
    included), so the within-band grating dispersion does not survive as an
    angular blur. What does survive is a lateral spread of the exit position
    across the eye pupil: after m = L/(2 t tan th_d) bounce periods the exit
    position is x = m * 2 t * tan(th_d), and at fixed m,
        d x / d lambda = L / (sin th_d cos th_d) * d th_d / d lambda
                       = L / (n * Lambda * sin th_d * cos^2 th_d).
    The LED band (sigma_lam = FWHM/2.355) therefore spreads the exiting beam
    laterally by
        sigma_x = L * sigma_lam / (n Lambda sin th_d cos^2 th_d)   [mm],
    which vignettes the eye pupil, since a displaced collimated beam does not
    shift the retinal image of a distant object. It is fully differentiable
    and grows smoothly as th_d approaches grazing, which is the physical
    reason designs near the edge of the guiding window are poor."""
    n, period = theta[:, 0], theta[:, 5]
    ang = _diffraction_angles_rgb(n, period)                     # [B,3] rad
    sin_d = torch.sin(ang).clamp(min=0.05)
    cos_d = torch.cos(ang).clamp(min=0.05)
    sig_lam = LED_FWHM_NM / 2.355                                # nm RMS
    dth = dispersion_rad_per_nm(n_rgb(n), period, ang)           # rad/nm
    # d x / d lambda = L/(sin cos) * d th/d lambda; sig_lam converts to mm
    return L_PROP_MM * sig_lam * dth / (sin_d * cos_d)


def chromatic_walkoff_mm(theta: torch.Tensor) -> torch.Tensor:
    """Spec metric [2]: photopically weighted RMS chromatic pupil walk-off in
    mm (see pupil_walkoff_mm)."""
    w = V_PHOTOPIC.to(theta.device).unsqueeze(0)
    return (w * pupil_walkoff_mm(theta)).sum(dim=1)


def n_bounces(t: torch.Tensor, theta_d: torch.Tensor) -> torch.Tensor:
    """TIR bounce count over the coupler separation.

    The horizontal advance between SUCCESSIVE surface hits (top->bottom or
    bottom->top) is t*tan(theta_d), so over a coupler separation L the beam
    hits a surface N = L / (t tan(theta_d)) times, and the total in-glass
    path is N * t / cos(theta_d) = L / sin(theta_d). Counting L/(2 t tan)
    instead would count one hit per full zig-zag period rather than two, which
    makes both N_b and the Beer-Lambert path a factor of two too short.
    Clamped to [1, N_BOUNCE_MAX]."""
    return (L_PROP_MM / (t * torch.tan(theta_d))).clamp(1.0, N_BOUNCE_MAX)


def fresnel_T(n: torch.Tensor, field_deg: float, pol: str) -> torch.Tensor:
    """Exact single-interface air->medium power transmittance at incidence
    field_deg for polarization 'TE' (s) or 'TM' (p). (Hecht, Optics, ch. 4;
    Born & Wolf ch. 1.) By reciprocity the medium->air transmittance at the
    conjugate internal angle is identical, so a two-interface path costs
    fresnel_T**2. This factor is used in the full-material (scalar) cascade
    and in the validation suite; in PMMA mode the RCWA efficiency already
    contains the interface losses, so it is not applied there."""
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
    """Single-polarization double-coupler throughput, resolved per primary.

    Returns [B,3] for blue, green, and red. Every wavelength-dependent term is
    evaluated at its own wavelength and its own material index: the grating
    equation, the guiding window, the in-guide angle, the bounce count, the
    Beer-Lambert path, the Debye-Waller roughness exponent, the Fresnel
    coefficients, and the RCWA coupling efficiency. Evaluating the cascade at
    green alone and calling the result full-color hides exactly the effect
    that matters most here, since the red order sits closest to the edge of
    the guiding window and couples several times more weakly than blue.

    This is a relative figure of merit for ranking designs at one eye
    position. It does not model exit-pupil expansion, so it is not a device
    efficiency.

    Cascade (all terms dimensionless power fractions, per primary):
      guided-mask x eta_in x Beer-Lambert x Tien-roughness
      x re-interaction survival x eta_out

    The device has surface-relief couplers on the faces the light enters and
    exits through. The RCWA efficiency is normalized to incident power with
    the wave launched from air through the corrugated interface, so eta_in
    already includes the entry-face reflection loss, and by reciprocity
    eta_out includes the exit-face loss; there is no separate flat-interface
    Fresnel factor in the PMMA path. Between the couplers the light propagates
    by lossless TIR, with bulk absorption and roughness scatter carrying the
    losses.

    One approximation is worth stating: the RCWA grid is solved at normal
    incidence and reused across the field, so the angular dependence carried
    here is the guiding window and the grating equation rather than a
    field-resolved coupling efficiency. Over the few-degree guided window of a
    low-index substrate that is a small effect, and the guiding window is the
    term that actually decides the field edge.
    """
    n, alpha, sigma, Lc, t, period, depth, duty = theta.unbind(dim=1)

    n_k = n_rgb(n)                                  # [B,3] per-primary index
    t_k = t.unsqueeze(1)                            # [B,1] broadcast over color
    lam_mm = (WL.to(theta.device) * 1e-6).unsqueeze(0)   # [1,3] mm

    # grating geometry at this field angle, per primary
    x = grating_x(period, WL, field_deg)            # [B,3]
    mask = guided_mask(x, n_k)                      # [B,3] guiding window
    ang = diffraction_angle(x, n_k)                 # [B,3] in-guide angle

    # bulk Beer-Lambert loss along each primary's own zig-zag path
    NB = n_bounces(t_k, ang)                        # [B,3]
    path_mm = NB * t_k / torch.cos(ang)
    T_bulk = torch.exp(-alpha.unsqueeze(1) * path_mm)

    # Roughness scatter per TIR bounce (Tien 1971, Appl. Opt. 10, 2395): a
    # specular loss per bounce of exp[-(4 pi sigma n cos(theta)/lambda)^2],
    # weighted by a Payne & Lacey (1994) correlation term S(Lc) in (0,1] where
    # a longer correlation length keeps more of the scattered lobe inside the
    # guided beam. Both the wavelength in the exponent and the bounce count are
    # per primary, which is why a rough surface shifts the colour balance
    # rather than just dimming the image.  # SYNC S(Lc)
    per_bounce = (4 * torch.pi * (sigma.unsqueeze(1) * 1e-6) * n_k
                  * torch.cos(ang) / lam_mm) ** 2
    S_corr = (1.0 / (1.0 + (Lc / 3e5))).unsqueeze(1)
    T_scatter = torch.exp(-per_bounce * S_corr * NB)

    # Grating coupling, rigorous and polarization-resolved at all three
    # primaries. In the full-material space the calibration grid does not
    # apply, so the scalar formula is evaluated per primary instead.
    if PMMA_MODE:
        eta = eta_rgb(n, period, depth, duty, pol=pol)          # [B,3]
    else:
        eta = torch.stack([eta_first_order(n, period, depth, duty,
                                           wl_idx=k, pol=pol)
                           for k in range(3)], dim=1)

    # Interface loss, which depends on the branch: the RCWA efficiency already
    # carries the corrugated-interface reflection, but the scalar Goodman
    # efficiency is a pure phase-grating term with no interface physics, so the
    # full-material branch keeps the flat-interface factor, evaluated at each
    # primary's own index, and the PMMA branch does not.
    T_iface = 1.0 if PMMA_MODE else fresnel_T(n_k, field_deg, pol) ** 2

    # In-coupler re-interaction. While the bounce advance 2 t tan(th_d) is
    # shorter than the in-coupler aperture W_IN_MM, the guided beam re-hits the
    # grating and each encounter re-diffracts a fraction of order eta back out
    # (the Zhao et al. 2024 mechanism, taken to first order with the same eta
    # by reciprocity). Red advances fastest per bounce, so it re-crosses the
    # coupler fewest times. The relu keeps the factor exactly 1 for thick
    # guides whose first bounce already clears the coupler.
    m_re = torch.relu(W_IN_MM / (2.0 * t_k * torch.tan(ang)) - 1.0)
    T_reint = (1.0 - eta.clamp(max=0.95)) ** m_re

    T_grating = eta * T_reint * eta                 # in x survival x out

    return mask * T_iface * T_bulk * T_scatter * T_grating


def transmission_rgb(theta: torch.Tensor, field_deg: float = 0.0,
                     pol: str = "unpol") -> torch.Tensor:
    """Per-primary double-coupler throughput, [B,3] for blue, green, red.
    pol in {'TE','TM','unpol'}; unpolarized is (T_TE + T_TM)/2, the average of
    the two power transmissions."""
    if pol == "unpol":
        return 0.5 * (_transmission_pol(theta, field_deg, "TE")
                      + _transmission_pol(theta, field_deg, "TM"))
    return _transmission_pol(theta, field_deg, pol)


def transmission(theta: torch.Tensor, field_deg: float = 0.0,
                 pol: str = "unpol") -> torch.Tensor:
    """White-balanced double-coupler throughput, [B]: the WORST primary.

    A full-colour display has to hit a white point, so the usable throughput
    of a fixed projector budget is set by the weakest channel; the stronger
    channels are driven down to match it. Reporting green alone, or an average
    over primaries, credits the design for light it cannot actually use, which
    is what a green-only cascade did. Use transmission_rgb() for the
    per-primary numbers and colour_balance() for how far apart they sit.

    This remains a relative figure of merit for ranking designs at one eye
    position, not a device efficiency."""
    return transmission_rgb(theta, field_deg, pol).min(dim=1).values


def transmission_photopic(theta: torch.Tensor, field_deg: float = 0.0,
                          pol: str = "unpol") -> torch.Tensor:
    """Photopically weighted mean over primaries, [B]. Defensible only for a
    projector whose primary powers are provisioned to the same weights; state
    the assumed powers if this is quoted. transmission() (worst primary) is
    the conservative default."""
    w = V_PHOTOPIC.to(theta.device).unsqueeze(0)
    return (w * transmission_rgb(theta, field_deg, pol)).sum(dim=1)


def colour_balance(theta: torch.Tensor, field_deg: float = 0.0,
                   pol: str = "unpol") -> torch.Tensor:
    """Ratio of the weakest to the strongest primary throughput, [B] in (0,1].
    1.0 is a perfectly balanced white; small values mean the image is usable
    only after throwing away most of the strong channels."""
    T = transmission_rgb(theta, field_deg, pol)
    return T.min(dim=1).values / (T.max(dim=1).values + 1e-12)


def transmission_polarized(theta: torch.Tensor, field_deg: float = 0.0):
    """Convenience: dict with per-primary and white-balanced TE, TM, unpol,
    plus the TE/TM diattenuation of the white-balanced number.

    Note that the white balance is a minimum over primaries and therefore does
    not commute with the TE/TM average: the weakest primary under TE need not
    be the weakest under TM. The unpolarized entry here is the min of the
    per-primary unpolarized transmission, which is the physically meaningful
    order of operations and matches transmission(). The (T_TE + T_TM)/2
    identity holds per primary, on the _rgb entries."""
    te = _transmission_pol(theta, field_deg, "TE")
    tm = _transmission_pol(theta, field_deg, "TM")
    unpol_rgb = 0.5 * (te + tm)
    te_w, tm_w = te.min(dim=1).values, tm.min(dim=1).values
    return {"TE": te_w, "TM": tm_w, "unpol": unpol_rgb.min(dim=1).values,
            "TE_rgb": te, "TM_rgb": tm, "unpol_rgb": unpol_rgb,
            "diattenuation": (te_w - tm_w) / (te_w + tm_w + 1e-12)}


# ----------------------------------------------------------------------------
# Feasibility / FOV analysis
# ----------------------------------------------------------------------------

def tir_penalty(theta: torch.Tensor, field_deg=None,
                margin: float = 0.01) -> torch.Tensor:
    """Differentiable penalty > 0 when any RGB wavelength leaves the guiding
    window 1+margin < x < n-margin. By default it is evaluated at the field
    centre and both field edges (+/- FOV_DEG), not just at normal incidence:
    checking only the centre lets a design sit just past evanescence at the
    field edge while the sigmoid mask's tail still scores it. `margin` is the
    explicit guard band in x-space.

    field_deg: None -> (0, +FOV_DEG, -FOV_DEG); or a single float."""
    n, period = theta[:, 0], theta[:, 5]
    fields = (0.0, float(FOV_DEG), -float(FOV_DEG)) if field_deg is None \
        else (float(field_deg),)
    pen = torch.zeros_like(n)
    nk = n_rgb(n)                                              # [B,3]
    for fd in fields:
        x = grating_x(period, WL, fd)                          # [B,3]
        lo = torch.relu((1.0 + margin) - x)
        hi = torch.relu(x - (nk - margin))
        pen = pen + (lo + hi).sum(dim=1)
    return pen


def hard_guided_ok(theta: torch.Tensor, field_deg=None) -> torch.Tensor:
    """Boolean audit with no sigmoid and no relaxation: True iff every RGB
    order is strictly inside 1 < x < n at every checked field angle. Use it to
    re-score finalists; a design that fails it is exploiting the soft mask."""
    return tir_penalty(theta, field_deg=field_deg, margin=0.0) <= 0.0


def fov_window_deg(theta: torch.Tensor):
    """Exact guided field window per design (degrees, in air), i.e. the range
    of incidence angles theta_i with 1 < sin(theta_i)+lambda/period < n for
    all RGB wavelengths simultaneously. Returns (lo_deg, hi_deg, width_deg);
    width 0 means no common full-RGB window. This is the fundamental
    index-limited FOV of a diffractive waveguide (Kress & Chatterjee 2021)."""
    n, period = theta[:, 0], theta[:, 5]
    lam = WL.to(period.device)                                 # [3]
    lo_sin = (1.0 - lam.unsqueeze(0) / period.unsqueeze(1))    # [B,3]
    hi_sin = (n_rgb(n) - lam.unsqueeze(0) / period.unsqueeze(1))
    lo = lo_sin.max(dim=1).values.clamp(-_SAFE_SIN, _SAFE_SIN)
    hi = hi_sin.min(dim=1).values.clamp(-_SAFE_SIN, _SAFE_SIN)
    lo_deg = torch.rad2deg(torch.asin(lo))
    hi_deg = torch.rad2deg(torch.asin(hi))
    return lo_deg, hi_deg, (hi_deg - lo_deg).clamp(min=0.0)


# ----------------------------------------------------------------------------
# MTF cascade (batched: theta [B,8] -> [B])
# ----------------------------------------------------------------------------

def watson_eye_mtf(f_cyc_mm, pupil_mm) -> torch.Tensor:
    """Watson (2013), J. Vis. 13(6):18, mean human optical MTF.

        M(u, d) = sqrt(D(u, d, 555 nm)) * [1 + (u/u1(d))^2]^(-0.62)
        u1(d)   = 21.95 - 5.512 d + 0.3922 d^2      [cyc/deg]

    where D is the diffraction-limited MTF of a circular pupil of diameter
    d mm at 555 nm and u is spatial frequency in cyc/deg. This is Watson's
    fit to the population-mean wavefront-derived MTF of 200 eyes (his
    "Formula 8 x sqrt(MTF_DL)" variant with the Lorentzian exponent fixed at
    0.62). Because real eyes are aberrated, it sits well below the diffraction
    limit, about 0.49 against 0.85 at 3 mm and 11.9 cyc/deg. The fit is valid
    for d in [2, 6] mm; callers clamp d >= 1 mm, and values in [1, 2) mm are a
    smooth extrapolation.

    f_cyc_mm: retinal spatial frequency in cyc/mm (converted internally via
    EYE_FL_MM). pupil_mm: tensor of pupil diameters, mm. Differentiable."""
    d = pupil_mm
    u = f_cyc_mm * EYE_FL_MM * torch.pi / 180.0        # cyc/mm -> cyc/deg
    u = torch.as_tensor(u, dtype=d.dtype, device=d.device)
    # diffraction-limited component at 555 nm: cutoff u0 in cyc/deg
    u0 = (d * 1e6 / 555.0) * torch.pi / 180.0
    r = (u / u0).clamp(max=0.999)
    D = (2 / torch.pi) * (torch.acos(r) - r * torch.sqrt(1 - r ** 2))
    u1 = 21.95 - 5.512 * d + 0.3922 * d ** 2
    return torch.sqrt(D) * (1 + (u / u1) ** 2) ** (-0.62)


def mtf_system(theta: torch.Tensor) -> torch.Tensor:
    """Five-component multiplicative MTF cascade at F0_CYC_PER_MM (green)."""
    n, alpha, sigma, Lc, t, period, depth, duty = theta.unbind(dim=1)
    f = F0_CYC_PER_MM  # cyc/mm at the retina-conjugate image plane

    # 1) Human-eye optical MTF from the Watson (2013) mean-eye formula,
    #    evaluated per primary at a pupil reduced by the walk-off: the
    #    within-band chromatic walk-off sigma_x shrinks the common filled
    #    aperture to d_eff = D - sigma_x (smoothly clamped to >= 1 mm), which
    #    is where the non-cancelling chromatic penalty lands, as pupil
    #    vignetting rather than retinal blur. The overlap D - sigma_x is a
    #    first-order aperture model; the walk-off length itself is exact
    #    geometry.  # SYNC overlap model
    sig_x = pupil_walkoff_mm(theta)                              # [B,3] mm
    d_eff = 1.0 + torch.nn.functional.softplus(
        PUPIL_MM - sig_x - 1.0, beta=10.0)                       # [B,3] >= ~1
    w = V_PHOTOPIC.to(theta.device).unsqueeze(0)                 # [1,3]
    mtf_eye = (w * watson_eye_mtf(f, d_eff)).sum(dim=1)          # [B]

    # 2) Roughness MTF: Gaussian attenuation from scatter-induced angular blur
    #    (heuristic coefficient)  # SYNC
    blur_rough = (sigma / 6.0) ** 2 * (1.0 / (1.0 + Lc / 3e5)) * 8e-3
    mtf_rough = torch.exp(-2 * (torch.pi * blur_rough * f) ** 2)

    # 3) Chromatic MTF. The only angular chromatic residual of a
    #    matched-coupler combiner is the in/out period-mismatch tolerance
    #    (RESID_DISP, about 0.1% nanoimprint reproducibility). Both the
    #    line-centre RGB displacement and the within-band fan survive at that
    #    fractional level, so both terms below scale with RESID_DISP. The
    #    non-cancelling within-band effect, the pupil walk-off, is carried by
    #    term (1), not here. This is the exact modulus of the photopically
    #    weighted three-line complex sum (Thibos 1987), each line attenuated by
    #    its finite-bandwidth Gaussian envelope.
    ang = _diffraction_angles_rgb(n, period)                     # [B,3] rad
    dtheta = ang - ang[:, 1:2]
    x_k = EYE_FL_MM * RESID_DISP * dtheta                        # [B,3] mm
    dth_dlam = dispersion_rad_per_nm(n_rgb(n), period, ang)      # rad/nm
    sig_lam = LED_FWHM_NM / 2.355
    x_sig = EYE_FL_MM * RESID_DISP * dth_dlam * sig_lam          # [B,3] mm
    env = torch.exp(-2 * (torch.pi * f * x_sig) ** 2)            # [B,3]
    re = (w * env * torch.cos(2 * torch.pi * f * x_k)).sum(dim=1)
    im = (w * env * torch.sin(2 * torch.pi * f * x_k)).sum(dim=1)
    mtf_chrom = torch.sqrt(re ** 2 + im ** 2 + 1e-12)

    # 4) Grating MTF: contrast loss from periodic wavefront modulation, scaling
    #    with phase depth (heuristic coefficient 0.15)  # SYNC
    phi = 2 * torch.pi * depth * (n - 1.0) / 532.0
    mtf_grat = 1.0 - 0.15 * torch.sin(phi / 2) ** 2

    # 5) Coupler MTF: contrast degradation from finite diffraction efficiency
    #    across two coupling events (Goodsell et al. 2024 framing). eta comes
    #    from eta_first_order(), normalized by its attainable ceiling: the
    #    grid's own unpolarized 532 nm maximum in PMMA mode, and the 4/pi^2
    #    scalar ceiling in the full space (heuristic)  # SYNC
    eta = eta_first_order(n, period, depth, duty, wl_idx=1, pol="unpol")
    ceil = _rcwa_grid()["coup_ceil_532"] if PMMA_MODE else 0.4053
    mtf_coup = 0.80 + 0.20 * (eta / ceil)

    return mtf_eye * mtf_rough * mtf_chrom * mtf_grat * mtf_coup


def forward_model(theta: torch.Tensor) -> torch.Tensor:
    """theta [B,8] -> spec y [B,4]: [MTF, T_FOM, walkoff_mm, T_FOM_at_FOV].
    Transmissions are unpolarized and white-balanced, meaning the worst
    primary; use transmission_rgb() for the per-primary numbers and
    transmission_polarized() for TE/TM."""
    return torch.stack([
        mtf_system(theta),
        transmission(theta, field_deg=0.0),
        chromatic_walkoff_mm(theta),
        transmission(theta, field_deg=FOV_DEG),
    ], dim=1)


# Spec normalization scales, to keep the losses balanced across metrics and to
# set the relative weight of each term in the design objective.
#
# Indices 1 and 3 are throughput. The 0.10 used before the cascade became
# wavelength-resolved was sized for the old green-only number, which was about
# a hundred times larger than the white-balanced one. Left at 0.10 the
# throughput term contributed under 1% of the objective, so the search was
# effectively blind to it and drifted on MTF and walk-off alone. The
# white-balanced throughput has a median of 2.8e-4 and a ceiling near 1.4e-3
# over the guided design space, so 1e-3 puts a good design at order unity,
# comparable to the MTF term.
#
# Index 2 is the walk-off in mm scaled by the eye-pupil diameter, so
# walkoff/PUPIL is the fraction of the pupil lost to chromatic vignetting.
SPEC_SCALE = torch.tensor([1.0, 1e-3, PUPIL_MM, 1e-3])


def normalize_spec(y: torch.Tensor) -> torch.Tensor:
    return y / SPEC_SCALE.to(y.device)


if __name__ == "__main__":
    torch.manual_seed(0)
    use_pmma()
    th = sample_theta(5)
    y = forward_model(th)
    print("theta sample:\n", th)
    print("spec [MTF, T_FOM, walkoff_mm, T_fov]:\n", y)
    # sanity: n must be pinned to the material constant in PMMA mode
    assert torch.allclose(th[:, 0], torch.full_like(th[:, 0], PMMA_N_532)), \
        "PMMA mode must pin n to PMMA_N_532"
    # sanity: the Watson mean-eye MTF at 3 mm and 40 cyc/mm should sit near
    # 0.49 for an aberrated eye, well below the 0.85 diffraction limit
    m_eye = watson_eye_mtf(F0_CYC_PER_MM, torch.tensor([PUPIL_MM]))
    print(f"Watson eye MTF @40cyc/mm, 3mm pupil: {m_eye.item():.4f} "
          "(diffraction limit would be 0.847)")
    assert 0.40 < m_eye.item() < 0.60, "Watson term out of expected range"
    # per-primary throughput: the spread across colours is the whole point of
    # the wavelength-resolved cascade
    Trgb = transmission_rgb(th)
    print("T per primary (B,G,R):")
    for row, bal in zip(Trgb.tolist(), colour_balance(th).tolist()):
        print(f"   B {row[0]:.5f}  G {row[1]:.5f}  R {row[2]:.5f}   "
              f"worst/best = {bal:.3f}")
    print("T white-balanced (worst primary):", transmission(th).tolist())
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
    bad[:, 5] = 680.0  # a period outside the guided window
    print("unguided-design T (must be ~0):", transmission(bad).max().item())
    # sanity: the calibrated coupling at the reference design (n=1.5, period
    # 438 nm, depth 400 nm, duty 0.5) should reproduce the rigorous RCWA
    # values TE 0.0479, TM 0.0878, unpol 0.0679, nowhere near the scalar 0.347
    rec = (torch.tensor([1.5]), torch.tensor([437.98]),
           torch.tensor([400.0]), torch.tensor([0.5001]))
    e_te = eta_rcwa(*rec, wl_idx=1, pol="TE")
    e_tm = eta_rcwa(*rec, wl_idx=1, pol="TM")
    print(f"RCWA eta at reference design (532nm): TE {e_te.item():.4f} "
          f"TM {e_tm.item():.4f} unpol {(0.5*(e_te+e_tm)).item():.4f}  "
          f"(rigorous: TE 0.0479 TM 0.0878 unpol 0.0679; scalar would say 0.347)")
