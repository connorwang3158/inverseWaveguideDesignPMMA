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

============================ v3 PHYSICS REVISION ==============================
RCWA-CALIBRATED GRATING COUPLING (PMMA mode). The 2026-07-13 rigor audit
(results/design_rcwa_check_na.csv) showed that at the TIR-mandated PMMA
periods (430-449 nm, below the visible wavelengths) the scalar first-order
efficiency overestimates rigorous RCWA by ~5x at 532 nm and ~15x at 635 nm,
and drives the depth optimum to the 400 nm bound when the rigorous optimum is
near 200 nm (Pommet et al., JOSA A 11, 1827 (1994) predict exactly this
scalar breakdown at ~lambda-scale features). In PMMA mode the engine now
interpolates eta_1(period, depth, duty; lambda, pol) from a rigorous grcwa
grid (physics/rcwa_eta_grid.npz, built by physics/calibrate_rcwa.py,
off-grid-verified in results/rcwa_calibration_check.csv). Consequences:
  * absolute transmission and the depth optimum are now quoted at the
    rigorous (L2) level inside the design loop itself;
  * grating coupling is polarization-resolved (TE vs TM), completing FIX-3 —
    scalar theory was pol-blind;
  * the grid carries its own refractive-index axis (n = 1.48/1.49/1.50):
    eta_TM was measured to move ~20% across the PMMA index bounds, so n
    interpolates like the geometry instead of being pinned at the midpoint;
  * full-material mode (use_full) keeps the scalar term — the grid covers
    only the PMMA window; full-space numbers remain v2-level accuracy;
  * records/tables from the v2 engine are NOT comparable — searchers key
    their hall-of-fame and run tables on ENGINE_VERSION below.
The physics-probe checkpoint system (networks/surrogate.py) detects this
revision automatically and refuses stale v2-trained surrogates.
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

import os

import torch

# Engine revision. Bumped whenever the physics changes enough to reset
# cross-run comparability (v2 = TIR-constrained scalar coupling; v3 =
# RCWA-calibrated coupling). Searchers and run tables key their output
# filenames on this so records from different engines never mix.
ENGINE_VERSION = "v4"

# Source spectral width per display primary (LED FWHM, nm). Drives the
# non-cancelling chromatic metric (FIX-8) and the finite-bandwidth chromatic
# MTF (FIX-9). Typical micro-LED / LED-illuminated LCoS primaries: 20-25 nm.
LED_FWHM_NM = 25.0

# FIX-11 (v4): n IS A MATERIAL PROPERTY, NOT THREE FREE CHOICES. The design
# variable theta[0] is the index AT GREEN (532 nm); in PMMA mode the blue and
# red indices follow PMMA's Sellmeier dispersion (Sultanova, Kasarova &
# Nikolov, Acta Phys. Pol. A 116, 585 (2009)): n(450)=1.5006, n(532)=1.4937,
# n(635)=1.4886 -> fixed offsets relative to green. The v3 engine let the
# optimizer pick a single n=1.500 for all three colours — a value PMMA does
# not have at red, exactly where the x < n constraint binds (2026-07-17
# audit, §2.8). Offsets are zero in full-material mode (unknown material).
PMMA_DISP_OFFSET = torch.tensor([+0.00692, 0.0, -0.00506])  # B, G, R vs green
_DISP_ACTIVE = False  # toggled by use_pmma()/use_full()


def n_rgb(n: torch.Tensor) -> torch.Tensor:
    """Per-primary refractive index [B,3]: design n (green) + material
    dispersion offsets (PMMA mode only)."""
    off = PMMA_DISP_OFFSET.to(n.device) if _DISP_ACTIVE \
        else torch.zeros(3, device=n.device)
    return n.unsqueeze(1) + off.unsqueeze(0)

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
RESID_DISP = 0.001            # v4 (FIX-9): residual dispersion fraction
                              # reaching the retina. For matched in/out coupler
                              # periods the out-coupler cancels the in-guide
                              # dispersion EXACTLY for a collimated input
                              # (sin(th_out) = n sin(th_d) - lambda/period
                              # = sin(th_i), wavelength-independent), including
                              # within each primary's band. What survives is
                              # the in/out PERIOD MISMATCH from fabrication:
                              # d(sin th_out) = (lambda/period)(dP/P). NIL
                              # period reproducibility ~0.1% -> 0.001. The v3
                              # value 0.10 was an uncalibrated fudge that both
                              # exaggerated retinal colour 100x and created
                              # the fringe-lottery failure (2026-07-17 audit
                              # §2.1/§2.4).
ACCEPT_SIN = 0.35             # in-coupler angular acceptance width (sin space)
                              # cf. Zhao et al., Opt. Express 32, 12340 (2024)  # SYNC
FOV_DEG = 20.0                # design half-field angle for the FOV metric
                              # (use_pmma() lowers it -- see FIX-1/FIX-2 note)
TIR_SOFTNESS = 0.005          # sigmoid width of the guiding mask (sin space)

# Parameter bounds (min, max) — from Paper 1's literature-reported material ranges
# NOTE: BOUNDS is mutated in place by use_pmma()/use_full(); FULL_BOUNDS below
# keeps a pristine copy so the full design space can always be restored.
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
    global FOV_DEG, PMMA_MODE, _DISP_ACTIVE
    BOUNDS.copy_(PMMA_BOUNDS)
    FOV_DEG = 5.0
    PMMA_MODE = True
    _DISP_ACTIVE = True   # FIX-11: PMMA Sellmeier dispersion on
    _rcwa_grid()   # v3: fail loudly NOW if the calibration grid is missing —
    # a silent scalar fallback would be exactly the "stale physics steering"
    # bug class the checkpoint probe system exists to prevent
    print("[physics] PMMA-only mode: material pinned, geometry free, "
          f"period restricted to RGB-guided window, FOV metric at "
          f"{FOV_DEG:.0f} deg, RCWA-calibrated coupling ({ENGINE_VERSION})")


PMMA_MODE = False


def use_full():
    """Restore the full material design space (inverse of use_pmma()),
    including the full-space FOV metric angle. Needed because BOUNDS is
    mutated in place — without this, a process that once called use_pmma()
    could never get the full space (or its 20-deg FOV metric) back."""
    global FOV_DEG, PMMA_MODE, _DISP_ACTIVE
    BOUNDS.copy_(FULL_BOUNDS)
    FOV_DEG = _FULL_FOV_DEG
    PMMA_MODE = False
    _DISP_ACTIVE = False  # FIX-11: unknown material -> no dispersion offsets


# ----------------------------------------------------------------------------
# v3: RCWA-calibrated grating coupling (see the v3 PHYSICS REVISION block)
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
                f"{_RCWA_GRID_PATH} not found — the {ENGINE_VERSION} engine's "
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
    sorted 1-D axis. q is clamped to the axis range (queries only leave it by
    numerical noise — BOUNDS and the grid cover the same window); gradients
    flow through the fractional weight."""
    qc = q.clamp(axis[0].item(), axis[-1].item())
    idx = torch.searchsorted(axis, qc.detach().contiguous(), right=True)
    idx = idx.clamp(1, len(axis) - 1)
    x0, x1 = axis[idx - 1], axis[idx]
    return idx - 1, (qc - x0) / (x1 - x0)


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
    """RCWA-calibrated first-order coupling efficiency (v3, PMMA window)."""
    return _interp_eta(_rcwa_grid(), n, period, depth, duty, wl_idx, pol)


def eta_first_order(n, period, depth, duty, wl_idx: int = 1,
                    pol: str = "unpol") -> torch.Tensor:
    """The engine's single source of truth for grating coupling efficiency.

    PMMA mode (v3): rigorous RCWA-calibrated interpolation, polarization-
    resolved. Full-material mode: the scalar binary-phase-grating formula
    (Goodman, Fourier Optics), eta_1 = 4 (sin(pi*duty)/pi)^2 sin^2(phi/2)
    with phi = 2 pi depth (n-1)/lambda — the calibration grid covers only
    the PMMA window, and scalar theory is polarization-blind, so `pol` is
    ignored there (Pommet et al. 1994 quantify the scalar error at
    ~1-lambda features; full-space numbers remain v2-level accuracy)."""
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
    """First-order in-guide angles for the RGB design wavelengths, [B,3] rad.
    FIX-11: uses the per-primary material index n_rgb(n)."""
    x = grating_x(period, WL, field_deg)          # [B,3]
    return diffraction_angle(x, n_rgb(n))


def internal_angle_spread_deg(theta: torch.Tensor) -> torch.Tensor:
    """DIAGNOSTIC ONLY (was the v2/v3 'chromatic spread' spec metric).

    Red-blue first-order IN-GUIDE angle difference (deg). In a combiner with
    matched in/out-coupler periods the out-coupler exactly cancels this
    dispersion for a collimated input (sin(theta_out) = n sin(theta_d)
    - lambda/period = sin(theta_i)); it is therefore NOT an output chromatic
    aberration, and using it as one drove the v3 optimizer toward the top of
    the period window to minimise a quantity a real combiner cancels
    (2026-07-17 audit, §2.1). Kept for geometry diagnostics; removed from the
    spec vector."""
    ang = _diffraction_angles_rgb(theta[:, 0], theta[:, 5])
    return torch.rad2deg(ang[:, 2] - ang[:, 0])


def chromatic_spread_deg(theta: torch.Tensor) -> torch.Tensor:
    """FIX-8 (v4): NON-CANCELLING chromatic blur from source spectral width.

    Each display primary has finite bandwidth (LED FWHM ~ LED_FWHM_NM). The
    in-guide angular dispersion of the grating,
        d(theta_d)/d(lambda) = 1 / (n * period * cos(theta_d)),
    spreads that band into a fan of guided angles. Unlike the line-centre
    dispersion, the WITHIN-BAND fan does not cancel at a matched out-coupler
    into a single collimated output: the angular fan maps into spatial
    walk-off across the pupil and residual angular blur (the true chromatic
    blur source in matched-grating combiners; 2026-07-17 audit §2.1
    recommends exactly d(theta_d)/d(lambda) * d(lambda)_LED).

    Metric: photopically weighted RGB average of
        rad2deg( LED_FWHM_NM / (n * period * cos(theta_d_k)) )   [deg].
    Reduces smoothly toward 0 for large periods/small dispersion; fully
    differentiable."""
    n, period = theta[:, 0], theta[:, 5]
    ang = _diffraction_angles_rgb(n, period)                     # [B,3] rad
    dth_dlam = 1.0 / (n_rgb(n) * period.unsqueeze(1)
                      * torch.cos(ang).clamp(min=0.05))          # rad/nm
    blur = torch.rad2deg(dth_dlam * LED_FWHM_NM)                 # [B,3] deg
    w = V_PHOTOPIC.to(theta.device).unsqueeze(0)
    return (w * blur).sum(dim=1)


def n_bounces(t: torch.Tensor, theta_d: torch.Tensor) -> torch.Tensor:
    """TIR bounce count over the coupler separation (FIX-4, corrected FIX-7).

    The horizontal advance between SUCCESSIVE surface hits (top->bottom or
    bottom->top) is t*tan(theta_d), so over a coupler separation L the beam
    hits a surface N = L / (t tan(theta_d)) times, and the total in-glass
    path is N * t / cos(theta_d) = L / sin(theta_d). The v2/v3 engine used
    L/(2 t tan) — one hit per full zig-zag PERIOD instead of two — making
    both N_b and the Beer-Lambert path exactly 2x short (2026-07-17 audit,
    §2.9). Clamped to [1, N_BOUNCE_MAX]."""
    return (L_PROP_MM / (t * torch.tan(theta_d))).clamp(1.0, N_BOUNCE_MAX)


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

    # --- Grating coupling (v3): eta_first_order() owns the mode branch —
    #     rigorous RCWA-calibrated per-polarization efficiency in PMMA mode
    #     (scalar theory overshoots ~5-15x at the sub-wavelength PMMA periods
    #     and is pol-blind, see the v3 REVISION block), scalar Goodman formula
    #     in the full space. Angular acceptance applies to the in-coupler only
    #     (FIX-5); the out-coupler sees the guided angle by construction.
    eta = eta_first_order(n, period, depth, duty, wl_idx=1, pol=pol)
    s_i = torch.sin(torch.deg2rad(torch.tensor(float(field_deg),
                                               device=theta.device)))
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

def tir_penalty(theta: torch.Tensor, field_deg=None,
                margin: float = 0.01) -> torch.Tensor:
    """Differentiable penalty > 0 when any RGB wavelength leaves the guiding
    window 1+margin < x < n-margin. FIX-10 (v4): evaluated at BOTH the field
    centre and the field edges (+/- FOV_DEG) by default. The v3 penalty was
    normal-incidence-only, so the search parked the record design with red
    0.85 sigmoid-widths PAST evanescence at theta_i = +FOV (2026-07-17
    audit, §2.2); the sigmoid mask's tail let it score anyway. `margin` is
    the explicit guard band in x-space.

    field_deg: None -> (0, +FOV_DEG, -FOV_DEG); or a single float."""
    n, period = theta[:, 0], theta[:, 5]
    fields = (0.0, float(FOV_DEG), -float(FOV_DEG)) if field_deg is None \
        else (float(field_deg),)
    pen = torch.zeros_like(n)
    nk = n_rgb(n)                                              # FIX-11 [B,3]
    for fd in fields:
        x = grating_x(period, WL, fd)                          # [B,3]
        lo = torch.relu((1.0 + margin) - x)
        hi = torch.relu(x - (nk - margin))
        pen = pen + (lo + hi).sum(dim=1)
    return pen


def hard_guided_ok(theta: torch.Tensor, field_deg=None) -> torch.Tensor:
    """Boolean audit (no sigmoid, no relaxation): True iff every RGB order is
    STRICTLY inside 1 < x < n at every checked field angle. Use to re-score
    finalists — a design that fails this is farming the soft mask."""
    return tir_penalty(theta, field_deg=field_deg, margin=0.0) <= 0.0


def fov_window_deg(theta: torch.Tensor):
    """Exact guided field window per design (degrees, in air), i.e. the range
    of incidence angles theta_i with 1 < sin(theta_i)+lambda/period < n for
    ALL RGB wavelengths simultaneously. Returns (lo_deg, hi_deg, width_deg);
    width 0 means no common full-RGB window. This is the fundamental
    index-limited FOV of a diffractive waveguide (Kress & Chatterjee 2021)."""
    n, period = theta[:, 0], theta[:, 5]
    lam = WL.to(period.device)                                 # [3]
    lo_sin = (1.0 - lam.unsqueeze(0) / period.unsqueeze(1))    # [B,3]
    hi_sin = (n_rgb(n) - lam.unsqueeze(0) / period.unsqueeze(1))  # FIX-11
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
    xx = torch.tensor(min(f / fc, 0.999), device=theta.device)
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
    #    FIX-9 (v4): each primary is a FINITE-BANDWIDTH line, not a delta.
    #    The within-band grating dispersion d(theta_d)/d(lambda) maps the
    #    LED spectrum (sigma_lam = FWHM/2.355) into a Gaussian spread of
    #    retinal displacements per primary; each phasor is attenuated by the
    #    Gaussian line's own MTF, exp[-2 (pi f x_sigma_k)^2]. This removes
    #    the 3-delta model's triangle-inequality floor (0.543) and the
    #    single-frequency fringe lottery (2026-07-17 audit, §2.4).
    ang = _diffraction_angles_rgb(n, period)                     # [B,3] rad
    dtheta = ang - ang[:, 1:2]
    x_k = EYE_FL_MM * RESID_DISP * dtheta                        # [B,3] mm
    dth_dlam = 1.0 / (n_rgb(n) * period.unsqueeze(1)
                      * torch.cos(ang).clamp(min=0.05))          # rad/nm
    sig_lam = LED_FWHM_NM / 2.355
    x_sig = EYE_FL_MM * RESID_DISP * dth_dlam * sig_lam          # [B,3] mm
    env = torch.exp(-2 * (torch.pi * f * x_sig) ** 2)            # [B,3]
    w = V_PHOTOPIC.to(theta.device).unsqueeze(0)                 # [1,3]
    re = (w * env * torch.cos(2 * torch.pi * f * x_k)).sum(dim=1)
    im = (w * env * torch.sin(2 * torch.pi * f * x_k)).sum(dim=1)
    mtf_chrom = torch.sqrt(re ** 2 + im ** 2 + 1e-12)

    # 4) Grating MTF: contrast loss from periodic wavefront modulation — scales
    #    with phase depth (heuristic L1 coefficient 0.15)  # SYNC
    phi = 2 * torch.pi * depth * (n - 1.0) / 532.0
    mtf_grat = 1.0 - 0.15 * torch.sin(phi / 2) ** 2

    # 5) Coupler MTF: contrast degradation from finite diffraction efficiency
    #    across two coupling events (Goodsell et al. 2024 framing); eta from
    #    eta_first_order(), normalized by its attainable ceiling — the grid's
    #    own unpolarized 532 nm maximum in PMMA mode (v3), the 4/pi^2 scalar
    #    ceiling in the full space (heuristic L1)  # SYNC
    eta = eta_first_order(n, period, depth, duty, wl_idx=1, pol="unpol")
    ceil = _rcwa_grid()["coup_ceil_532"] if PMMA_MODE else 0.4053
    mtf_coup = 0.80 + 0.20 * (eta / ceil)

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
    # v3 sanity: calibrated coupling at the v2 record design (n=1.5, period
    # 438 nm, depth 400 nm, duty 0.5) must reproduce the 2026-07-13 rigorous
    # audit (TE 0.0479, TM 0.0878, unpol 0.0679), nowhere near the scalar
    # 0.347 the v2 engine used
    rec = (torch.tensor([1.5]), torch.tensor([437.98]),
           torch.tensor([400.0]), torch.tensor([0.5001]))
    e_te = eta_rcwa(*rec, wl_idx=1, pol="TE")
    e_tm = eta_rcwa(*rec, wl_idx=1, pol="TM")
    print(f"v3 eta at v2 record (532nm): TE {e_te.item():.4f} TM {e_tm.item():.4f} "
          f"unpol {(0.5*(e_te+e_tm)).item():.4f}  (rigorous audit: TE 0.0479 "
          f"TM 0.0878 unpol 0.0679; scalar wrongly said 0.347)")
