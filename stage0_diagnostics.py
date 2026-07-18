"""Stage 0 diagnostics from the 2026-07-17 referee report (§5, Stage 0).

Six checks, no retraining. Each answers a gate question:
  D1  Five MTF factors separately at the record design  -> is MTF_sys just MTF_chrom?
  D2  theta_d(R) - theta_d(B) at the record             -> equals the reported 28.8 chrom?
  D3  T_FOV / T over 1000 random designs                -> is T_FOV = const * T?
  D4  Hard-mask (no sigmoid) re-score of top finalists  -> do they survive?
  D5  S(L_c) across the sampled range                   -> ever meaningfully < 1?
  D6  Sobol total-effect sensitivity of the 8->4 map    -> how many params matter?
Plus the §2.2 edge check: sigmoid offsets and the true full-RGB guided FOV window.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "physics"))
import torch
import waveguide_physics as wp

torch.manual_seed(0)
wp.use_pmma()

# Record design (results/best_design_ever_v3.csv, 2026-07-17)
rec = torch.tensor([[1.5, 5.0047e-05, 0.79866, 3.9954e+05,
                     0.8978, 448.13, 199.53, 0.42448]])
print("=" * 72)
print("RECORD:", {k: v for k, v in zip(
    "n alpha sigma Lc t period depth duty".split(), rec[0].tolist())})
y = wp.forward_model(rec)[0]
print(f"engine spec  MTF={y[0]:.5f}  T={y[1]:.6f}  chrom={y[2]:.3f} deg  "
      f"T_fov={y[3]:.6f}   (csv: 0.78382 / 0.012491 / 28.827 / 0.011736)")

# ---------------------------------------------------------------- D1
print("\n[D1] MTF cascade factors at the record design")
n, alpha, sigma, Lc, t, period, depth, duty = rec.unbind(dim=1)
f = wp.F0_CYC_PER_MM
lam_mm = 532e-6
fc = wp.PUPIL_MM / (lam_mm * wp.EYE_FL_MM)
xx = torch.tensor(min(f / fc, 0.999))
mtf_diff = (2 / torch.pi) * (torch.acos(xx) - xx * torch.sqrt(1 - xx ** 2))
blur = (sigma / 6.0) ** 2 * (1.0 / (1.0 + Lc / 3e5)) * 8e-3
mtf_rough = torch.exp(-2 * (torch.pi * blur * f) ** 2)
ang = wp._diffraction_angles_rgb(n, period)
x_k = wp.EYE_FL_MM * wp.RESID_DISP * (ang - ang[:, 1:2])
w = wp.V_PHOTOPIC.unsqueeze(0)
re = (w * torch.cos(2 * torch.pi * f * x_k)).sum(1)
im = (w * torch.sin(2 * torch.pi * f * x_k)).sum(1)
mtf_chrom = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
phi = 2 * torch.pi * depth * (n - 1.0) / 532.0
mtf_grat = 1.0 - 0.15 * torch.sin(phi / 2) ** 2
eta = wp.eta_first_order(n, period, depth, duty, wl_idx=1, pol="unpol")
ceil = wp._rcwa_grid()["coup_ceil_532"]
mtf_coup = 0.80 + 0.20 * (eta / ceil)
prod = mtf_diff * mtf_rough * mtf_chrom * mtf_grat * mtf_coup
for name, v in [("diff", mtf_diff), ("rough", mtf_rough[0]),
                ("chrom", mtf_chrom[0]), ("grat", mtf_grat[0]),
                ("coup", mtf_coup[0]), ("PRODUCT", prod[0])]:
    print(f"   MTF_{name:<7s} = {float(v):.5f}")
print(f"   photopic weights actually in code: {wp.V_PHOTOPIC.tolist()}"
      f"  (report assumed [0.194, 0.771, 0.034] order R,G,B?)")
wB, wG, wR = wp.V_PHOTOPIC.tolist()
print(f"   triangle-inequality floor of MTF_chrom = {wG - wB - wR:.4f}")

# ---------------------------------------------------------------- D2
print("\n[D2] internal guided-angle spread at record")
angd = torch.rad2deg(ang)[0]
print(f"   theta_d  B={angd[0]:.2f}  G={angd[1]:.2f}  R={angd[2]:.2f} deg")
print(f"   R-B spread = {angd[2]-angd[0]:.3f} deg   vs reported chrom 28.827")

# ---------------------------------------------------------------- D3
print("\n[D3] T_FOV / T over 1000 random PMMA designs")
th = wp.sample_theta(1000)
T0 = wp.transmission(th, 0.0)
Tf = wp.transmission(th, wp.FOV_DEG)
ratio = (Tf / (T0 + 1e-30))
ok = T0 > 1e-8
r = ratio[ok]
A5 = float(torch.exp(-(torch.sin(torch.deg2rad(torch.tensor(wp.FOV_DEG)))
                       / wp.ACCEPT_SIN) ** 2))
print(f"   FOV_DEG in PMMA mode = {wp.FOV_DEG} deg;  A(FOV)/A(0) = {A5:.5f}")
print(f"   over {int(ok.sum())} designs with T>1e-8:  ratio mean={r.mean():.5f}"
      f"  std={r.std():.5f}  min={r.min():.5f}  max={r.max():.5f}")

# ---------------------------------------------------------------- D4
print("\n[D4] hard-mask re-score of top finalists (optimal_designs_na.csv)")
import csv as _csv
finalists = []
with open(os.path.join(os.path.dirname(__file__),
                       "results", "optimal_designs_na.csv")) as fh:
    for i, row in enumerate(_csv.DictReader(fh)):
        if i >= 10:
            break
        finalists.append([float(row[k]) for k in
                          ("n", "alpha(1/mm)", "sigma(nm)", "Lc(nm)", "t(mm)",
                           "period(nm)", "depth(nm)", "duty")])
fin = torch.tensor(finalists)

def J_of(y):
    return y[:, 0] + y[:, 1] / 0.10 + 0.3 * y[:, 3] / 0.10 - 0.5 * y[:, 2] / 30.0

y_soft = wp.forward_model(fin)
J_soft = J_of(y_soft)

_orig_mask = wp.guided_mask
wp.guided_mask = lambda x, n: ((x > 1.0) & (x < n)).to(x.dtype)   # hard window
y_hard = wp.forward_model(fin)
J_hard = J_of(y_hard)
wp.guided_mask = _orig_mask

print("   rank   J_soft    J_hard    T_soft%   T_hard%   Tfov_soft% Tfov_hard%")
for i in range(len(fin)):
    print(f"   {i+1:>4d}  {J_soft[i]:.4f}   {J_hard[i]:.4f}   "
          f"{100*y_soft[i,1]:.3f}    {100*y_hard[i,1]:.3f}     "
          f"{100*y_soft[i,3]:.4f}    {100*y_hard[i,3]:.4f}")

# §2.2 edge positions + true simultaneous-RGB FOV window at the record
print("\n[D4b] sigmoid-edge positions at the record (widths of 0.005)")
for fd in (0.0, wp.FOV_DEG):
    xs = wp.grating_x(rec[:, 5], wp.WL, fd)[0]
    for k, lab in enumerate("BGR"):
        x = float(xs[k])
        print(f"   th_i={fd:>4.1f}  {lab}: x={x:.5f}  (x-1)/w={(x-1)/0.005:+8.2f}"
              f"  (x-n)/w={(x-1.5)/0.005:+8.2f}")
lo, hi, wdt = wp.fov_window_deg(rec)
print(f"   exact full-RGB guided window: [{lo[0]:.2f}, {hi[0]:.2f}] deg, "
      f"width {wdt[0]:.2f} deg  (metric evaluated at FOV_DEG={wp.FOV_DEG})")

# ---------------------------------------------------------------- D5
print("\n[D5] S(Lc) over the sampled PMMA Lc range")
for LcV in (2e5, 3e5, 4e5):
    print(f"   Lc={LcV:.0e} nm  ->  S = {1/(1+LcV/3e5):.4f}")
print("   full-space range 300 .. 1.2e6:",
      f"S(300)={1/(1+300/3e5):.5f}  S(1.2e6)={1/(1+1.2e6/3e5):.4f}")

# ---------------------------------------------------------------- D6
print("\n[D6] Sobol total-effect indices (Jansen estimator, N=4096) on J")
N, D = 4096, 8
g = torch.Generator().manual_seed(7)
A = wp.sample_theta(N, generator=g)
B = wp.sample_theta(N, generator=g)
def Jphys(th):
    return J_of(wp.forward_model(th))
fA, fB = Jphys(A), Jphys(B)
varY = torch.cat([fA, fB]).var()
names = "n alpha sigma Lc t period depth duty".split()
print(f"   Var[J] = {varY:.5f}")
tot = []
for i in range(D):
    ABi = A.clone(); ABi[:, i] = B[:, i]
    fABi = Jphys(ABi)
    ST = ((fA - fABi) ** 2).mean() / (2 * varY)
    tot.append(float(ST))
for nm, s in sorted(zip(names, tot), key=lambda z: -z[1]):
    print(f"   S_T({nm:<6s}) = {s:.4f}")
print("=" * 72)
