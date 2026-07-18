# Image Manifest — for Overleaf

Copy the `figs/` folder into your Overleaf project root (same level as
`waveguide_paper.tex`) so the four `\includegraphics{figs/...}` calls
resolve unchanged. All four are referenced by relative path `figs/<name>`,
so no path edits are needed if the folder structure is preserved.

| # | Filename | Referenced as | Figure in paper | Used in section | What it shows |
|---|---|---|---|---|---|
| 1 | `fig_guided_window.png` | `figs/fig_guided_window.png` | Fig. 1 | Section III-A | The full-RGB guided window: x = sin(theta_i) + lambda/period plotted vs. grating period for the three RGB primaries, with the TIR edge (x=1), evanescent edge (x=n), the calibration grid's period range, and the record design's period marked. |
| 2 | `fig_scalar_vs_rcwa.png` | `figs/fig_scalar_vs_rcwa.png` | Fig. 2 | Section IV-A | 20-point rigorous RCWA depth scan (TE, TM, unpolarized) vs. scalar diffraction theory at the scalar-optimal geometry (period 438 nm, duty 0.5, 532 nm), showing scalar theory's overestimate and its incorrect 400 nm depth optimum against the true interior optimum near 200 nm. |
| 3 | `memorization_audit.png` | `figs/memorization_audit.png` | Fig. 3 | Section VI (memorization audit) | Four-part memorization audit for the tandem inverse network (nearest-training-neighbor distance, held-out vs. train-set error distributions, etc.). |
| 4 | `neural_adjoint_run.png` | `figs/neural_adjoint_run.png` | Fig. 4 | Section VI-C (neural-adjoint search) | Left: 400-restart neural-adjoint search convergence trajectory. Right: finalists' surrogate-predicted vs. exact-physics-scored per-metric values (surrogate-vs-physics honesty check). |

## Notes for your own edits/audits

- All four images are PNG, generated at 150-200 dpi from the project's own
  matplotlib scripts (not hand-drawn), so they will look soft if you scale
  them up significantly in Overleaf. Regenerate at higher dpi from the
  source scripts if you need print resolution.
- Figure 1 (`fig_guided_window.png`) is currently schematic-only. The
  Round 2 revision considered adding the record design's guard-margin
  bands to it, and adding a fifth figure (a heatmap of the
  depth-optimum-by-period-duty sweep discussed in Section VIII), but
  neither is included in this revision; see the response letter for why.
- No image was added, removed, or renamed in this revision; only the
  manuscript text around them changed.
