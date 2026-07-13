# Archived results from the pre-revision (v1) physics engine

The 2026-07 physics audit (see the `v2 PHYSICS REVISION` block at the top of
`waveguide_physics.py`) added the TIR guiding constraint, angle-resolved
in-coupling, polarization-resolved Fresnel transmission, the geometric bounce
count, and the exact chromatic MTF. Results produced under the old engine are
NOT comparable with results produced after the fix — in particular the v1
hall-of-fame record (J = 1.8746 at period ≈ 700 nm) is a design whose
first-order beams are not TIR-guided at all; the corrected engine scores its
transmission at ~0.

Files here are kept for provenance only:

- `best_design_ever_v1.csv`   — old-physics hall of fame (leaky record design)
- `surrogate_runs_v1.csv`     — surrogate training rows from before the fix,
  plus two `--quick` smoke-test rows (8k samples / 15 epochs — not converged,
  not paper data)
- `training_runs_v1.csv`      — inverse-network rows from before the fix,
  plus one `--quick` smoke-test row

Current records live in `../best_design_ever_v2.csv`; current training tables
restart cleanly in `../surrogate_runs.csv` and `../training_runs.csv`.
