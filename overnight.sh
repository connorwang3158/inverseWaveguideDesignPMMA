#!/bin/bash
# Overnight neural-network training pipeline — TIME-BUDGETED (>= $HOURS hours).
# Leave it running; in the morning open results_report.html.
#
#   Stage 1/6  train the forward surrogate network, 5 seeds (learns the physics)
#   Stage 2/6  train the inverse network THROUGH the surrogate, 5 seeds (tandem)
#   Stage 3/6  physics-decoder ablation run (for the paper's comparison table)
#   Stage 4/6  design search: neural adjoint + gradient/Pareto baselines
#   Stage 5/6  figures, 3D model, HTML report
#   Stage 6/6  BONUS LAPS: until the time budget is spent, keep adding one more
#              surrogate seed + inverse seed + a fresh record hunt per lap, so
#              every extra hour buys statistical power or a better design.
#
# The first Stage-1 seed automatically replaces any surrogate checkpoint that
# was trained under an older waveguide_physics.py (physics-probe check), so it
# is always safe to run this right after editing the physics engine.
#
# Usage:
#   bash overnight.sh                 # >= 12 hours (default)
#   HOURS=16 bash overnight.sh        # longer night
#   HOURS=0  bash overnight.sh        # core protocol only, no bonus laps
# Watch progress from another Terminal tab:  tail -f overnight_log.txt
cd "$(dirname "$0")"

HOURS=${HOURS:-12}            # minimum total runtime in hours
SAMPLES=${SAMPLES:-150000}    # training-set size per run
SUR_EPOCHS=${SUR_EPOCHS:-250} # surrogate epochs per seed
INV_EPOCHS=${INV_EPOCHS:-400} # inverse-network epochs per seed
NA_STARTS=${NA_STARTS:-4000}  # neural-adjoint parallel starting designs
NA_STEPS=${NA_STEPS:-600}     # gradient steps (measured: converges by ~400)
SEEDS=${SEEDS:-"0 1 2 3 4"}   # paper protocol: 5 seeds
LOG=overnight_log.txt
T0=$(date +%s)

# caffeinate stops macOS from idle-sleeping mid-run; skipped on other systems
CAF=""
command -v caffeinate >/dev/null && CAF="caffeinate -i"

mark() { echo "=== $1  $(date) ===" | tee -a "$LOG"; }

echo "Overnight run started: $(date)   (time budget: >= ${HOURS}h)" | tee -a "$LOG"

for SEED in $SEEDS; do
  mark "stage 1/6: surrogate seed $SEED"
  $CAF python3 networks/surrogate.py --pmma \
    --samples "$SAMPLES" --epochs "$SUR_EPOCHS" --batch 512 --seed "$SEED" \
    >> "$LOG" 2>&1
done

for SEED in $SEEDS; do
  mark "stage 2/6: inverse (tandem through surrogate) seed $SEED"
  $CAF python3 networks/train_inverse.py --pmma --decoder surrogate \
    --samples "$SAMPLES" --epochs "$INV_EPOCHS" --batch 256 --seed "$SEED" \
    >> "$LOG" 2>&1
done

# full 5-seed ablation so the physics-decoder arm has the same statistical
# weight as the surrogate arm in the paper's comparison table
for SEED in $SEEDS; do
  mark "stage 3/6: ablation (tandem through exact physics) seed $SEED"
  $CAF python3 networks/train_inverse.py --pmma --decoder physics \
    --samples "$SAMPLES" --epochs "$INV_EPOCHS" --batch 256 --seed "$SEED" \
    >> "$LOG" 2>&1
done

mark "stage 4/6: design search (neural adjoint + baselines)"
$CAF python3 networks/neural_adjoint.py --starts "$NA_STARTS" --steps "$NA_STEPS" >> "$LOG" 2>&1
$CAF python3 baselines/optimize_pmma.py   >> "$LOG" 2>&1
$CAF python3 baselines/sweep_pareto.py    >> "$LOG" 2>&1

mark "stage 5/6: figures, 3D model, audit, report"
python3 visuals/make_3d_model.py        >> "$LOG" 2>&1
python3 networks/audit_memorization.py  >> "$LOG" 2>&1
python3 visuals/make_report.py          >> "$LOG" 2>&1

# ---- stage 6: spend the remaining budget. Each lap = one extra surrogate
# seed + one extra inverse seed + one fresh multi-start record hunt, then a
# report rebuild so results_report.html stays current all night.
EXTRA=5
while [ $(( $(date +%s) - T0 )) -lt $(( HOURS * 3600 )) ]; do
  mark "stage 6/6: bonus lap, seed $EXTRA (time budget not yet spent)"
  $CAF python3 networks/surrogate.py --pmma \
    --samples "$SAMPLES" --epochs "$SUR_EPOCHS" --batch 512 --seed "$EXTRA" \
    >> "$LOG" 2>&1
  $CAF python3 networks/train_inverse.py --pmma --decoder surrogate \
    --samples "$SAMPLES" --epochs "$INV_EPOCHS" --batch 256 --seed "$EXTRA" \
    >> "$LOG" 2>&1
  $CAF python3 networks/neural_adjoint.py --starts "$NA_STARTS" --steps "$NA_STEPS" \
    --seed "$EXTRA" >> "$LOG" 2>&1
  python3 visuals/make_3d_model.py >> "$LOG" 2>&1
  python3 networks/audit_memorization.py >> "$LOG" 2>&1
  python3 visuals/make_report.py   >> "$LOG" 2>&1
  EXTRA=$((EXTRA + 1))
done

ELAPSED_MIN=$(( ( $(date +%s) - T0 ) / 60 ))
echo "Overnight run finished: $(date)   (total $((ELAPSED_MIN / 60))h $((ELAPSED_MIN % 60))m)" | tee -a "$LOG"
echo "DONE. Open results_report.html and waveguide_3d.html; records live in results/best_design_ever_v3.csv"
