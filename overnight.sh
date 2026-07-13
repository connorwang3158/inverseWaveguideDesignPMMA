#!/bin/bash
# Overnight neural-network training pipeline. Leave it running; in the morning
# open results_report.html.
#
#   Stage 1  train the forward surrogate network, 5 seeds (learns the physics)
#   Stage 2  train the inverse network THROUGH the surrogate, 5 seeds (tandem)
#   Stage 3  physics-decoder ablation run (for the paper's comparison table)
#   Stage 4  neural-adjoint design search through the surrogate + baselines
#   Stage 5  figures, 3D model, HTML report
#
# Usage:  bash overnight.sh        (laptop plugged in; lid open)
# Watch progress from another Terminal tab:  tail -f overnight_log.txt
cd "$(dirname "$0")"

# caffeinate stops macOS from idle-sleeping mid-run; skipped on other systems
CAF=""
command -v caffeinate >/dev/null && CAF="caffeinate -i"

echo "Overnight run started: $(date)" | tee -a overnight_log.txt

for SEED in 0 1 2 3 4; do
  echo "=== surrogate seed $SEED started $(date) ===" | tee -a overnight_log.txt
  $CAF python3 surrogate.py --pmma \
    --samples 100000 --epochs 150 --batch 512 --seed $SEED \
    >> overnight_log.txt 2>&1
done

for SEED in 0 1 2 3 4; do
  echo "=== inverse (surrogate decoder) seed $SEED started $(date) ===" | tee -a overnight_log.txt
  $CAF python3 train_inverse.py --pmma --decoder surrogate \
    --samples 100000 --epochs 300 --batch 256 --seed $SEED \
    >> overnight_log.txt 2>&1
done

echo "=== physics-decoder ablation started $(date) ===" | tee -a overnight_log.txt
$CAF python3 train_inverse.py --pmma --decoder physics \
  --samples 100000 --epochs 300 --batch 256 --seed 0 \
  >> overnight_log.txt 2>&1

echo "=== design search started $(date) ===" | tee -a overnight_log.txt
$CAF python3 neural_adjoint.py --starts 800 --steps 800 >> overnight_log.txt 2>&1
$CAF python3 optimize_pmma.py   >> overnight_log.txt 2>&1
$CAF python3 sweep_pareto.py    >> overnight_log.txt 2>&1

python3 make_3d_model.py        >> overnight_log.txt 2>&1
python3 make_report.py          >> overnight_log.txt 2>&1
echo "Overnight run finished: $(date)" | tee -a overnight_log.txt
echo "DONE. Open results_report.html and waveguide_3d.html"
