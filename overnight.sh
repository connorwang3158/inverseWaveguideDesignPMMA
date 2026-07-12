#!/bin/bash
# Overnight rigorous training: all 5 seeds, big dataset, ~117k iterations each.
# Usage:  bash overnight.sh          (keep Mac plugged in; lid open)
# caffeinate stops macOS from idle-sleeping while this runs.
cd "$(dirname "$0")"
echo "Overnight run started: $(date)" | tee -a overnight_log.txt
for SEED in 0 1 2 3 4; do
  echo "=== seed $SEED started $(date) ===" | tee -a overnight_log.txt
  caffeinate -i python3 train_inverse.py --pmma \
    --samples 100000 --epochs 300 --batch 256 --seed $SEED \
    >> overnight_log.txt 2>&1
done
# design search + trade-off menu with the final physics, then build the report
caffeinate -i python3 optimize_pmma.py   >> overnight_log.txt 2>&1
caffeinate -i python3 sweep_pareto.py    >> overnight_log.txt 2>&1
python3 make_report.py                   >> overnight_log.txt 2>&1
echo "Overnight run finished: $(date)" | tee -a overnight_log.txt
echo "DONE. Open results_report.html and training_runs.csv"
