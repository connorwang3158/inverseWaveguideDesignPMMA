# Diffractive AR Waveguide — Inverse-Design Engine

Differentiable physics engine and ML inverse-design pipeline for PMMA
surface-relief-grating AR waveguides. This repository contains only the
source code that runs and calculates the model.

## Setup
    pip install torch numpy matplotlib grcwa

## Build the RCWA calibration grid (required once)
    python3 physics/calibrate_rcwa.py

## Run
    python3 physics/validate.py
    python3 networks/surrogate.py --pmma
    python3 networks/train_inverse.py --pmma
    python3 networks/neural_adjoint.py
    bash overnight.sh

## Layout
    physics/     differentiable engine, rigorous RCWA layer, validation
    networks/    forward surrogate, tandem inverse net, neural-adjoint search
    baselines/   non-neural gradient / Pareto searches
    visuals/     figure and 3D-model generators
    metagrating/ separate metagrating / SRG research thread
    v4engine/    solver cross-checks

Runtime outputs (results/, figures/, checkpoints/, logs, reports) are
generated on run and are not tracked.
