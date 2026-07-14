# Citations — Sources Used in Constructing This Model

Grouped by which part of the codebase they anchor. **Verify every DOI/page number
against the publisher site before manuscript submission** — entries marked (†) are
foundational references cited from standard knowledge and must be double-checked.

**Provenance note:** Group A intentionally overlaps Paper 1's bibliography — the
analytic engine implements Paper 1's physics, so it inherits those anchors.
Groups B–E are independent of Paper 1. Group D is the rigorous electromagnetic
canon (verified against publisher listings) and carries the paper's L2 claims.

## A. Physics formulas in `waveguide_physics.py`

1. Watson, A. B. (2013). A formula for the mean human optical modulation transfer
   function as a function of pupil size. *Journal of Vision*, 13(6):18.
   https://doi.org/10.1167/13.6.18 — diffraction-limited eye MTF.
2. Tien, P. K. (1971). Light waves in thin films and integrated optics. *Applied
   Optics*, 10(11), 2395–2413. (†) — per-TIR-bounce roughness loss
   exp[−(4πσn cosθ/λ)²].
3. Payne, F. P., & Lacey, J. P. R. (1994). A theoretical analysis of scattering
   loss from planar optical waveguides. *Optical and Quantum Electronics*, 26,
   977–986. https://doi.org/10.1007/BF00708339 — correlation-length weighting.
4. Goodman, J. W. *Introduction to Fourier Optics* (any ed.), diffraction-grating
   chapter. (†) — scalar binary phase-grating efficiency η₁=4(sin πD/π)²sin²(φ/2).
5. Thibos, L. N. (1987). Calculation of the influence of lateral chromatic
   aberration on image quality across the visual field. *JOSA A*, 4(8), 1673.
   https://doi.org/10.1364/JOSAA.4.001673 — chromatic blur treatment.
6. Nilsen, K., Ding, Y., Lee, S.-L., & Wu, S.-T. (2025). Comparisons of glass and
   plastic waveguides for augmented reality glasses. *Optics Express*, 33, 20051.
   https://doi.org/10.1364/OE.562679 — PMMA material parameters.
7. Zhao, Z., et al. (2024). Theoretical efficiency limit of diffractive input
   couplers in augmented reality waveguides. *Optics Express*, 32, 12340.
   https://doi.org/10.1364/OE.519027 — coupler efficiency limits, angular acceptance.
8. Goodsell, J., et al. (2024). Framework for optimizing AR waveguide in-coupler
   architectures. *Optics Express*, 32, 9967. https://doi.org/10.1364/OE.515544.
9. Kress, B. C., & Chatterjee, I. (2020). Waveguide combiners for mixed reality
   headsets: a nanophotonics design perspective. *Nanophotonics*, 10, 41–74.
   https://doi.org/10.1515/nanoph-2020-0410 — system architecture framing.

## B. Machine-learning method in `surrogate.py`, `train_inverse.py`, `neural_adjoint.py`

10. Liu, D., Tan, Y., Khoram, E., & Yu, Z. (2018). Training deep neural networks
    for the inverse design of nanophotonic structures. *ACS Photonics*, 5(4),
    1365–1369. (†) — the tandem architecture `train_inverse.py` implements
    (inverse net trained through the frozen learned forward model).
11. Peurifoy, J., et al. (2018). Nanophotonic particle simulation and inverse
    design using artificial neural networks. *Science Advances*, 4(6), eaar4206.
    (†) — the forward-surrogate framework `surrogate.py` implements: a deep
    network trained on simulated data to emulate the electromagnetic response.
12. Ren, S., Padilla, W., & Malof, J. (2020). Benchmarking deep inverse models
    over time, and the neural-adjoint method. *NeurIPS 33*. (†) — the
    neural-adjoint method `neural_adjoint.py` implements: gradient design search
    through the frozen trained surrogate.
13. Ma, W., Liu, Z., Kudyshev, Z. A., Boltasseva, A., Cai, W., & Liu, Y. (2021).
    Deep learning for the design of photonic structures. *Nature Photonics*,
    15, 77–90. (†) — review anchoring the surrogate + tandem + adjoint taxonomy.
14. AutoTandemML (2025). arXiv:2502.15643 — active-learning tandem variant
    (comparison baseline candidate).

## C. Closest prior work (positioning; see research_framework.md §4)

13. Tandem NN slanted waveguide grating design. *Optics Express*, 32, 12587 (2024).
    https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-7-12587
14. Inverse design and uniformity optimization of diffractive waveguides for
    AR-HUD. *Applied Optics*, 64, 3536 (2025).
    https://opg.optica.org/ao/abstract.cfm?uri=ao-64-13-3536
15. Data-driven inverse design of hybrid waveguide gratings (tandem + cVAE).
    *Optics*, 6(4), 61 (2025). https://doi.org/10.3390/opt6040061
16. End-to-end differentiable design of geometric waveguide displays.
    arXiv:2601.04370 (2026). https://arxiv.org/abs/2601.04370
17. Tian, Z., et al. (2025). An achromatic metasurface waveguide for augmented
    reality displays. *Light: Science & Applications*, 14, 94.
    https://doi.org/10.1038/s41377-025-01761-w

## D. Rigorous electromagnetic canon in `rigorous_solver.py` (VERIFIED)

18. Moharam, M. G., & Gaylord, T. K. (1981). Rigorous coupled-wave analysis of
    planar-grating diffraction. *JOSA*, 71(7), 811–818. (†) — RCWA's origin.
19. Moharam, M. G., Grann, E. B., Pommet, D. A., & Gaylord, T. K. (1995).
    Formulation for stable and efficient implementation of the rigorous
    coupled-wave analysis of binary gratings. *JOSA A*, 12(5), 1068–1076.
    — VERIFIED — the formulation modern RCWA codes (incl. grcwa) implement.
20. Moharam, M. G., Pommet, D. A., Grann, E. B., & Gaylord, T. K. (1995).
    Stable implementation of the rigorous coupled-wave analysis for
    surface-relief gratings: enhanced transmittance matrix approach.
    *JOSA A*, 12(5), 1077–1086.
    https://opg.optica.org/josaa/abstract.cfm?uri=josaa-12-5-1077 — VERIFIED.
21. Pommet, D. A., Moharam, M. G., & Grann, E. B. (1994). Limits of scalar
    diffraction theory for diffractive phase elements. *JOSA A*, 11(6),
    1827–1834. https://opg.optica.org/josaa/abstract.cfm?uri=josaa-11-6-1827
    — VERIFIED — proves scalar theory errs >±5% when feature size < 14λ; AR
    coupler gratings sit at ~1λ, mandating the vectorial treatment. THIS is
    the citation that justifies rigorous_solver.py's existence and explains
    the ~50% scalar-vs-RCWA discrepancy found in rcwa_validation.csv.
22. Li, L. (1996). Use of Fourier series in the analysis of discontinuous
    periodic structures. *JOSA A*, 13(9), 1870–1876. (†) — factorization rules.
23. grcwa — autograd-capable RCWA (W. Jin). Docs: https://grcwa.readthedocs.io ;
    source: https://github.com/weiliangjinca/grcwa — the solver wrapped here.
24. Kim, C., & Lee, B. (2023). TORCWA: GPU-accelerated Fourier modal method and
    gradient-based optimization for metasurface design. *Computer Physics
    Communications*, 282, 108552.
    https://www.sciencedirect.com/science/article/abs/pii/S0010465522002715
    — VERIFIED — GPU alternative.
25. Meent: differentiable electromagnetic simulator for machine learning.
    arXiv:2406.12904. https://arxiv.org/pdf/2406.12904 — differentiable-RCWA
    option for future in-loop training.
26. Oskooi, A. F., et al. (2010). Meep: A flexible free-software package for
    electromagnetic simulations by the FDTD method. *Computer Physics
    Communications*, 181, 687–702. (†) — FDTD for non-periodic shapes.

## E. Architecture models in `architectures.py`

23. eLight review (2023). Waveguide-based augmented reality displays:
    perspectives and challenges. *eLight*, 3.
    https://link.springer.com/article/10.1186/s43593-023-00057-z
24. Lumus geometric waveguide technical overview. https://lumus.com/how-it-works/
    — qualitative anchor only; vendor figures are NOT evidence (framework §5).
25. SCHOTT reflective waveguides product documentation.
    https://www.schott.com/en-gb/products/schott-reflective-waveguides-p1001190
26. US Patent 11,514,828 — AR headset display (embedded mirror geometry).

## F. Author baseline

27. Wang, C. *Modeling Diffractive Singular Flat AR Waveguide Optical
    Performance* (Paper 1) and repository:
    github.com/connorwang3158/ModelingSingularFlatDiffractiveWaveguidesWithinARGlasses
    — used as the system-metric baseline; superseded at the coupler level by
    the rigorous RCWA treatment (claims ladder L2).

## G. Shortlist from the 2026-07 literature sweep (candidate upgrades)

Frameworks and methods evaluated for the next iteration; none are load-bearing
in the current results. Verify DOIs before citing.

### G1. Differentiable rigorous solvers (for the v3 RCWA-calibrated engine)

28. Huang, Y., et al. (2024). Eigendecomposition-free inverse design of
    meta-optics devices (TorchRDIT: differentiable R-DIT, PyTorch, reports
    ~16× speedup over RCWA-based inverse design). *Optics Express*, 32(8),
    13986. https://github.com/yi-huang-1/torchrdit — fastest path to putting
    rigorous efficiencies INSIDE the training loop.
29. NIDN — Neural Inverse Design of Nanostructures (ESA, 2022). *Scientific
    Reports*, 12. https://www.nature.com/articles/s41598-022-26312-w — fully
    differentiable RCWA derived from grcwa; closest drop-in for our stack.
    (torcwa and meent already listed in Group D, items 24–25.)

### G2. Ill-posedness benchmarks & the boundary-loss neural adjoint

30. Ren, S., Mahendra, A., Khatib, O., Deng, Y., Padilla, W. J., & Malof, J. M.
    (2022). Inverse deep learning methods and benchmarks for artificial
    electromagnetic material design. *Nanoscale*, 14. arXiv:2112.10254; code:
    https://github.com/BensonRen/AEM_DIM_Bench — benchmark protocol worth
    mirroring in §6.3, and the boundary-loss NA variant: adds a penalty that
    keeps adjoint searches off the design-space walls — directly relevant
    because our current record pins n, depth, Lc, and alpha at their bounds.
31. Extended neural adjoint with regression activation mapping (2025).
    arXiv:2507.18644 — interpretability layer over NA for multilayer optics.

### G3. Multimodal inverse methods (one-to-many; candidate baselines)

32. Unni, R., Yao, K., & Zheng, Y. (2020). Deep convolutional mixture density
    network for inverse design of layered photonic structures. *ACS Photonics*,
    7(10). https://doi.org/10.1021/acsphotonics.0c00630 — MDN outputs a
    DISTRIBUTION of designs per target; natural extra baseline next to the
    tandem net and naive regressor.
33. Transfer-learning-assisted MDN for nanophotonics (2024). arXiv:2401.12254.
34. AdjointDiffusion: physics-guided, fabrication-aware diffusion models for
    photonic inverse design (2025). arXiv:2504.17077 — generative frontier;
    data-hungry, flag as future work rather than a baseline.

### G4. System-level efficiency context (for framing PMMA's numbers)

35. Breaking the in-coupling efficiency limit in waveguide AR displays with
    polarization volume gratings (2024). *Light: Science & Applications*, 13.
    https://www.nature.com/articles/s41377-024-01537-8 — reports ~10% (20° FOV)
    and ~3% (30° FOV) system efficiencies for conventional diffractive
    combiners: the honest comparison window for our PMMA transmission numbers.
36. Single-layer full-color waveguide display based on a broadband efficient
    meta-grating (2025). *Nanomaterials*, 15(19), 1493.
    https://doi.org/10.3390/nano15191493 — >60% RGB coupling on thin glass;
    the high-index performance ceiling PMMA trades away for cost/weight.
37. Physics-constrained neural networks for surrogate modeling of lossless
    periodic structures (2026). arXiv:2606.28119 — energy-conservation
    constraints inside the surrogate; candidate regularizer if the v3
    RCWA-trained surrogate shows unphysical predictions.
