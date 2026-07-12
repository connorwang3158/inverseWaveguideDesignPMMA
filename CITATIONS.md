# Citations — Sources Used in Constructing This Model

Grouped by which part of the codebase they anchor. **Verify every DOI/page number
against the publisher site before manuscript submission** — entries marked (†) are
foundational references cited from standard knowledge and must be double-checked.

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

## B. Machine-learning method in `train_inverse.py`

10. Liu, D., Tan, Y., Khoram, E., & Yu, Z. (2018). Training deep neural networks
    for the inverse design of nanophotonic structures. *ACS Photonics*, 5(4),
    1365–1369. (†) — the tandem architecture this code implements.
11. Ren, S., et al. (2020). Benchmarking deep inverse models over time, and the
    neural-adjoint method. *NeurIPS 33*. (†) — physics-in-the-loop inverse design.
12. AutoTandemML (2025). arXiv:2502.15643 — active-learning tandem variant
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

## D. Rigorous solvers in `rigorous_solver.py`

18. Moharam, M. G., & Gaylord, T. K. (1981). Rigorous coupled-wave analysis of
    planar-grating diffraction. *JOSA*, 71(7), 811–818. (†) — RCWA itself.
19. Li, L. (1996). Use of Fourier series in the analysis of discontinuous
    periodic structures. *JOSA A*, 13(9), 1870–1876. (†) — factorization rules
    RCWA implementations rely on.
20. grcwa — autograd-capable RCWA (W. Jin). Docs: https://grcwa.readthedocs.io ;
    source: https://github.com/weiliangjinca/grcwa — the solver wrapped here.
21. Kim, C., & Lee, B. (2023). TORCWA: GPU-accelerated Fourier modal method and
    gradient-based optimization for metasurface design. *Computer Physics
    Communications*, 282, 108552.
    https://www.sciencedirect.com/science/article/abs/pii/S0010465522002715
22. Oskooi, A. F., et al. (2010). Meep: A flexible free-software package for
    electromagnetic simulations by the FDTD method. *Computer Physics
    Communications*, 181, 687–702. (†) — FDTD option for non-periodic shapes.

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
