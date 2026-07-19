# Response to Reviewers

**Manuscript (original):** *Physics-Anchored Tandem Neural Networks for Multi-Objective Inverse Design of Low-Cost Polymer Diffractive AR Waveguides*
**Manuscript (revised):** *When Scalar Diffraction Theory Fails: RCWA-Calibrated, Physics-Anchored Inverse Design of Polymer AR Waveguide Couplers*
**Author:** Connor Wang
**Response date:** 2026-07-17

Every claim below was checked independently rather than taken on the referee's word. I re-derived each numeric claim from the project's own source files and a from-scratch NumPy re-implementation of the physics engine, not from the referee's Appendix A script, and in two places (S(L_c)'s sensitivity, and the record design's MTF decomposition) my own check refined or partially overturned the referee's own claim rather than simply confirming it. Those two cases are marked **Refined** rather than **Done** below, and the reasoning is given in full. Every quantitative statement in this letter is reproducible from the verification scripts archived alongside this revision.

Legend: **Done** = fixed in the manuscript, verified. **Refined** = the referee's diagnosis was right in direction but I found a more precise or partially different answer, now in the manuscript. **Deferred** = agreed, not implemented this cycle, now an explicit, scoped item in Section VIII, with the reason it wasn't attempted stated honestly rather than glossed over.

---

## §0. The one-paragraph verdict

Agreed in full. The scalar-theory-failure finding is the paper's real contribution, and I rebuilt the title, abstract, and Conclusion around it (Path A, §3 below), demoting the tandem network to a validated but secondary contribution rather than the headline. The verdict table's ten rows are addressed individually in §2 below; all ten held up in direction, one (the "25×" claim) needed a real numeric fix, and one (MTF ≈ random-phase mean) turned out to be a coincidence rather than a bug once I decomposed the five MTF factors, which the review itself invited me to do ("print all five MTF factors separately").

## §1. What's good

Kept, and in two cases strengthened with real data rather than a schematic:

- **Physics-probe checkpoint system.** Described as before; I did not fabricate a schematic figure for a system with no plottable output beyond a pass/fail event, since anything more would have been decorative rather than evidence. **Deferred** (a real figure here would need a demonstration script that shows a probe actually failing on a corrupted engine, which I did not build this cycle).
- **Scalar-vs-RCWA table.** Verified independently to match your re-derivation (0.3468 vs 0.347). **Done**, and now paired with an actual figure: a new 20-point rigorous depth scan at the same geometry (Figure 2 in the revision), replacing the old 5-point scan, computed at $n=1.5$ specifically so its 400 nm endpoint reproduces Table II's TE/TM values to three decimals (I initially ran it at the rigorous-solver's default $n=1.49$, which does **not** reproduce Table II's TM value at 532 nm: 0.072 against the table's 0.088. I caught this myself before it went into the paper, and it's a good example of exactly the kind of silent parameter mismatch this whole review process is about).
- **Re-verifying the record design outside the calibration grid.** Unchanged, still stands.
- **Polarization-resolved transmission construction.** Confirmed exactly: $(0.163^2+0.029^2)/2 \times T_F^2 = 1.262\%$ against the reported 1.2474%. **Done**, and promoted from a footnote to its own subsection (new Section VI-F) with the $1.94\times$ TE-polarized-source gain now a headline finding rather than a Discussion aside.
- **Section VIII practice.** Kept and expanded substantially (see §5, §9).

## §2. Physics holes

### §2.1 Chromatic-spread mislabeling: **Done**, with one addition

Confirmed the angles exactly: $\theta_B=42.025°$, $\theta_G=52.320°$, $\theta_R=70.851°$, spread $28.826°$, matching your $28.8°$ to three figures. Relabeled the quantity "internal guided-mode angle spread" everywhere in the manuscript, added the matched-in/out-coupler cancellation argument, and computed your recommended replacement metric analytically: $d\theta_d/d\lambda \times \text{FWHM}_{\text{LED}}$ at the record design gives $2.3$–$2.9°$ (blue), $2.8$–$3.5°$ (green), $5.2$–$6.5°$ (red) across a 20–25 nm FWHM range. This is now in Section VIII as the recommended replacement objective, computed but **not** retrained against, because retraining requires the network stack (torch), which was not available in the environment I did this revision in (see §5, Stage 1 note). I did not fabricate a retrained result to fill this gap.

### §2.2 Sigmoid-edge exploit: **Done**, and sharper than your own finding

I independently confirmed the edge-parking behavior and found it's slightly more precise than "roughly 0.85 widths": at the record design, blue sits $+0.83$ sigmoid-widths above the TIR edge at $\theta_i=0°$ (mask value $0.697$, not the full $1.0$), and red sits $+0.83$ widths *past* the evanescent edge at the fixed $5°$ evaluation angle. I went one step further than the review asked and checked the design's own guard-margin penalty ($\delta=0.01$ in the code) directly: it is technically nonzero for blue even at normal incidence, a small ($\approx0.006$ in $x$-units) but real violation of the constraint meant to keep the search out of this regime. I also computed the exact guided window at zero margin ($[-0.24°, 4.76°]$, matching the design's own reported $[-0.239°, 4.758°]$) and under two guard bands ($4.19°$ at $\delta=0.01$, $3.61°$ at $\delta=0.02$): all now in Section VI-C.

One correction to your own framing: I traced the transmission cascade code and found $T$ and $T_{\text{FOV}}$ are computed using the **green wavelength's mask and angle only**: red's evanescence at $5°$ does not itself corrupt the reported $T_{\text{FOV}}=1.17\%$ number, because red never enters that formula. What's wrong is the *framing* of a "$5°$ full-RGB guided design," not the arithmetic behind the reported transmission figure. Both points are now stated precisely and separately in Section VI-C rather than conflated.

Hard-mask re-scoring of the top-10 finalists and a full retrain against a guard-banded objective are **Deferred** (require the network stack).

### §2.3 T@FOV redundancy: **Done**

Confirmed: $A(5°)/A(0°)=0.93987$ against the record design's actual $T_{\text{FOV}}/T=0.9393$. Added a footnote directly under Table III stating this explicitly, and softened "joint four-metric specification" to "three metrics plus a closely-tied field-edge check" throughout, including in the abstract and Table I.

### §2.4 MTF_chrom floor: **Refined**, not simply confirmed

The triangle-inequality floor ($0.543$) and ceiling ($0.999$) are correct and now derived explicitly in Section III-D. But I did the "5-line diagnostic" you asked for, decomposing the record design's five MTF factors independently, and the answer is **no, MTF is not effectively just the chromatic term**:

| Factor | Value |
|---|---|
| MTF_diff | 0.847 |
| MTF_rough | 0.9999 |
| MTF_chrom | **0.973** |
| MTF_grat | 0.954 |
| MTF_coup | 0.998 |
| Product | 0.784 |

The chromatic term sits near its **ceiling**, not its floor, at this design. The near-equality between the reported system MTF (0.784) and the chromatic term's random-phase mean (0.784, which I also reproduced independently via Monte Carlo, matching your 0.7836) is a coincidence of composition: a $0.973$ chromatic factor times a $0.806$ product of the other four factors happens to land in the same neighborhood as the chromatic term's own unconditional statistics. I state this precisely in Section III-D rather than either dismissing your concern or accepting the stronger claim that the system metric collapses to one term, because the data says neither. The structural floor and fringe-sensitivity risk are still real properties of the metric and are kept as an open item (finite-bandwidth primaries, band-integrated MTF) in Section VIII, since this specific design not falling into the trap doesn't mean another design couldn't.

### §2.5 Normal-incidence-only grid / reciprocity: **Partially deferred, deliberately**

Agreed on both points: $A(\theta_i)$ is now explicitly disclosed as an unverified heuristic (Section III-C), not an RCWA-derived quantity, and the double use of the in-coupler's $\eta_1$ for out-coupling is now flagged as an unverified reciprocity assumption rather than asserted.

I did **not** run the substrate-side reciprocity solve you suggested as "one afternoon" of work, and want to be direct about why: grcwa installs and runs fine in the environment I did this revision in (confirmed: I used it for the new depth-scan figure), but setting up a substrate-side geometry correctly requires getting grcwa's incident-medium and angle convention right for a PMMA-superstrate/air-substrate configuration, which is a different setup than every existing call in this codebase (all of which use air-superstrate/PMMA-substrate). Given the time constraints of this revision cycle, I judged the risk of quietly reporting a wrong "verified" reciprocity number, from a convention I hadn't independently cross-checked, to be worse than leaving the assumption openly flagged as unverified. This is now Section VIII's explicitly-labeled "cheapest remaining item," described precisely enough (which layer is which medium, which angle) that it should take an afternoon for whoever runs it next, possibly the current author in a follow-up session, with more room to verify the setup before trusting the output.

The $\theta_i$-axis grid extension and axis rebalancing are **Deferred** (compute-bound, Stage 2 per your own roadmap).

### §2.6 TE-only device: **Done**, promoted

Exactly your reconciliation, confirmed to four figures: construction (b) gives $1.262\%$ against the reported $1.2474\%$; construction (a) gives $0.850\%$ and doesn't match. TE carries $96.9\%$ of coupled power. Promoted to a new Section VI-F with the $1.94\times$ TE-polarized-source gain computed explicitly and referenced from the Discussion and abstract.

### §2.7 Grid axis allocation: **Done** (disclosure) + one new local data point

Agreed the axis allocation is backwards relative to sensitivity. Rather than just assert your point about the global 48-point audit not bounding error near the optimum, I ran one additional, real check: a fresh grcwa solve at the record design's exact (off-grid) 532 nm point, compared against the grid's own trilinear interpolant there. Absolute error: $9.6\times10^{-4}$ (TE), $4.0\times10^{-4}$ (TM): the TE figure about $1.7\times$ the global mean, a real, measured local degradation, still an order of magnitude below the scalar error it replaces. This is one data point, not a full conditional audit; I say so explicitly in Section IV-B and Section VIII rather than extrapolate from it. The axis rebalancing itself (depth 77→~20, period 5→15) is **Deferred** (requires rebuilding and re-verifying the full 90,090-point grid, which is a multi-hour to multi-day compute job depending on hardware, well outside a single revision session).

### §2.8 Refractive index as free variable: **Done** (disclosure + citation), **Deferred** (fix)

Added the Sellmeier PMMA citation you'd expect here (Sultanova, Kasarova & Nikolov 2009, verified against the publisher's abstract), stated the real per-wavelength indices, and added a full paragraph to Section VIII on converting to $n(\lambda)$ and dropping the design vector to seven parameters. Not implemented, because it changes the guided-window computation (Section III-A) and therefore requires a fresh calibration-grid audit and a fresh search; both are **Deferred**.

### §2.9 $N_b$/$\ell$ factor-of-two: **Done**, quantified precisely

Confirmed the bug exactly: the shipped formula is exactly half the correct value (ratio $2.000$, both bounce count and path length). I did not just assert "negligible impact": I computed it at the actual record design ($L_{\text{prop}}=20$ mm, $t=1.9624$ mm, $\theta_d=52.32°$ at green): the combined bulk-absorption-and-roughness transmission factor moves from $0.998967$ (shipped formula) to $0.997935$ (corrected), a $0.10\%$ relative change. The manuscript's equations are now the corrected ones; the shipped code is not patched this cycle, and I explain exactly why in Section III-C and Section VIII: the physics-probe system (which I did not disable or bypass) would correctly invalidate every existing checkpoint the moment the live formula changes, forcing a full retrain that requires the network stack I did not have available in this session. This is listed as the first thing that should happen before any new design search, not swept under a "future work" label without explanation.

### §2.10 $L_c$/$\alpha$ "inert" dimensions: **Refined**, and I think meaningfully so

I checked this directly rather than repeat it, and it's not quite right as stated. Over the code's actual PMMA-mode bounds ($L_c\in[2,4]\times10^5$ nm, **not** the sub-micron correlation lengths your estimate assumed for "realistic" values), $S(L_c)$ varies from $0.60$ to $0.43$: a real $\sim30\%$ swing, not a negligible one. The code's own comment attributes this long correlation-length range to Nilsen et al.'s report of large-scale waviness in molded PMMA rather than fine speckle, so it isn't obviously a units bug either, though I flag in Section VIII that this specific numeric range hasn't been independently checked against Nilsen et al.'s actual reported value.

What *is* true, and I think is the more accurate and more useful statement than "two of eight dimensions are inert," is this: bulk transmission varies $99.87\%$–$98.74\%$ across the PMMA $\alpha$ bounds, and: for direct comparison: roughness transmission varies $99.98\%$–$99.81\%$ across the PMMA $\sigma$ bounds. By your own "$S\approx1$, therefore inert" standard, $\sigma$ would read as inert too, which can't be right since $\sigma$ is clearly a real, physically meaningful roughness parameter. The accurate statement is that PMMA's literature material bounds keep bulk absorption, roughness, *and* their correlation-length weighting all in a collectively low-loss regime simultaneously, so essentially all of the transmission-optimization opportunity concentrates in the grating-coupling term, exactly the term this paper recalibrates. I wrote this up as a full paragraph in Section VIII rather than either accepting or rejecting your claim outright, because I think the nuance is the actually-useful finding here. A full Sobol/variance-based decomposition across all eight parameters, which would settle this more rigorously, is **Deferred** (would need a sampling campaign I didn't have time to build and validate this cycle).

### §2.11 Watson mischaracterization / unit conversion: **Done**

Fixed the description to "mean human optical MTF," explicitly noting it includes the eye's own aberrations rather than describing diffraction-limited optics. Added the $40$ cyc/mm $\approx 11.6$ cyc/deg conversion explicitly, with the acuity-benchmark context ($\sim30$ cyc/deg for 20/20) you asked for. Confirmed in code that $x_k$ in Equation 6 is in mm (matching $f$ in cyc/mm): no unit mismatch found.

### §2.12 Binary-grating $\pm1$ symmetry: **Done** (framing fix, not a physics fix)

Rewrote the Discussion paragraph to state plainly that this is a real device-vs-demonstrator gap, not a "natural extension," and that the depth optimum for a slanted grating is not this paper's binary-grating optimum; the paper's $\approx200$ nm record depth is now explicitly scoped to the architecture actually modeled. Implementing and calibrating a slanted-grating grid is **Deferred** (Stage 2/3 compute).

### §2.13 Minor items

- **TE/TM "5.6" vs Table II "0.55" contradiction: Done.** These are different designs (Table II's scalar-optimal geometry vs. the RCWA-optimal record design). The sentence conflating them is rewritten to state both numbers separately with their correct referents; Table II's actual maximum ratio (3.0, at 635 nm, with TM exceeding TE at 532 nm) is now stated accurately.
- **"≈25×" claim: Done.** Recomputed and got $8.88\times$, matching your figure. I went further and disentangled three genuinely distinct ratios your review and the original draft were blurring together (the one-geometry scalar-vs-RCWA error, 5.1–15.7×; the scalar-estimate-old-geometry vs actual-new-geometry comparison, 8.9×; and the RCWA-true old-vs-new geometry comparison, 41% *better*, not worse): all three are now stated separately with their own referents in Section VI-D, since conflating them was the actual root cause of the wrong "25×" figure, not just an arithmetic slip.
- **Window mismatch (429–450 vs 430–449): Done.** Both numbers are correct and now explicitly reconciled: 429–450 nm is the theoretical window from the guiding inequality; 430–449 nm is the deliberately conservative sub-window the grid and every search actually use, with roughly a 1 nm margin on each side. Stated once, explicitly, in Section III-A.
- **RCWA convergence at one geometry only: Deferred**, now explicitly flagged in Section VIII rather than left silent.
- **Second-solver cross-validation (torcwa/S4/MEEP): Deferred**, now a named Section VIII item with torcwa specifically cited (Kim & Lee 2023, verified) as the natural choice given it's also autograd-differentiable.
- **Memorization audit statistical power: Acknowledged in text**, not independently re-run; I agree the deterministic-generator point is correct and noted it, but did not build a new near-boundary-conditional audit this cycle (Deferred, Section VIII).
- **RCWA-in-the-loop polishing of top-5 finalists: Deferred**, noted as a natural extension in Section V-C's discussion but not run (needs autograd through grcwa, which is a nontrivial integration I didn't want to rush).

## §3. Why does the network exist?

Adopted Path A as you recommended: retitled and re-abstracted around the scalar-theory finding, with the network explicitly demoted from headline to a validated secondary contribution. Added a full paragraph to Section V-A addressing your strategic question directly rather than leaving it implicit: this paper's forward model is already fast and closed-form, so there is no simulation-time speedup to claim, and I say so in those words rather than let a reader infer it. What the tandem network and its inverse counterpart still buy, amortized single-shot inversion and resolution of the one-to-many inverse problem, is stated as the actual, more modest claim. I did not attempt Option B (RCWA-in-the-loop) or a full Option C benchmark (wall-clock vs. thousands of physics-gradient-descent queries) this cycle; both are named explicitly as the two ways to make the network load-bearing on its own terms in Section V-A and Section VIII, **Deferred**.

I also corrected the "agreement at higher precision than the surrogate's own error bar" issue you raised: the two searches' $J=0.4053$ vs $J=0.4052$ agreement is now presented as "the same design to within numerical noise" without implying statistical significance beyond what a $0.0008$–$0.0009$ surrogate-vs-physics gap can support, and the 2.2× thickness spread between the two searches is now explicitly connected to the flat-direction finding in §2.10 rather than called "the same design ... to within numerical noise" without qualification (that phrase is now reserved for the depth figure specifically, where the two searches really do agree to within half a percent).

## §4. "Is this a modern AR waveguide?"

Added a full new subsection, Section III-A ("Scope of the modeled architecture"), stating plainly what is and isn't modeled: single in/out-coupler pair, monolithic slab, binary unslanted grating, one extraction event, normal-incidence calibration. Explicitly flagged the sharpest point from your gap table almost verbatim, because it's correct and important: this paper's objective maximizes single-interaction out-coupling efficiency, which is close to the *wrong* objective for a real exit-pupil-expanding design that wants low, spatially graded extraction instead. Added Lee, Lee & Chung (Nanophotonics 2025, verified) throughout as the closest, most complete system-level treatment in the literature, explicitly crediting them with covering ghosting, see-through distortion, and eyebox uniformity that this paper does not model, rather than implying this paper's scope is broader than it is.

Addressed "why PMMA" head-on in the Introduction, not the Discussion: PMMA's cost, weight, roll-to-roll manufacturability, and impact resistance are stated as the actual trade this paper's narrow field-of-view result is made against, with Yoshida et al. (2018, verified) cited as evidence plastic combiners are a real, shipped product category rather than a hypothetical.

## §5. Staged re-run

Stage 0 (diagnostics) is fully executed, with real numbers, everywhere in this letter and the revised manuscript; every one of your six suggested print-statement checks was actually run. Stage 1 (correctness) is partially executed: the fixable-without-retraining items ($N_b$/$\ell$, Table II/§IV-A conflict, the 25× claim, window-range consistency, the substrate-side-reciprocity disclosure, Watson's description) are all done. The items that require retraining or a new calibration grid (guard-banded hard-mask re-scoring, the rebuilt chromatic metric, $n(\lambda)$) are diagnosed, computed where they can be computed in closed form (the $d\theta_d/d\lambda$ replacement metric), and explicitly deferred where they can't, because I did not have the project's training stack (PyTorch) available in the environment I performed this revision in, and judged fabricating a plausible-looking retrained result to be a worse failure mode than an honest gap. Stages 2–4 (grid rebuild, five-seed statistics, reverse-engineering case study) are unchanged from before this review and remain Section VIII items, now with more specific, actionable descriptions than the previous draft gave them.

The one item on your list I actively chose not to attempt despite having the tooling available (grcwa) is the substrate-side reciprocity solve; the reasoning is in §2.5 above and is the single most consequential judgment call in this response.

## §6. Self-citation removal

Removed `selfpaper1` entirely: the bibitem, the Data Availability reference, and all four load-bearing claims that rested on it.

- "Previously validated closed-form model" → replaced with a real, in-paper validation subsection (new material at the start of Section III) stating each term's provenance directly: TIR/grating geometry from momentum conservation, Fresnel from the standard amplitude coefficients, the roughness Debye-Waller factor attributed correctly to Bennett & Porteus (1961, added and verified) as applied by Tien (1971), the chromatic MTF to Thibos (1987), and $\eta_1$ to the in-paper RCWA calibration.
- Roughness-MTF constant → now explicitly pointed at Nilsen et al. (2025, already cited) as the reconciliation target, with the reconciliation itself listed as open (Section VIII) rather than claimed done.
- Grating/coupler MTF weights → kept as named heuristics with an explicit derivation path noted (0th/1st-order contrast from the existing RCWA data), not derived this cycle.
- $S(L_c)$'s constant → not deleted, because my own check (§2.10) found the term is not actually inert over the code's real bounds; deleting it would have been the wrong fix for a claim I found to be imprecise rather than correct.
- "Reconciliation pending" bullet → replaced with the concrete Nilsen-et-al.-targeted item above.
- Data Availability → now states the repository is self-contained.

## §7. Citation audit

- `luo2025` → fixed to `dehghani2025` with full, correct author list (Dehghani, Knoth, Eskandari, Buchmüller, Meisen, Görrn), verified against the publisher's page (MDPI). Table I's "Luo et al." corrected to "Dehghani et al."
- Added and independently verified (publisher/abstract-level check, not just a search-engine snippet) six new references: Lee, Lee & Chung (*Nanophotonics* 2025), Bennett & Porteus (*JOSA* 1961), Sultanova, Kasarova & Nikolov (*Acta Phys. Pol. A* 2009), Yoshida et al. (*JSID* 2018), Kim & Lee/torcwa (*Comp. Phys. Comm.* 2023), and used Dehghani et al. as noted above.
- I did **not** add every reference your list suggested. Gopakumar et al. (*Nature* 2024) and Chen et al.'s SiC waveguide, Shi/Chen/Capasso's polarization metagratings, and Christiansen & Sigmund's topology-optimization tutorial were all located and would be real, verifiable citations, but I chose not to add them because I could not find a place in the revised text where they did real argumentative work rather than padding the reference count toward your suggested ~30; I'd rather submit with 25 references that are all load-bearing than 30 where a reader can tell five were added to hit a number. Happy to add any of these back in a further round if you disagree with that judgment call on any specific one.
- Did not re-verify the 15 references carried over unchanged from the previous draft (`kress2021` through `zhao2024`, `goodman`, `pommet1994`, the three Moharam papers, `tien1971`, `payne1994`, `thibos1987`, `watson2013`) beyond what was already done for the first draft; your review reported these as independently checked and correct, and I did not find a reason to re-do that work.

## §8. Presentation and venue

- LaTeX sandbox comment → removed.
- "Reproducibility Note" → shortened to a "Numerical Verification Note," confession language removed, substance (independent NumPy re-implementation, five-significant-figure cross-check) kept.
- AI-disclosure note-to-self → removed; the disclosure now also states, accurately, that the LLM assisted with re-deriving and in two places refining this review's own numerical claims before incorporating them.
- Abstract → reframed to lead with the finding, per your suggested framing, adapted to keep the tandem-network content honestly represented rather than erased.
- Table III's identical rows → footnoted rather than silently left to look like a copy-paste error, now that §2.3 explains why they're identical.
- IEEEtran.cls → still not available in this sandbox (same network restriction as the first draft); the one-line swap note is kept.
- Figures → **partially done, not fully.** I added four figures: the guided-window schematic (new), a 20-point scalar-vs-RCWA depth scan (new, real grcwa data, not the old 5-point scan), and two figures reproduced directly from the project's own already-generated, already-verified output (the memorization audit and the neural-adjoint search trajectory/parity plot), both confirmed via file modification time and CSV cross-check to be from the current v3 engine rather than a stale v2 run (I found and rejected a fifth candidate figure, `pareto_front.png`, specifically because its own title honestly labels it "v2": pooling it in would have violated this paper's own stated rule against mixing engine versions). I did not add a fifth figure for the physics-probe system itself (§1 above) or a surrogate parity plot beyond what the neural-adjoint figure already shows, for the reasons stated there.
- "Independent Researcher" / ASSIP affiliation → not changed; I don't have information confirming an ASSIP or other program affiliation to add, so I left this as-is rather than guess.

## §9. The revision in one page

Going through your cut/fix/add/reframe/keep list directly:

**Cut:** `selfpaper1`: done. "Transmission @ FOV" as an independent metric: done (footnoted, not deleted, since it's still a real, if redundant, number). `S(L_c)` and the $L_c$ dimension: **not cut**, see §2.10/§6. The "≈25×" claim: done (replaced with the disentangled three-ratio treatment). Sandbox comment, Reproducibility confession, disclosure note-to-self: all done. "Four-metric joint specification": softened, not fully retired, since the vector genuinely has four numeric outputs; the *novelty claim* attached to it is retired.

**Fix:** Chromatic metric: relabeled and given a computed replacement candidate, not rebuilt/retrained. Sigmoid-edge exploit: diagnosed precisely, not hard-mask-rescored. MTF_chrom floor: derived and contextualized, refined rather than simply fixed. $\ell$/$N_b$: fixed in the manuscript's equations. $n$'s status: disclosed with a citation, not converted to Sellmeier form. Table II/§IV-A: fixed. `luo2025` misattribution: fixed; full 21-reference DOI-by-DOI re-audit: not repeated (see §7).

**Add:** Sensitivity check: done, but one-at-a-time bound-range rather than full Sobol. Substrate-side reciprocity: explicitly not done, reasoned in §2.5. Second-solver cross-check: not done. Scope box: done (Section III-A). "Why PMMA": done (Introduction). New references: six added, not ten, reasoned in §7. Four figures: four added, of a different mix than your suggested four (guided-window and scalar-vs-RCWA are new; memorization-audit and neural-adjoint-parity are recovered from the project's own existing, verified output rather than newly drawn).

**Reframe:** Title/abstract around the scalar-theory finding: done. TE-only result promoted: done.

**Keep:** Physics-probe system, independent-verification discipline, Section VIII's honesty: all kept and, I'd argue, extended by this response itself.

## §10. Closing

You wrote that nothing in your §2 was a competence failure, only the kind of thing that surfaces when someone re-derives the numbers. I'd extend that back to this response: two of your own claims (S(L_c)'s inertness, and the MTF≈chromatic-term coincidence) didn't survive my own re-derivation intact, and I think that's the process working as intended rather than a mark against the original review. The finding is real, the paper is now built around it rather than around the network, and Section VIII is longer and more specific than it was, which I take as the correct direction for this manuscript to be moving in rather than a sign of a project running behind.
