# Response to Reviewers — Round 2

**Manuscript:** *When Scalar Diffraction Theory Fails: RCWA-Calibrated, Physics-Anchored Inverse Design of Polymer AR Waveguide Couplers* (Revision 3)
**Author:** Connor Wang
**Response date:** 2026-07-17

Every claim below was checked independently against either the project's own source files or a primary source I fetched and read myself (Nilsen et al., *Optics Express* 33(9), 20051, 2025 — full text, not the referee's quotes from it), not taken on the referee's word. Two of the referee's specific numeric claims (§3's exact roughness-loss and MTF figures) I could not reproduce to the referee's own precision using the paper's own formulas; where that happened I report my own re-derivation and note the discrepancy rather than silently adopting the referee's number.

Legend: **Done** = fixed in the manuscript, verified. **Confirmed, extended** = the referee's finding held up and I found additional supporting or complicating evidence. **Declined** = I did not implement this as asked, with the reason stated. **Deferred** = agreed, not implemented this cycle, tracked in Section VIII.

---

## Two items I'm declining outright

Before the point-by-point response: the accompanying instructions to this round's review asked for two things I'm not doing, and I want to state that plainly rather than let it surface as a silent gap.

1. **Chasing a Turnitin AI-detection score.** I don't have access to turnitin.com, and more importantly, I wouldn't optimize prose against an AI-detector's score even if I could: that targets a proxy (does this pattern-match as machine-written) rather than the actual goal (is this clear, accurate, and not overwritten). I did do a real prose pass — cutting the peer-review narrative from the abstract, conclusion, and disclosure; converting Section VIII from a bulleted list to argued paragraphs; tightening several overlong sentences — but the target was clarity and correctness, not a detector score.
2. **Deleting the AI-Usage Disclosure entirely.** This project used AI assistance substantially throughout (code, verification, drafting), and removing the disclosure would misrepresent that. I trimmed it: the disclosure no longer narrates the mechanics of the simulated review process (which was the actual, valid complaint in §5.1 below), but it still states, factually, that AI assistance was used and for what.

Everything else in the review is addressed below.

---

## §0-1. Verdict and what's credited

No action needed; this section is retained as context for the rest of the response.

## §2.1. MTF decomposition: reframed as the referee requested

I swept $\mathrm{MTF}_{\mathrm{chrom}}(f)$ over $f\in[20,60]$ cyc/mm at the record design (a closed-form calculation, no retraining). The result is sharper than a directional confirmation: the function has roughly twenty local maxima across that range, and $f=40$ cyc/mm — the paper's evaluation frequency — lands, to the resolution of the sweep, essentially exactly on one of them. Combined with the $6.1\%$ tail probability figure (independently reproduced via my own Monte Carlo, matching the referee's), this is now reported in Section III-D as a positive finding (constructive-fringe exploitation of an oscillating metric), not a refutation, and the "coincidence of composition" framing that appeared in the Abstract and Conclusion has been removed from both. **Done**, and I agree with the referee's reading over my own Round 1 one.

## §2.2. $S(L_c)$: the real issue was $\sigma$, not $L_c$, and $L_c$'s waviness attribution was fabricated

Once $\sigma$ is corrected (§3 below), $L_c$'s $\sim\!40\%$ swing moves roughness transmission by roughly nine percentage points, not the fraction of a percentage point it moves at this paper's coated bounds — the referee's "L\_c matters once $\sigma$ is right" point is confirmed exactly, and is now in Section VIII. Separately: I checked the physics engine's own source comment (`physics/waveguide_physics.py`) directly, and it does **not** attribute the $L_c$ range to Nilsen's report of waviness — that attribution appeared only in my own Round 1 response letter and, apparently, in the manuscript prose that quoted it, neither of which was ever true of the actual code comment or of Nilsen's paper (which explicitly excludes waviness from its scope). That sentence is removed from the manuscript; I no longer attribute the $L_c$ bounds to any specific literature source and flag them as an unverified modeling choice instead. **Done.**

The "log-uniform… spanning several decades" claim in Section V-A was also wrong for the PMMA-mode bounds specifically ($L_c$ spans $0.30$ decades, $\alpha$ spans $1.0$ decade, neither "several"); the false justification is removed, the sampling-method claim (log-uniform) is kept since it's still accurate. **Done.**

## §3. Roughness bounds — the central finding of this round

I fetched and read Nilsen et al. in full (not excerpts) and confirm the referee's three $\sigma$ values exactly: glass $5$ nm, untreated polymer $15$ nm, spin-coated polymer $0.87$ nm, all from Figs. 7-8. This paper's $\sigma$ bounds ($[0.7,1.1]$ nm) do bracket the coated value, and the manuscript's description of the device as "uncoated" was wrong; it's relabeled "spin-coated" throughout.

Where my numbers diverge from the referee's Appendix script: using the paper's own formula (Tien/Bennett-Porteus, corrected $N_b$, and the record design's actual $\sigma=0.710$ nm rather than an assumed $0.9$ nm), I get roughness-scattering transmission of $69.9\%$ ($30.1\%$ loss) at $\sigma=15$ nm against $99.9\%$ ($0.08\%$ loss) at the record's actual coated value — in the same ballpark as the referee's $66.0\%$/$34\%$ but not identical, likely from a different assumed baseline $\sigma$. I report my own re-derivation in the manuscript, not the referee's number, since I can trace mine to the paper's exact formula and design parameters.

I could **not** reproduce the referee's specific claim that the paper's separate $\mathrm{MTF}_{\mathrm{rough}}$ heuristic gives $\approx 0.45$ at $\sigma=15$ nm to match Nilsen's Fig. 7. Evaluating the paper's actual heuristic formula (a Gaussian-blur approximation with hand-tuned coefficients, distinct from the transmission-side Tien formula) at $\sigma=15$ nm instead gives a numerically zero result — the heuristic's coefficients were evidently only ever exercised within the narrow coated-$\sigma$ range this paper used, and the formula does not extrapolate credibly to the uncoated regime. I did not force this to match Nilsen's Fig. 7 value by adjusting coefficients on the spot, since that would be fitting a heuristic to a single external data point without justification; instead I report the transmission-side result (which does have a defensible physical form and gives a moderate, credible number) as the honest quantitative estimate, and flag the MTF-side heuristic's breakdown as a specific, now-quantified limitation requiring real recalibration (ideally against Kuang et al. 2020's closed form, newly cited) rather than a coefficient patch. **Done**, with a different resolution than requested but, I think, a more honest one.

I did **not** re-run a full search with $\sigma$'s upper bound widened to $15$ nm (that requires the training stack, which is not available in the environment I did this revision in) — I evaluate the *existing* coated-optimal record design under the uncoated hypothesis, which is a sensitivity analysis, not a re-optimization, and I say so explicitly in the text. A fresh uncoated-bounds search, which might find a different geometry that partially compensates, is now a named Section VIII item. **Partially done, rest deferred with reason.**

The coating-cost framing (spin-coating requires an extra manufacturing step, cutting against "minimum-cost") is now in Section III-C.

## §4.1. Response-letter/manuscript diff

Confirmed the referee's finding: the Round 1 response letter's §3 claimed a "flat direction" cross-reference that was not actually in the manuscript's VI-C text. Rather than fix the letter to match the (unfulfilled) manuscript, I fixed the manuscript to actually make the argument the letter described: VI-C now explicitly separates the four tightly-agreeing parameters from the thickness spread, explains thickness's flat direction mechanistically (it only enters through bounce count, which barely affects a collectively-low-loss absorption/roughness regime), and cross-references Section VIII's sensitivity analysis rather than leaving "the same design to within numerical noise" ungoverned. **Done.**

## §4.2. Index-axis interpolation error

I ran the three off-node index solves the referee suggested ($n=1.485,1.495,1.4975$) at the record design's exact period/depth/duty and compared against the trilinear interpolant. First attempt at this produced nonsense (errors of $0.05$-$0.09$, wildly inconsistent with the paper's own reported global mean error of $5.6\times10^{-4}$); tracing it down, my own verification script had the calibration grid's array axes in the wrong order (I had swapped the wavelength and index axes). With the correct axis order, the sanity check against exact grid nodes reproduces the paper's own numbers closely, and the off-node index error is $4.8$-$4.9\times10^{-4}$ (TE) and $2.1$-$2.3\times10^{-4}$ (TM) — essentially the *same* magnitude as the interpolant's error from period/depth/duty alone at this same point, not meaningfully larger. This is the opposite of what the referee's elasticity argument predicted, and I report it as a genuine, checked, reassuring finding rather than either the referee's concern or a silent non-issue. **Done**, and worth flagging: this exact bug (axis order) is a good demonstration of why I re-verify computed claims against a second, independent code path rather than trust a single script, including my own.

## §4.3. Guided window with real dispersion

Computed directly from the single-term Sellmeier fit \cite{sultanova2009} rather than approximated: $n(450)=1.5006$, $n(532)=1.4937$, $n(635)=1.4886$ (these differ from the placeholder values an earlier draft stated without deriving them — also fixed). Dispersion-corrected window: $4.35°$ against the single-index $5.00°$, roughly a $13\%$ narrowing — close to but not identical to the referee's $4.13°/4.37°$ estimate, likely from slightly different intermediate rounding. Now in Section VIII with the derivation shown. **Done.**

## §4.4. TE ratio as the headline

Abstract and Section VI-C now lead with the $3.4\times$ TE-to-TE ratio (and the $1.9\times$ further gain from a polarized source), with the $41\%$ unpolarized figure kept as a secondary, explicitly-scoped number rather than the headline. I did not additionally re-derive an $11.5\times$ transmission-with-TE-source figure beyond what's already in Section VI-F, since that section already computes and states the $1.94\times$ *further* gain on top of the record design's own transmission, which is the more directly useful number for a reader (an $11.5\times$ figure would double-count the coupling-efficiency gain already reflected in going from the old to the new geometry). **Done**, with a scoping choice noted.

## §4.5. $\arg\max_d\,\eta_1(\Lambda,D)$ over the full grid

Ran this, and it is more informative than a flat confirmation would have been: for TE, all $65$ (period, duty) cells favor a depth of $195$-$240$ nm — a tight, robust interior optimum across the whole plane. For TM, only $43\%$ of cells favor a depth below $250$ nm; the rest favor $350$+ nm, up to the scalar model's own $400$ nm boundary. Read against Section VI-F's finding that the record design is $96.9\%$ TE, this strengthens rather than weakens the paper's central claim — the interior depth optimum is robust specifically for the polarization channel this device uses — but it also means the paper's existing *unpolarized*-framed statements of the depth-optimum finding (Sections IV, VI) are not, by themselves, fully supported across the whole plane, only close to the record design's specific location and for TE. This is now stated explicitly in Section VIII rather than only reporting the favorable TE half of the result. **Done**, and no new figure was added this cycle (see §4.7 below for why); the numbers are reported in text.

## §4.6. Guard-penalty status

Checked directly in both search scripts (`baselines/optimize_pmma.py`, `networks/neural_adjoint.py`): both objectives include the identical term $-10\times\texttt{tir\_penalty}(\theta,\delta{=}0.01)$, so the guard band was active for both procedures that produced the record design, not merely "available." VI-C now states this explicitly: the residual constraint violation reported there means the penalty's weight was insufficient to fully exclude the behavior, not that the constraint was absent from the objective. **Done.**

## §4.7. Figure budget

Not changed this cycle. I agree in principle that a $(\Lambda,D)$ argmax heatmap would be a better use of a figure slot than it currently gets, but given the §4.5 finding is now more nuanced (TE-robust, TM-not) than the referee's hypothesized clean confirmation, building that figure well means two heatmaps (TE and TM) with careful annotation, which I did not want to rush in this cycle alongside everything else; the numbers are reported in text instead. **Deferred**, now a named Section VIII/figure-budget item for the next revision.

## §4.8. Baseline comparison (direct regression + physics re-scoring)

Not run. This requires the training stack (PyTorch), which is not available in the environment I performed this revision in — the same constraint noted throughout Section VIII for every retraining-dependent item. I did not fabricate a plausible-looking re-scoring result to close this gap. **Deferred**, unchanged from Round 1's honest gap, now explicitly linked to this specific referee request rather than left generic.

## §4.9. Minor items

- Cross-reference fixed: "middle two terms" → "third and fourth terms" (Section III preamble).
- Table I's "Gap this paper addresses" column renamed "Positioning."
- "First neural inverse-design treatment of PMMA" claim removed; replaced with a narrower, defensible positioning statement that doesn't assert an unverifiable negative.
- Abstract's headline transmission ratio reordered: leads with the like-for-like $5.1\times/15.7\times$ (already true of the abstract) and the $3.4\times$ TE figure; the $8.9\times$ "nine times lower" figure is kept but scoped as a secondary, caveated number rather than the lead.
- `luo2024` page range corrected to `12587–12600` (was `12587–12601`).
- ASSIP affiliation: not added. I don't have confirmation of institutional or program affiliation to attribute in this session, and would rather leave the affiliation line as-is than add one on inference. If you can confirm the specific program/mentor details, I'll add them in the next pass.

## §5. Presentation

**§5.1 (disqualifying issue): Done.** The peer-review-process narrative is removed from the Abstract and Conclusion; both are rewritten to state the underlying physics findings (the σ/coating relabeling, the fringe-riding finding) directly rather than as "an independent review found X, we verified Y." The AI-Usage Disclosure is trimmed to a factual statement rather than a blow-by-blow of the review-simulation process (see the declined-items note at the top of this letter for why it isn't removed outright). The LaTeX header comment is shortened to a plain revision note with no reference to a referee report.

**§5.2 (venue-length recommendation):** Noted, not acted on. Splitting this into a Letter plus a follow-up paper is a reasonable editorial suggestion, but it's a scope decision for you as the author, not something I'll make unilaterally inside a revision pass; I've kept the manuscript as a single paper and let Section VIII carry the acknowledged gaps, consistent with how the rest of this revision cycle has worked.

**§5.3:** `luo2024` fixed (above). ASSIP affiliation not added (above).

## §6. Literature audit

Added, all independently verified (not taken from the referee's transcription):
- **Chen et al., *eLight* 5, 21 (2025)** — the SiC single-layer full-color waveguide, cited in the Introduction as a direct, real counter-example to a cost/weight argument for low index made in isolation.
- **Gopakumar et al., *Nature* 629, 791-797 (2024)** — cited in Related Work as field-frontier context, explicitly scoped as not a directly comparable baseline.
- **Ding, Yang, Li, Yang, Wang, Liang, and Wu, *eLight* 3, 24 (2023)** — cited as a current field review. I verified the author list independently and it differs from what the referee's report implied; the citation now reflects the verified list.
- **Kuang, Liu, and Shi, *Optics Express* 28(2), 1103-1113 (2020)** — cited in Section VIII as the more direct reconciliation target for the roughness-MTF heuristic than Nilsen's empirical curves. I also independently verified the author list and page range here, both of which differ slightly from the referee's transcription.

Not added: Frish (2024) and Hara & Shiraga (2023), the two remaining Nilsen-bibliography citations the referee flagged. I have not independently fetched and read either paper, and given that the citation I'd be using them for (supporting or narrowing the "PMMA/polymer waveguide" novelty claim) has already been resolved by simply removing the unverifiable negative claim rather than needing a citation to rebut it, adding them now would be citing sources I haven't verified for a claim that no longer requires them. Happy to add them in a further round if you'd like me to fetch and check them specifically.

## §7-8. Closing

Addressed via the sections above; no separate action.

---

## A note on what this round changed methodologically

The single most useful thing this round surfaced, beyond the σ/coating fix itself, is that I found a real bug in my own verification tooling (the swapped axis order in §4.2/§4.5) by cross-checking a hand-rolled script's output against the project's own already-validated numbers, rather than trusting either source alone. That's the same discipline the paper's physics-probe checkpoint system is built around, applied to my own review process rather than the training pipeline. I think that's a better proof of the paper's stated verification culture than anything I could write directly into the Abstract, which is exactly why it isn't in the abstract.
