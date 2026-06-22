# potentialplans.md — Expansions for the Chronos circuit paper

*Planning doc for turning the current result (change-detection and periodicity are **distributed** across attention
heads, MLP layers, and mid-encoder MLP features in Chronos-T5, Base and Large) into a stronger paper. Each expansion
below is scoped against the existing de-confounded pipeline: counterfactual interchange ablation, the structural
metrics (`changepoint_recovery`, `period_power_fraction`), faithfulness + completeness + selectivity (vs motif) +
sufficiency, SNR sweeps, and bootstrap CIs. Honesty discipline carries over: lead with what is established, label
exploratory work, report a minimum detectable effect so any null reads as de-confounded.*

## Status context (not part of this doc's plans)
- **Cross-layer feature circuit (L9–L17, union-ablated): IN PROGRESS.** This is the highest-leverage open item and
  the one untested granularity; it is already being run, so it is excluded from the plans below.
- **ETT real-data confirmation: NON-NEGOTIABLE, not an "expansion."** Everything so far is synthetic, and synthetic
  saturation is the first thing a reviewer attacks on a null. Reproducing "no localized circuit; structure survives
  small ablation" on real seasonality and real regime shifts closes the central objection. Do this regardless of
  which expansions below are pursued.

## Leverage summary

| Expansion | What it changes | Tier impact | Feasibility / risk |
|---|---|---|---|
| 1. Generalize the SAE-vs-circuit discrepancy | A standalone methods claim | **Tier-changer** (highest novelty) | Moderate; needs multiple features / a 2nd SAE |
| 2. Steering | A positive, interactive result | High value | Low-moderate; likely works for periodicity |
| 3. Attribution graphs | Characterizes the distributed computation | Methods/prestige upgrade | Moderate-high; transcoder-grade, can be diffuse |
| 4. Second behavior | Broadens "what is distributed in TSFMs" | Breadth; possible positive circuit | Low-moderate; pick a clean structural metric |
| 5. Across models / scale | Universality; turns nulls into a trend | Tier-changer if scale-dependent | Scale sweep low; cross-arch high |

---

## 1. Generalize the SAE-versus-circuit discrepancy (the standalone methods contribution)

**What it is.** The current discrepancy rests on one feature in one model: Mishra's top mid-encoder level-shift SAE
feature is interpretable but is not a sufficient cause of change-detection (denoising gain ~0). Turn that into a
general claim: SAE features systematically overstate causal localization.

**Why it matters.** This is the most novel, most topical, and most *standalone* contribution available; it does not
depend on the Chronos result landing a particular way, and "an interpretable SAE feature need not be a sufficient
cause" is a live methodological question for the whole SAE-interpretability program. It is the best candidate for a
main-track headline.

**How to do it.** Take several of Mishra's interpretable features (the top-N level-shift and seasonality features,
not just the single best), and for each run the sufficiency (denoising) and completeness tests from the corrected
Phase 5 v2 pipeline. Quantify, per feature, the gap between an interpretability score (activation correlation with
the labeled event, the basis Mishra used) and causal sufficiency (the denoising gain / completeness collapse).
Report the distribution of (interpretability vs causal sufficiency) across features, with CIs. Strengthen by
replicating on a second SAE (a different seed or width) and ideally a second model.

**Outcomes.** If most interpretable features fail sufficiency, you have "SAE features systematically overstate
causal localization," a clean general claim. If some pass, you have a taxonomy of which interpretable features are
and are not causal, which is also publishable and arguably more useful.

**Risk / fit.** Moderate; depends on SAE availability and on the pattern holding across features. Becomes the
paper's headline methods section; the Chronos circuit results become the worked example that motivates it.

## 2. Steering (the positive, interactive result)

**What it is.** The denoising direction already built *is* steering; it returns null for change-detection. Test it
where it should work: inject or scale a feature set to induce or suppress **periodicity**, or to amplify a detected
level shift.

**Why it matters.** A working steering demo is a positive, memorable, interactive result that distinguishes the
paper from Mishra (ablation-only) and from time2time (layer-level concept steering). It gives the paper a control
knob and a figure reviewers remember, even if the headline scientific result is a distributed null.

**How to do it.** Identify the feature set associated with periodicity (or with the shift signal), inject it into a
context that lacks the behavior, and sweep the steering strength; measure the structural-metric response (period-P
power for periodicity, changepoint recovery for shifts) as a dose-response with CIs. Run the suppression direction
too (ablate-to-induce-flat). Selectivity check: steering periodicity should not move the changepoint metric, and
vice versa.

**Outcomes.** Working steering on any behavior is a positive result and a demo. Failed steering everywhere further
strengthens the distributed claim (the computation cannot be grabbed by any small handle).

**Risk / fit.** Low-moderate; periodicity steering is more likely to work than change-detection (which came back
distributed). A dedicated "controllability" section plus a dose-response figure.

## 3. Attribution graphs (the prestige characterization)

**What it is.** An end-to-end attribution graph from input tokens through attention and MLP features to the forecast
logits, for change-detection and periodicity.

**Why it matters.** Everything so far is ablation and patching. Attribution graphs are the current frontier of
circuit work, and even for a *distributed* computation, a full graph that shows the boundary or period signal
fanning out across many features is a stronger positive artifact than a faithfulness curve: it visualizes *why* the
computation is distributed rather than only asserting it.

**How to do it.** Build the attribution graph using the transcoder / circuit-tracing approach (attribute the metric
back through feature nodes and the residual stream, prune to significant edges). Produce the graph for both
behaviors and annotate where the signal concentrates versus diffuses. Carry the error nodes, as in Phase 5 v2.

**Outcomes.** A characterization figure and a graph object; "here is the distributed change-detection computation,"
which converts the negative into a positive structural description.

**Risk / fit.** Moderate-high; transcoder-grade and research-effort, and for periodicity (least linearly accessible,
per Pandey) the graph may be diffuse and hard to read. The methods and visualization upgrade for the paper.

## 4. Second behavior beyond periodicity and change-detection (breadth)

**What it is.** A third controlled computation run through the same pipeline, for example trend extrapolation,
missing-data / imputation handling, or variance / regime-volatility shifts.

**Why it matters.** Two behaviors is a case study; three or more makes the paper about TSFM computation generally,
and broadens the central claim from "these two are distributed" to "the computations we tested are distributed."

**How to do it.** Design controlled stimuli and a clean structural metric for the new behavior (avoid trend-slope,
which was unstable here; prefer a behavior with a well-posed, monotone metric), then run the full de-confounded
pipeline: counterfactual minimal pairs, interchange ablation in both directions, faithfulness + completeness +
selectivity + sufficiency, SNR sweep, CIs.

**Outcomes.** Distributed again strengthens the general claim. A *localized* third behavior would be the first
positive circuit in the project and a significant result on its own.

**Risk / fit.** Low-moderate, contingent on a clean metric. Another point on the distributed-versus-localized axis;
a localized hit here would be a headline.

## 5. Across multiple models and scales (universality)

**What it is.** Replicate beyond a single Large checkpoint. Two sub-tracks.

**5a. Scale sweep within the Chronos family (Tiny / Small / Base / Large).** Cleanest, same tokenization, reuses the
pipeline directly. Asks whether localization is scale-dependent: does change-detection stay distributed at every
scale, or sharpen / concentrate as the model grows (the Olsson-style phase-transition question)? "Distributed at
every scale" is a real systematic finding; any sharpening is more interesting than a single-model result and would
turn the nulls into a trend.

**5b. Cross-architecture (TimesFM, Moirai, Chronos-2).** The universality move that established credibility in LLM
interp, and the highest-variance item. The obstacle is structural: patch-based models break the lag-in-tokens
correspondence the pipeline assumes, so "lag" and the counterfactual must be redefined relative to patch boundaries,
and Chronos-2 has no decoder cross-attention. Frame as "does an analogous distributed pattern appear," and be
prepared for it to be a limitations section rather than a clean result.

**Outcomes.** Distributed across scale and architecture is a strong universality claim. Scale-dependent localization
is the more interesting trend and a possible tier-changer.

**Risk / fit.** 5a is low-moderate and high-value; do it. 5b is high-risk and messy; pursue only for a longer arc.
Together they reframe the single-model negatives as a systematic study of TSFM computation.

---

## How these combine into paper framings

- **The ICLR-positive path:** cross-layer circuit lands (in progress) → add steering (Section 2) + the scale sweep
  (5a) → "here is the cross-layer change-detection circuit, it is controllable, and its structure is consistent
  across scale." A positive, systematic, main-track story.
- **The methods-paper path:** generalize the SAE-vs-circuit discrepancy (Section 1) + attribution graphs
  (Section 3) → "interpretable SAE features systematically overstate causal localization, demonstrated with full
  attribution graphs across features and behaviors." Stands on its own regardless of the Chronos circuit result.
- **The breadth / systematic-study path:** second behavior (Section 4) + scale sweep (5a) + ETT → "across behaviors
  and scales, TSFM computations are distributed, established with de-confounded causal methods." A rigorous,
  complete negative, strong at a workshop and defensible at AAAI with the methods framing.

## Recommended order
1. (In progress) cross-layer feature circuit, the tier-decider.
2. ETT real-data, the non-negotiable credibility item.
3. Steering on periodicity, the cheapest positive result.
4. Scale sweep (5a), turns the nulls into a trend at low cost.
5. Generalize the SAE discrepancy, the highest-ceiling standalone contribution.
6. Attribution graphs and a second behavior, the depth and breadth upgrades.
7. Cross-architecture (5b), only for a longer arc.

**Do not** add more methods to the change-detection *null* specifically; it is thoroughly established across three
granularities, and re-confirming it a sixth way has diminishing returns. The leverage is in a positive result
elsewhere (cross-layer, steering, a second behavior) or a generalized method (the SAE discrepancy), not in
re-establishing the negative.
