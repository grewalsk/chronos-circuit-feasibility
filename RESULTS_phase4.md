# Phase 4 — Results (change-detection circuit; localization test)

*Draft results section for the change-detection arc of the Chronos circuit study. Numbers verified against
[`phase4_pilot_a100.json`](phase4_pilot_a100.json) (chronos-t5-large, 24 encoder layers, 384 heads/site, 3 seeds,
32 series, 2-seed 5-point dose-response sweep; CTX 256, PRED 64, tau at 0.65 of context, obs noise 0.30). Figure:
[`fig5_phase4_pilot_a100.png`](fig5_phase4_pilot_a100.png).*

## Results

To test whether Chronos detects level shifts via a localizable attention circuit, we reused the Phase 3b protocol
on Chronos-T5-Large (24 encoder layers, 384 heads per site), gating on a delta-normalized changepoint-recovery
metric (1 = the forecast tracks the new post-shift level, 0 = it reverts), with full-site ablation measuring
necessity and head-level ablation reserved as the only localization test. Against a size-matched random-head null
(changepoint p95 = 0.243), full-site ablation of encoder-self collapsed changepoint recovery to 0.660 (95% CI
0.530 to 0.784) and decoder-cross to 0.485 (0.326 to 0.646), while decoder-self (0.168) fell below the null.
Neither necessary site was structurally selective: ablating encoder-self collapsed periodicity almost as much as
change-detection (motif 0.549 versus changepoint 0.660), and decoder-cross degraded the two about equally (motif
0.461 versus changepoint 0.485); the trend-slope control was unstable (negative collapses) and is reported as a
soft control only, so selectivity rests on the motif contrast. Critically, the head-level test found no circuit:
within encoder-self the best single head (L9H4) accounted for a 0.039 relative collapse, and a greedy group of the
top eight behavioral candidates reached only 0.081, about 12% of the 0.660 site effect, with no sufficient set of
eight or fewer heads recovering it. Five of the eight single heads did exceed a near-zero within-site null (p95
approximately 0.000), which is why the automated verdict reads "SPLIT," but the partial-localization tail is faint
and we report the result as distributed with a weak localized component, not as a circuit. The dose-response sweep
confirms this shape: encoder-self changepoint collapse is an endpoint effect, near zero through f = 0.5 (0.03,
-0.01, 0.00) and rising only as the ablated fraction approaches one (0.42 at f = 0.75, 0.70 at f = 1.0), while
non-target structure collapses early (0.34 already at f = 0.1); decoder-cross behaves the same (changepoint 0.00 to
0.31 across f, with the low-f severance check passing) and the random arm stays near zero until f = 1. The method
therefore detected large effects (encoder-self 0.660, far above the 0.243 null) but no selective or localized one,
indicating a de-confounded null rather than insufficient power. Because this run is on Chronos-T5-Large, Mishra's
own model, the "base is too small" escape hatch is closed: the change-detection computation that Mishra localized
as mid-encoder SAE features is distributed across attention heads.

Exploratory mechanism probes on the top three encoder-self candidates (L9H4, L1H9, L13H14) are consistent with
attention routing boundary information rather than computing the detection, and we label them directional because
nothing localized. The heads' contribution was invariant to the absolute signal level (boundary-locality
correlation -0.015) and concentrated on the most recent regime boundary (recency mass 1.90 recent versus 0.73
older), but scaled only weakly with shift magnitude (delta slope +0.093, R-squared 0.264) and did not localize
under node patching. A relative-depth comparison to Mishra is likewise exploratory and weak: the behavioral
candidates spread across encoder depth (relative depths 0.00, 0.04, 0.13, 0.39, 0.52, 0.57), with only L12 (0.52)
inside the mid-encoder band [0.45, 0.55], L13 (0.57) just above, and the strongest single head L9 (0.39) below it. A
monotonicity check validated the metric and stimuli (clean changepoint recovery rose from 0.58 to 0.99 with shift
magnitude, Spearman rho = +1.0). The honest reconciliation with Mishra is that attention carries boundary and
recency signals but the level-shift detection is plausibly MLP or feature mediated, which we did not trace and flag
as the headline future-work direction. Taken with Phase 3b, both forecasting computations we have examined,
periodicity and change-detection, are distributed across attention in the original Chronos-T5 at both Base and
Large scales. This is a circuit-level adjudication of the attention-degeneration debate that corroborates Mishra
(2603.10071) and Pandey (2511.15324) at the causal head-circuit level while nuancing them: the structure their
probes and SAEs surface is present in attention, only diffusely, not as a localizable head circuit.

## Figure 5 caption

**Figure 5. Localization test for a change-detection circuit in Chronos-T5-Large.** (5a) Per-site structural
selectivity under full-site mean-ablation: encoder-self and decoder-cross collapse changepoint recovery above the
size-matched random-head null (dashed line), but neither is selective, each collapsing periodicity (motif) about as
much, and decoder-self falls below the null; the trend control is unstable and shown for completeness only. (5b)
Dose-response sweep (multi-seed) ablating a random fraction f of each site: changepoint collapse is an endpoint
effect, near zero until f approaches one, while non-target structure collapses at low-to-mid f, and the cross low-f
severance check passes, so no site localizes change-detection at a non-severing fraction. (5c) Head-level
localization within encoder-self: a greedy top-k group of behavioral candidates reaches only about 12% of the
full-site collapse (best single head 0.039; top-8 group 0.081 versus site 0.660) against a near-zero within-site
null, so there is no sufficient small head set. (5d) Exploratory mechanism on the top candidates: boundary-locality
(level-invariant) and recency (prefers the most recent boundary) with weak magnitude-scaling, plus a relative-depth
comparison placing only one candidate (L12, depth 0.52) inside Mishra's mid-encoder band, consistent with attention
routing boundary information rather than computing change-detection.
