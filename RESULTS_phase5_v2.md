# Phase 5 v2 — Results (feature-level localization; counterfactual interchange ablation)

*Draft Results for the feature-level arc of the Chronos circuit study, the corrected Phase 5 (supersedes the
layer-level Phase 5 for the localization question). Numbers verified against
[`phase5v2_pilot_a100.json`](phase5v2_pilot_a100.json) (chronos-t5-large; enc-MLP L14, relative depth 0.61;
8192-feature TopK SAE, reconstruction loss 0.062; 32 counterfactual pairs; K-grid 1 to 128; SNR deltas
[1.5,3.0] / [0.6,1.0] / [0.3,0.45]; metric = changepoint recovery, clean 0.896). Figure:
[`fig6_phase5v2_pilot_a100.png`](fig6_phase5v2_pilot_a100.png).*

## Results

To test whether change-detection is a localized *feature* circuit, as Mishra's mid-encoder level-shift SAE features
suggest, we ran counterfactual interchange feature-ablation on Chronos-T5-Large: minimal pairs (a level shift at
tau versus a flat series at the old level, sharing the same noise realization), an 8192-feature TopK SAE on the
mid-encoder enc-MLP layer (L14, chosen by a non-gating orientation scan, reconstruction loss 0.062), the SAE error
term carried as a node, and features substituted with their value on the matched counterfactual run in both
directions, noising (clean to corrupt, necessity) and denoising (corrupt to clean, sufficiency). The decisive
result is the sufficiency test: injecting the top change-detection features into a no-shift run induced essentially
no level change (induced 0.008 versus a corrupt baseline of 0.008, maximum gain over all set sizes -0.0003 against
a 0.15 threshold), so the features cannot cause the behavior. Completeness agrees: ablating the top-k features
collapsed changepoint recovery by at most 0.012 (about 1.3% of the clean 0.896) even at k=128, barely above a
random-feature null (0.003), with periodicity untouched (motif drop approximately 0). Faithfulness, selectivity,
and completeness do all nominally pass at small set size (a selective set that beats the completeness null at k=1),
but this is a degenerate artifact, not a circuit, and crucially the set is *not* sufficient: keeping any single
feature preserves about 0.81 recovery, the top-k features are no better than the random-k null (both about 0.81
across k=1 to 128), and ablating essentially all of L14's features costs only about 0.08 recovery, so L14 is
largely bypassable for change-detection. This is precisely why the verdict requires faithfulness and completeness
and selectivity and sufficiency jointly, and why sufficiency correctly vetoes: the result is distributed at the
feature level, not a small sufficient causal circuit. The SNR sweep confirms the shape is not a saturation
artifact: the faithfulness fraction stays flat to slightly above one as the shift magnitude drops toward the noise
floor (about 0.92 at delta [1.5,3.0], rising to about 1.02 at [0.3,0.45], clean recovery 0.91 to 0.73), so
localization does not sharpen where detection is hard. With a minimum detectable effect of 0.094, a localized
feature set of the relevant size would have been resolved, so this is a de-confounded null, not insufficient power,
and the clean-model metric is well-posed (changepoint recovery monotone in shift magnitude, Spearman rho 0.94).

This closes the loop on the SAE-versus-circuit question on Mishra's own model and at his own granularity. Across
every level we have tested, attention heads (periodicity in 3b, change-detection in 4), MLP layers (the prior Phase
5), and now mid-encoder MLP features here, change-detection in Chronos-T5-Large is distributed; Mishra's mid-encoder
level-shift SAE features do not form a small sufficient causal circuit. The SAE-feature-versus-causal-circuit
correspondence is therefore a positive methodological finding rather than an open gap: a feature that an SAE
surfaces as a localized, interpretable change-detector need not be a sufficient cause of the behavior. One caveat
bounds the claim, and we state it plainly: this traced a single layer, L14, the orientation-best, but L14's own
single-layer necessity is only 0.021 (the strong 0.506 enc-MLP necessity reported earlier was the entire 24-layer
site), so a cross-layer feature circuit, sparse features spread across roughly L9 to L17 and union-ablated, is the
one granularity still formally untested. We frame that as the remaining future-work tightening, not as closed (it
is built as Phase 5 v3). Selectivity throughout is against the motif control only (the unstable trend metric is
dropped), and every headline effect carries a bootstrap confidence interval.

## Figure 6 caption

**Figure 6. Counterfactual feature-level localization of change-detection in Chronos-T5-Large (enc-MLP L14,
8192-feature SAE).** (6a) Faithfulness versus feature-set size (keep top-k, ablate the complement to the
counterfactual): flat at about 0.81 to 0.84 across k=1 to 128 and identical to the random-k null, so the top-k
features are no better than random and L14 is largely bypassable. (6b) Completeness (ablate the top-k to the
counterfactual) versus the motif control and a random-feature null, with bootstrap CIs: changepoint collapse is at
most 0.012, about 1.3% of clean, near the random null, with periodicity untouched. (6c) SNR sweep: the
faithfulness fraction stays flat to slightly above one as the shift magnitude approaches the noise floor, so
localization does not sharpen where detection is hard. (6d) Denoising and sufficiency: injecting the change-features
into no-shift runs induces no level change (induced approximately equal to the corrupt baseline, gain approximately
zero), the decisive negative, so the features cannot steer the behavior.
