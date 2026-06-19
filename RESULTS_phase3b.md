# Phase 3b — Results (site-isolated causal localization)

*Draft results section for the Chronos selective periodic-induction study. Numbers verified against
[`phase3b_pilot_t4_full.json`](phase3b_pilot_t4_full.json) (chronos-t5-base, full profile: 3 seeds, 32 series,
5-point fraction sweep). Figure: [`fig4b_site_isolated.png`](fig4b_site_isolated.png).*

## Results

To localize the periodicity computation, we ablated each attention site in isolation (all 144 encoder-self,
decoder-self, or decoder-cross heads of Chronos-T5-base) and measured the structural collapse of period-P
power, rather than global CRPS, against a size-matched random-head null (motif p95 = 0.22). All three sites
produced significant period-P collapse exceeding the null: encoder-self 0.55 (95% CI 0.34 to 0.76),
decoder-cross 0.45 (0.28 to 0.63), and decoder-self 0.26 (0.13 to 0.38). None, however, was structurally
selective. Ablating encoder-self collapsed changepoint-level recovery (0.73) more than periodicity (0.55), a
change-detection signature, and decoder-cross degraded periodic and non-periodic structure about equally (0.46
versus 0.45). Because ablating all cross heads severs the encoder-to-decoder channel, we resolved the cross
case with a dose-response sweep over a random ablated fraction f of each site. Periodic collapse was an
endpoint effect, near zero at low fractions for every site (encoder-self at f = 0.25: -0.05) and rising only as
f approached 1, whereas non-periodic structure collapsed earlier and harder (encoder-self at f = 0.25: +0.49;
decoder-cross at f = 0.50: +0.29 versus periodic +0.04), and no site showed a selective periodic effect at low
f. Large effects were detected (encoder-self 0.55, far above the 0.22 null) but no selective one, indicating a
de-confounded null rather than insufficient power. We therefore find no localized attention circuit for
periodicity in Chronos-T5; the lag-tracking structure identified in Phase 1 is distributed across attention.
This corroborates Mishra (2603.10071) and Pandey (2511.15324) at the circuit level, with change-detection
dominating and periodicity not localizable, while nuancing them: the periodic structure their probes and SAEs
could not localize is nonetheless present in attention, only diffusely. MLP and feature-level circuits were not
traced and remain future work.

## Figure 4 caption

**Figure 4. Site-isolated causal localization of periodicity in Chronos-T5-base.** (4b) Per-site structural
collapse under full-site mean-ablation: every site collapses period-P power significantly above the
size-matched random-head null (dashed line), yet none does so selectively, with encoder-self collapsing
changepoint recovery (0.73) more than periodicity (0.55), a change-detection signature. (4c) Dose-response
sweep ablating a random fraction f of each site shows periodic collapse to be an endpoint effect (near zero
until f approaches 1) while non-periodic structure collapses at low-to-mid f, so no site localizes periodicity
at a non-severing fraction; the sweep is single-seed, so the endpoint, non-selective shape is robust but exact
low-f values are noisy.
