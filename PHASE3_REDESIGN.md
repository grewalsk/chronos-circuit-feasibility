# Phase 3 Redesign — Causal Validation, Properly Structured

**Status:** ✅ IMPLEMENTED (2026-06-15) and verified end-to-end in `mock_cpu` (the actual `.ipynb` executes
with zero errors). Decisions taken: stimuli = **motif + noisy-sine**; ETT = **kept in Phase 4**; resample
cross-check = **included (confirmatory only)**; mechanism decomposition = **included now**. A second 4-agent
adversarial review hardened it further (see "Review refinements" at the bottom).
**Context:** the `pilot_t4` run (chronos-t5-base) gave Phase 1 GREEN (77 distributed lag-trackers; `dec_self L8H1` T=0.991 but *does not copy*), Phase 2 split (trackers ≠ copiers), Phase 3 FAIL (ablating the cross copiers → periodic ΔCRPS ≈ 0, CI includes 0; causal mass localizes to `enc_self`). The interim "multi-target + any-passes" patch was a mistake (see C1). This doc fixes the methodology before we re-run.

---

## TL;DR

Selective causal ablation is the correct pillar and most of the harness is sound. But two things must change before the Phase-3 verdict is defensible:

1. **The gate must be confirmatory, not a fishing expedition.** Test the *independently identified* H1 candidate (lag-tracking ∩ copying). The EAP-selected / "any-of-4-targets-passes" logic is selection bias + multiple comparisons and would manufacture a green.
2. **Output-ablation ΔCRPS cannot, by itself, distinguish "epiphenomenal" from "real-but-redundant"** in a model we already know is redundant. We need a redundancy-aware ladder, a mechanism-targeted metric (period-P spectral peak, not just global CRPS), and induction-diagnostic stimuli.

The redesign keeps the spec's standing controls and outcome tree; it makes the causal test actually able to separate the three outcomes (A = causal selective head / B = no circuit, distributed / split).

---

## ✅ What's GOOD (keep as-is)

- **K1. Mean-ablation, never zero** — correct per spec/IOI. Keep. (Add resample-ablation as a *cross-check*, not a replacement; see C5.)
- **K2. Selective contrast across conditions** — periodic vs trend-only vs changepoint-only is exactly the right selectivity test and the headline framing. Keep.
- **K3. Bootstrap CIs + a pre-registered selectivity ratio** — sound statistics for the per-condition deltas. Keep (but apply multiple-comparison discipline, C1).
- **K4. Period-altered corrupt for patching/EAP** — correct: it genuinely breaks the lag-P structure, unlike phase-scramble (which preserves the lag-P autocorrelation by Wiener–Khinchin). Keep.
- **K5. EAP (NLL proxy) → exact path-patch verification of top edges + low-rank false-negative probe + ACDC scoped to the EAP region** — this is good *localization* machinery and the AtP*-style verification is correct. Keep, but demote to exploratory (C1).
- **K6. Encoder-self vs decoder-cross localization** — the right question ("where does the causal mass live") and the pilot already gave a real answer (encoder). Keep as exploratory.
- **K7. Common-random-numbers paired clean/ablated CRPS** — variance reduction done right. Keep.
- **K8. The grouped/redundancy instinct** (the interim "trackers_group" target) — directionally correct; it just needs to become a principled *ladder* (C4) rather than one arbitrary group.
- **K9. The CRPS-floor diagnostic** (printing clean periodic CRPS) — keep; it makes the floor problem visible. The *fix* for the floor should change, though (C3).

---

## 🔧 What's CHANGEABLE (fix), by severity

### C1 — [BLOCKER] Confirmatory vs exploratory separation; kill "any-target-passes"
**Problem.** The interim patch selects heads by their attribution to the periodic NLL (EAP, clean=P vs corrupt=P′) and then ablates those same heads and "finds" they degrade periodic forecasting. The periodic axis is shared between selection and validation → the positive result is partly baked in (the mock `ratio=27.68 PASS` is this circularity, not a finding). Testing 4 targets and passing if *any* clears a 95% CI also inflates the false-positive rate. This violates the spec's "do not tune to manufacture a green."
**Fix.**
- **One pre-registered confirmatory target = the Phase-1 ∩ Phase-2 candidate** (a head that *both* tracks lag *and* copies). In the pilot the strongest tracker (`dec_self L8H1`) does **not** copy, so the H1 candidate is whichever head does both — likely a `cross` head from Phase 2's `selective_periodic_induction` set that *also* has high Phase-1 T. Define the H1 candidate set as `argmax over heads passing both signatures`.
- **Everything else (EAP-causal heads, groups, layers) is exploratory** — reported, never gating.
- **Verdict (C6)** keys only on the confirmatory target. Multiple exploratory targets are fine *as exploration* but get a clear "(exploratory)" label and no influence on PASS/FAIL.

### C2 — [MAJOR] Redundancy makes single/small output-ablation near-uninformative
**Problem.** Chronos is known-redundant (2510.03358). Self-repair / backup heads (Hydra effect, McGrath 2023; backup name-movers, IOI) mean **ΔCRPS ≈ 0 is expected even for a genuinely participating head.** So the pilot's null is *ambiguous*, not negative — and better target-picking doesn't fix it, because compensation happens downstream of the knockout.
**Fix.** Add the **redundancy ladder** (C4) so the experiment can say "the computation is in attention but distributed across N heads" rather than silently reading redundancy as absence.

### C3 — [MAJOR] CRPS is the wrong readout for *this* hypothesis; add a period-structure metric
**Problem.** Global CRPS asks "did the forecast get worse." H1 predicts a *specific* failure: loss of **period-P structure / phase alignment**. Under redundancy, CRPS can stay flat while the mechanism is clearly perturbed.
**Fix.** Add a **spectral/structure metric**: power (or autocorrelation) at the known period P of the *forecast*, and report **Δ(spectral-peak-at-P)** under ablation alongside ΔCRPS. A head that, when ablated, flattens the P-peak even when CRPS barely moves is strong mechanistic evidence. Keep ΔCRPS as the headline number; the spectral metric is the sharp probe and is far more robust to redundancy and to the CRPS floor.

### C4 — [MAJOR] Replace the ad-hoc group with a principled redundancy ladder
**Problem.** One arbitrary "top-6 trackers" group can't characterize a distributed mechanism.
**Fix.** Ablate a **monotone ladder** and report where the periodic spectral peak (C3) and CRPS finally collapse:
`single H1 candidate → group(all lag-tracking heads) → all dec_self → all cross → all decoder attention → all attention`.
This *is* the split/distributed result and directly adjudicates attention-degeneration (the paper's second contribution): if the peak only collapses at "all attention," periodicity is in attention but fully distributed; if it never collapses, it's MLP-mediated.

### C5 — [MINOR] Cross-check ablation type
**Problem.** Mean-ablation can leave signal in the preserved mean and pushes activations off-distribution.
**Fix.** Add **resample (noising) ablation** — replace the head's activation with its value on a *period-altered* input — as a cross-check on the confirmatory target. If mean- and resample-ablation disagree, that's itself diagnostic. Low priority; do it for the confirmatory target only.

### C6 — [MAJOR] Honest verdict logic
**Problem.** "GO if any target selective" rewards fishing.
**Fix.**
- **GO** ⇔ the *confirmatory* H1 candidate is causally selective (periodic ΔCRPS CI excludes 0 **and** selective vs trend/changepoint **and** selective collapse of the P-spectral-peak), reproduced across seeds.
- **Otherwise the pilot's distributed picture stands → the paper is the split / degeneration-adjudication result** (outcome B/split), with the ladder (C4) showing *how* distributed and the localization (K6) showing *where*. Explicitly: a non-confirmatory result is **not** a failure of the project — it's outcome B/split, which the spec calls the most probable and arguably most informative.

### C7 — [MAJOR] Stimuli are too easy in the wrong way
**Problem.** Gaussian `obs_noise` lifts the floor but mostly inflates *irreducible* noise (estimator variance), not the difficulty of the *periodic structure*. A clean sine is forecastable by many mechanisms, so nailing it is uninformative about *which*.
**Fix.** Add **repeated-motif stimuli** (sharp, non-sinusoidal features; we already use one in the Phase-0 plumbing test) where the *only* way to get the fine structure right is to **copy the exact value one period back** — making the task itself induction-diagnostic. Keep a noisy-sine condition for comparability; make motif the primary causal-test stimulus. Optionally pull a small **real-data (ETT) spot-check** forward from Phase 4, since synthetic saturation is exactly where the floor bites.

### C8 — [STRETCH] Mechanism decomposition: attention-pattern vs OV
**Problem.** Output-ablation conflates "selects the wrong lag" (QK) with "copies the wrong value" (OV) — but the spec's staged structure (estimate → aggregate → **select** → copy) wants them separated.
**Fix.** On the confirmatory candidate only: **attention-pattern patching** (freeze the head's pattern to the period-altered one → does it lose the right lag?) vs **OV/value patching** (→ does it copy the wrong bin?). This is the most mechanistically informative experiment but the most custom; flag as stretch for this pass.

---

## Redesigned Phase 3 (target structure)

1. **Confirmatory gate (pre-registered).** H1 candidate = head(s) that *both* track lag (Phase 1, high T, slope≈integer×P) *and* copy (Phase 2). Test for: (a) selective ΔCRPS (periodic degrades, trend/changepoint do not), and (b) selective collapse of the period-P spectral peak. Seed-replicated. **This alone decides GO.**
2. **Redundancy ladder (exploratory).** single → lag-tracker group → all-dec_self → all-cross → all-attention; report ΔCRPS *and* Δ(P-peak) at each rung. Locates how distributed the mechanism is.
3. **Mechanism decomposition (exploratory/stretch).** attention-pattern vs OV patching on the confirmatory candidate.
4. **EAP / path-patch / ACDC (exploratory).** Localization only ("where"), never a gate.
5. **Verdict.** GO ⇔ confirmatory candidate is causally selective; else the distributed/split result stands (still a strong paper).

---

## Implementation sketch (what actually changes in the notebook)

- **CONFIG:** add `confirmatory: "phase1_and_phase2"` (how the H1 candidate is chosen); `ablation_ladder: [...]`; `stimulus_mode: "motif" | "sine" | "both"`; keep `obs_noise`; optional `resample_ablation: true`; optional `ett_spotcheck: false`.
- **Stimuli (Sec 4):** add `make_motif_periodic` (repeated sharp motif). Selectivity conditions unchanged.
- **Metric:** add `spectral_peak(forecast_samples, P)` and report Δ under ablation next to ΔCRPS everywhere a CRPS number appears.
- **Phase 3 (Sec 7):** replace the "4 targets + any-passes" block with: (i) confirmatory target derived from `PHASE1 ∩ PHASE2`; (ii) the ladder as a labeled exploratory sweep; (iii) keep EAP/ACDC but tag exploratory; (iv) verdict from confirmatory only.
- **Report (Sec 8):** GO/NO-GO/PIVOT keyed to the confirmatory candidate; print the ladder collapse point and the dominant locus as the "what we learned" line.
- **Honesty:** if no head satisfies Phase-1 ∩ Phase-2 (possible — pilot's best tracker doesn't copy), say so explicitly and route to the distributed-result framing rather than forcing a candidate.

---

## Cost / runtime (free T4, base)

- Confirmatory target: ~1 small head-set × 3 conditions × ~32 series × 2 generates ≈ unchanged from one current target (~5–8 min).
- Ladder: ~5 rungs × 3 conditions × (fewer series, e.g. 16) — the big rungs ("all attention") are *cheaper* per-series to interpret, but add the spectral metric. Budget ~15–25 min total. Caps stay logged.
- Spectral metric is ~free (FFT on the sampled forecasts we already draw).
- Net: comparable to the current Phase 3; the multi-target combinatorial blowup goes away because only the confirmatory target gates.

---

## Decisions I need from you (before implementing)

1. **Stimuli:** motif-only, or motif + noisy-sine for comparability? (Recommend: both, motif primary.)
2. **ETT spot-check:** pull a tiny real-data check forward into this pass, or keep it Phase-4? (Recommend: keep Phase-4 unless the motif stimuli still saturate.)
3. **Resample ablation (C5):** include as a cross-check now, or defer? (Recommend: include, confirmatory target only — cheap.)
4. **Mechanism decomposition (C8):** in this pass or next? (Recommend: next pass — it's the most custom and the confirmatory gate + ladder already answer the GO/NO-GO question.)

---

## What this does NOT change (guardrails)

Original Chronos-T5 only; HF-hook backend; mean-ablation as the primary op; per-head scalars (no raw-attention caching); the selective/multi-lag signature and fixed-offset null in Phase 1; checkpointing/re-runnability; mock-vs-pilot labeling; honest negatives. Scope stays Phases 0–3.

---

## Review refinements (second adversarial pass, applied)

The redesign passed an independent 4-agent review (the core anti-circularity logic was confirmed sound — the
gated candidate is selected from Phase-1∩Phase-2 independently of the causal metric; the mechanism-decomposition
tensor algebra was verified, `recon_clean ≈ 0`). These additional fixes were then applied:

- **Period-structure metric → power fraction.** Switched from normalized acf@P (amplitude-blind: halving the
  periodic amplitude leaves acf≈1) to the **fraction of forecast power in the period-P FFT band** — sensitive
  to both shape and amplitude loss, bounded [0,1], comparable across `obs_noise`, and well-defined for a
  flattened forecast (where acf-of-near-zero was an arbitrary artifact).
- **Collapse criterion → relative + CI.** A "collapse" now requires the structure drop to be **≥30% of the
  clean structure AND ≥0.02 absolute AND the bootstrap CI to exclude 0** — consistent in the confirmatory gate
  and the ladder, and not fooled by ~1e-7 floating-point drops or by `obs_noise` scaling.
- **Ladder → genuinely nested.** Rungs are now strict supersets `H1 ⊆ +lag_trackers ⊆ +dec_self ⊆ +cross ⊆
  all_attention` (nesting asserted in code), so the "first collapse" point is interpretable. `all_attention`
  is flagged the **trivial ceiling** and excluded from collapse localization.
- **Honest selectivity reporting.** Kept the spec's selectivity rule (non-periodic CI includes 0 **or** the
  periodic effect is ≥`selectivity_ratio_min`× larger) but the prose no longer overclaims "must stay flat";
  a pass via the ratio clause while a non-periodic condition is significant is flagged as *relative*
  selectivity. Periodic-sine is labelled a **secondary** comparability readout; only motif gates.
- **Robustness.** Per-stage Phase-3 checkpointing (`confirm_mean`, `confirm_resample`, each ladder rung) so a
  Colab disconnect resumes mid-phase; a **stale-schema guard** on checkpoints (old-schema JSON is a cache miss,
  not a crash); the plumbing positive-control sweep is capped (16 heads, not all 144) on the pilot.
- **Smaller fixes.** Mechanism head chosen by max Phase-2 copy_mass (principled, not row order); the dead
  `cand_rows` parameter removed; resample relabelled "corrupt-distribution mean-ablation" (it is not a
  position-preserving resample); H1-set size surfaced as a caveat on ΔCRPS magnitude; explicit banner when no
  head satisfies both signatures.

**Known residual (acceptable for feasibility):** the confirmatory gate ablates the *whole* H1 set jointly, so
ΔCRPS magnitude scales with set size — surfaced as a caveat, not hidden. A stronger future check would add a
single-best-head confirmatory result alongside the set result.
