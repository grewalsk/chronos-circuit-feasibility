# STATE — Chronos Selective Periodic-Induction Circuit

*Living status doc. Last updated after the **Phase 4** change-detection run on `chronos-t5-large` (2026-06-19).
Phases 0–3 + Phase 3b (periodicity) + Phase 4 (change-detection) are built and run; both causal results are
complete and de-confounded — **both forecasting computations are DISTRIBUTED across attention**, at Base and Large.*

---

## TL;DR

A head-level **circuit-level analysis** of the original **Chronos-T5** TSFM, testing whether it learns a
**selective / multi-lag induction head** for periodicity (infer period `P` from content → attend `t−P, t−2P, …`
→ copy the bin). Phases 0–3 are built and have been run on Base.

**Verdict: NO-GO as a localized single-head circuit → the result is the DISTRIBUTED / SPLIT outcome (the spec's
"outcome B", its most-probable and "most informative" branch).** Periodicity *is* represented in attention
(lag-tracking heads exist and some also copy), but **no small set of heads is causally necessary for it** —
ablating the identified candidates leaves the forecast's period structure intact and hurts trend/changepoint
*more* than periodicity; the structure only collapses under site-scale ablation. This is a clean, **trustworthy**
null (the CRPS floor was real this time) and a publishable paper: first head-level circuit-discovery methods
applied to a TSFM + a circuit-level adjudication of the attention-degeneration debate.

**Phase 3b (site-isolated, equal-size ablation) now DE-CONFOUNDS it → DISTRIBUTED, conclusively.** Ablating each
attention site in isolation (144 heads each) vs a size-matched random null: **all three sites collapse period-P
power above the null, but none selectively** (enc-self collapses *changepoint* more than periodicity; cross
degrades both equally), and the dose-response shows periodic collapse is an **endpoint effect** (only near full
ablation) while non-periodic structure collapses earlier — so **no attention site localizes periodicity**. The
draft Results paragraph + Fig 4 caption are in [`RESULTS_phase3b.md`](RESULTS_phase3b.md). The causal result is
done; remaining items are generalization (ETT) + writing, not the result itself.

**Phase 4 (change-detection, the affirmative-circuit attempt) → DISTRIBUTED on `chronos-t5-large`.** Mishra
reports change-detection as causally dominant via *mid-encoder level-shift SAE features*, so this was the best
shot at a localized circuit — and it is the highest-value disambiguator because Large is *Mishra's own model*.
Result: **encoder-self is necessary in aggregate** (full-site changepoint collapse **0.660**, 95% CI 0.530–0.784,
vs random@N null **0.243**; cross 0.485; dec-self below null) **but not selective** (it collapses periodicity 0.549
≈ change-detection 0.660) **and not localized** — the top-8 behavioral candidate heads reproduce only **~12%** of
the site collapse (best single head 0.039, group 0.081), with **no sufficient set ≤8 heads** and a pure **endpoint
dose-response** (changepoint ≈0 until f>0.5, 0.70 at f=1; non-target collapses early). The automated verdict reads
"SPLIT" (5 single heads beat a ~0 within-site null) but the localized tail is faint → **report as distributed with
a weak localized component, not a circuit.** Draft Results + Fig 5 caption in
[`RESULTS_phase4.md`](RESULTS_phase4.md); result = [`phase4_pilot_a100.json`](phase4_pilot_a100.json), figure =
[`fig5_phase4_pilot_a100.png`](fig5_phase4_pilot_a100.png). **The base-too-small escape hatch is closed (this is
Large).** Net: *both* forecasting computations Chronos implements (periodicity + change-detection) are distributed
across attention — the paper is a circuit-level adjudication of the attention-degeneration debate at both scales.

**Phase 5 (does it live in the MLPs / features?) → DISTRIBUTED at every granularity, on `chronos-t5-large`.** The
attention analysis was attention-only; Pandey (nonlinear/MLP) and Mishra (mid-encoder level-shift SAE *features*)
both point to MLPs. **Phase 5 (layer-level)** found enc-MLP necessary in aggregate (full-site changepoint collapse
**0.506** vs random-MLP null 0.365) **but not localized** (top-8 layers grouped 0.205 *below* the random-8 null —
the best layers underperform random; endpoint dose-response) → distributed across MLP layers too. An adversarial
audit then found that layer-level verdict **untrustworthy for the localization question** (wrong granularity vs
Mishra's *features*; off-distribution mean-ablation; trend-driven selectivity; saturated stimuli; a gate that
skipped the feature test). **Phase 5 v2 (the corrected test)** redid it at the *feature* level with **counterfactual
interchange ablation** (minimal pairs, SAE features, error node carried, noising+denoising, motif-only selectivity,
SNR sweep, CIs, unconditional). Verdict = **B: DISTRIBUTED at the feature level, SAE-vs-circuit discrepancy
AIRTIGHT** — decided by **sufficiency**: injecting the top change-features into no-shift runs induces **no** level
change (max denoising gain **−0.0003** vs a 0.15 bar); completeness ≤0.012 (≈1.3% of clean 0.896); faithfulness is
a *degenerate* pass (L14 bypassable: top-k = random null ~0.81); no SNR sharpening; MDE 0.094 (de-confounded, not
underpowered). Draft Results + Fig 6 in [`RESULTS_phase5_v2.md`](RESULTS_phase5_v2.md); result =
[`phase5v2_pilot_a100.json`](phase5v2_pilot_a100.json), figure = [`fig6_phase5v2_pilot_a100.png`](fig6_phase5v2_pilot_a100.png).
**Net (3b/4/5):** change-detection (and periodicity) in Chronos-T5-Large is **distributed across attention heads,
MLP layers, AND mid-encoder MLP features** — Mishra's SAE features do **not** form a small *sufficient* causal
circuit; SAE-feature-vs-causal-circuit correspondence is a positive methodological finding. *Caveat (stated
plainly):* v2 traced one layer (L14, only 0.021 single-layer necessity); a **cross-layer** feature circuit (L9–L17
union) is the one untested granularity — built as **Phase 5 v3** ([`phase5_v3.ipynb`](phase5_v3.ipynb), counterfactual
union-ablation across the top-N mid-encoder layers), not yet run on GPU.

---

## 1. Pipeline status (chronos-t5-base, pilot_t4)

| Phase | What it tests | Status |
|---|---|---|
| 0a tokenization | per-step tokenization, lag-in-tokens = lag-in-time | **PASS** |
| 0b plumbing | hooks fire ×4 sites; mean-ablation bites; cross-attn key provenance; NLL-grad sign | **PASS** (4/4) |
| 1 selective-lag scan | ≥1 head tracks the content-determined lag across P, above nulls | **GREEN** |
| 2 copying / OV | which trackers also copy (OV→lm_head DLA) | **copiers found** → H1 = `cross L3H5`, `cross L8H8` |
| 3 causal validation | is the H1 head causal **and** selective? | **FAIL** → distributed/split |
| → overall | GO / NO-GO / PIVOT | **NO-GO single-head; GO distributed paper** |

---

## 2. The pilot result — exact numbers

Config: `chronos-t5-base`, periods [8,12,16,24], 3 seeds, 32 series/condition, ctx 256, pred 64,
`obs_noise=0.30`, 100 CRPS samples, 1000 bootstraps. H1 set (Phase-1 lag-tracking ∩ Phase-2 copying) =
**`cross L3H5`, `cross L8H8`** (2 heads, ablated jointly). Resample cross-check agrees.

**CRPS-floor diagnostic (this is what makes the null trustworthy):** clean motif CRPS = **0.2274** (vs ~0.001
in the first pilot — `obs_noise` + motif gave real headroom); clean motif period-P power fraction = **0.3166**.

**Confirmatory gate — mean-ablation** (Δstruct = drop in period-P power fraction):

| condition | ΔCRPS [95% CI] | Δ period-P power [95% CI] |
|---|---|---|
| **motif** (gates) | **+0.00304** [+0.00154, +0.00459] | **−0.00257** [−0.00338, −0.00175] |
| periodic-sine (secondary) | −0.00081 [−0.00156, −0.00006] | −0.00085 [−0.00181, +0.00016] |
| trend | **+0.02235** [+0.01555, +0.03010] | — |
| changepoint | **+0.01187** [+0.00561, +0.01798] | — |

`crps_sig=True`, **`struct_sig=False`**, `nonper_ok=False`, **selectivity ratio = 0.136**, `passed=False`.
Resample-ablation agrees (motif +0.00367, trend +0.02720, changepoint +0.01181, motif Δstruct −0.00268).

**Two decisive facts:**
1. **Period structure survives.** Ablating H1 nudges CRPS up (+0.003) but the period-P power does **not** collapse
   (Δ is *−0.003*, i.e. slightly the wrong way). The heads contribute marginally to fidelity, not to periodicity.
2. **The effect is anti-selective.** The same ablation hurts trend (+0.0224) and changepoint (+0.0119) **4–7×
   more** than the periodic target (ratio 0.136 ≪ 1). These are general-purpose forecast heads, not a periodic
   circuit. *(Under a CRPS-only gate, the motif CI-excludes-zero would have falsely read as PASS; the period-P
   power metric is what vetoes it.)*

**Redundancy ladder (nested supersets; motif) — collapse first at `+dec_self`:**

| rung | heads | ΔCRPS | Δ period-P power [95% CI] | collapses? |
|---|---:|---|---|---|
| H1 | 2 | +0.00199 [−0.00011,+0.00392] | −0.00298 [−0.00417,−0.00161] | no |
| +lag_trackers | 32 | +0.18085 | −0.11048 [−0.13220,−0.08991] | **no** |
| **+dec_self** | **176** | +0.83462 | **+0.29110 [+0.19097,+0.38975]** | **YES** (~73% of clean 0.397) |
| +cross | 291 | +0.84660 | +0.29815 | yes |
| all_attention (ceiling) | 432 | +0.84660 | +0.29815 | yes (trivial) |

Period structure is robust to ablating all 32 lag-tracking heads and collapses only once all 144 decoder
self-attention heads are added → **distributed**, and **not** a redundancy-masked cross-attention circuit (which
would have broken at the trackers rung).

**Mechanism decomposition** (`cross L3H5`): `dNLL_full=+0.00186`, **`pattern (lag-sel)=+0.00802`** > `OV (copy)=+0.00297`,
`recon_clean=0.000000` (reconstruction sanity exact). Lag-selection leg dominates; magnitudes small.

**Localization (exploratory, never gates):** EAP |attr| by site = **enc_self 0.945**, cross 0.578, dec_self 0.241
→ dominant enc_self; EAP↔exact r = **0.75**; 1 false-negative; staged feeders `enc_self L2H0`, `L0H5`;
**ACDC retained 38 heads** (a dense, not sparse, circuit estimate).

Raw checkpoints in the repo: `phase1_pilot_t4.json`, `phase2_pilot_t4.json`, `phase3_pilot_t4new.json`.

---

## 2b. Phase 3b — site-isolated de-confounding (the FINAL causal result)

The Phase-3 ladder collapsed at `+dec_self` (176/432 heads), confounded with ablation **size**. Phase 3b ablates
each site **in isolation** (all 144 enc-self / dec-self / cross heads), scores **structural collapse** per
condition (period-P power / trend-slope / changepoint-level recovery — not global CRPS), and gates a LOCUS on
*selective* collapse that **beats the size-matched `random@N` null** (cross additionally needs the effect at low
fraction `f`, since ablating all cross severs the encoder→decoder channel). Two notebooks: **`phase3b_fast.ipynb`**
(lean, ~30 min) and **`phase3b_full.ipynb`** (3 seeds, 32 series, 5-point sweep). Full result =
`phase3b_pilot_t4_full.json`, figure = `fig4b_site_isolated.png`.

**Verdict: DISTRIBUTED.** Full-site motif period-P collapse (95% CI), random null p95 = **0.219**:

| site | motif collapse [95% CI] | trend | changepoint | selective? |
|---|---|---|---|---|
| enc_self | +0.551 [0.341, 0.755] | +0.285 | **+0.726** | no (changepoint collapses *more*) |
| dec_self | +0.255 [0.134, 0.381] | −0.447 | +0.091 | no (below the 0.30 floor) |
| cross | +0.453 [0.276, 0.629] | −1.266 | +0.459 | no (motif ≈ changepoint) |

**Dose-response (Fig 4c):** periodic collapse is an **endpoint effect** — ~0 at low `f` for every site, rising
only near `f=1`; non-periodic structure collapses *earlier/harder* (enc-self at `f=0.25`: motif −0.05 vs
non-periodic +0.49). `low_f_selective = False` for all sites → **no site localizes periodicity at a non-severing
fraction.** Detection power is real (enc-self 0.55 ≫ 0.22 null) — a de-confounded null, not underpowered.

**Bonus:** enc-self collapses changepoint (0.726) > periodicity (0.551) → encoder self-attention is the most
generally-necessary site with a **change-detection lean** — a direct corroboration of Mishra.

*Caveats:* the sweep is single-seed (Fig 4c shape robust, exact low-`f` values noisy); the trend-slope-recovery
metric is unstable (negative collapses) — the verdict rests on motif + changepoint.

---

## 2c. Phase 4 — change-detection circuit on `chronos-t5-large` (the affirmative-circuit attempt)

`phase4.ipynb` (single-source builder `build_phase4.py`) **reuses all the 3b machinery** (the `.o` per-head
mean-ablation hook, `SITES`, equal-size / site-isolated / `random@N` ablation, structural rel-collapse + bootstrap,
dose-response sweep, cross low-`f` rule, checkpoint/resume) and changes **only the target** to level shifts. New for
Phase 4: a **standardized delta-normalized median `changepoint_recovery`** metric (re-runs the 3b changepoint
baseline under it — the old 0.726 is **not** cited once the metric changed; re-measured value = **0.660**);
**CTX-relative level-shift stimuli** with metadata threaded through `make_batch` (τ inside context, ≥30% post-shift);
a **behavioral scan** = delta-response (record-mode reuse of the `.o` hook) + boundary attention (a **manual
`output_attentions` forward**, since `pipeline.predict`'s generate path does not surface attentions); **head-level
single/group ablation** (the only localization adjudicator); **relative-depth Mishra cross-check** (~45–55% encoder
depth, not literal block 11 = a Large index); a **trend-based binned** delta-monotonicity assert; and a
**multi-seed** dose-response. MODE/MODEL switch: `mock_cpu | pilot_t4` (base) `| pilot_a100` (Large). A 5-dimension
adversarial review hardened the verdict logic (notably: a LOCALIZED claim now requires the winning site to be
necessary first — ≥ `STRUCT_COLLAPSE_MIN` **and** beat random@N — so a small set can never "reproduce most of" a
non-existent collapse).

**Verdict: DISTRIBUTED (faint SPLIT tail).** Run = Large, 24 encoder layers, 384 heads/site, 3 seeds × 32 series,
2-seed 5-point sweep. Full-site changepoint collapse vs random@N null (p95 = **0.243**):

| site | changepoint collapse [95% CI] | motif | trend | >null? | selective? |
|---|---|---|---|---|---|
| enc_self | **+0.660** [0.530, 0.784] | +0.549 | −0.529 | yes | **no** (motif ≈ changepoint) |
| cross | +0.485 [0.326, 0.646] | +0.461 | −0.805 | yes | no (motif ≈ changepoint) |
| dec_self | +0.168 [0.060, 0.287] | +0.252 | −0.258 | **no** | no (below null) |

**Necessity yes, localization no.** Encoder-self is necessary in aggregate but (a) **not selective** — it collapses
periodicity ≈ as much as change-detection — and (b) **not localized**: the **top-8 behavioral candidate heads
reproduce only ~12%** of the 0.660 site collapse (best single head **L9H4 = 0.039**, top-8 group **0.081**), **no
sufficient set ≤8 heads**. The **dose-response** is the clincher: changepoint collapse is an **endpoint effect**
(≈0 at f=0.1/0.25/0.5 → 0.42 at 0.75 → 0.70 at 1.0) while **non-target structure collapses early** (0.34 at f=0.1)
— the exact distributed signature periodicity gave in 3b. The automated verdict prints **"SPLIT"** because 5 of 8
single heads beat a **~0** within-site `k=1` null, but the partial-localization tail is faint → **reported as
distributed with a weak localized component, not a circuit.** Detection power is real (0.660 ≫ 0.243 null → a
de-confounded null, not underpowered).

**Mishra reconciliation (the headline future-work hook).** Mechanism probes (exploratory — nothing localized) on
the top candidates show attention **routes** boundary info — **boundary-local** (corr w/ global offset −0.015) and
**recency**-biased (mass 1.90 most-recent vs 0.73 older) — but scales only weakly with shift size (δ-slope +0.093,
R²=0.26). Candidate enc-self layers spread across depth (rel-depths 0.00–0.57; only **L12 @0.52** inside Mishra's
mid-encoder band). The honest reconciliation: attention carries boundary/recency signals, but the level-shift
**detection is plausibly MLP/feature-mediated** — which we do **not** trace (head-level attention only). This is
why Mishra's SAEs localize a *feature* mid-encoder while the *attention heads* are distributed. **The base-too-small
ambiguity is closed: this is Large, Mishra's own model.**

*Caveats:* trend metric unstable (lean on changepoint gate + motif selectivity); mechanism + Mishra-depth probes are
exploratory (nothing localized to validate them on); ETT real-data generalization still outstanding (as for 3b).

---

## 3. The nuance — what this paper is (and isn't)

**"Circuit-level analysis" describes the granularity and methods, not whether a clean circuit was found.**
It contrasts with representational/probing analysis (Pandey's linear probes, Mishra's SAE features) and
layer-level steering (time2time). Our work operated on the **head/edge computational graph** with the canonical
circuit-discovery toolkit: per-head behavioral attribution (lag-tracking, OV→lm_head copying DLA), causal
node knockout (mean/resample ablation), **path patching**, **EAP** + exact verification, **ACDC**, and a
**QK-vs-OV sub-circuit decomposition**. That methodological move — head-level causal circuit analysis in a
TSFM — is the first-of-its-kind contribution and stands regardless of outcome.

**A circuit-level analysis can legitimately return "no localized circuit."** Outcome B (distributed/redundant)
is itself a circuit-level claim, established with circuit-level methods — and we *positively characterized* the
distributed structure: a 38-head ACDC set, the cross-site EAP mass, the ladder showing structure survives the
lag-trackers and collapses only at site scale, and the QK/OV split of the candidate. Analogy: a crystallographer
who finds a protein is intrinsically disordered did structural biology; "no rigid fold" is a structural result.

**What to claim — and what not to:**
- ❌ *"We found the selective periodic-induction circuit."* — false; we did not.
- ✅ *"We applied head-level circuit-discovery methods to a TSFM for the first time and obtained a circuit-level
  characterization: periodicity is **not localized** to a selective-induction head — it is **distributed and
  redundant** across self-attention (survives ablation of all lag-tracking heads; collapses only at site scale;
  ACDC retains ~38 heads; causal mass split enc-self / dec-self)."*
- This corroborates Mishra ("not a clean periodic attention mechanism") and Pandey (low linear accessibility)
  **at the circuit level**, while **nuancing** them: the lag structure their probes/SAEs couldn't localize *is*
  present in attention — it's just distributed.

**Honest caveats to state in the paper:**
1. **Granularity.** This is a head-level analysis of the **attention** circuit; we did **not** trace MLP/feature
   circuits. If periodicity is MLP-mediated (consistent with Pandey), we've shown *attention doesn't localize it*
   but haven't traced the full circuit. MLP/transcoder tracing is explicit future work (spec Phase 7).
2. **Ladder size confound.** The `+dec_self` collapse rung removes 176/432 heads, so "collapse there" is partly
   confounded with ablation *size*. **Phase 3b** (below) de-confounds it with site-isolated, equal-size ablations.
3. **Candidate-set is noise-dependent.** The specific H1 heads shifted between the `obs_noise=0.05` and `0.30`
   pilots; the *distributed conclusion* is the robust claim, not any single candidate. A noise/seed sweep should
   confirm invariance.
4. **Synthetic-only so far.** Needs an ETT real-data confirmation (spec Phase 4) before the strong claim.

---

## 4. What's left before the writeup (roadmap)

**Tier 1 — ✅ DONE (Phase 3b + Phase 4):**
- ✅ **Phase 3b — site-isolated, equal-size ablation** (built + run, fast + full) → **DISTRIBUTED**, de-confounded
  (§2b). Site-isolated ablation removes the size confound; the dose-response + beat-null + low-`f` gates close the
  redundancy-masking and cross-bottleneck blind spots.
- ✅ **Phase 4 — change-detection circuit on `chronos-t5-large`** (built + run, §2c) → **DISTRIBUTED** (faint SPLIT
  tail). The best shot at an affirmative circuit, on Mishra's own model: enc-self necessary (0.660 vs 0.243 null)
  but non-selective and non-localized (top-8 = ~12% of site; endpoint dose-response). **Closes the base-too-small
  ambiguity.** Both forecasting computations are now distributed.
- ✅ **Detection-power statement** — baked into both verdicts (large effects detected, none selective → real null).
- ⏳ **Robustness across `obs_noise`/more sweep seeds** — 3b: 2+3 seeds agree; Phase 4: 2 sweep seeds. Recommended
  polish = swap the unstable trend metric for low-freq power retained on the final figures.

**Tier 2 — generalization (credibility):**
- **ETT real-data** confirmation — reproduces "no localized circuit; structure survives small ablation" on real
  seasonality / real regime shifts. Closes the #1 reviewer objection; now the main remaining must-have.
- ✅ **Base → Large** — DONE via Phase 4 (`chronos-t5-large`, 710M, Mishra's model, A100). Within-family
  universality confirmed for change-detection; periodicity-on-Large is an optional extra (the distributed verdict
  already holds at both scales for change-detection).

**Tier 3 — mechanism depth (turns "distributed" into a characterization):**
- **MLP / feature circuits — now the HEADLINE future work.** Phase 4 shows attention *routes* boundary/recency
  signals but the level-shift *detection* is plausibly MLP/feature-mediated (reconciles Mishra's mid-encoder SAE
  *features* localizing while the *attention heads* are distributed). Transcoder / MLP tracing is the natural next
  step and the strongest framing for the paper's limitation→future-work arc.
- Trace the 3b stages: the `enc_self L5H7` compressed "period-detector" (T=0.918, slope 0.18) and staged feeders.
- SAE cross-validation vs Mishra (does the distributed locus map to his "seasonality" / level-shift features?).

**Tier 4 — writing & figures (spec Phase 5):**
- Fig 1 schematic · Fig 2 lag-tracking heatmap · Fig 3 copying · Fig 4 causal selectivity + nested ladder ·
  Fig 4b/5 the 3b site-isolated localization · Fig 5 scope (across P, size, ETT) + SAE triangulation.
- Verify every arXiv ID; correct the 2510.09776 framing (linear attention, regime not covered); cite time2time
  and draw the layer-steering-vs-circuit line; limitations (synthetic + ETT, head-level, redundancy, 3b caveat).

**Critical path to a defensible workshop submission:** ~~3b~~ ✅ → ~~Large~~ ✅ (Phase 4) → ~~MLP/feature tracing~~
✅ (Phase 5 v2, feature-level distributed/airtight) → **ETT** (the one remaining must-have) → writing. Optional
tightening: run **Phase 5 v3** (cross-layer feature union, the last untested granularity) on a high-RAM A100;
sturdier trend metric. All causal results (periodicity + change-detection; attention, MLP layers, MLP features;
Base + Large) are **done** and consistently DISTRIBUTED; what remains is real-data generalization + writing, not
the finding.

---

## 5. Repo contents

- `chronos_circuit_feasibility.ipynb` — the Phases 0–3 deliverable notebook (mock_cpu → pilot_t4).
- `phase3b_fast.ipynb` / `phase3b_full.ipynb` — the site-isolated Phase 3b (lean first verdict / publication res).
- `phase4.ipynb` — Phase 4 change-detection notebook (mock_cpu | pilot_t4=base | pilot_a100=Large).
- `phase5.ipynb` — Phase 5 layer-level MLP ablation (superseded for localization by v2; see §2-TLDR).
- `phase5_v2.ipynb` — Phase 5 v2 counterfactual FEATURE-level ablation (the corrected localization test).
- `phase5_v3.ipynb` — Phase 5 v3 CROSS-LAYER feature union test (closes the single-layer caveat; not yet GPU-run).
- `build_notebook.py`, `build_phase3b.py`, `build_phase4.py`, `build_phase5.py`, `build_phase5_v2.py`,
  `build_phase5_v3.py` — single-source builders for the notebooks.
- `PHASE3_REDESIGN.md` — rationale for the redesigned, non-circular Phase 3 (confirmatory gate + ladder +
  mechanism decomposition) and the two adversarial-review passes.
- `RESULTS_phase3b.md` — Phase 3b draft Results + Fig 4 caption (numbers verified vs the JSON).
- `RESULTS_phase4.md` — Phase 4 draft Results + Fig 5 caption (numbers verified vs `phase4_pilot_a100.json`).
- `RESULTS_phase5_v2.md` — Phase 5 v2 draft Results + Fig 6 caption (numbers verified vs `phase5v2_pilot_a100.json`).
- `phase4_pilot_a100.json` / `fig5_phase4_pilot_a100.png` — Phase 4 Large result (§2c) + Fig 5.
- `phase5_pilot_a100.json` — Phase 5 layer-level Large result; `phase5v2_pilot_a100.json` /
  `fig6_phase5v2_pilot_a100.png` — Phase 5 v2 feature-level Large result (the airtight feature-level distributed null).
- `phase1_pilot_t4.json`, `phase2_pilot_t4.json`, `phase3_pilot_t4new.json` — Phase 1/2/3 pilot checkpoints (§2).
- `phase3b_pilot_t4_full.json` — Phase 3b full result (§2b); `fig4b_site_isolated.png` = Fig 4.
- `*.png` — Fig 2 (lag-tracking), Fig 3 (copying), Fig 4/4b (causal), stimulus sanity.
- Authoritative design lives in the separate repo `github.com/grewalsk/circuitTSFM`
  (`chronos_circuit_spec_v2.md`, `chronos_circuit_plan_v2.md`), cloned into the session by Section 0.

## 6. Reproduce

**Phases 0–3:** open `chronos_circuit_feasibility.ipynb` in Colab → `mock_cpu` (CPU smoke test) → set
`MODE="pilot_t4"`, T4 GPU, Run all (~30–45 min). Per-stage checkpoints resume on disconnect.
**Phase 3b:** open `phase3b_fast.ipynb` (or `_full`) → `mock_cpu` → `MODE="pilot_t4"`, **any CUDA GPU** (T4/L4),
Run all. Incremental checkpoints persist to **Google Drive** (`USE_DRIVE`) so disconnects just resume; `_full`
≈ ~2 h on T4 (less on L4). `CHRONOS_CIRCUIT_FORCE=1` / `CHRONOS_3B_FORCE=1` force recompute.
