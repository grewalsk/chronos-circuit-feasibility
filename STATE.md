# STATE — Chronos Selective Periodic-Induction Circuit

*Living status doc. Last updated after the redesigned `pilot_t4` run on `chronos-t5-base` (2026-06-15).*

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

**Tier 1 — must-have, cheap (free-T4):**
- **Phase 3b — site-isolated, equal-size ablation.** all-enc_self vs all-dec_self vs all-cross vs 144 random
  heads (matched size), ΔCRPS + Δ period-P power on motif. De-confounds the locus from ablation size. *(Not yet
  implemented; the one immediate next coding step.)*
- **Detection-power statement.** The method *can* catch a localized head if one exists (the ladder's +dec_self
  Δstruct +0.291, CI excluding 0, is the positive control) — state so the null isn't read as underpowered.
- **Robustness.** Confirm the distributed verdict is invariant across seeds and `obs_noise ∈ {0.15, 0.30, 0.45}`.

**Tier 2 — generalization (credibility):**
- **ETT real-data** confirmation (spec Phase 4) — reproduces "no localized circuit; structure survives small
  ablation" on real seasonality. Closes the #1 reviewer objection.
- **Base → Large** (`chronos-t5-large`, 710M, Mishra's model; A100) — within-family universality. The one
  expensive item; a strengthener for the distributed result, not a blocker for a workshop submission.

**Tier 3 — mechanism depth (turns "distributed" into a characterization):**
- Trace the stages: the `enc_self L5H7` compressed "period-detector" (T=0.918, slope 0.18) and the staged
  feeders → the dec_self heads carrying the structure (spec's estimate → aggregate → select).
- SAE cross-validation vs Mishra (does the distributed locus map to his "seasonality" features?).

**Tier 4 — writing & figures (spec Phase 5):**
- Fig 1 schematic · Fig 2 lag-tracking heatmap · Fig 3 copying · Fig 4 causal selectivity + nested ladder ·
  Fig 4b/5 the 3b site-isolated localization · Fig 5 scope (across P, size, ETT) + SAE triangulation.
- Verify every arXiv ID; correct the 2510.09776 framing (linear attention, regime not covered); cite time2time
  and draw the layer-steering-vs-circuit line; limitations (synthetic + ETT, head-level, redundancy, 3b caveat).

**Critical path to a defensible workshop submission:** `3b → robustness/power → ETT` (all free-T4, a day or two).
Large and the mechanism-depth tracing elevate it to the conference-length version.

---

## 5. Repo contents

- `chronos_circuit_feasibility.ipynb` — the deliverable notebook (Phases 0–3; mock_cpu → pilot_t4).
- `build_notebook.py` — single-source builder that generates the notebook (+ a `_mirror.py` for local testing).
- `PHASE3_REDESIGN.md` — the rationale for the redesigned, non-circular Phase 3 (confirmatory gate + ladder +
  mechanism decomposition) and the two adversarial-review passes applied.
- `phase1_pilot_t4.json`, `phase2_pilot_t4.json`, `phase3_pilot_t4new.json` — the pilot checkpoints (the numbers
  in §2 come from these).
- `*.png` — Fig 2 (lag-tracking), Fig 3 (copying), Fig 4 (causal), stimulus sanity.
- Authoritative design lives in the separate repo `github.com/grewalsk/circuitTSFM`
  (`chronos_circuit_spec_v2.md`, `chronos_circuit_plan_v2.md`), cloned into the session by Section 0.

## 6. Reproduce

Open in Colab → run `mock_cpu` (default, CPU smoke test, ~minutes, **not interpretable**) → set
`CONFIG["MODE"]="pilot_t4"`, switch to a T4 GPU, Run all (~30–45 min) for the real verdict. Checkpoints are
per-stage, so a disconnect resumes mid-phase. `CHRONOS_CIRCUIT_FORCE=1` forces recompute.
