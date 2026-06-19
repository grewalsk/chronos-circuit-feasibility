# Chronos Selective Periodic‑Induction Circuit — Feasibility Notebook (Phases 0–3)

A single, self‑contained Google Colab notebook that runs the **feasibility pass** of a circuit‑level
mechanistic‑interpretability study of the original **Chronos‑T5** time‑series foundation model.

> **TL;DR** — Does Chronos *learn* a **selective / multi‑lag induction head** (the periodicity analog of an
> induction head)? The notebook scans every attention head for a circuit that **infers a series' own period
> `P` from content**, **attends to the same‑phase positions one or more periods back** (`t−P, t−2P, …`), and
> **copies** the bin value found there into the forecast — then tests whether that head is **causal and
> selective**. It is **win‑either‑way**: a clean null is an informative result that routes to a
> change‑detection circuit instead.

The deliverable is one file: [`chronos_circuit_feasibility.ipynb`](chronos_circuit_feasibility.ipynb).

> **Current result (chronos‑t5‑base): DISTRIBUTED.** Phase 1 finds lag‑tracking heads *are* present in attention
> and some also copy, but Phase 3 + the de‑confounded **Phase 3b** (site‑isolated, equal‑size ablation) show **no
> attention site causally localizes periodicity** — period structure survives partial ablation and collapses only
> near full‑site ablation, while non‑periodic structure collapses earlier. So Chronos *represents* periodicity in
> attention but does **not** localize it to a selective‑induction head; the computation is **distributed/redundant**
> — a circuit‑level adjudication of the attention‑degeneration debate that corroborates Mishra & Pandey while
> nuancing them. See [`STATE.md`](STATE.md) (status + exact numbers) and [`RESULTS_phase3b.md`](RESULTS_phase3b.md)
> (draft Results paragraph). Phase 3b notebooks: [`phase3b_fast.ipynb`](phase3b_fast.ipynb) /
> [`phase3b_full.ipynb`](phase3b_full.ipynb).

---

## Why this is interesting

Chronos‑T5 is encoder–decoder, autoregressive, with **per‑time‑step tokenization** into a 4,096‑bin
vocabulary and **T5 relative‑position bias**. It has no seasonality inductive bias, yet forecasts periodic
signals well — *something* implements periodicity. This notebook asks **where**, at head‑level granularity,
which (to our knowledge) has not been done for any TSFM.

The hypothesis is the **selective induction head** of d'Angelo, Croce & Flammarion, *Selective Induction
Heads* (ICLR 2025, [arXiv:2509.08184](https://arxiv.org/abs/2509.08184)): a circuit that selects the correct
lag from content rather than attending at a position‑fixed offset.

**The central confound** is T5's relative‑position bias — a head can attend at a fixed offset `P` for purely
positional reasons. The defense is the **selective / multi‑lag signature**: a genuine selective‑induction
head must **track the correct lag as `P` changes across series**; a positional artifact attends at a constant
offset and cannot track. Identification therefore rests on **lag‑tracking across `P`** (calibrated against a
**fixed‑offset null**), never a raw fixed‑`P` lag score.

---

## Two hard gates

| Gate | Question | Verdict printed |
|---|---|---|
| **Phase 1** | Does any head *track the correct lag as `P` varies*? (the selective signature) | `GREEN \| PIVOT \| AMBIGUOUS` |
| **Phase 3** | Is that head *causal and selective*? (ablation degrades periodic forecasting but not trend/changepoint) | `PASS \| FAIL` |

Headline causal metric: **selective ΔCRPS**.

---

## How to run

The notebook has **one switch** — `CONFIG["MODE"]` — and the same code path serves both modes.

1. **`mock_cpu` (run this first).** A few‑minute **pipeline smoke test** on CPU with `chronos‑t5‑tiny`. Its
   purpose is only to confirm the pipeline executes end‑to‑end; **its numbers are NOT scientifically
   interpretable** and are labeled as such everywhere.
2. **`pilot_t4` (the real verdict).** Flip `CONFIG["MODE"] = "pilot_t4"` and run on a **free Colab T4** with
   `chronos‑t5‑base` for the actual feasibility verdict.

Open in Colab, Runtime → Run all. The first cells `pip install` Chronos/nnsight and clone the design repo into
the session. Scores checkpoint to disk (`/content/ckpt`), so each phase is independently re‑runnable after a
disconnect (set `CHRONOS_CIRCUIT_FORCE=1` to recompute).

---

## Notebook structure

| Section | Phase | Pass condition |
|---|---|---|
| 0 — Setup & repo context | 0 | CONFIG resolves; design spec/plan cloned into session |
| 1 — Install & imports | 0 | libraries import; versions print |
| 2 — Load Chronos & validate tokenization | 0a | `PER‑STEP TOKENIZATION: PASS` (lag‑in‑tokens = lag‑in‑time) |
| 3 — Hook validation | 0b | **`PLUMBING: PASS`** — hooks fire at all 4 sites; mean‑ablation bites; cross‑attn key provenance; NLL‑proxy gradient sign |
| 4 — Stimulus generators | 0 | periodic / AR(1) / phase‑scrambled / period‑altered / **motif** / trend / changepoint |
| 5 — Selective‑lag scan **(GATE)** | 1 | **`GREEN \| PIVOT \| AMBIGUOUS`** + lag‑tracking heatmap (Fig 2) |
| 6 — Copying / OV leg | 2 | tracks‑lag‑and‑copies vs tracks‑lag‑only (Fig 3) |
| 7 — Causal validation **(GATE)** | 3 | **`PHASE 3: PASS \| FAIL`** — confirmatory gate on the lag‑tracking ∩ copying head (selective ΔCRPS + period‑P power collapse), nested redundancy ladder, attention‑pattern vs OV decomposition, EAP localization (Fig 4) |
| 8 — Feasibility report | — | overall **`GO \| NO‑GO \| PIVOT`** recommendation |

> **Phase 3 is built around one principle** ([PHASE3_REDESIGN.md](PHASE3_REDESIGN.md)): the causal **GO** gate
> tests the head we identified *independently* (Phase 1 lag‑tracking ∩ Phase 2 copying) — never heads selected
> by the same causal metric it then validates on. Because Chronos is known‑redundant (so a single head's
> ablation can read null even if it participates), a non‑confirmatory result is **not** a project failure: the
> **nested redundancy ladder** (`H1 ⊆ +lag‑trackers ⊆ +dec_self ⊆ +cross ⊆ all‑attention`) shows *how
> distributed* the periodic computation is, which is itself the publishable split / attention‑degeneration
> result. The headline causal readouts are **selective ΔCRPS** and the **collapse of period‑P forecast power**.

---

## Methodology notes (the parts that make it correct)

- **Original Chronos‑T5 only** (`amazon/chronos‑t5‑{tiny,…,base}`) — *not* Chronos‑Bolt or Chronos‑2, which
  patch and/or use a different architecture for which the decoder‑cross‑attention copy hypothesis has no
  referent. Per‑step tokenization is verified at runtime before any lag index is computed.
- **Introspection, not assumption.** The model architecture is printed and the four intervention sites
  (encoder self‑attn, decoder self‑attn, decoder cross‑attn, and the per‑head input to each attention output
  projection `.o`), `lm_head`, and `relative_attention_bias` are located by walking the module tree.
- **Validated HF forward hooks** are the intervention backend for every scored number (nnsight is probed for
  interactive use; the spec sanctions HF hooks as the trustworthy fallback). TransformerLens is **not** used —
  it is decoder‑only; Chronos‑T5 is encoder–decoder.
- **Mean‑ablation, never zero‑ablation.** **EAP** uses a differentiable **categorical‑NLL proxy** (never
  backprop through sampling); the headline causal numbers use **gradient‑free sampled CRPS**.
- **No raw attention tensors are cached** — only per‑head scalar scores, computed on the fly.
- **The aperiodic controls are chosen carefully.** Phase‑scrambling preserves the power spectrum and therefore
  (Wiener–Khinchin) the lag‑`P` autocorrelation, so it is a *weak* control for a lag‑based head — the notebook
  uses **AR(1)** as the genuine lag‑collapse control and a **period‑altered** series as the Phase‑3 corrupt.
- **Multi‑lag, not single‑lag.** Candidacy tracks integer multiples of `P` (slope ≈ `m ∈ {1,2,3}`), so a head
  copying from `t−2P` is not falsely rejected.
- **Exact verification of attribution‑patched edges**, including a low‑EAP‑rank probe for the AtP* false
  negatives that motivate exact verification; ACDC is scoped to the EAP‑surfaced region only.
- **Determinism** (seeded python/numpy/torch; fixed, versioned mean‑scaling regime) and **honest branching**
  (a red Phase 1 skips Phases 2–3 and recommends the change‑detection pivot — no parameter tuning to
  manufacture a green).

---

## Requirements

- Python 3.10+, `torch`, `chronos-forecasting`, `nnsight` (optional), `scipy`, `matplotlib`.
- Verified on `torch 2.8`, `transformers 4.51`, `chronos-forecasting 2.2.2`. Runs CPU‑only for the smoke test;
  a free Colab T4 for the pilot.

The notebook installs its own dependencies in the first cells, so on Colab you only need to run the cells.

### Regenerating the notebook

The notebook is generated from a single‑source builder so the code lives in exactly one place:

```bash
python build_notebook.py   # writes chronos_circuit_feasibility.ipynb (+ _mirror.py for local smoke testing)
```

---

## Authoritative design

The experimental design — the selective‑induction definition, the standing controls, the per‑phase
deliverables, the outcome tree, and the reference list — lives in **[`circuitTSFM`](https://github.com/grewalsk/circuitTSFM)**
(`chronos_circuit_spec_v2.md`, `chronos_circuit_plan_v2.md`). Section 0 of the notebook clones it into the
Colab session so the design is visible while the notebook runs. This repo is the **feasibility implementation**
of Phases 0–3; Phases 4–7 (ETT, the Large final numbers, cross‑size universality, submission) follow only
after this gate passes.

---

## Method lineage / references

Olsson et al. 2022 (induction matching/copying scores) · Wang et al. 2022 (IOI circuit‑validation template) ·
Goldowsky‑Dill et al. 2023 (path patching) · Syed et al. 2023 (edge attribution patching) ·
Kramár et al. 2024 (AtP\* false‑negative caveats) · Conmy et al. 2023 (ACDC) ·
Heimersheim & Nanda 2024 (activation‑patching guide) · Fiotto‑Kaufman et al. (nnsight) ·
d'Angelo, Croce & Flammarion 2025 (selective induction heads) · Ansari et al. 2024 (Chronos).
