#!/usr/bin/env python3
# Single-source builder for phase3b.ipynb (site-isolated, equal-size ablation).
# Emits phase3b.ipynb (deliverable) + _mirror_3b.py (local smoke test of the exact code).
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []
def md(s):  CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ============================================================================
md(r"""
# Phase 3b — Site-Isolated, Equal-Size Ablation (Chronos-T5)

**What this is.** A standalone follow-up to the Phases 0–3 feasibility notebook. The pilot's *nested* redundancy
ladder showed the forecast's period-P structure collapsed only at the `+dec_self` rung (176 of 432 heads) — a
collapse **confounded with ablation size** (remove a third of the heads and any model degrades). And the three
localization methods disagreed (EAP mass highest in `enc_self`; ladder collapsed at `dec_self`; the H1
candidates were `cross`). **Phase 3b removes the size confound** and decides *where* the distributed periodicity
lives, or confirms it is genuinely distributed.

**Design.** For each attention site, ablate **only that site's heads, non-nested**, and compare every site
against a **size-matched random-head null** (`random@N`, `N` = heads-per-site, drawn across all sites, `K`
independent draws). On `chronos-t5-base` all three sites are 144 heads (12 layers × 12 heads), so they are
mutually comparable; `random@N` controls for "you removed N heads." Conditions: **motif** (gates the verdict),
**trend** and **changepoint** (selectivity controls). Metrics per condition, bootstrapped over series:
**ΔCRPS** and **Δ period-P power** (positive = period structure collapsed; the sharp, gating metric on motif).

> **Run order:** run `MODE="mock_cpu"` first (CPU, tiny random-init T5, **not interpretable** — plumbing only),
> then set `MODE="pilot_t4"` on a **T4 GPU** for the real verdict (loads Chronos-T5-base).

### ⚠️ Methodological caveat baked into the verdict (read this)
Equal head-**count** is not equal functional **disruption**. Ablating **all** of `cross` severs the entire
encoder→decoder channel — a categorically harsher lesion than `random@N` (which leaves ~2/3 of the cross channel
intact) or than ablating one self-attention stream. So a `cross` "collapse" can mean *"the decoder needs the
encoder"*, not *"cross localizes periodicity."* We disambiguate with (i) the **selectivity contrast** (a real
locus must hurt motif **more** than trend/changepoint), and (ii) requiring the site to **beat the size-matched
random null**, not merely exclude zero. `GRADED_VARIANT` (config, off by default) adds a matched-*fraction*
within-site ablation that resolves the `cross` bottleneck cleanly if you turn it on.

Guardrails (unchanged): original Chronos-T5 only; **HF forward-hook** backend (no TransformerLens / nnsight);
per-head **mean-ablation**; **motif** is the gating stimulus; **honest negatives** — a DISTRIBUTED verdict is a
valid, stronger result, not a failure.
""")

# ============================================================================
md("## 1. CONFIG (the single MODE switch + mock overrides)")
code(r"""
import os
CONFIG = {
    "MODE": "mock_cpu",                 # flip to "pilot_t4" on a T4 GPU for the real verdict
    "model_id": "amazon/chronos-t5-base",
    "SEED0": 0,
    "PERIODS": [8, 12, 16, 24],
    "N_SEEDS": 3,
    "N_SERIES": 32,
    "CTX": 256,
    "PRED": 64,
    "OBS_NOISE": 0.30,
    "N_CRPS_SAMPLES": 100,
    "N_BOOTSTRAP": 1000,
    "N_RANDOM_DRAWS": 8,                 # K independent size-matched random@N draws (the null distribution)
    "RANDOM_NULL_PCT": 95,               # a site must beat this percentile of the random@N null to "localize"
    "CONDITIONS_3B": ["motif", "trend", "changepoint"],
    "SITES_3B": ["enc_self", "dec_self", "cross"],
    "GRADED_VARIANT": False,             # if True, also run a matched-FRACTION within-site ablation (cross fix)
    # ---- mock_cpu overrides (tiny random-init T5; numbers NOT interpretable) ----
    "mock_cpu": {
        "PERIODS": [6, 8], "N_SEEDS": 2, "N_SERIES": 6, "CTX": 48, "PRED": 24,
        "N_CRPS_SAMPLES": 12, "N_BOOTSTRAP": 50, "N_RANDOM_DRAWS": 3,
    },
}
MODE = os.environ.get("CHRONOS_3B_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4"), MODE
if MODE == "mock_cpu":
    CONFIG.update(CONFIG["mock_cpu"])    # apply overrides in-place
IS_MOCK = (MODE == "mock_cpu")
MOCK_TAG = "  [MOCK_CPU — NOT INTERPRETABLE]" if IS_MOCK else ""
ON_COLAB = os.path.isdir("/content")
CKPT_DIR = "/content" if ON_COLAB else os.path.abspath(".")
print(f"MODE={MODE}{MOCK_TAG}  periods={CONFIG['PERIODS']} seeds={CONFIG['N_SEEDS']} "
      f"series={CONFIG['N_SERIES']} ctx={CONFIG['CTX']} pred={CONFIG['PRED']} graded={CONFIG['GRADED_VARIANT']}")
""")

# ============================================================================
md("## 2. Setup — imports, device, install")
code(r"""
import sys, json, subprocess
def _ensure(pkg, imp):
    if os.environ.get("CHRONOS_3B_SKIP_INSTALL") == "1": return
    try: __import__(imp)
    except Exception:
        print("installing", pkg, "..."); subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
if not IS_MOCK:
    _ensure("chronos-forecasting", "chronos")

import numpy as np, torch
import matplotlib
if not ON_COLAB: matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(CONFIG["SEED0"]); np.random.seed(CONFIG["SEED0"])
DEVICE = "cuda" if (not IS_MOCK and torch.cuda.is_available()) else "cpu"
if not IS_MOCK and DEVICE == "cpu": print("WARN: pilot requested but no CUDA — running on CPU (slow).")
DTYPE = torch.float32
print("device:", DEVICE)
""")

# ============================================================================
md(r"""
## 3. Model loading + attention-site parsing (doubles as the Phase-0 sanity check)

`pilot_t4` loads the original Chronos-T5 inner T5; `mock_cpu` builds a tiny **random-init** T5 (no download).
Sites are parsed by walking the module tree and classifying `T5Attention` modules by name. We **assert all three
sites are equal size** (the premise of the experiment).
""")
code(r"""
def classify_attention_modules(model):
    sites = {"enc_self": [], "dec_self": [], "cross": []}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ != "T5Attention":
            continue
        if name.startswith("encoder") and "SelfAttention" in name:
            sites["enc_self"].append((name, mod))
        elif name.startswith("decoder") and "layer.0.SelfAttention" in name:
            sites["dec_self"].append((name, mod))
        elif name.startswith("decoder") and "EncDecAttention" in name:
            sites["cross"].append((name, mod))
    return sites

def _nheads(mod):
    for a in ("n_heads", "num_heads"):
        if hasattr(mod, a): return int(getattr(mod, a))
    return int(mod.config.num_heads)
def _dkv(mod):
    for a in ("key_value_proj_dim", "d_kv"):
        if hasattr(mod, a): return int(getattr(mod, a))
    return int(mod.config.d_kv)

if IS_MOCK:
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=256, d_model=64, d_kv=32, d_ff=128,
                   num_layers=2, num_decoder_layers=2, num_heads=2,
                   decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    INNER = T5ForConditionalGeneration(cfg).eval()
    VOCAB = cfg.vocab_size
    PIPE = None
else:
    from chronos import ChronosPipeline
    PIPE = ChronosPipeline.from_pretrained(CONFIG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)
    INNER = PIPE.inner_model.eval()
    VOCAB = INNER.config.vocab_size

SITES = classify_attention_modules(INNER)
for s in SITES: assert len(SITES[s]) > 0, f"no modules classified for site {s}"
print("per-site (modules, heads):",
      {s: (len(SITES[s]), _nheads(SITES[s][0][1])) for s in SITES})

def site_size(sites):
    sizes = {s: len(sites[s]) * _nheads(sites[s][0][1]) for s in sites}
    assert len(set(sizes.values())) == 1, f"sites NOT equal size: {sizes}"
    return next(iter(sizes.values()))

N = site_size(SITES)
print(f"EQUAL-SIZE CHECK: PASS — each site = {N} heads  (random@N will match this)")
""")

# ============================================================================
md("## 4. Stimuli — motif (induction-diagnostic), trend, changepoint")
code(r"""
def make_motif(P, rng, L):
    # sharp, NON-sinusoidal repeated motif (spike + step): the fine structure can only be produced by copying
    # the exact value one period back -> induction-diagnostic.
    motif = rng.standard_normal(P)
    motif[rng.integers(P)] += 3.0 * (1 if rng.random() > 0.5 else -1)
    motif[P // 2:] += 1.5
    motif = motif - motif.mean()
    x = np.tile(motif, L // P + 2)[:L]
    return x + CONFIG["OBS_NOISE"] * rng.standard_normal(L)

def make_trend(P, rng, L):
    slope = rng.uniform(0.5, 1.5) / L; off = rng.uniform(-1, 1)
    return slope * np.arange(L) + off + CONFIG["OBS_NOISE"] * rng.standard_normal(L)

def make_changepoint(P, rng, L):
    cp = int(L * 0.6); jump = rng.uniform(2, 4) * (1 if rng.random() > 0.5 else -1)
    x = CONFIG["OBS_NOISE"] * rng.standard_normal(L); x[cp:] += jump
    return x

_GEN = {"motif": make_motif, "trend": make_trend, "changepoint": make_changepoint}

def make_batch(cond, rng):
    # returns contexts (list of CTX arrays), targets (n_series, PRED), periods (n_series,)
    L = CONFIG["CTX"] + CONFIG["PRED"]; gen = _GEN[cond]
    ctx, tgt, Ps = [], [], []
    for i in range(CONFIG["N_SERIES"]):
        P = CONFIG["PERIODS"][i % len(CONFIG["PERIODS"])]
        full = gen(P, rng, L)
        ctx.append(full[:CONFIG["CTX"]]); tgt.append(full[CONFIG["CTX"]:]); Ps.append(P)
    return ctx, np.array(tgt), Ps
""")

# ============================================================================
md("## 5. Metrics — sample CRPS, period-P power fraction, bootstrap CI")
code(r"""
def crps_samples(samples, target):                 # samples (n_samples, H), target (H,)
    samples = np.asarray(samples, float); target = np.asarray(target, float)
    term1 = np.abs(samples - target[None, :]).mean(axis=0)
    pair  = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    return float((term1 - 0.5 * pair).mean())

def period_power_fraction(forecast_1d, P):
    # fraction of (non-DC) power within +/-1.5 bins of f0=1/P and its 2nd harmonic. In [0,1].
    x = np.asarray(forecast_1d, float); x = x - x.mean()
    H = len(x); power = np.abs(np.fft.rfft(x)) ** 2; freqs = np.fft.rfftfreq(H)
    if len(freqs) < 2: return 0.0
    df = freqs[1] - freqs[0]; total = power[1:].sum() + 1e-12; f0 = 1.0 / P
    band = (np.abs(freqs - f0) <= 1.5 * df) | (np.abs(freqs - 2 * f0) <= 1.5 * df); band[0] = False
    return float(power[band].sum() / total)

def bootstrap_ci(x, pct=(2.5, 97.5)):
    x = np.asarray(x, float)
    if len(x) == 0: return [0.0, 0.0]
    rng = np.random.default_rng(0)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(CONFIG["N_BOOTSTRAP"])]
    return [float(np.percentile(bs, pct[0])), float(np.percentile(bs, pct[1]))]
""")

# ============================================================================
md(r"""
## 6. Ablation harness — per-head mean-ablation `.o` pre-hook + site/random modes

The pre-hook reads `_ablate_heads` and `key_value_proj_dim` from the **parent T5Attention captured in the
closure** (not from the `.o` linear). Head `h` is the head-major column slice `[h*d_kv:(h+1)*d_kv]` of the
unshaped input to `.o`. **Mean-ablation** replaces those columns with their batch+position mean (never zero).
Site ablations are **non-nested** (only that site). `random@N` draws `N=site_size` heads across all sites,
`replace=False`, `K` times.
""")
code(r"""
def _make_pre_hook(attn):                 # attn = parent T5Attention (captured)
    d_kv = _dkv(attn)
    def hook(o_module, args):
        heads = getattr(attn, "_ablate_heads", None)
        if not heads:
            return None
        x = args[0].clone()
        for h in heads:
            sl = slice(h * d_kv, (h + 1) * d_kv)
            seg = x[..., sl]
            m = seg.mean(dim=tuple(range(seg.dim() - 1)), keepdim=True)   # batch+pos mean -> (1,..,1,d_kv)
            x[..., sl] = m
        return (x,)
    return hook

def install_hooks(sites):
    handles = []
    for lst in sites.values():
        for _, mod in lst:
            mod._ablate_heads = set()
            handles.append(mod.o.register_forward_pre_hook(_make_pre_hook(mod)))
    return handles

def clear_ablations(sites):
    for lst in sites.values():
        for _, mod in lst:
            mod._ablate_heads = set()

def build_head_pool(sites):
    return [(mod, h) for lst in sites.values() for _, mod in lst for h in range(_nheads(mod))]

def set_site_ablation(sites, site):                 # ablate ONLY this site (non-nested, all its heads)
    clear_ablations(sites)
    for _, mod in sites[site]:
        mod._ablate_heads = set(range(_nheads(mod)))

def set_random_ablation(sites, pool, n, rng):        # size-matched random@N null
    clear_ablations(sites)
    for idx in rng.choice(len(pool), size=n, replace=False):
        mod, h = pool[idx]; mod._ablate_heads.add(h)

def set_graded_site_ablation(sites, site, frac, rng):  # matched-FRACTION within-site (the cross-bottleneck fix)
    clear_ablations(sites)
    heads = [(mod, h) for _, mod in sites[site] for h in range(_nheads(mod))]
    k = max(1, int(round(frac * len(heads))))
    for idx in rng.choice(len(heads), size=k, replace=False):
        mod, h = heads[idx]; mod._ablate_heads.add(h)

HANDLES = install_hooks(SITES)
HEAD_POOL = build_head_pool(SITES)
print(f"hooks installed on {len(HANDLES)} '.o' modules | head pool = {len(HEAD_POOL)} heads | N(site)={N}")
""")

# ============================================================================
md("## 7. Forecast backends + plumbing assert (ablating a site must change the output)")
code(r"""
def forecast_pilot(contexts, n_samples):
    torch.manual_seed(CONFIG["SEED0"])     # common random numbers: clean vs ablated share sampling noise
    ctx = [torch.tensor(np.asarray(c), dtype=DTYPE) for c in contexts]
    fc = PIPE.predict(ctx, prediction_length=CONFIG["PRED"], num_samples=n_samples)
    return fc.detach().cpu().numpy()        # (n_series, n_samples, PRED)

def forecast_mock(contexts, n_samples):
    # drive a forward pass so the '.o' hooks FIRE, then return toy samples that DEPEND on the forward output
    # (so ablation demonstrably changes the forecast). Numbers are not interpretable.
    n = len(contexts); H = CONFIG["PRED"]
    ids = np.zeros((n, 32), dtype=np.int64)
    for i, c in enumerate(contexts):
        c = np.asarray(c, float); rng_ptp = (c.max() - c.min()) + 1e-9
        q = np.clip(((c - c.min()) / rng_ptp * (VOCAB - 3)).astype(int) + 2, 0, VOCAB - 1)
        q = q[-32:]; ids[i, :len(q)] = q
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE)
    dec = torch.zeros((n, H), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = INNER(input_ids=inp, decoder_input_ids=dec)            # hooks fire here
    sig = out.logits.float().mean(dim=-1).cpu().numpy()              # (n, H): ablation-sensitive
    samples = np.zeros((n, n_samples, H)); rng = np.random.default_rng(123)
    for i in range(n):
        c = np.asarray(contexts[i], float)
        base = np.resize(c[-H:] if len(c) >= H else np.resize(c, H), H)
        amp = 1.0 + 0.3 * np.tanh(sig[i].mean())                     # global ablation-dependence
        perturb = 0.5 * (sig[i] - sig[i].mean())                     # per-step ablation-dependence (changes power)
        samples[i] = amp * base[None, :] + perturb[None, :] + 0.1 * rng.standard_normal((n_samples, H))
    return samples

FORECAST = forecast_mock if IS_MOCK else forecast_pilot

# plumbing assert: ablating a site changes the output (hooks are wired)
_ctx, _tgt, _Ps = make_batch("motif", np.random.default_rng(0))
clear_ablations(SITES); _a = FORECAST(_ctx[:2], 4)
set_site_ablation(SITES, "dec_self"); _b = FORECAST(_ctx[:2], 4); clear_ablations(SITES)
_changed = not np.allclose(_a, _b)
print(f"PLUMBING: ablating dec_self changed the forecast = {_changed}  (max|Δ|={np.abs(_a-_b).max():.4g})")
assert _changed, "ablation did NOT change the output — hooks not wired correctly"
print("PLUMBING: PASS" + MOCK_TAG)
""")

# ============================================================================
md(r"""
## 8. Phase 3b loop

For each seed × condition: forecast **clean** once (shared baseline), then each **site** (isolated), then the
**`random@N`** null (`K` draws). `Δ period-P power` is computed on the **mean over samples** of the forecast,
per series, only on **motif** (the gating condition). Everything checkpoints to a flat record list.
""")
code(r"""
def _crps_vec(fc, tgt):
    return np.array([crps_samples(fc[i], tgt[i]) for i in range(len(tgt))])

def _power_vec(fc, Ps):
    return np.array([period_power_fraction(fc[i].mean(0), Ps[i]) for i in range(len(Ps))])

def run_phase3b():
    records = []
    for seed in range(CONFIG["N_SEEDS"]):
        rng = np.random.default_rng(CONFIG["SEED0"] + seed)
        for cond in CONFIG["CONDITIONS_3B"]:
            ctx, tgt, Ps = make_batch(cond, rng)
            clear_ablations(SITES)
            fc0 = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
            crps0 = _crps_vec(fc0, tgt)
            pw0 = _power_vec(fc0, Ps) if cond == "motif" else None

            # --- each SITE in isolation (non-nested) ---
            for site in CONFIG["SITES_3B"]:
                set_site_ablation(SITES, site)
                fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                dcrps = _crps_vec(fc, tgt) - crps0
                rec = dict(seed=seed, cond=cond, kind=site,
                           dcrps_mean=float(dcrps.mean()), dcrps_ci=bootstrap_ci(dcrps))
                if cond == "motif":
                    dpw = pw0 - _power_vec(fc, Ps)
                    rec.update(dpower_mean=float(dpw.mean()), dpower_ci=bootstrap_ci(dpw))
                records.append(rec)

            # --- size-matched random@N null (K draws) ---
            nd, npw = [], []
            for d in range(CONFIG["N_RANDOM_DRAWS"]):
                set_random_ablation(SITES, HEAD_POOL, N, rng)
                fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                nd.append(float((_crps_vec(fc, tgt) - crps0).mean()))
                if cond == "motif":
                    npw.append(float((pw0 - _power_vec(fc, Ps)).mean()))
            records.append(dict(seed=seed, cond=cond, kind="random",
                                dcrps_draws=nd, dpower_draws=npw))

            # --- optional GRADED matched-fraction within-site (resolves the cross-bottleneck) ---
            if CONFIG["GRADED_VARIANT"]:
                frac = 1.0 / len(CONFIG["SITES_3B"])   # ~1/3 of a site == random's per-site hit
                for site in CONFIG["SITES_3B"]:
                    set_graded_site_ablation(SITES, site, frac, rng)
                    fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                    dcrps = _crps_vec(fc, tgt) - crps0
                    rec = dict(seed=seed, cond=cond, kind=f"graded:{site}",
                               dcrps_mean=float(dcrps.mean()), dcrps_ci=bootstrap_ci(dcrps))
                    if cond == "motif":
                        rec.update(dpower_mean=float((pw0 - _power_vec(fc, Ps)).mean()))
                    records.append(rec)
            clear_ablations(SITES)
        print(f"  seed {seed} done")
    return records

print("running Phase 3b ..." + MOCK_TAG)
RECORDS = run_phase3b()
print(f"collected {len(RECORDS)} records")
""")

# ============================================================================
md(r"""
## 9. Verdict — LOCUS vs DISTRIBUTED (de-confounded)

**LOCUS = site** iff, on **motif**: (i) the site's Δ period-P power **beats the size-matched `random@N` null**
(> the `RANDOM_NULL_PCT` percentile), **and** (ii) it is **periodicity-specific** — the same site's ablation does
not damage trend/changepoint as much (motif ΔCRPS ≤ max(trend,changepoint) ΔCRPS) **or** the collapse is large
(> 2× the null percentile). Otherwise **DISTRIBUTED** (the de-confounded outcome B — the stronger claim).
The verdict keys on **motif only**; trend/changepoint feed the selectivity check. We also print a
**detection-power** line: beating the null at full-site scale shows the method *can* localize; if even the
most-collapsing site fails to beat `random@N`, the null is real, not underpowered.
""")
code(r"""
def _agg(kind, cond, field):
    vals = [r[field] for r in RECORDS if r.get("kind") == kind and r["cond"] == cond and field in r]
    return float(np.mean(vals)) if vals else float("nan")

def summarize_3b():
    null_draws = [v for r in RECORDS if r["kind"] == "random" and r["cond"] == "motif" for v in r["dpower_draws"]]
    null_pct = float(np.percentile(null_draws, CONFIG["RANDOM_NULL_PCT"])) if null_draws else 0.0
    null_crps = [v for r in RECORDS if r["kind"] == "random" and r["cond"] == "motif" for v in r["dcrps_draws"]]

    rows = []
    for site in CONFIG["SITES_3B"]:
        dpw   = _agg(site, "motif", "dpower_mean")
        c_mot = _agg(site, "motif", "dcrps_mean")
        c_trd = _agg(site, "trend", "dcrps_mean")
        c_cp  = _agg(site, "changepoint", "dcrps_mean")
        beats_null = dpw > null_pct
        selective  = (c_mot <= max(c_trd, c_cp)) or (dpw > 2 * null_pct)
        rows.append(dict(site=site, dpower=dpw, beats_null=bool(beats_null), selective=bool(selective),
                         dcrps_motif=c_mot, dcrps_trend=c_trd, dcrps_changepoint=c_cp,
                         locus=bool(beats_null and selective)))
    loci = [r for r in rows if r["locus"]]
    best = max(rows, key=lambda r: (r["dpower"] if np.isfinite(r["dpower"]) else -1e9))
    if loci:
        win = max(loci, key=lambda r: r["dpower"])
        verdict = f"LOCUS = {win['site']}"
    else:
        verdict = "DISTRIBUTED"

    print("=" * 78)
    print(f"PHASE 3b VERDICT: {verdict}{MOCK_TAG}")
    print("=" * 78)
    print(f"random@N null (motif Δpower, p{CONFIG['RANDOM_NULL_PCT']}) = {null_pct:+.4f}  "
          f"(N={N} heads/draw, K={CONFIG['N_RANDOM_DRAWS']}×{CONFIG['N_SEEDS']} draws)")
    print(f"  {'site':9s} {'Δpower(motif)':>13s} {'beats_null':>10s} {'selective':>9s}   "
          f"{'ΔCRPS mot':>9s} {'trend':>8s} {'chgpt':>8s}  LOCUS")
    for r in rows:
        print(f"  {r['site']:9s} {r['dpower']:+13.4f} {str(r['beats_null']):>10s} {str(r['selective']):>9s}   "
              f"{r['dcrps_motif']:+9.4f} {r['dcrps_trend']:+8.4f} {r['dcrps_changepoint']:+8.4f}  "
              f"{'*' if r['locus'] else ''}")
    # detection-power statement
    if best["dpower"] > null_pct:
        print(f"\nDETECTION POWER: site '{best['site']}' beats the size-matched null "
              f"(Δpower {best['dpower']:+.4f} > {null_pct:+.4f}) -> the method CAN localize beyond size.")
    else:
        print(f"\nDETECTION POWER: even the most-collapsing site ('{best['site']}', Δpower {best['dpower']:+.4f}) "
              f"does NOT beat the size-matched null ({null_pct:+.4f}) -> the distributed null is REAL, not underpowered.")
    if any(r["site"] == "cross" and r["locus"] for r in rows):
        print("NOTE: 'cross' flagged as locus — recall ablating ALL cross severs the encoder→decoder channel; "
              "confirm with the selectivity row and (if needed) GRADED_VARIANT before claiming a cross locus.")
    return dict(verdict=verdict, null_pct=null_pct, null_crps_mean=float(np.mean(null_crps)) if null_crps else 0.0,
                rows=rows, mode=MODE, mock=IS_MOCK, N=N, config={k: CONFIG[k] for k in
                ("PERIODS","N_SEEDS","N_SERIES","CTX","PRED","OBS_NOISE","N_CRPS_SAMPLES","N_BOOTSTRAP",
                 "N_RANDOM_DRAWS","RANDOM_NULL_PCT","GRADED_VARIANT")})

SUMMARY = summarize_3b()
""")

# ============================================================================
md("## 10. Fig 4b — per-site Δ period-P power vs random null (left); per-site ΔCRPS by condition (right)")
code(r"""
try:
    rows = SUMMARY["rows"]; sites = [r["site"] for r in rows]; xs = np.arange(len(sites))
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    # left: motif Δpower with random-null line
    ax[0].bar(xs, [r["dpower"] for r in rows], 0.6,
              color=["#c0392b" if r["locus"] else "#7f8c8d" for r in rows])
    ax[0].axhline(SUMMARY["null_pct"], color="k", ls="--", lw=1,
                  label=f"random@N p{CONFIG['RANDOM_NULL_PCT']}")
    ax[0].axhline(0, color="gray", lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels(sites)
    ax[0].set_ylabel("Δ period-P power (motif; drop)")
    ax[0].set_title(f"Fig 4b: site-isolated locus  [{SUMMARY['verdict']}]" + MOCK_TAG, fontsize=10); ax[0].legend()
    # right: ΔCRPS by condition (selectivity)
    w = 0.25
    ax[1].bar(xs - w, [r["dcrps_motif"] for r in rows], w, label="motif", color="#c0392b")
    ax[1].bar(xs,     [r["dcrps_trend"] for r in rows], w, label="trend", color="#7f8c8d")
    ax[1].bar(xs + w, [r["dcrps_changepoint"] for r in rows], w, label="changepoint", color="#bdc3c7")
    ax[1].axhline(0, color="k", lw=0.8); ax[1].set_xticks(xs); ax[1].set_xticklabels(sites)
    ax[1].set_ylabel("ΔCRPS under ablation"); ax[1].set_title("selectivity by condition"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(CKPT_DIR, "fig4b_site_isolated.png"), dpi=90)
    plt.show(); plt.close(fig)
    print("saved fig4b_site_isolated.png")
except Exception as e:
    print("fig4b skipped:", repr(e)[:160])
""")

# ============================================================================
md("## 11. Checkpoint to JSON")
code(r"""
out = dict(summary=SUMMARY, records=RECORDS)
p = os.path.join(CKPT_DIR, f"phase3b_{MODE}.json")
with open(p, "w") as f: json.dump(out, f, indent=2)
print("wrote", p)
print("\nPHASE 3b COMPLETE." + MOCK_TAG, "->", SUMMARY["verdict"])
""")

# ---- assemble ----------------------------------------------------------------------------------
nb = new_notebook()
nb.cells = [new_markdown_cell(s) if t == "md" else new_code_cell(s) for (t, s) in CELLS]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}
with open("phase3b.ipynb", "w") as f: nbf.write(nb, f)
with open("_mirror_3b.py", "w") as f:
    f.write("\n".join(["# auto-mirror of phase3b code cells (local smoke test)"] +
                      ["\n# " + "=" * 60 + "\n" + s for t, s in CELLS if t == "code"]))
print(f"wrote phase3b.ipynb ({sum(t=='code' for t,_ in CELLS)} code cells) + _mirror_3b.py")
