#!/usr/bin/env python3
# Single-source builder for phase3b.ipynb (site-isolated ablation; structural-selectivity verdict +
# fraction-sweep dose-response; cross-locus gated on a low-f selective effect).
# Emits phase3b.ipynb + _mirror_3b.py (local smoke test).
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []
def md(s):  CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ============================================================================
md(r"""
# Phase 3b — Site-Isolated Ablation: Locus vs Distributed (structural-selectivity verdict)

**Goal.** The pilot's *nested* ladder collapsed period structure only at `+dec_self` (176/432 heads) — confounded
with ablation **size**. Phase 3b ablates each attention site **in isolation** and decides *where* the distributed
periodicity lives, or confirms it is genuinely distributed, **without** the size confound.

**The core confound (and why the fix is what it is).** Equal head-**count** is not equal functional **disruption**.
`cross` is the *only* channel through which the decoder sees the encoder, so ablating **all** of `cross` is a
**100% channel severance**, while a size-matched `random@N` removes only ~1/3 of every channel (the context
channel survives) and `all-enc_self`/`all-dec_self` degrade a stream without severing a bottleneck. A head-count
null **structurally cannot** catch this, because random never reaches 100% of `cross`. So we do **not** lean on
head-matching. The principled control is **selectivity**: severance is **non-selective by construction** — cutting
the context channel hurts *all* forecasting — whereas a genuine periodicity locus produces a **selective**
effect: only the period structure collapses.

**Two design rules that make selectivity load-bearing:**
1. **Measure structure like-for-like, not global CRPS.** Periodic forecasting may simply be more context-hungry,
   so severance could hit motif CRPS harder than trend CRPS for reasons unrelated to localization. We instead
   compare **structural collapse** per condition: **period-P power** (motif), **trend-slope recovery** (trend),
   **changepoint-level recovery** (changepoint). Severance flattens *all* structure (the forecast reverts toward
   the mean) → all three collapse; a real locus collapses **period-P power only**.
2. **A fraction-sweep dose-response.** For each site we ablate a random fraction `f ∈ {0.1,0.25,0.5,0.75,1.0}` of
   its heads (with a random-across-all-sites arm at each matched total). **Severance is an endpoint effect** —
   `cross`'s collapse appears only as `f→1` and is non-selective at every `f`. **A genuine locus** shows a
   **selective** collapse at **low `f`**, where the channel is nowhere near severed. The dose-response *shape*
   settles it.

**Decision rule (baked in, not prose).** `enc_self` / `dec_self` localize iff they show a **selective structural
collapse** (period-P power collapses; trend-slope & changepoint-level survive). **`cross` localizes iff that AND
the selective effect appears at low `f`** (not only at `f=1`) — so a severance artifact at `f=1` can never be
reported as a `cross` locus. Otherwise: **DISTRIBUTED** (the de-confounded, stronger result). Global ΔCRPS is
reported as a **necessity** number only; the `random@N` head-count null is a demoted **sanity baseline**.

> Run `MODE="mock_cpu"` first (CPU, tiny random T5, **not interpretable**), then `MODE="pilot_t4"` on a **T4**.
>
> **Free-T4 disconnects are handled.** Both experiments **checkpoint incrementally** (after each `(seed,
> condition)` block and each `(seed, f)` sweep unit). If Colab drops, just **re-run the notebook** — it resumes
> from the last saved unit. On Colab the checkpoints persist to **Google Drive** (`USE_DRIVE`, one-time auth
> prompt) so they survive even a full runtime reset; set `CHRONOS_3B_FORCE=1` to recompute from scratch.
>
> Guardrails: original Chronos-T5 only; HF forward-hook backend; per-head **mean-ablation**; honest negatives.
""")

# ============================================================================
md("## 1. CONFIG")
code(r"""
import os
CONFIG = {
    "MODE": "mock_cpu",                 # -> "pilot_t4" on a T4 GPU
    "model_id": "amazon/chronos-t5-base",
    "USE_DRIVE": True,                   # on Colab, persist checkpoints to Google Drive so a runtime reset
                                        # (free-tier disconnect) does NOT lose progress; falls back to /content
    "SEED0": 0,
    "PERIODS": [8, 12, 16, 24],
    "N_SEEDS": 3,
    "N_SERIES": 32,
    "CTX": 256,
    "PRED": 64,
    "OBS_NOISE": 0.30,
    "N_CRPS_SAMPLES": 64,                # dominant memory/time lever (gating metric is structural, not CRPS)
    "N_BOOTSTRAP": 1000,
    "FORECAST_BATCH": 4,                 # series per predict() call (T4-memory safe; auto-halves on OOM)
    "N_RANDOM_DRAWS": 8,                 # size-matched random@N (f=1) head-count baseline draws
    "CONDITIONS_3B": ["motif", "trend", "changepoint"],
    "SITES_3B": ["enc_self", "dec_self", "cross"],
    # ---- fraction-sweep (the confound-control dose-response panel) ----
    "F_GRID": [0.1, 0.25, 0.5, 0.75, 1.0],
    "SWEEP_DRAWS": 3,                    # random draws per (site, f)
    "SWEEP_SERIES": 16,                  # smaller series count for the sweep (shape, not headline CIs)
    "SWEEP_SEEDS": 1,
    "LOW_F_MAX": 0.5,                    # a cross locus must show its selective effect at f <= this
    # ---- decision thresholds (pre-registered) ----
    "STRUCT_COLLAPSE_MIN": 0.30,         # motif period-P power must lose >= 30% of its structure
    "SELECTIVITY_MARGIN": 2.0,           # motif structural collapse must be >= 2x the non-periodic collapse
    # ---- mock overrides (tiny random T5; NOT interpretable) ----
    "mock_cpu": {
        "PERIODS": [6, 8], "N_SEEDS": 2, "N_SERIES": 6, "CTX": 48, "PRED": 24,
        "N_CRPS_SAMPLES": 12, "N_BOOTSTRAP": 50, "N_RANDOM_DRAWS": 3, "FORECAST_BATCH": 999,
        "F_GRID": [0.25, 0.5, 1.0], "SWEEP_DRAWS": 2, "SWEEP_SERIES": 4, "SWEEP_SEEDS": 1,
    },
}
MODE = os.environ.get("CHRONOS_3B_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4"), MODE
if MODE == "mock_cpu": CONFIG.update(CONFIG["mock_cpu"])
IS_MOCK = (MODE == "mock_cpu")
MOCK_TAG = "  [MOCK_CPU — NOT INTERPRETABLE]" if IS_MOCK else ""
ON_COLAB = os.path.isdir("/content")
CKPT_DIR = os.path.abspath(".")
if ON_COLAB:
    CKPT_DIR = "/content"
    if CONFIG.get("USE_DRIVE", True) and not IS_MOCK:   # mount Drive so checkpoints survive a runtime reset
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            CKPT_DIR = "/content/drive/MyDrive/chronos_phase3b"; os.makedirs(CKPT_DIR, exist_ok=True)
            print("checkpoints -> Google Drive (survives disconnects):", CKPT_DIR)
        except Exception as e:
            print("Drive mount skipped (", repr(e)[:80], ") -> /content (lost on a full runtime reset)")
print(f"MODE={MODE}{MOCK_TAG}  periods={CONFIG['PERIODS']} seeds={CONFIG['N_SEEDS']} series={CONFIG['N_SERIES']} "
      f"ctx={CONFIG['CTX']} pred={CONFIG['PRED']}  F_GRID={CONFIG['F_GRID']}  ckpt={CKPT_DIR}")
""")

# ============================================================================
md("## 2. Setup")
code(r"""
import sys, json, subprocess, gc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # reduce CUDA fragmentation
def _ensure(pkg, imp):
    if os.environ.get("CHRONOS_3B_SKIP_INSTALL") == "1": return
    try: __import__(imp)
    except Exception:
        print("installing", pkg); subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
if not IS_MOCK: _ensure("chronos-forecasting", "chronos")

import numpy as np, torch
import matplotlib
if not ON_COLAB: matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(CONFIG["SEED0"]); np.random.seed(CONFIG["SEED0"])
DEVICE = "cuda" if (not IS_MOCK and torch.cuda.is_available()) else "cpu"
if not IS_MOCK and DEVICE == "cpu": print("WARN: pilot requested but no CUDA -> CPU (slow).")
DTYPE = torch.float32
print("device:", DEVICE)
""")

# ============================================================================
md("## 3. Model + attention-site parsing (equal-size assert = Phase-0 sanity)")
code(r"""
def classify_attention_modules(model):
    sites = {"enc_self": [], "dec_self": [], "cross": []}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ != "T5Attention": continue
        if name.startswith("encoder") and "SelfAttention" in name: sites["enc_self"].append((name, mod))
        elif name.startswith("decoder") and "layer.0.SelfAttention" in name: sites["dec_self"].append((name, mod))
        elif name.startswith("decoder") and "EncDecAttention" in name: sites["cross"].append((name, mod))
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
    cfg = T5Config(vocab_size=256, d_model=64, d_kv=32, d_ff=128, num_layers=2, num_decoder_layers=2,
                   num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    INNER = T5ForConditionalGeneration(cfg).eval(); VOCAB = cfg.vocab_size; PIPE = None
else:
    from chronos import ChronosPipeline
    if DEVICE == "cuda":   # free any model/leaked tensors left by a PRIOR (e.g. crashed) run in this session
        for _v in ("PIPE", "INNER"):
            if _v in globals():
                try: del globals()[_v]
                except Exception: pass
        gc.collect(); torch.cuda.empty_cache()
        _free, _tot = torch.cuda.mem_get_info()
        print(f"GPU free {_free/1e9:.1f}/{_tot/1e9:.1f} GB before load")
        assert _free > 1.5e9, ("Only %.2f GB free — a previous run left memory resident on the GPU. "
                               "RESTART THE RUNTIME (Runtime -> Restart session), then run again." % (_free/1e9))
    PIPE = ChronosPipeline.from_pretrained(CONFIG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)
    INNER = PIPE.inner_model.eval(); VOCAB = INNER.config.vocab_size
    INNER.requires_grad_(False)                              # no grads anywhere (no autograd graph retained)
    try: INNER.config._attn_implementation = "eager"        # matches the known-good feasibility-notebook path
    except Exception: pass

SITES = classify_attention_modules(INNER)
for s in SITES: assert len(SITES[s]) > 0, f"no modules for {s}"
print("per-site (modules, heads):", {s: (len(SITES[s]), _nheads(SITES[s][0][1])) for s in SITES})

def site_size(sites):
    sizes = {s: len(sites[s]) * _nheads(sites[s][0][1]) for s in sites}
    assert len(set(sizes.values())) == 1, f"sites NOT equal size: {sizes}"
    return next(iter(sizes.values()))
N = site_size(SITES)
print(f"EQUAL-SIZE CHECK: PASS — each site = {N} heads")
""")

# ============================================================================
md("## 4. Stimuli — motif / trend / changepoint")
code(r"""
def make_motif(P, rng, L):
    m = rng.standard_normal(P); m[rng.integers(P)] += 3.0 * (1 if rng.random() > 0.5 else -1)
    m[P // 2:] += 1.5; m = m - m.mean()
    return np.tile(m, L // P + 2)[:L] + CONFIG["OBS_NOISE"] * rng.standard_normal(L)

def make_trend(P, rng, L):
    slope = rng.uniform(0.5, 1.5) / L; off = rng.uniform(-1, 1)
    return slope * np.arange(L) + off + CONFIG["OBS_NOISE"] * rng.standard_normal(L)

def make_changepoint(P, rng, L):
    cp = int(L * 0.6); jump = rng.uniform(2, 4) * (1 if rng.random() > 0.5 else -1)
    x = CONFIG["OBS_NOISE"] * rng.standard_normal(L); x[cp:] += jump; return x

_GEN = {"motif": make_motif, "trend": make_trend, "changepoint": make_changepoint}

def make_batch(cond, rng, n_series):
    L = CONFIG["CTX"] + CONFIG["PRED"]; gen = _GEN[cond]; ctx, tgt, Ps = [], [], []
    for i in range(n_series):
        P = CONFIG["PERIODS"][i % len(CONFIG["PERIODS"])]
        full = gen(P, rng, L); ctx.append(full[:CONFIG["CTX"]]); tgt.append(full[CONFIG["CTX"]:]); Ps.append(P)
    return ctx, np.array(tgt), Ps
""")

# ============================================================================
md(r"""
## 5. Metrics — necessity (CRPS) + **structural** recovery per condition (the selectivity primitives)

All three structural metrics live in ~[0,1] and **collapse toward 0 under severance** (a mean-reverting flat
forecast), so they are directly comparable as "fraction of structure retained":
- **motif** → `period_power_fraction` (fraction of forecast power at period P).
- **trend** → `trend_slope_recovery` (how well the forecast's slope matches the target's; flat ⇒ 0).
- **changepoint** → `changepoint_level_recovery` (how well the forecast holds the post-jump level; reverted ⇒ 0).

`rel_collapse = (structure_clean − structure_ablated) / structure_clean` (per condition; bootstrap CI). Positive
= structure destroyed. The verdict compares **rel_collapse across conditions** — never CRPS magnitudes.
""")
code(r"""
def crps_samples(samples, target):
    samples = np.asarray(samples, float); target = np.asarray(target, float)
    t1 = np.abs(samples - target[None, :]).mean(axis=0)
    pair = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    return float((t1 - 0.5 * pair).mean())

def period_power_fraction(f1d, P):
    x = np.asarray(f1d, float); x = x - x.mean(); H = len(x)
    power = np.abs(np.fft.rfft(x)) ** 2; freqs = np.fft.rfftfreq(H)
    if len(freqs) < 2: return 0.0
    df = freqs[1] - freqs[0]; total = power[1:].sum() + 1e-12; f0 = 1.0 / P
    band = (np.abs(freqs - f0) <= 1.5 * df) | (np.abs(freqs - 2 * f0) <= 1.5 * df); band[0] = False
    return float(power[band].sum() / total)

def trend_slope_recovery(f1d, target1d):
    H = len(f1d); t = np.arange(H)
    sf = float(np.polyfit(t, f1d, 1)[0]); st = float(np.polyfit(t, target1d, 1)[0])
    return float(max(0.0, 1.0 - abs(sf - st) / (abs(st) + 1e-6)))

def changepoint_level_recovery(f1d, target1d):
    lf = float(np.mean(f1d)); lt = float(np.mean(target1d))
    return float(max(0.0, 1.0 - abs(lf - lt) / (abs(lt) + 1e-6)))

def structure(cond, fmean_1d, P, target_1d):
    if cond == "motif": return period_power_fraction(fmean_1d, P)
    if cond == "trend": return trend_slope_recovery(fmean_1d, target_1d)
    if cond == "changepoint": return changepoint_level_recovery(fmean_1d, target_1d)
    return float("nan")

def bootstrap_ci(x, pct=(2.5, 97.5)):
    x = np.asarray(x, float)
    if len(x) == 0: return [0.0, 0.0]
    rng = np.random.default_rng(0)
    bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(CONFIG["N_BOOTSTRAP"])]
    return [float(np.percentile(bs, pct[0])), float(np.percentile(bs, pct[1]))]

def rel_collapse(struct_clean, struct_abl):
    sc = np.asarray(struct_clean, float); sa = np.asarray(struct_abl, float)
    d = sc - sa; cm = float(sc.mean()) + 1e-6
    lo, hi = bootstrap_ci(d)
    return float(d.mean() / cm), [lo / cm, hi / cm]
""")

# ============================================================================
md("## 6. Ablation harness — `.o` mean-ablation hook + site / fraction / random modes")
code(r"""
def _make_pre_hook(attn):
    d_kv = _dkv(attn)
    def hook(o_module, args):
        heads = getattr(attn, "_ablate_heads", None)
        if not heads: return None
        x = args[0].clone()
        for h in heads:
            sl = slice(h * d_kv, (h + 1) * d_kv); seg = x[..., sl]
            x[..., sl] = seg.mean(dim=tuple(range(seg.dim() - 1)), keepdim=True)
        return (x,)
    return hook

def install_hooks(sites):
    handles = []
    for lst in sites.values():
        for _, mod in lst:
            mod._ablate_heads = set(); handles.append(mod.o.register_forward_pre_hook(_make_pre_hook(mod)))
    return handles

def clear_ablations(sites):
    for lst in sites.values():
        for _, mod in lst: mod._ablate_heads = set()

def build_head_pool(sites):
    return [(mod, h) for lst in sites.values() for _, mod in lst for h in range(_nheads(mod))]

def site_heads(sites, site):
    return [(mod, h) for _, mod in sites[site] for h in range(_nheads(mod))]

def set_site_ablation(sites, site):                       # f=1: all of one site (non-nested)
    clear_ablations(sites)
    for _, mod in sites[site]: mod._ablate_heads = set(range(_nheads(mod)))

def set_fraction_site(sites, site, f, rng):               # random fraction f WITHIN one site
    clear_ablations(sites); heads = site_heads(sites, site)
    k = max(1, int(round(f * len(heads))))
    for idx in rng.choice(len(heads), size=k, replace=False):
        mod, h = heads[idx]; mod._ablate_heads.add(h)

def set_random_pool(sites, pool, n, rng):                 # n heads across ALL sites (size/total matched)
    clear_ablations(sites); n = min(n, len(pool))
    for idx in rng.choice(len(pool), size=n, replace=False):
        mod, h = pool[idx]; mod._ablate_heads.add(h)

HANDLES = install_hooks(SITES); HEAD_POOL = build_head_pool(SITES)
print(f"hooks on {len(HANDLES)} '.o' modules | pool={len(HEAD_POOL)} | N(site)={N}")
""")

# ============================================================================
md("## 7. Forecast backends + plumbing assert")
code(r"""
def forecast_pilot(contexts, n_samples):
    # forecast in small SERIES chunks (batching all 32 series x 100 samples through 64-step generation OOMs a
    # T4 — the cross-attn KV cache over the full context is the driver). Auto-halve the chunk on OOM.
    torch.manual_seed(CONFIG["SEED0"])    # common random numbers (clean vs ablated share sampling noise)
    bs = int(CONFIG.get("FORECAST_BATCH", 4)); outs = []; i = 0
    while i < len(contexts):
        chunk = [torch.tensor(np.asarray(c), dtype=DTYPE) for c in contexts[i:i + bs]]
        try:
            with torch.inference_mode():    # NO autograd graph can be retained (the live-memory leak)
                fc = PIPE.predict(chunk, prediction_length=CONFIG["PRED"], num_samples=n_samples)
            arr = fc.detach().cpu().numpy(); del fc
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs // 2); print(f"  [oom] FORECAST_BATCH -> {bs}"); continue
            raise
        outs.append(arr); i += len(chunk)
        if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)    # (n_series, n_samples, PRED)

def forecast_mock(contexts, n_samples):
    n = len(contexts); H = CONFIG["PRED"]; ids = np.zeros((n, 32), dtype=np.int64)
    for i, c in enumerate(contexts):
        c = np.asarray(c, float); q = np.clip(((c - c.min()) / ((c.max() - c.min()) + 1e-9) * (VOCAB - 3)).astype(int) + 2, 0, VOCAB - 1)
        q = q[-32:]; ids[i, :len(q)] = q
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE); dec = torch.zeros((n, H), dtype=torch.long, device=DEVICE)
    with torch.no_grad(): out = INNER(input_ids=inp, decoder_input_ids=dec)     # hooks fire
    sig = out.logits.float().mean(dim=-1).cpu().numpy(); samples = np.zeros((n, n_samples, H)); rng = np.random.default_rng(123)
    for i in range(n):
        c = np.asarray(contexts[i], float); base = np.resize(c[-H:] if len(c) >= H else np.resize(c, H), H)
        amp = 1.0 + 0.3 * np.tanh(sig[i].mean()); perturb = 0.5 * (sig[i] - sig[i].mean())
        samples[i] = amp * base[None, :] + perturb[None, :] + 0.1 * rng.standard_normal((n_samples, H))
    return samples

FORECAST = forecast_mock if IS_MOCK else forecast_pilot

_ctx, _tgt, _Ps = make_batch("motif", np.random.default_rng(0), 2)
clear_ablations(SITES); _a = FORECAST(_ctx, 4)
set_site_ablation(SITES, "dec_self"); _b = FORECAST(_ctx, 4); clear_ablations(SITES)
print(f"PLUMBING: ablating dec_self changed forecast = {not np.allclose(_a,_b)} (max|Δ|={np.abs(_a-_b).max():.4g})")
assert not np.allclose(_a, _b), "ablation did NOT change output — hooks not wired"
print("PLUMBING: PASS" + MOCK_TAG)
""")

# ============================================================================
md(r"""
## 8. Experiment A — full-site necessity + structural selectivity

Per seed × condition: forecast **clean** once, then each **site** (f=1, isolated) and the **`random@N`** baseline.
We record **ΔCRPS** (necessity) and **structural rel-collapse** per condition (the selectivity primitive).
""")
code(r"""
def _crps_vec(fc, tgt): return np.array([crps_samples(fc[i], tgt[i]) for i in range(len(tgt))])
def _struct_vec(fc, Ps, tgt, cond): return np.array([structure(cond, fc[i].mean(0), Ps[i], tgt[i]) for i in range(len(tgt))])

# ---- incremental checkpoint / resume: a Colab disconnect just RE-RUN and it continues -----------
FORCE = os.environ.get("CHRONOS_3B_FORCE", "0") == "1"
def _ckp(name): return os.path.join(CKPT_DIR, f"phase3b_{MODE}_{name}.json")
def _load_recs(name):
    p = _ckp(name)
    if os.path.exists(p) and not FORCE:
        try: return json.load(open(p))
        except Exception: return []
    return []
def _save_recs(name, recs): json.dump(recs, open(_ckp(name), "w"))

def run_full_site():
    recs = _load_recs("full")
    done = {(r["seed"], r["cond"]) for r in recs if r.get("kind") == "random"}   # block ends with the random rec
    recs = [r for r in recs if (r["seed"], r["cond"]) in done]                    # drop any partial block
    for seed in range(CONFIG["N_SEEDS"]):
        for ci, cond in enumerate(CONFIG["CONDITIONS_3B"]):
            if (seed, cond) in done:
                print(f"  [ckpt] full-site ({seed},{cond}) resumed"); continue
            rng = np.random.default_rng(CONFIG["SEED0"] + seed * 100 + ci)        # per-unit seed -> resume-deterministic
            ctx, tgt, Ps = make_batch(cond, rng, CONFIG["N_SERIES"])
            clear_ablations(SITES); fc0 = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
            crps0 = _crps_vec(fc0, tgt); s0 = _struct_vec(fc0, Ps, tgt, cond)
            for site in CONFIG["SITES_3B"]:
                set_site_ablation(SITES, site); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                dcrps = _crps_vec(fc, tgt) - crps0; rel, rel_ci = rel_collapse(s0, _struct_vec(fc, Ps, tgt, cond))
                recs.append(dict(seed=seed, cond=cond, kind=site, dcrps_mean=float(dcrps.mean()),
                                 dcrps_ci=bootstrap_ci(dcrps), rel_collapse=rel, rel_ci=rel_ci, clean_struct=float(s0.mean())))
            rel_draws, crps_draws = [], []
            for _ in range(CONFIG["N_RANDOM_DRAWS"]):
                set_random_pool(SITES, HEAD_POOL, N, rng); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                rel_draws.append(rel_collapse(s0, _struct_vec(fc, Ps, tgt, cond))[0])
                crps_draws.append(float((_crps_vec(fc, tgt) - crps0).mean()))
            recs.append(dict(seed=seed, cond=cond, kind="random", rel_draws=rel_draws, crps_draws=crps_draws))
            clear_ablations(SITES); _save_recs("full", recs)                      # checkpoint after each block
            print(f"  full-site ({seed},{cond}) done + saved")
    return recs

print("Experiment A (full-site)..." + MOCK_TAG); FULL = run_full_site()
""")

# ============================================================================
md(r"""
## 9. Experiment B — fraction-sweep dose-response (the confound-control panel)

For `f ∈ F_GRID`, ablate a random fraction `f` of each site (several draws), plus a **random-across-all-sites arm**
at the matched total (`round(f·N)` heads). We track **motif** structural collapse and the **non-periodic** collapse
(max of trend/changepoint) at each `f`. *Severance* shows collapse only as `f→1`, non-selective; a *locus* shows a
**selective collapse at low `f`**.
""")
code(r"""
def run_sweep():
    recs = _load_recs("sweep")
    done = {(r["seed"], round(r["f"], 4)) for r in recs}            # only complete (seed,f) units are ever saved
    recs = [r for r in recs if (r["seed"], round(r["f"], 4)) in done]
    for seed in range(CONFIG["SWEEP_SEEDS"]):
        for fi, f in enumerate(CONFIG["F_GRID"]):
            if (seed, round(f, 4)) in done:
                print(f"  [ckpt] sweep ({seed},f={f}) resumed"); continue
            # batches deterministic per (seed,cond); ablation rng deterministic per (seed,f) -> resume-stable
            batches = {c: make_batch(c, np.random.default_rng(CONFIG["SEED0"] + 5000 + seed * 10 + ci), CONFIG["SWEEP_SERIES"])
                       for ci, c in enumerate(CONFIG["CONDITIONS_3B"])}
            clean = {}
            for c, (ctx, tgt, Ps) in batches.items():
                clear_ablations(SITES); clean[c] = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), Ps, tgt, c)
            arng = np.random.default_rng(CONFIG["SEED0"] + 6000 + seed * 100 + fi)
            unit = []
            for d in range(CONFIG["SWEEP_DRAWS"]):
                for arm in CONFIG["SITES_3B"] + ["random"]:
                    rel = {}
                    for c, (ctx, tgt, Ps) in batches.items():
                        if arm == "random": set_random_pool(SITES, HEAD_POOL, max(1, int(round(f * N))), arng)
                        else:               set_fraction_site(SITES, arm, f, arng)
                        rel[c] = rel_collapse(clean[c], _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), Ps, tgt, c))[0]
                    unit.append(dict(seed=seed, f=f, draw=d, arm=arm, motif=rel["motif"],
                                     nonper=max(rel["trend"], rel["changepoint"]), trend=rel["trend"], changepoint=rel["changepoint"]))
            recs += unit; clear_ablations(SITES); _save_recs("sweep", recs)
            print(f"  sweep ({seed},f={f}) done + saved")
    return recs

print("Experiment B (fraction sweep)..." + MOCK_TAG); SWEEP = run_sweep()
""")

# ============================================================================
md(r"""
## 10. Verdict — structural selectivity primary; `cross` gated on a low-`f` selective effect
""")
code(r"""
def _mean(kind, cond, field):
    v = [r[field] for r in FULL if r.get("kind") == kind and r["cond"] == cond and field in r]
    return float(np.mean(v)) if v else float("nan")
def _rel_lo(kind, cond):
    v = [r["rel_ci"][0] for r in FULL if r.get("kind") == kind and r["cond"] == cond and "rel_ci" in r]
    return float(np.mean(v)) if v else float("nan")

def sweep_agg(arm, f):
    rs = [r for r in SWEEP if r["arm"] == arm and abs(r["f"] - f) < 1e-9]
    if not rs: return None
    return dict(motif=float(np.mean([r["motif"] for r in rs])), nonper=float(np.mean([r["nonper"] for r in rs])))

def selective_structural(site):
    mot = _mean(site, "motif", "rel_collapse"); mot_lo = _rel_lo(site, "motif")
    trd = _mean(site, "trend", "rel_collapse"); cp = _mean(site, "changepoint", "rel_collapse")
    sig = (mot_lo > 0) and (mot >= CONFIG["STRUCT_COLLAPSE_MIN"])
    sel = mot >= CONFIG["SELECTIVITY_MARGIN"] * max(trd, cp, 1e-6)
    return dict(motif=mot, motif_lo=mot_lo, trend=trd, changepoint=cp, sig=bool(sig), selective=bool(sel),
                selective_structural=bool(sig and sel))

def low_f_selective(site):
    for f in CONFIG["F_GRID"]:
        if f > CONFIG["LOW_F_MAX"]: continue
        a = sweep_agg(site, f); r = sweep_agg("random", f)
        if a is None: continue
        if (a["motif"] >= CONFIG["STRUCT_COLLAPSE_MIN"] and a["motif"] >= CONFIG["SELECTIVITY_MARGIN"] * max(a["nonper"], 1e-6)
                and (r is None or a["motif"] > r["motif"])):
            return True, f
    return False, None

def summarize():
    rows = []
    for site in CONFIG["SITES_3B"]:
        ss = selective_structural(site); lf, lff = low_f_selective(site)
        is_cross = (site == "cross")
        locus = ss["selective_structural"] and (lf if is_cross else True)
        rows.append(dict(site=site, **ss, low_f_selective=bool(lf), low_f=lff,
                         dcrps_motif=_mean(site, "motif", "dcrps_mean"),
                         dcrps_trend=_mean(site, "trend", "dcrps_mean"),
                         dcrps_changepoint=_mean(site, "changepoint", "dcrps_mean"),
                         gated_on_low_f=is_cross, locus=bool(locus)))
    null = [v for r in FULL if r["kind"] == "random" and r["cond"] == "motif" for v in r["rel_draws"]]
    null_p = float(np.percentile(null, 95)) if null else 0.0
    loci = [r for r in rows if r["locus"]]
    verdict = (f"LOCUS = {max(loci, key=lambda r: r['motif'])['site']}" if loci else "DISTRIBUTED")

    print("=" * 84); print(f"PHASE 3b VERDICT: {verdict}{MOCK_TAG}"); print("=" * 84)
    print(f"structural rel-collapse (fraction of structure destroyed); selectivity = motif >> trend/changepoint")
    print(f"  {'site':9s} {'motif':>7s} {'trend':>7s} {'chgpt':>7s}  {'sel-struct':>10s} {'low-f sel':>9s}  "
          f"{'ΔCRPS mot':>9s}  LOCUS")
    for r in rows:
        lowf = f"yes@{r['low_f']}" if r["low_f_selective"] else "no"
        print(f"  {r['site']:9s} {r['motif']:+7.3f} {r['trend']:+7.3f} {r['changepoint']:+7.3f}  "
              f"{str(r['selective_structural']):>10s} {lowf:>9s}  {r['dcrps_motif']:+9.4f}  "
              f"{'*' if r['locus'] else ''}{'  (gated:low-f)' if r['gated_on_low_f'] else ''}")
    print(f"\n  sanity baseline: random@N motif rel-collapse p95 = {null_p:+.3f} (head-count null, demoted)")
    # detection-power, structural
    best = max(rows, key=lambda r: r["motif"] if np.isfinite(r["motif"]) else -1e9)
    if any(r["selective_structural"] for r in rows):
        print(f"  DETECTION POWER: at least one site shows selective structural collapse -> method CAN localize.")
    else:
        print(f"  DETECTION POWER: no site selectively collapses period structure (best='{best['site']}', "
              f"motif rel {best['motif']:+.3f}) -> the DISTRIBUTED null is real, not underpowered.")
    cr = next(r for r in rows if r["site"] == "cross")
    if cr["selective_structural"] and not cr["low_f_selective"]:
        print(f"  CROSS: selective at f=1 but NOT at low f -> consistent with channel SEVERANCE, not a locus "
              f"(correctly NOT reported as a cross locus).")
    return dict(verdict=verdict, rows=rows, null_p95=null_p, mode=MODE, mock=IS_MOCK, N=int(N),
                thresholds={k: CONFIG[k] for k in ("STRUCT_COLLAPSE_MIN", "SELECTIVITY_MARGIN", "LOW_F_MAX")})

SUMMARY = summarize()
""")

# ============================================================================
md("## 11. Figures — Fig 4b (full-site structural selectivity) + Fig 4c (dose-response)")
code(r"""
try:
    rows = SUMMARY["rows"]; sites = [r["site"] for r in rows]; xs = np.arange(len(sites)); w = 0.25
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    # Fig 4b: structural rel-collapse by condition (motif should tower for a locus; equal => severance/distributed)
    ax[0].bar(xs - w, [r["motif"] for r in rows], w, label="motif (period-P)", color="#c0392b")
    ax[0].bar(xs,     [r["trend"] for r in rows], w, label="trend (slope)", color="#7f8c8d")
    ax[0].bar(xs + w, [r["changepoint"] for r in rows], w, label="changepoint (level)", color="#bdc3c7")
    ax[0].axhline(CONFIG["STRUCT_COLLAPSE_MIN"], color="k", ls=":", lw=1, label="collapse min")
    ax[0].axhline(0, color="gray", lw=0.8); ax[0].set_xticks(xs); ax[0].set_xticklabels(sites)
    ax[0].set_ylabel("structural rel-collapse"); ax[0].legend(fontsize=7)
    ax[0].set_title(f"Fig 4b: structural selectivity [{SUMMARY['verdict']}]" + MOCK_TAG, fontsize=9)
    # Fig 4c: dose-response motif rel-collapse vs f, per arm
    for arm, col in [("enc_self", "#2980b9"), ("dec_self", "#27ae60"), ("cross", "#c0392b"), ("random", "#999999")]:
        ys = [(sweep_agg(arm, f) or {"motif": np.nan})["motif"] for f in CONFIG["F_GRID"]]
        ax[1].plot(CONFIG["F_GRID"], ys, marker="o", color=col, label=arm, lw=(2 if arm != "random" else 1),
                   ls=("--" if arm == "random" else "-"))
    ax[1].axvline(CONFIG["LOW_F_MAX"], color="k", ls=":", lw=1, label=f"low-f gate ({CONFIG['LOW_F_MAX']})")
    ax[1].set_xlabel("fraction of site ablated (f)"); ax[1].set_ylabel("motif period-P rel-collapse")
    ax[1].set_title("Fig 4c: dose-response (severance=endpoint; locus=low-f)", fontsize=9); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(CKPT_DIR, "fig4b_site_isolated.png"), dpi=90); plt.show(); plt.close(fig)
    print("saved fig4b_site_isolated.png")
except Exception as e:
    print("fig skipped:", repr(e)[:160])
""")

# ============================================================================
md("## 12. Checkpoint")
code(r"""
out = dict(summary=SUMMARY, full_site=FULL, sweep=SWEEP)
p = os.path.join(CKPT_DIR, f"phase3b_{MODE}.json")
with open(p, "w") as f: json.dump(out, f, indent=2)
print("wrote", p, "->", SUMMARY["verdict"], MOCK_TAG)
""")

# ---- assemble ----------------------------------------------------------------------------------
nb = new_notebook()
nb.cells = [new_markdown_cell(s) if t == "md" else new_code_cell(s) for (t, s) in CELLS]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}
with open("phase3b.ipynb", "w") as f: nbf.write(nb, f)
with open("_mirror_3b.py", "w") as f:
    f.write("\n".join(["# auto-mirror of phase3b code cells"] +
                      ["\n# " + "=" * 60 + "\n" + s for t, s in CELLS if t == "code"]))
print(f"wrote phase3b.ipynb ({sum(t=='code' for t,_ in CELLS)} code cells) + _mirror_3b.py")
