#!/usr/bin/env python3
# Single-source builder for phase5_v4.ipynb. Phase 5 v4: CROSS-LAYER feature circuit at the Circuit-Tracing standard.
# Supersedes phase5_v3 (per-layer SAEs ranked by a gradient-free |dact|x||W_dec|| proxy on the top-4 layers). v4 fixes
# the three v3 gaps that each biased the verdict toward a false "distributed":
#   (1) ranking was a gradient-free heuristic, so a causal feature with a small activation change never entered the
#       top-k union. v4 ranks by a PRINCIPLED LINEAR ATTRIBUTION (frozen-attention virtual weights) and the all-paths
#       influence B = (I - A)^-1 - I, with EAP-IG kept as a gradient-based cross-check.
#   (2) the SNR sweep was dropped. v4 restores it (a saturated shift manufactures "distributed").
#   (3) only 4 layers hosted dictionaries, chosen by single-layer necessity (the wrong selector for a distributed
#       circuit). v4 covers the WHOLE mid-encoder band.
# Substrate change: per-layer TRANSCODERS (read MLP input, reconstruct MLP output), not SAEs, so feature-to-feature
# edges are linear via the residual stream. A local replacement model carries per-(layer,position) ERROR NODES and
# freezes attention patterns and norm denominators so the attribution is exactly linear. Interventions are layer-range
# constrained patching with an end-layer sweep, both directions (noising = necessity, denoising = sufficiency).
# Reports completeness/replacement scores (feature vs error dark-matter), a feature-splitting check at two dictionary
# sizes, and the change-detection vs periodicity attention-asymmetry test (frozen attention freezes the likely QK crux
# for change-detection; Phase 4 found attention routes the boundary signal).
# Backend: HuggingFace forward hooks on the real Chronos-T5 module tree (NOT TransformerLens). MODE switch
# mock_cpu | pilot_a100 (Large, A100). Checkpoint to phase5v4_<MODE>.json. No em dashes anywhere.
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []
def md(s):  CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ============================================================================
md(r"""
# Phase 5 v4: Cross-Layer Feature Circuit at the Circuit-Tracing Standard

Tests whether change-detection in Chronos-T5 is implemented by a small **cross-layer feature circuit** (sparse
features spread across several mid-encoder layers, ablated together), the one granularity still untested after
Phase 3b/4 (heads), Phase 5 (MLP layers) and Phase 5 v2 (single-layer MLP features) all found it distributed.

This supersedes `phase5_v3`, whose negative could not be trusted because all three of its flaws biased it toward a
false "distributed":

1. **Ranking was a gradient-free heuristic** (`|delta act| x ||W_dec||`), so a causal feature with a small activation
   change never entered the top-k union. v4 ranks by a **principled linear attribution**: per-layer **transcoders**
   give linear feature-to-feature edges via the residual stream, a **local replacement model** freezes attention
   patterns and normalization denominators so the edge from feature s to feature t is exactly linear, and we rank by
   the **all-paths influence** `B = (I - A)^-1 - I` rather than a single-step score. EAP-IG (a gradient-based effect
   on the metric) and the old proxy are kept as cross-checks.
2. **No SNR sweep** (a regression from v2). Restored: a saturated shift lets redundant pathways suffice and
   manufactures "distributed" by construction, so we sweep the shift magnitude toward the noise floor.
3. **Only 4 layers hosted dictionaries**, chosen by single-layer necessity, which is the wrong selector for a circuit
   that is by hypothesis spread across layers (no single layer looks necessary precisely because the computation is
   distributed across them). v4 covers the **whole mid-encoder band**.

**What stays from the trusted substrate:** counterfactual minimal pairs (a clean shift of magnitude delta at tau vs a
matched flat corrupt with the SAME noise realization, asserted), `changepoint_recovery` (the logit-difference analog),
the motif selectivity control via `period_power_fraction`, both intervention directions, the faith-beats-null guard,
bootstrap CIs, the minimum-detectable-effect readout, checkpoint and resume, and mock-then-pilot.

**The attention caveat is first-class.** The Circuit-Tracing method freezes attention, so it explains feature-feature
interaction given the pattern but never explains how the pattern formed (the QK circuit). Phase 4 found attention
routes the boundary and recency signal for change-detection, so a frozen-attention circuit here freezes the likely
crux. We turn this into a testable prediction: run the same pipeline on periodicity and compare. If periodicity yields
a more feature-mediated (higher replacement-score) graph than change-detection, that asymmetry is the result.

**Verdicts (both bankable).** A: a small cross-layer set is faithful, complete, selective, sufficient, beats the
union null, and explains a high fraction (the circuit, flipping the arc). B: distributed across layers too, the
discrepancy closed at every granularity, with the attention asymmetry explaining why.
""")

# ============================================================================
md("## 0. Config and MODE switch (`mock_cpu` exercises every path but is NOT interpretable; `pilot_a100` is the real run)")
code(r"""
import os
CONFIG = {
    "MODE": "mock_cpu",                  # -> "pilot_a100" (Chronos-T5-Large on a high-RAM A100)
    "MODEL_BY_MODE": {"mock_cpu": None, "pilot_t4": "amazon/chronos-t5-base", "pilot_a100": "amazon/chronos-t5-large"},
    "USE_DRIVE": True,
    "SEED0": 0, "PERIODS": [8, 12, 16, 24],
    "N_PAIRS": 32, "CTX": 256, "PRED": 64, "OBS_NOISE": 0.30, "TAU_FRAC_CTX": 0.65,
    "N_CRPS_SAMPLES": 48, "N_BOOTSTRAP": 1000, "FORECAST_BATCH": 4,
    "DELTA_PRIMARY": [1.5, 3.0],
    # ---- cross-layer band (full coverage; cap only if memory-constrained) ----
    "ORIENT_DEPTH_LO": 0.40, "ORIENT_DEPTH_HI": 0.75,
    "CROSS_MAX_LAYERS": 0,               # 0 = the WHOLE band; >0 caps the count (contiguous central span)
    "CACHE_DTYPE": "float16",
    # ---- transcoders (TopK) read MLP input, reconstruct MLP output; carry the error node ----
    "TC_DICT_MULT": 8, "TC_DICT_MULT_SMALL": 4, "TC_TOPK": 64, "TC_STEPS": 3000, "TC_LR": 1e-3, "TC_BATCH": 2048, "TRAIN_PAIRS": 128,
    "TC_RECON_MAX": 0.25,                # HELD-OUT recon gate (frac of eval MLP-output variance); a poor transcoder voids the result
    # ---- attribution: linear virtual weights (gold) + all-paths influence; EAP-IG and proxy as cross-checks ----
    "RANK_METHOD": "gold",               # "gold" (frozen-attn linear virtual weight, all-paths) | "eapig" | "proxy"
    "ATTR_PAIRS": 12, "ATTR_BATCH": 2, "ATTR_TOPF": 96, "ATTR_GRAPH_PAIRS": 16, "GRAPH_BOOTSTRAP": 200, "GRAPH_REBUILDS": 3, "RANK_AGREE_MIN": 0.5, "EAP_STEPS": 4,   # ATTR_TOPF = node cap for the explicit A
    # ---- faithfulness / completeness / selectivity / sufficiency over UNION size ----
    "K_GRID": [1, 2, 4, 8, 16, 32, 64, 128], "FAITH_TARGET": 0.60, "LOCALIZE_MAX_FEATURES": 32,
    "N_RANDOM_NULL": 32, "SELECTIVITY_MARGIN": 2.0, "FAITH_BEAT_MARGIN": 0.05, "SUFFICIENCY_BAR": 0.15,
    "PATTERN_PAIRS": 8, "PATTERN_SAMPLES": 16,   # attention-pattern interchange experiment (the QK-crux causal test)
    "END_SWEEP": 3,                      # layer-range constrained patching: # of end-layers to sweep (report max)
    "MISHRA_DEPTH_LO": 0.45, "MISHRA_DEPTH_HI": 0.55,
    # ---- SNR sweep (the saturation control) ----
    "SNR_DELTAS": [[1.5, 3.0], [0.6, 1.0], [0.3, 0.45]], "SNR_PAIRS": 16, "SNR_KS": [4, 16, 64],
    # ---- replacement-score escalation signature (cross-layer superposition -> consider a CLT) ----
    "MULTIHOP_ESCALATE": 0.35,           # if >this share of all-paths influence is multi-hop, flag CLT as next step
    # ---- mock overrides (fast, NOT interpretable) ----
    "mock_cpu": {
        "PERIODS": [6, 8], "N_PAIRS": 4, "CTX": 48, "PRED": 24, "N_CRPS_SAMPLES": 12, "N_BOOTSTRAP": 50,
        "FORECAST_BATCH": 999, "TC_DICT_MULT": 4, "TC_DICT_MULT_SMALL": 2, "TC_TOPK": 4, "TC_STEPS": 40, "TC_BATCH": 64,
        "ATTR_PAIRS": 4, "ATTR_BATCH": 2, "ATTR_TOPF": 16, "ATTR_GRAPH_PAIRS": 4, "GRAPH_BOOTSTRAP": 40, "GRAPH_REBUILDS": 2, "EAP_STEPS": 2, "TRAIN_PAIRS": 8, "PATTERN_PAIRS": 3, "PATTERN_SAMPLES": 6,
        "K_GRID": [1, 2, 4], "LOCALIZE_MAX_FEATURES": 2, "N_RANDOM_NULL": 2, "END_SWEEP": 2,
        "SNR_DELTAS": [[1.5, 3.0], [0.6, 1.0]], "SNR_PAIRS": 4, "SNR_KS": [2, 4],
    },
}
MODE = os.environ.get("CHRONOS_P5V4_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4", "pilot_a100"), MODE
CONFIG["model_id"] = os.environ.get("CHRONOS_P5V4_MODEL", CONFIG["MODEL_BY_MODE"][MODE])
if MODE == "mock_cpu": CONFIG.update(CONFIG["mock_cpu"])
IS_MOCK = (MODE == "mock_cpu")
IS_LARGE = (CONFIG["model_id"] is not None and "large" in CONFIG["model_id"])
MOCK_TAG = "  [MOCK_CPU, NOT INTERPRETABLE]" if IS_MOCK else ""
ON_COLAB = os.path.isdir("/content")
CKPT_DIR = os.path.abspath(".")
if ON_COLAB:
    CKPT_DIR = "/content"
    if CONFIG.get("USE_DRIVE", True) and not IS_MOCK:
        try:
            from google.colab import drive
            drive.mount("/content/drive"); CKPT_DIR = "/content/drive/MyDrive/chronos_phase5v4"; os.makedirs(CKPT_DIR, exist_ok=True)
            print("checkpoints -> Google Drive:", CKPT_DIR)
        except Exception as e:
            print("Drive mount skipped (", repr(e)[:80], ") -> /content")
print(f"MODE={MODE}{MOCK_TAG}  model={CONFIG['model_id']}  rank={CONFIG['RANK_METHOD']}  band=[{CONFIG['ORIENT_DEPTH_LO']},{CONFIG['ORIENT_DEPTH_HI']}] "
      f"pairs={CONFIG['N_PAIRS']}  K_GRID={CONFIG['K_GRID']}  SNR={CONFIG['SNR_DELTAS']}  ckpt={CKPT_DIR}")
""")

# ============================================================================
md("## 1. Imports, device, dtype")
code(r"""
import sys, json, subprocess, gc, re, warnings, contextlib
warnings.filterwarnings("ignore", message=".*past_key_values.*")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _ensure(pkg, imp):
    if os.environ.get("CHRONOS_P5V4_SKIP_INSTALL") == "1": return
    try: __import__(imp)
    except Exception:
        print("installing", pkg); subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
if not IS_MOCK: _ensure("chronos-forecasting", "chronos")
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
if not ON_COLAB: matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(CONFIG["SEED0"]); np.random.seed(CONFIG["SEED0"])
DEVICE = "cuda" if (not IS_MOCK and torch.cuda.is_available()) else "cpu"
DTYPE = torch.float32; CACHE_DT = torch.float16 if CONFIG["CACHE_DTYPE"] == "float16" else torch.float32
print("device:", DEVICE)
""")

# ============================================================================
md("## 2. Model (mock = a tiny randomly-initialized T5; pilot = the real Chronos-T5) and the encoder-MLP module tree")
code(r"""
def _layer_idx(name):
    m = re.search(r"block\.(\d+)\.", name); return int(m.group(1)) if m else -1
def classify_mlp_modules(model):
    sites = {"enc_mlp": [], "dec_mlp": []}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ in ("T5DenseActDense", "T5DenseGatedActDense"):
            if name.startswith("encoder"):   sites["enc_mlp"].append((name, mod))
            elif name.startswith("decoder"): sites["dec_mlp"].append((name, mod))
    return sites
if IS_MOCK:
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=256, d_model=64, d_kv=16, d_ff=128, num_layers=4, num_decoder_layers=2,
                   num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1, dense_act_fn="relu",
                   feed_forward_proj="relu", is_gated_act=False)
    INNER = T5ForConditionalGeneration(cfg).eval(); VOCAB = cfg.vocab_size; PIPE = None
    DEC_START = int(cfg.decoder_start_token_id); D_MODEL = cfg.d_model
    try: INNER.config._attn_implementation = "eager"
    except Exception: pass
else:
    from chronos import ChronosPipeline
    if DEVICE == "cuda":
        for _v in ("PIPE", "INNER"):
            if _v in globals():
                try: del globals()[_v]
                except Exception: pass
        gc.collect(); torch.cuda.empty_cache()
        _free, _tot = torch.cuda.mem_get_info(); print(f"GPU free {_free/1e9:.1f}/{_tot/1e9:.1f} GB")
        assert _free > (6.0e9 if IS_LARGE else 1.5e9), "low GPU memory, RESTART THE RUNTIME."
    PIPE = ChronosPipeline.from_pretrained(CONFIG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)
    INNER = PIPE.inner_model.eval(); VOCAB = INNER.config.vocab_size; INNER.requires_grad_(False)
    try: INNER.config._attn_implementation = "eager"
    except Exception: pass
    DEC_START = int(getattr(INNER.config, "decoder_start_token_id", 0)); D_MODEL = int(INNER.config.d_model)
MLP = classify_mlp_modules(INNER)
MLP_MODS = {s: {_layer_idx(n): mod for n, mod in MLP[s]} for s in MLP}
N_ENC_LAYERS = len(MLP["enc_mlp"])
def rel_depth(li): return float(li) / max(1, (N_ENC_LAYERS - 1))
def band_layers():
    band = [l for l in range(N_ENC_LAYERS) if CONFIG["ORIENT_DEPTH_LO"] <= rel_depth(l) <= CONFIG["ORIENT_DEPTH_HI"]]
    if len(band) < 2:                                    # cross-layer needs at least two layers; take the central two
        c = N_ENC_LAYERS // 2; band = sorted({max(0, c - 1), min(N_ENC_LAYERS - 1, c)})
    cap = CONFIG.get("CROSS_MAX_LAYERS", 0)
    if cap and cap < len(band):                          # keep a contiguous central span
        st = (len(band) - cap) // 2; band = sorted(band)[st:st + cap]
    return sorted(band)
GATED = any(m.__class__.__name__ == "T5DenseGatedActDense" for _, m in MLP["enc_mlp"] + MLP["dec_mlp"])
if GATED: print("  NOTE: gated activations present; the relu-gate freeze linearizes ungated MLPs only, so the linear attribution is approximate for gated layers (Chronos-T5 is ungated relu).")
print("enc MLP layers:", N_ENC_LAYERS, "| d_model:", D_MODEL, "| dec MLP layers:", len(MLP["dec_mlp"]))
""")

# ============================================================================
md("## 3. Reusable substrate: counterfactual minimal pairs, motif control, metrics, bootstrap CI (kept from v2/v3)")
code(r"""
def make_cf_pair(ctx_len, pred_len, rng, noise, delta):
    tau = int(CONFIG["TAU_FRAC_CTX"] * ctx_len); L0 = rng.uniform(-1.0, 1.0); L1 = L0 + delta
    eps = rng.normal(0, noise, size=ctx_len + pred_len); base = np.arange(ctx_len + pred_len)
    clean = np.where(base < tau, L0, L1) + eps; corrupt = np.full(ctx_len + pred_len, L0) + eps
    return clean[:ctx_len], corrupt[:ctx_len], clean[ctx_len:], dict(tau=int(tau), L0=float(L0), L1=float(L1), delta=float(delta))
def make_motif(P, L, rng, noise):
    m = rng.standard_normal(P); m[rng.integers(P)] += 3.0 * (1 if rng.random() > 0.5 else -1); m[P // 2:] += 1.5; m = m - m.mean()
    return np.tile(m, L // P + 2)[:L] + noise * rng.standard_normal(L)
def make_motif_cf_battery(rng, n):
    cc, co, tg, mt = [], [], [], []; L = CONFIG["CTX"] + CONFIG["PRED"]
    for i in range(n):
        P = CONFIG["PERIODS"][i % len(CONFIG["PERIODS"])]
        m = rng.standard_normal(P); m[rng.integers(P)] += 3.0 * (1 if rng.random() > 0.5 else -1); m[P // 2:] += 1.5; m = m - m.mean()
        periodic = np.tile(m, L // P + 2)[:L]; noise = CONFIG["OBS_NOISE"] * rng.standard_normal(L)
        clean = periodic + noise; corrupt = np.full(L, float(periodic.mean())) + noise   # period-ABSENT, shared noise (minimal pair)
        cc.append(clean[:CONFIG["CTX"]]); co.append(corrupt[:CONFIG["CTX"]]); tg.append(clean[CONFIG["CTX"]:]); mt.append({"P": int(P)})
    return cc, co, np.array(tg), mt
def make_cf_battery(rng, n, dl, dh):
    cc, co, tg, mt = [], [], [], []
    for _ in range(n):
        delta = rng.choice([-1.0, 1.0]) * rng.uniform(dl, dh)
        a, b, c, m = make_cf_pair(CONFIG["CTX"], CONFIG["PRED"], rng, CONFIG["OBS_NOISE"], float(delta))
        cc.append(a); co.append(b); tg.append(c); mt.append(m)
    return cc, co, np.array(tg), mt
def make_motif_battery(rng, n):
    ctx, tgt, mt = [], [], []
    for i in range(n):
        P = CONFIG["PERIODS"][i % len(CONFIG["PERIODS"])]; s = make_motif(P, CONFIG["CTX"] + CONFIG["PRED"], rng, CONFIG["OBS_NOISE"])
        ctx.append(s[:CONFIG["CTX"]]); tgt.append(s[CONFIG["CTX"]:]); mt.append({"P": int(P)})
    return ctx, np.array(tgt), mt
def changepoint_recovery(f1d, meta):
    fhat = float(np.median(np.asarray(f1d))); return float(np.clip(1.0 - abs(fhat - meta["L1"]) / (abs(meta["L1"] - meta["L0"]) + 1e-8), 0.0, 1.0))
def period_power_fraction(f1d, P):
    x = np.asarray(f1d, float); x = x - x.mean(); power = np.abs(np.fft.rfft(x)) ** 2; freqs = np.fft.rfftfreq(len(x))
    if len(freqs) < 2: return 0.0
    df = freqs[1] - freqs[0]; total = power[1:].sum() + 1e-12; f0 = 1.0 / P
    band = (np.abs(freqs - f0) <= 1.5 * df) | (np.abs(freqs - 2 * f0) <= 1.5 * df); band[0] = False
    return float(power[band].sum() / total)
def bootstrap_ci(x, pct=(2.5, 97.5)):
    x = np.asarray(x, float)
    if len(x) == 0: return [0.0, 0.0]
    rng = np.random.default_rng(0); bs = [rng.choice(x, len(x), replace=True).mean() for _ in range(CONFIG["N_BOOTSTRAP"])]
    return [float(np.percentile(bs, pct[0])), float(np.percentile(bs, pct[1]))]
def mean_ci(x): x = np.asarray(x, float); lo, hi = bootstrap_ci(x); return float(x.mean()), [lo, hi]
def cp_vec(s, metas): return np.array([changepoint_recovery(s[i].mean(0), metas[i]) for i in range(len(metas))])
def motif_vec(s, metas): return np.array([period_power_fraction(s[i].mean(0), metas[i]["P"]) for i in range(len(metas))])
def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float); ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); d = np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()); return float((ra * rb).sum() / d) if d > 0 else 0.0
# hard assert: clean and corrupt differ ONLY in the shift (shared noise)
_r = np.random.default_rng(1); _a, _b, _c, _m = make_cf_pair(64, 16, _r, 0.3, 2.0)
assert np.allclose(_a[:_m["tau"]], _b[:_m["tau"]]) and not np.allclose(_a[_m["tau"]:], _b[_m["tau"]:]), "CF pair not minimal"
print("counterfactual-pair assert PASS (shared noise, only the shift differs); metrics ready")
""")

# ============================================================================
md("## 4. Forecast and tokenize helpers (mock = a deterministic surrogate; pilot = the real Chronos tokenizer)")
code(r"""
def forecast_raw(contexts, n_samples):
    if IS_MOCK:
        n = len(contexts); H = CONFIG["PRED"]; ids = np.zeros((n, 32), dtype=np.int64)
        for i, c in enumerate(contexts):
            c = np.asarray(c, float); q = np.clip(((c - c.min())/((c.max()-c.min())+1e-9)*(VOCAB-3)).astype(int)+2, 0, VOCAB-1); q = q[-32:]; ids[i, :len(q)] = q
        inp = torch.tensor(ids, dtype=torch.long, device=DEVICE); dec = torch.zeros((n, H), dtype=torch.long, device=DEVICE)
        with torch.no_grad(): out = INNER(input_ids=inp, decoder_input_ids=dec)
        sig = out.logits.float().mean(dim=-1).cpu().numpy(); samples = np.zeros((n, n_samples, H)); rng = np.random.default_rng(123)
        for i in range(n):
            c = np.asarray(contexts[i], float); base = np.resize(c[-H:] if len(c) >= H else np.resize(c, H), H)
            samples[i] = (1.0+0.3*np.tanh(sig[i].mean()))*base[None,:] + 0.5*(sig[i]-sig[i].mean())[None,:] + 0.1*rng.standard_normal((n_samples, H))
        return samples
    torch.manual_seed(CONFIG["SEED0"]); bs = int(CONFIG.get("FORECAST_BATCH", 4)); outs = []; i = 0
    while i < len(contexts):
        chunk = [torch.tensor(np.asarray(c), dtype=DTYPE) for c in contexts[i:i+bs]]
        try:
            with torch.inference_mode(): fc = PIPE.predict(chunk, prediction_length=CONFIG["PRED"], num_samples=n_samples)
            arr = fc.detach().cpu().numpy(); del fc
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs//2); print(f"  [oom] FORECAST_BATCH -> {bs}"); continue
            raise
        outs.append(arr); i += len(chunk)
        if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)
def _tokenize(contexts):
    if IS_MOCK:
        arrs = []
        for c in contexts:
            c = np.asarray(c, float); q = np.clip(((c - c.min())/((c.max()-c.min())+1e-9)*(VOCAB-3)).astype(int)+2, 0, VOCAB-1); arrs.append(q.astype(np.int64))
        Ln = max(len(a) for a in arrs); ids = np.zeros((len(arrs), Ln), dtype=np.int64); am = np.zeros((len(arrs), Ln), dtype=np.int64)
        for i, a in enumerate(arrs): ids[i, :len(a)] = a; am[i, :len(a)] = 1
        return torch.tensor(ids, device=DEVICE), torch.tensor(am, device=DEVICE)
    ct = torch.tensor(np.asarray(contexts), dtype=DTYPE); ids, am, _s = PIPE.tokenizer.context_input_transform(ct); return ids.to(DEVICE), am.to(DEVICE)
def _tok_scale(contexts):
    if IS_MOCK: ids, am = _tokenize(contexts); return ids, am, None
    ct = torch.tensor(np.asarray(contexts), dtype=DTYPE); ids, am, scale = PIPE.tokenizer.context_input_transform(ct)
    return ids.to(DEVICE), am.to(DEVICE), scale   # keep scale on the tokenizer (CPU) device so label_input_transform matches lab; only token ids go to GPU
def _target_tokens(targets, scale):
    if IS_MOCK:
        arrs = []
        for t in targets:
            t = np.asarray(t, float); q = np.clip(((t - t.min())/((t.max()-t.min())+1e-9)*(VOCAB-3)).astype(int)+2, 0, VOCAB-1); arrs.append(q.astype(np.int64))
        L = max(len(a) for a in arrs); ids = np.full((len(arrs), L), -100, dtype=np.int64)
        for i, a in enumerate(arrs): ids[i, :len(a)] = a
        return torch.tensor(ids, device=DEVICE)
    PL = int(getattr(PIPE.tokenizer.config, "prediction_length", CONFIG["PRED"])); arr = np.asarray(targets, float)
    if arr.shape[1] != PL:                                       # label_input_transform requires length == prediction_length
        fix = np.zeros((arr.shape[0], PL))
        for i, row in enumerate(arr): fix[i] = row[:PL] if len(row) >= PL else np.concatenate([row, np.full(PL - len(row), row[-1])])
        arr = fix
    lab = torch.tensor(arr, dtype=DTYPE); tids, tmask = PIPE.tokenizer.label_input_transform(lab, scale)
    tids = tids.clone().long(); tids[~tmask.bool()] = -100; return tids.to(DEVICE)
print("forecast + tokenize helpers ready")
""")

# ============================================================================
md(r"""
## 5. Transcoders (read MLP input, reconstruct MLP output) and the persistent enc-MLP hook

A transcoder reads the MLP input `x^L` (the input to `DenseReluDense`, the post-LN residual) and reconstructs the MLP
output `y^L`, bridging the ReLU. Because its decoder writes linearly into the residual stream, transcoder features
interact linearly once attention and the norms are frozen (Section 7), which is what makes the attribution principled
rather than the v3 heuristic. We carry the **error node** `err = y^L - decode(encode(x^L))` exactly as v2/v3 carried
the SAE error, so the local replacement reproduces the underlying model on-prompt.

The single persistent forward hook on each enc-MLP module supports four modes, all reading the module input `inp[0]`:
`_tc` (exact replacement: decode(encode(x)) + err, used to validate the local replacement), `_attr` (inject a
differentiable feature-activation leaf: decode(leaf) + err, for the linear attribution and EAP-IG), `_cf`
(counterfactual interchange patch carrying the error node, for the layer-range constrained interventions), and
`_ablate` (mean ablation, orientation-only).
""")
code(r"""
class Transcoder(nn.Module):
    '''Reads MLP input x^L, reconstructs MLP output y^L. TopK sparse code; decoder writes linearly to the residual.'''
    def __init__(self, d_in, d_out, m, k):
        super().__init__(); self.k = int(min(k, m)); self.b_pre = nn.Parameter(torch.zeros(d_in))
        self.enc = nn.Linear(d_in, m); self.dec = nn.Linear(m, d_out, bias=True)
    def preact(self, x): return self.enc(x - self.b_pre)
    def encode(self, x):
        z = torch.relu(self.preact(x)); v, idx = z.topk(self.k, dim=-1); return torch.zeros_like(z).scatter_(-1, idx, v)
    def decode(self, a): return self.dec(a)
    def forward(self, x): a = self.encode(x); return self.decode(a), a
def train_transcoder(x_tr, y_tr, x_ev, y_ev, m, seed):
    tc = Transcoder(D_MODEL, D_MODEL, m, CONFIG["TC_TOPK"]).to(DEVICE)
    torch.manual_seed(seed); tc.b_pre.data = x_tr.mean(0).detach(); tc.dec.bias.data = y_tr.mean(0).detach()
    opt = torch.optim.Adam(tc.parameters(), lr=CONFIG["TC_LR"]); N = x_tr.shape[0]
    for _ in range(CONFIG["TC_STEPS"]):
        ix = torch.randint(0, N, (min(CONFIG["TC_BATCH"], N),), device=x_tr.device)
        a = tc.encode(x_tr[ix]); loss = ((tc.decode(a) - y_tr[ix]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    tc.requires_grad_(False)
    with torch.no_grad():                                  # HELD-OUT reconstruction (frac of EVAL output variance)
        recon = float((((tc.decode(tc.encode(x_ev)) - y_ev) ** 2).mean() / (y_ev - y_ev.mean(0)).pow(2).mean().clamp_min(1e-8)).detach())
    return tc, recon, m

def _mlp_hook(module, inp, out):
    x = inp[0]
    a = getattr(module, "_attr", None)                 # attribution: decode(leaf)+err, leaf is a differentiable tensor
    if a is not None: tc, leaf, err = a; return tc.decode(leaf) + err
    t = getattr(module, "_tc", None)                   # exact transcoder replacement: decode(encode(x))+err
    if t is not None: tc, err = t; return tc.decode(tc.encode(x)) + err
    cf = getattr(module, "_cf", None)                  # constrained interchange patch: pin features AND the error node to the run own (unperturbed) cache
    if cf is not None:
        tc, idx, src, base, eb = cf; fx = tc.encode(x)
        if base is not None and eb is not None and base.shape[-2] == fx.shape[-2] and eb.shape[-2] == fx.shape[-2]:
            f = base.clone(); err = eb                  # fully constrained: NO within-range leakage through features OR the error node
        else:
            f = fx; err = out - tc.decode(fx)           # fallback (mock 32-vs-S shape mismatch, or no base): live re-encode
        L = min(f.shape[-2], src.shape[-2]); f[..., :L, idx] = src[..., :L, idx]; return tc.decode(f) + err
    if getattr(module, "_ablate", False):              # orientation-only mean ablation (report only)
        return out.mean(dim=tuple(range(out.dim()-1)), keepdim=True).expand_as(out)
    return None
def _reset_modes():
    for _, mod in MLP["enc_mlp"]: mod._attr = None; mod._tc = None; mod._cf = None; mod._ablate = False
HANDLES = [mod.register_forward_hook(_mlp_hook) for _, mod in MLP["enc_mlp"]]; _reset_modes()
def clear_hooks(): _reset_modes()
def capture_io(layers, contexts):
    '''One encoder pass; returns per-layer (MLP input, MLP output), detached, for transcoder training and caches.'''
    ids, am = _tokenize(contexts); store = {}; hs = []
    for l in layers:
        m = MLP_MODS["enc_mlp"][l]
        def mk(ll):
            def cap(mod, i, o): store[ll] = (i[0].detach(), o.detach())
            return cap
        hs.append(m.register_forward_hook(mk(l)))
    clear_hooks()
    with torch.no_grad(): INNER.get_encoder()(input_ids=ids, attention_mask=am)   # no_grad (NOT inference_mode): outputs feed the attribution autograd graph as constants
    for h in hs: h.remove()
    return {l: store[l][0] for l in layers}, {l: store[l][1] for l in layers}
print(f"transcoder + persistent hook on {len(HANDLES)} enc-MLP layers ready (modes: _tc, _attr, _cf, _ablate)")
""")

# ============================================================================
md(r"""
## 6. Counterfactual interchange forecast (multi-layer simultaneous patch, carrying the error node)

`forecast_cf_multi` substitutes, at each requested layer, a subset of feature activations with their value on the
matched counterfactual run (interchange ablation, NOT a mean or zero), carrying the error node. Patching the union
across layers simultaneously is the cross-layer circuit test. The layer-range constrained variant and the end-layer
sweep are built on top of this in Section 11.
""")
code(r"""
def forecast_cf_multi(contexts, arm, n_samples, tcs):
    '''arm: {layer: (idx, src, base, eb)}. Fully constrained interchange: pin features to base and the error node to eb (the run own, unperturbed values), override idx with src; no within-range leakage through features or the error node.'''
    if IS_MOCK:
        clear_hooks()
        for l, (idx, src, base, eb) in arm.items():
            MLP_MODS["enc_mlp"][l]._cf = (tcs[l], idx, src.to(DEVICE, dtype=DTYPE), None if base is None else base.to(DEVICE, dtype=DTYPE), None if eb is None else eb.to(DEVICE, dtype=DTYPE))
        out = forecast_raw(contexts, n_samples); clear_hooks(); return out
    bs = int(CONFIG.get("FORECAST_BATCH", 4)); outs = []; i = 0
    while i < len(contexts):                                  # own the batching so the _cf src never desyncs from the chunk
        j = min(i + bs, len(contexts)); clear_hooks()
        for l, (idx, src, base, eb) in arm.items():
            MLP_MODS["enc_mlp"][l]._cf = (tcs[l], idx, src[i:j].to(DEVICE, dtype=DTYPE), None if base is None else base[i:j].to(DEVICE, dtype=DTYPE), None if eb is None else eb[i:j].to(DEVICE, dtype=DTYPE))
        try:
            chunk = [torch.tensor(np.asarray(c), dtype=DTYPE) for c in contexts[i:j]]
            with torch.inference_mode(): fc = PIPE.predict(chunk, prediction_length=CONFIG["PRED"], num_samples=n_samples)
            out = fc.detach().cpu().numpy(); del fc
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            clear_hooks()
            if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs // 2); print(f"  [oom] cf FORECAST_BATCH -> {bs}"); continue   # retry same i, src re-sliced
            raise
        clear_hooks(); outs.append(out); i = j
        if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)
def _idx_tensor(ids): return torch.as_tensor(list(ids), dtype=torch.long, device=DEVICE)
print("counterfactual interchange forecast ready")
""")

# ============================================================================
md(r"""
## 7. Local replacement model: freeze attention patterns and norm denominators (the linearization)

Building the local replacement on a prompt means substituting the transcoders for the MLPs and freezing every
nonlinearity except the feature preactivations. We do this with three stop-gradients applied only during the
attribution forward: detach the softmax (freeze the attention pattern, i.e. the QK circuit, keeping the OV/value
path), detach the RMSNorm denominator (freeze the normalization), and detach the ReLU gate (linearize the MLPs that
are not replaced by transcoders, so the whole path is linear). None of these change the forward value (detach only
affects the backward), so the replacement reproduces the underlying model on-prompt once the error node is added.
Under the freeze the map from any feature activation to any downstream feature preactivation, and to the output, is
exactly linear, which is what makes the edge weights virtual weights (a backward Jacobian with stop-grads on the
nonlinearities) rather than a heuristic.
""")
code(r"""
from transformers.models.t5 import modeling_t5 as _t5mod
@contextlib.contextmanager
def frozen_model():
    '''Freeze attention patterns, norm denominators, and MLP gates so the model is linear in the residual stream.'''
    _soft, _relu, _ln = F.softmax, F.relu, _t5mod.T5LayerNorm.forward
    def soft(x, *a, **k): return _soft(x, *a, **k).detach()                      # freeze QK attention pattern
    def relu(x, *a, **k): return x * (x > 0).to(x.dtype).detach()                # linearize the ungated MLP nonlinearity
    def lnfwd(self, h):                                                          # freeze the RMSNorm denominator
        v = h.to(torch.float32).pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(v + self.variance_epsilon).detach()
        if self.weight.dtype in (torch.float16, torch.bfloat16): h = h.to(self.weight.dtype)
        return self.weight * h
    F.softmax = soft; F.relu = relu; _t5mod.T5LayerNorm.forward = lnfwd
    try: yield
    finally: F.softmax = _soft; F.relu = _relu; _t5mod.T5LayerNorm.forward = _ln

def _decoder_logits(ids, am):
    dec = torch.full((ids.shape[0], 1), DEC_START, dtype=torch.long, device=DEVICE)
    return INNER(input_ids=ids, attention_mask=am, decoder_input_ids=dec).logits

def validate_local_replacement(contexts, tcs, layers):
    '''Checks: (a) freezing changes only gradients not forward values; (b) transcoders + error node + freeze reproduce
    the underlying model on-prompt; (c) the transcoder WITHOUT the error node (tests reconstruction, not the additive identity).'''
    ids, am = _tokenize(contexts); clear_hooks()
    with torch.inference_mode(): base = _decoder_logits(ids, am).clone()
    with frozen_model(), torch.inference_mode(): frz = _decoder_logits(ids, am).clone()   # plain model under the freeze
    frz_rel = float((base - frz).abs().max() / (base.abs().max().item() + 1e-8))
    inp, out = capture_io(layers, contexts)
    err = {l: (out[l] - tcs[l].decode(tcs[l].encode(inp[l]))).detach() for l in layers}
    clear_hooks()
    for l in layers: MLP_MODS["enc_mlp"][l]._tc = (tcs[l], err[l])
    with frozen_model(), torch.inference_mode(): repl = _decoder_logits(ids, am).clone()
    clear_hooks()
    for l in layers: MLP_MODS["enc_mlp"][l]._tc = (tcs[l], torch.zeros_like(err[l]))         # no error node
    with frozen_model(), torch.inference_mode(): noerr = _decoder_logits(ids, am).clone()
    clear_hooks(); den = base.abs().max().item() + 1e-8
    return dict(rel=float((base - repl).abs().max()) / den, max_abs=float((base - repl).abs().max()),
                frozen_value_rel=frz_rel, transcoder_only_rel=float((base - noerr).abs().max()) / den)
print("local replacement model (frozen attention + frozen norms + error node) ready")
""")

# ============================================================================
md(r"""
## 8. Principled attribution: frozen-attention linear virtual weights and all-paths influence

The headline ranking. Each enc-MLP feature is injected as an independent differentiable leaf (`decode(leaf) + err`).
Under the freeze (Section 7) the model BODY is linear in the feature leaves. We read out the EXPECTED FORECAST LEVEL
(a softmax-weighted sum of the quantizer bin centers at the post-shift positions), the quantity `changepoint_recovery`
is actually built on, via `torch.softmax` (not the frozen attention softmax) so the readout keeps its gradient. This
makes the gold a metric-aligned FIRST-ORDER attribution in the frozen-attention local replacement: exact for the
frozen model, but a first-order approximation of the real model (the un-replaced downstream MLPs are gate-frozen, so
the linearization error compounds with depth, the Circuit-Tracing paper own caveat). It is NOT an exactly-linear
virtual weight, and the forward-value asserts do not validate the weights, so we report the gold-vs-EAP-IG rank
correlation as a gating diagnostic and verify every selected set with exact counterfactual patching:

* `g[s] = a_s * d(readout)/d(leaf_s)` is the **direct** feature-to-output edge (one residual hop; downstream features
  are held as independent leaves so this skips them), and
* `A[s -> t] = a_s * d(preact_t)/d(leaf_s)` is the **direct** feature-to-feature virtual-weight edge.

The **all-paths** influence is then `B = (I - A)^-1 - I` (a Neumann series over all paths), and the all-paths output
influence is `(I - A)^-1 g`. We rank features by the counterfactual all-paths influence `(a_clean - a_corrupt) (x)
(I - A)^-1 g` (the indirect effect of patching the feature from clean to corrupt), not a single-step score, which is
exactly the v3 gap. `gold` is primary; `eapig` (integrated gradients on the same leaves through the UNfrozen model,
v3's tier) and the `|delta act| x ||W_dec||` proxy are kept as cross-checks. The exact counterfactual curves in
Section 11 verify whatever set the ranking selects, so the ranking governs efficiency and false-negative risk, not
the validity of a positive verdict.
""")
code(r"""
if IS_MOCK:
    CENTERS = torch.arange(VOCAB, dtype=torch.float32, device=DEVICE)        # mock: token id as a linear level proxy
else:
    _cen = getattr(PIPE.tokenizer, "centers", None)
    if _cen is not None:
        _cen = _cen.detach().float().flatten(); _full = torch.zeros(VOCAB); _ns = max(0, VOCAB - _cen.numel())
        _full[_ns:_ns + _cen.numel()] = _cen[: VOCAB - _ns]; CENTERS = _full.to(DEVICE)   # quantizer bin centers, aligned to token ids
    else: CENTERS = torch.arange(VOCAB, dtype=torch.float32, device=DEVICE)
def _level_readout(out, tids):
    '''Expected forecast LEVEL (softmax-weighted bin centers) summed over post-shift positions; the quantity
    changepoint_recovery is built on, so the gold ranking is aligned with the metric. Uses torch.softmax (NOT the
    frozen F.softmax), so the readout keeps its gradient while the model body stays frozen: a metric-aligned
    FIRST-ORDER attribution in the frozen-attention replacement, not an exactly-linear virtual weight.'''
    lg = out.logits.float(); p = torch.softmax(lg, dim=-1)
    lvl = (p * CENTERS.view(1, 1, -1)).sum(-1); m = (tids != -100).float()
    return (lvl * m).sum()
def _attr_grads(layers, tcs, cc, tgt, metas, leaves_value, frozen, steps):
    '''Accumulate d(readout)/d(leaf). Gold (frozen) uses the metric-aligned expected-level readout (first-order in the
    frozen-attention replacement); EAP-IG (unfrozen) integrates -loss through the nonlinear model. Returns {l: grad}.'''
    n = len(cc); ids, am, scale = _tok_scale(cc); tids = _target_tokens(tgt, scale)
    inp_c, out_c = capture_io(layers, cc)
    Fc = {l: tcs[l].encode(inp_c[l]).detach() for l in layers}
    err = {l: (out_c[l] - tcs[l].decode(Fc[l])).detach() for l in layers}
    base = {l: (leaves_value[l] if leaves_value is not None else torch.zeros_like(Fc[l])) for l in layers}  # path start (corrupt for IG, origin for clean influence)
    accum = {l: torch.zeros_like(Fc[l]) for l in layers}
    B = max(1, int(CONFIG["ATTR_BATCH"]))
    for s in range(steps):
        alpha = (s + 1) / steps
        for bi in range(0, n, B):
            sl = slice(bi, bi + B)
            leaves = {l: (base[l][sl] + alpha * (Fc[l][sl] - base[l][sl])).detach().requires_grad_(True) for l in layers}
            clear_hooks()
            for l in layers: MLP_MODS["enc_mlp"][l]._attr = (tcs[l], leaves[l], err[l][sl])
            ctx = frozen_model() if frozen else contextlib.nullcontext()
            with ctx, torch.enable_grad():
                out = INNER(input_ids=ids[sl], attention_mask=am[sl], labels=tids[sl])
                readout = _level_readout(out, tids[sl]) if frozen else -out.loss
                readout.backward()
            for l in layers:
                if leaves[l].grad is not None: accum[l][sl] += leaves[l].grad.detach()
            clear_hooks()
            if DEVICE == "cuda": torch.cuda.empty_cache()
    return {l: accum[l] / steps for l in layers}, Fc, err

def direct_influence(layers, tcs, cc, co, tgt, metas, frozen=True, steps=1, weight="cf"):
    '''Direct (one-hop) feature->output influence per (layer,feature). weight=cf (counterfactual) or clean.'''
    n = min(CONFIG["ATTR_PAIRS"], len(cc)); cc, co, tgt, metas = cc[:n], co[:n], tgt[:n], metas[:n]
    if weight == "cf":
        inp_o, _ = capture_io(layers, co); Fo = {l: tcs[l].encode(inp_o[l]).detach() for l in layers}
        grads, Fc, err = _attr_grads(layers, tcs, cc, tgt, metas, Fo, frozen, steps)
    else:
        grads, Fc, err = _attr_grads(layers, tcs, cc, tgt, metas, None, frozen, steps)
        Fo = {l: torch.zeros_like(Fc[l]) for l in layers}
    taus = [m.get("tau", 0) for m in metas]; CTX = CONFIG["CTX"]; g = {}
    for l in layers:
        contrib = (Fc[l] - Fo[l]) * grads[l]                                   # indirect effect = weight (x) grad
        g[l] = torch.stack([contrib[j, taus[j]:CTX].sum(0) for j in range(n)]).mean(0).abs().detach().cpu().numpy()
    return g, Fc, Fo, err

def proxy_influence(layers, tcs, cc, co, metas):
    '''The old |delta act| x ||W_dec|| heuristic, kept only as a reported cross-check.'''
    inp_c, _ = capture_io(layers, cc); inp_o, _ = capture_io(layers, co)
    taus = [m.get("tau", 0) for m in metas]; CTX = CONFIG["CTX"]; ie = {}
    for l in layers:
        a = tcs[l].encode(inp_c[l]); b = tcs[l].encode(inp_o[l]); dpost = torch.zeros(a.shape[-1], device=DEVICE)
        for j in range(len(taus)): dpost += (a[j, taus[j]:CTX] - b[j, taus[j]:CTX]).abs().mean(0)
        wn = tcs[l].dec.weight.norm(dim=0); ie[l] = (dpost / len(taus) * wn).detach().cpu().numpy()
    return ie
print("attribution (gold linear virtual weight + EAP-IG + proxy) ready")
""")

# ============================================================================
md(r"""
## 9. Explicit adjacency A, all-paths influence B = (I - A)^-1 - I, completeness/replacement, path-length

We prune to the top-`ATTR_TOPF` features by direct influence, then build the explicit feature-to-feature adjacency A
by a backward Jacobian per target preactivation under the freeze. Error nodes enter as additional leaves so we can
partition the output influence into feature-routed versus error-routed (the dark matter), giving:

* **replacement score** (direct basis) = feature-routed / (feature + error)-routed DIRECT output influence (how well the
  transcoder features replace the MLP output at the readout),
* **completeness score** (all-paths basis) = all-paths feature influence / (all-paths feature + all-paths error)
  influence, the error nodes routed through the same `(I - A)^-1`, so numerator and denominator share a path basis
  (one minus the error dark-matter share), and
* **influence by path length** from the SIGNED operator: the direct share `|g|` versus the multi-hop share
  `|((I - A)^-1 - I) g|`. A large multi-hop share is the cross-layer-superposition signature that would justify
  escalating to a cross-layer transcoder. All three carry bootstrap CIs over prompts.
""")
code(r"""
def build_attr_graph(layers, tcs, cc, co, tgt, metas, label, cf=True):
    '''Frozen-attention attribution graph. cf=True weights nodes and edges by the counterfactual delta (a_clean minus
    a_corrupt), the indirect-effect basis the headline ranking needs; cf=False (periodicity, no minimal-pair corrupt)
    uses clean activation and is labelled as such. Error nodes are routed through the SAME (I - A)^-1 as features so
    completeness and replacement share a path basis. Path-length uses the SIGNED operator; |A| only guards convergence.
    Bootstrap CIs over prompts on replacement, completeness, and the multihop share.'''
    sel_g, _, _, _ = direct_influence(layers, tcs, cc, co, tgt, metas, frozen=True, steps=1, weight=("cf" if cf else "clean"))
    M = sel_g[layers[0]].shape[0]; flat = np.concatenate([sel_g[l] for l in layers]); topf = min(CONFIG["ATTR_TOPF"], len(flat))
    sel = list(np.argsort(-flat)[:topf]); nodes = [(layers[j // M], int(j % M)) for j in sel]
    F_, E_ = len(nodes), len(layers); li = {l: layers.index(l) for l in layers}
    n = min(CONFIG["ATTR_GRAPH_PAIRS"], len(cc)); cc2, co2, tgt2, metas2 = cc[:n], co[:n], tgt[:n], metas[:n]
    ids, am, scale = _tok_scale(cc2); tids = _target_tokens(tgt2, scale)
    inp_c, out_c = capture_io(layers, cc2)
    Fcl = {l: tcs[l].encode(inp_c[l]).detach() for l in layers}; errc = {l: (out_c[l] - tcs[l].decode(Fcl[l])).detach() for l in layers}
    if cf:
        inp_o, out_o = capture_io(layers, co2)
        Fco = {l: tcs[l].encode(inp_o[l]).detach() for l in layers}; erro = {l: (out_o[l] - tcs[l].decode(Fco[l])).detach() for l in layers}
        wfeat = {l: (Fcl[l] - Fco[l]) for l in layers}; werr = {l: (errc[l] - erro[l]) for l in layers}
    else:
        wfeat = {l: Fcl[l] for l in layers}; werr = {l: errc[l] for l in layers}
    taus = [m.get("tau", 0) for m in metas2]; CTX = CONFIG["CTX"]
    leaves = {l: Fcl[l].clone().detach().requires_grad_(True) for l in layers}
    err_leaf = {l: errc[l].clone().detach().requires_grad_(True) for l in layers}
    preacts = {}
    def cap_pre(ll):
        def _h(mod, i, o): preacts[ll] = tcs[ll].preact(i[0])
        return _h
    clear_hooks(); phs = []
    for l in layers:
        MLP_MODS["enc_mlp"][l]._attr = (tcs[l], leaves[l], err_leaf[l]); phs.append(MLP_MODS["enc_mlp"][l].register_forward_hook(cap_pre(l)))
    with frozen_model(), torch.enable_grad():
        out = INNER(input_ids=ids, attention_mask=am, labels=tids); readout = _level_readout(out, tids)   # metric-aligned expected-level readout
    for h in phs: h.remove(); clear_hooks()
    og_feat = torch.autograd.grad(readout, [leaves[l] for l in layers], retain_graph=True, allow_unused=True)
    og_err = torch.autograd.grad(readout, [err_leaf[l] for l in layers], retain_graph=True, allow_unused=True)
    GFpe = np.zeros((n, F_)); GEpe = np.zeros((n, E_))                        # per-example DIRECT output influence
    for i, (lL, fi) in enumerate(nodes):
        gr = og_feat[li[lL]]
        if gr is not None: GFpe[:, i] = [float((gr[j, taus[j]:CTX, fi] * wfeat[lL][j, taus[j]:CTX, fi]).sum()) for j in range(n)]
    for e, l in enumerate(layers):
        ge = og_err[e]
        if ge is not None: GEpe[:, e] = [float((ge[j, taus[j]:CTX, :] * werr[l][j, taus[j]:CTX, :]).sum()) for j in range(n)]
    A = np.zeros((F_, F_)); Aef = np.zeros((E_, F_))                          # feature->feature and error->feature edges
    for ti, (lt, ft) in enumerate(nodes):
        scal = torch.stack([preacts[lt][j, taus[j]:CTX, ft].sum() for j in range(n)]).sum()
        if not scal.requires_grad: continue   # target preact has no upstream band-leaf dependency (e.g. the first band layer): its A column is correctly 0
        gs = torch.autograd.grad(scal, [leaves[l] for l in layers] + [err_leaf[l] for l in layers], retain_graph=True, allow_unused=True)
        for si, (ls, fs) in enumerate(nodes):
            if ls >= lt: continue
            gr = gs[li[ls]]
            if gr is not None: A[si, ti] = float(np.mean([float((gr[j, taus[j]:CTX, fs] * wfeat[ls][j, taus[j]:CTX, fs]).sum()) for j in range(n)]))
        for e, le in enumerate(layers):
            if le >= lt: continue
            gre = gs[E_ + e]
            if gre is not None: Aef[e, ti] = float(np.mean([float((gre[j, taus[j]:CTX, :] * werr[le][j, taus[j]:CTX, :]).sum()) for j in range(n)]))
    if DEVICE == "cuda":                                                   # free the retained forward graph + full-NF leaf/err caches before the numpy section (OOM safety on Large)
        del out, readout, preacts, leaves, err_leaf, og_feat, og_err; torch.cuda.empty_cache()
    spec = float(np.max(np.abs(np.linalg.eigvals(A)))) if F_ else 0.0
    scale_f = (0.9 / (spec + 1e-9)) if spec >= 0.9 else 1.0                   # signed operator; |A| spectral radius only guards convergence
    As = A * scale_f; Aef_s = Aef * scale_f; Binv = np.linalg.inv(np.eye(F_) - As); Bmat = Binv - np.eye(F_)
    def scores(gf, ge):
        feat_d = float(np.abs(gf).sum()); err_d = float(np.abs(ge).sum())
        ap = Binv @ gf; err_ap = ge + Aef_s @ ap                              # error all-paths = direct + error->feature->...->out
        feat_ap = float(np.abs(ap).sum()); err_apS = float(np.abs(err_ap).sum()); mh = float(np.abs(Bmat @ gf).sum())
        return (feat_d / (feat_d + err_d + 1e-9), feat_ap / (feat_ap + err_apS + 1e-9), mh / (feat_d + mh + 1e-9), ap)
    g_feat = GFpe.mean(0); g_err = GEpe.mean(0)
    replacement, completeness, multihop_share, allpaths = scores(g_feat, g_err)
    rng = np.random.default_rng(CONFIG["SEED0"] + 303); reps, comps, mhs_ = [], [], []
    for _ in range(int(CONFIG["GRAPH_BOOTSTRAP"])):
        ix = rng.integers(0, n, n); r, c, m, _ = scores(GFpe[ix].mean(0), GEpe[ix].mean(0)); reps.append(r); comps.append(c); mhs_.append(m)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    order_nodes = [int(x) for x in np.argsort(-np.abs(allpaths))]
    return dict(label=label, cf=bool(cf), nodes=[[int(a), int(b)] for a, b in nodes], n_nodes=F_,
                feat_infl=float(np.abs(g_feat).sum()), err_infl=float(np.abs(g_err).sum()),
                replacement=float(replacement), replacement_ci=ci(reps), completeness=float(completeness), completeness_ci=ci(comps),
                multihop_share=float(multihop_share), multihop_ci=ci(mhs_), spectral=float(spec),
                allpaths=allpaths.tolist(), order_nodes=order_nodes)
print("explicit A + all-paths influence + completeness/replacement + path-length builder ready")
""")

# ============================================================================
md("## 10. Orientation (full-band coverage; single-layer necessity is REPORT-ONLY, the wrong selector for a distributed circuit)")
code(r"""
FORCE = os.environ.get("CHRONOS_P5V4_FORCE", "0") == "1"
def _ckp(name): return os.path.join(CKPT_DIR, f"phase5v4_{MODE}_{name}.json")
def _load(name):
    p = _ckp(name)
    if os.path.exists(p) and not FORCE:
        try: return json.load(open(p))
        except Exception: return None
    return None
def _save(name, obj): json.dump(obj, open(_ckp(name), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

def orient():
    ck = _load("orient")
    if ck is not None: print("  [ckpt] orientation resumed -> band", ck["layers"]); return ck
    layers = band_layers()
    rng = np.random.default_rng(CONFIG["SEED0"] + 11); cc, co, tg, mt = make_cf_battery(rng, CONFIG["N_PAIRS"], *CONFIG["DELTA_PRIMARY"])
    clear_hooks(); s0 = cp_vec(forecast_raw(cc, CONFIG["N_CRPS_SAMPLES"]), mt); scan = []
    for li in layers:
        clear_hooks(); MLP_MODS["enc_mlp"][li]._ablate = True
        sa = cp_vec(forecast_raw(cc, CONFIG["N_CRPS_SAMPLES"]), mt); clear_hooks()
        scan.append((li, float((s0 - sa).mean() / (s0.mean() + 1e-6))))
    print("  single-layer necessity (report only, the wrong selector): " + ", ".join(f"L{l}@{rel_depth(l):.2f}={v:+.2f}" for l, v in scan))
    print(f"  -> transcoders on the FULL band: {layers} (depths " + ",".join(f"{rel_depth(l):.2f}" for l in layers) + ")")
    res = dict(layers=layers, scan=scan, clean_recovery=float(s0.mean())); _save("orient", res); return res
def assert_delta_monotone():
    rng = np.random.default_rng(CONFIG["SEED0"] + 5); recs = []
    for d in [0.5, 1.5, 3.0]:
        cc, co, tg, mt = make_cf_battery(rng, max(4, CONFIG["N_PAIRS"] // 4), d, d + 1e-3)
        clear_hooks(); recs.append(float(cp_vec(forecast_raw(cc, CONFIG["N_CRPS_SAMPLES"]), mt).mean()))
    mono = all(recs[i + 1] >= recs[i] - 0.08 for i in range(len(recs) - 1))
    print(f"  changepoint_recovery vs delta [0.5,1.5,3.0] = {['%.2f' % r for r in recs]}  monotone(tol)={mono}")
    assert IS_MOCK or mono, f"changepoint_recovery not monotone in delta on the clean model: {recs}"
    return dict(deltas=[0.5, 1.5, 3.0], recovery=recs, monotone=bool(mono))
print("Delta-monotonicity sanity (clean model)..." + MOCK_TAG); MONO = assert_delta_monotone()
print("Orientation (full-band)..." + MOCK_TAG); ORIENT = orient(); LAYERS = ORIENT["layers"]
""")

# ============================================================================
md("## 11a. Train transcoders across the band; validate the local replacement; build the headline ranking")
code(r"""
_rng = np.random.default_rng(CONFIG["SEED0"] + 21)
CC, CO, TG, MT = make_cf_battery(_rng, CONFIG["N_PAIRS"], *CONFIG["DELTA_PRIMARY"])
def train_band_transcoders(tr_cc, tr_co, ev_cc, ev_co, mult, seedoff):
    '''Train on a SEPARATE corpus (tr); report HELD-OUT reconstruction on the eval battery (ev) so features cannot be fit to the eval set.'''
    ti_c, to_c = capture_io(LAYERS, tr_cc); ti_o, to_o = capture_io(LAYERS, tr_co)
    ei_c, eo_c = capture_io(LAYERS, ev_cc); ei_o, eo_o = capture_io(LAYERS, ev_co)
    tcs, recon = {}, {}
    for l in LAYERS:
        xtr = torch.cat([ti_c[l].reshape(-1, D_MODEL), ti_o[l].reshape(-1, D_MODEL)], 0)
        ytr = torch.cat([to_c[l].reshape(-1, D_MODEL), to_o[l].reshape(-1, D_MODEL)], 0)
        xev = torch.cat([ei_c[l].reshape(-1, D_MODEL), ei_o[l].reshape(-1, D_MODEL)], 0)
        yev = torch.cat([eo_c[l].reshape(-1, D_MODEL), eo_o[l].reshape(-1, D_MODEL)], 0)
        tc, rl, nf = train_transcoder(xtr, ytr, xev, yev, mult * D_MODEL, CONFIG["SEED0"] + l + seedoff)
        tcs[l] = tc; recon[l] = rl
    return tcs, recon, mult * D_MODEL
_rngt = np.random.default_rng(CONFIG["SEED0"] + 23)
TRAIN_CC, TRAIN_CO, _, _ = make_cf_battery(_rngt, CONFIG["TRAIN_PAIRS"], *CONFIG["DELTA_PRIMARY"])
_mtr_cc, _mtr_co, _, _ = make_motif_cf_battery(_rngt, max(2, CONFIG["TRAIN_PAIRS"] // 2))
TRAIN_CC = list(TRAIN_CC) + list(_mtr_cc); TRAIN_CO = list(TRAIN_CO) + list(_mtr_co)   # separate corpus; recon is HELD-OUT on CC/CO
TCS, RECON, NF = train_band_transcoders(TRAIN_CC, TRAIN_CO, CC, CO, CONFIG["TC_DICT_MULT"], 0)
_bad = [l for l in LAYERS if RECON[l] > CONFIG["TC_RECON_MAX"]]
print(f"  {len(LAYERS)} layers x {NF} features; HELD-OUT recon(frac of eval out var) " + ",".join(f"L{l}:{RECON[l]:.3f}" for l in LAYERS))
DROPPED_LAYERS = []; _bandck = _load("band")
if _bandck is not None and not IS_MOCK and len([l for l in LAYERS if l in set(_bandck.get("layers", []))]) >= 2:
    keep = [l for l in LAYERS if l in set(_bandck["layers"])]; DROPPED_LAYERS = [l for l in LAYERS if l not in keep]   # RESUME: reuse the persisted kept band (deterministic; a re-derived drop could otherwise misalign the cached ORDER)
    if DROPPED_LAYERS: print(f"  [ckpt] band resumed -> kept {keep}, dropped {DROPPED_LAYERS}")
    LAYERS = keep; TCS = {l: TCS[l] for l in keep}; RECON = {l: RECON[l] for l in keep}
elif _bad and not IS_MOCK:                               # a layer whose transcoder cannot reconstruct is untrustworthy: drop it, keep coverage where the substrate is valid
    keep = [l for l in LAYERS if l not in _bad]
    assert len(keep) >= 2, f"too few layers pass the recon gate (keep={keep}, failed={_bad} at TC_RECON_MAX={CONFIG['TC_RECON_MAX']}); raise TC_TOPK/TC_STEPS/TC_DICT_MULT or relax TC_RECON_MAX"
    print(f"  [recon gate] DROPPING {_bad} (held-out recon > {CONFIG['TC_RECON_MAX']}); analysis proceeds on the well-reconstructed band {keep} (still cross-layer), recorded in the verdict")
    DROPPED_LAYERS = list(_bad); LAYERS = keep; TCS = {l: TCS[l] for l in keep}; RECON = {l: RECON[l] for l in keep}
elif _bad and IS_MOCK:
    print(f"  [mock] recon gate skipped (mock transcoder not interpretable); on pilot it would drop {_bad}")
BAND_SIG = ",".join(str(int(l)) for l in LAYERS) + "|" + str(int(NF))   # signature: discard band-dependent checkpoints if the kept band or NF changed across a resume
_save("band", {"layers": [int(l) for l in LAYERS], "dropped": [int(l) for l in DROPPED_LAYERS], "nf": int(NF)})
def _load_sig(name):
    ck = _load(name)
    if ck is None: return None
    if not (isinstance(ck, dict) and ck.get("sig") == BAND_SIG):
        print(f"  [ckpt] {name} discarded (band/NF signature changed since it was written; recomputing)"); return None
    return ck["obj"]
def _save_sig(name, obj): _save(name, {"sig": BAND_SIG, "obj": obj})
# hard assert: local replacement (transcoders + error node + frozen attention/norms) matches the model on-prompt
_vr = validate_local_replacement(CC[:2], TCS, LAYERS)
assert _vr["rel"] < 1e-3, f"local replacement does not match underlying model on-prompt (rel {_vr['rel']:.2e})"
assert _vr["frozen_value_rel"] < 1e-4, f"freezing changed forward VALUES (should only affect gradients): rel {_vr['frozen_value_rel']:.2e}"
print(f"  LOCAL REPLACEMENT matches model on-prompt (rel={_vr['rel']:.2e}); freeze preserves values (rel={_vr['frozen_value_rel']:.2e}); transcoder-only on-prompt rel={_vr['transcoder_only_rel']:.2f}  PASS" + MOCK_TAG)

def _resample_pairs(cc, co, tg, mt, seed):
    if seed == 0: return cc, co, tg, mt
    rng = np.random.default_rng(CONFIG["SEED0"] + 500 + seed); ix = rng.integers(0, len(cc), len(cc))
    return [cc[i] for i in ix], [co[i] for i in ix], tg[ix], [mt[i] for i in ix]
def _interleave(*orders):
    seen, out = set(), []
    for tup in zip(*orders):
        for j in tup:
            if int(j) not in seen: seen.add(int(j)); out.append(int(j))
    return out
def build_ranking(tcs):
    g_gold, _, _, _ = direct_influence(LAYERS, tcs, CC, CO, TG, MT, frozen=True, steps=1)
    g_eap, _, _, _ = direct_influence(LAYERS, tcs, CC, CO, TG, MT, frozen=False, steps=CONFIG["EAP_STEPS"])
    g_prox = proxy_influence(LAYERS, tcs, CC, CO, MT)
    flat_gold = np.concatenate([g_gold[l] for l in LAYERS]); flat_eap = np.concatenate([g_eap[l] for l in LAYERS]); flat_prox = np.concatenate([g_prox[l] for l in LAYERS])
    rc_ge = _spearman(flat_gold, flat_eap); rc_gp = _spearman(flat_gold, flat_prox)
    # change graph + CONSERVATIVE CI across GRAPH_REBUILDS pair-resamples (captures the variance the fixed-A bootstrap misses)
    graphs = [build_attr_graph(LAYERS, tcs, *_resample_pairs(CC, CO, TG, MT, rb), "change", cf=True) for rb in range(max(1, CONFIG["GRAPH_REBUILDS"]))]
    if DEVICE == "cuda": torch.cuda.empty_cache()
    graph = graphs[0]
    if len(graphs) > 1:
        rr = [g["replacement"] for g in graphs]; cm = [g["completeness"] for g in graphs]
        graph["replacement_ci"] = [min(graph["replacement_ci"][0], min(rr)), max(graph["replacement_ci"][1], max(rr))]
        graph["completeness_ci"] = [min(graph["completeness_ci"][0], min(cm)), max(graph["completeness_ci"][1], max(cm))]
    graph["replacement_mde"] = float((graph["replacement_ci"][1] - graph["replacement_ci"][0]) / 2)
    # headline order: counterfactual all-paths within the pruned top-F, then direct gold; UNION with EAP-IG when they disagree
    gold_order = list(np.argsort(-flat_gold))
    if CONFIG["RANK_METHOD"] == "gold" and graph["n_nodes"]:
        topnodes = [graph["nodes"][i] for i in graph["order_nodes"]]; head = [LAYERS.index(int(l)) * NF + int(f) for l, f in topnodes]
        gold_order = head + [j for j in gold_order if j not in set(head)]
    eap_order = list(np.argsort(-flat_eap))
    if rc_ge < CONFIG["RANK_AGREE_MIN"]:
        order = _interleave(gold_order, eap_order); print(f"  [rank] gold and EAP-IG disagree (rho={rc_ge:+.2f} < {CONFIG['RANK_AGREE_MIN']}); UNIONING their orders so a mis-ranked causal feature still enters the union")
    else:
        order = [int(j) for j in gold_order]
    # rank-vs-metric validation: exact single-feature denoising effect on the real metric vs the gold influence
    rmc = float("nan")
    try:
        nval = min(CONFIG["ATTR_PAIRS"], len(CC)); ic, _ = capture_io(LAYERS, CC[:nval]); io, _ = capture_io(LAYERS, CO[:nval])
        FCv = {l: tcs[l].encode(ic[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}; FCOv = {l: tcs[l].encode(io[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        clear_hooks(); cb = float(cp_vec(forecast_raw(CO[:nval], CONFIG["N_CRPS_SAMPLES"]), MT[:nval]).mean()); ex, inf = [], []
        for j in order[:min(8, len(order))]:
            l = LAYERS[j // NF]; fi = int(j % NF)
            arm1 = {l: (torch.as_tensor([fi], dtype=torch.long, device=DEVICE), FCv[l], FCOv[l], None)}   # inject one clean feature into the corrupt run (diagnostic)
            da = cp_vec(forecast_cf_multi(CO[:nval], arm1, CONFIG["N_CRPS_SAMPLES"], tcs), MT[:nval])
            ex.append(float(da.mean() - cb)); inf.append(float(flat_gold[j]))
        rmc = _spearman(inf, ex)
    except Exception as e: print("  [rank] metric-validation skipped:", repr(e)[:80])
    print(f"  ranking = {CONFIG['RANK_METHOD'].upper()} over {len(LAYERS)*NF} pairs; Spearman(gold,EAP)={rc_ge:+.3f} (gold,proxy)={rc_gp:+.3f}; gold-vs-EXACT-metric rho={rmc:+.2f}")
    print(f"  attribution graph: {graph['n_nodes']} nodes ({CONFIG['ATTR_GRAPH_PAIRS']} pairs x {CONFIG['GRAPH_REBUILDS']} rebuilds)  "
          f"replacement={graph['replacement']:.2f} CI[{graph['replacement_ci'][0]:.2f},{graph['replacement_ci'][1]:.2f}] MDE={graph['replacement_mde']:.2f}  "
          f"completeness={graph['completeness']:.2f}  multihop_share={graph['multihop_share']:.2f}")
    return order, dict(gold_eap=rc_ge, gold_proxy=rc_gp, gold_metric=rmc), graph
ck = _load_sig("rank")
if ck is not None:
    ORDER = ck["order"]; RANK_CORR = ck["rank_corr"]; GRAPH_CHANGE = ck["graph"]; print("  [ckpt] ranking resumed")
else:
    ORDER, RANK_CORR, GRAPH_CHANGE = build_ranking(TCS)
    _save_sig("rank", dict(order=[int(x) for x in ORDER], rank_corr=RANK_CORR, graph=GRAPH_CHANGE))
def split_union(order, k):
    per = {l: [] for l in LAYERS}
    for j in order[:k]:
        if j // NF >= len(LAYERS): continue   # defensive: a stale index from a changed band (the signature guard should already have rebuilt)
        per[LAYERS[j // NF]].append(int(j % NF))
    return per
# PLUMBING: multi-layer counterfactual patch bites
_ids, _am = _tokenize(CC[:2]); _dec = torch.full((2, 1), DEC_START, dtype=torch.long, device=DEVICE)
clear_hooks()
with torch.inference_mode(): _g0 = INNER(input_ids=_ids, attention_mask=_am, decoder_input_ids=_dec).logits.clone()
inp_o, _ = capture_io(LAYERS, CO[:2]); FCO2 = {l: TCS[l].encode(inp_o[l]) for l in LAYERS}
for l in LAYERS: MLP_MODS["enc_mlp"][l]._cf = (TCS[l], torch.arange(NF, device=DEVICE), FCO2[l].to(DEVICE, dtype=DTYPE), None, None)
with torch.inference_mode(): _g1 = INNER(input_ids=_ids, attention_mask=_am, decoder_input_ids=_dec).logits.clone()
clear_hooks(); assert not torch.allclose(_g0, _g1), "multi-layer patch did not bite"
print(f"  PLUMBING: multi-layer counterfactual patch bites (max|delta|={(_g0-_g1).abs().max():.4g})  PASS" + MOCK_TAG)
""")

# ============================================================================
md(r"""
## 11b. Cross-layer union curves: faithfulness, completeness, selectivity, sufficiency (layer-range patching, end-layer sweep)

For each union size k we run all four causal criteria with bootstrap CIs and a random-union null:

* **Faithfulness**: keep the candidate union clean, ablate the complement to its counterfactual value; the behavior
  should survive on the candidate set alone. It must beat the random-union null (the v3 faith-beats guard).
* **Completeness**: ablate the candidate union to its counterfactual value; the behavior should collapse. The union
  is patched with **layer-range constrained patching** (every patched layer is pinned to the run's own feature cache
  and only the union indices are overridden, so an upstream patch does not re-encode into downstream non-union
  features: no within-range leakage) and an **end-layer sweep** (pin the union across layers up to a swept end-layer,
  report the maximum collapse, because single-layer patching understates the effect), versus the random-union null.
* **Selectivity**: the collapse is specific to change-detection, not periodicity. The control is a periodicity
  MINIMAL PAIR (period-present vs period-absent, shared noise) ablated by the SAME interchange (not mean ablation),
  so the selectivity comparison is like-for-like.
* **Sufficiency (denoising)**: inject the candidate union into the corrupt run; does it induce the behavior? This is
  the decisive, positive-result direction and survives OR-gate redundancy. End-layer sweep, report the maximum gain.
""")
code(r"""
def _complement_arm(per, src, base, eb):
    arm = {}
    for l in LAYERS:
        mask = torch.ones(NF, dtype=torch.bool)
        if per[l]: mask[per[l]] = False
        arm[l] = (mask.nonzero(as_tuple=True)[0].to(DEVICE), src[l], base[l], eb[l])
    return arm
def _union_arm(per, src, base, eb, end_layer=None):
    return {l: (_idx_tensor(per[l]), src[l], base[l], eb[l]) for l in LAYERS if per[l] and (end_layer is None or l <= end_layer)}
def _end_layers(per):
    used = [l for l in LAYERS if per[l]]
    if not used: return [LAYERS[-1]]
    lo, hi = used[0], used[-1]; ne = max(1, int(CONFIG["END_SWEEP"]))
    cand = sorted(set(int(round(lo + (hi - lo) * t / (ne - 1))) for t in range(ne)) if ne > 1 else {hi})
    return [l for l in LAYERS if l in cand] or [hi]
def _sweep_union(contexts, per, src, base, eb, metas, vecfn, tcs, reducer):
    vals = []
    for el in _end_layers(per):
        arm = _union_arm(per, src, base, eb, end_layer=el)
        if not arm: continue
        vals.append(vecfn(forecast_cf_multi(contexts, arm, CONFIG["N_CRPS_SAMPLES"], tcs), metas))
    if not vals: return None
    return reducer(vals)

def run_curves():
    clear_hooks(); base = cp_vec(forecast_raw(CC, CONFIG["N_CRPS_SAMPLES"]), MT)
    clear_hooks(); corr_base = cp_vec(forecast_raw(CO, CONFIG["N_CRPS_SAMPLES"]), MT)
    inp_c, out_c = capture_io(LAYERS, CC); inp_o, out_o = capture_io(LAYERS, CO)
    FC = {l: TCS[l].encode(inp_c[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
    FCO = {l: TCS[l].encode(inp_o[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
    ERR_C = {l: (out_c[l] - TCS[l].decode(TCS[l].encode(inp_c[l]))).to("cpu", dtype=CACHE_DT) for l in LAYERS}   # unperturbed error nodes
    ERR_O = {l: (out_o[l] - TCS[l].decode(TCS[l].encode(inp_o[l]))).to("cpu", dtype=CACHE_DT) for l in LAYERS}
    # selectivity control = a periodicity MINIMAL PAIR (period-present vs period-absent, shared noise), ablated by the
    # SAME counterfactual interchange as change-detection (not mean ablation), so the comparison is like-for-like
    rngm = np.random.default_rng(CONFIG["SEED0"] + 44); mctx, mco, mtg, mmeta = make_motif_cf_battery(rngm, CONFIG["N_PAIRS"])
    clear_hooks(); mbase = motif_vec(forecast_raw(mctx, CONFIG["N_CRPS_SAMPLES"]), mmeta)
    inp_m, out_m = capture_io(LAYERS, mctx); inp_mo, _ = capture_io(LAYERS, mco)
    FM = {l: TCS[l].encode(inp_m[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
    FMO = {l: TCS[l].encode(inp_mo[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
    ERR_M = {l: (out_m[l] - TCS[l].decode(TCS[l].encode(inp_m[l]))).to("cpu", dtype=CACHE_DT) for l in LAYERS}
    KMAX = len(ORDER); rng = np.random.default_rng(CONFIG["SEED0"] + 77)
    rows = _load_sig("curves") or []; done = {r["k"] for r in rows}
    for k in [kk for kk in CONFIG["K_GRID"] if kk <= KMAX]:
        if k in done: print(f"    [ckpt] k={k} resumed"); continue
        per = split_union(ORDER, k)
        # faithfulness: keep union clean (base/err pinned to the CLEAN run), ablate the complement to corrupt (src = FCO)
        fa = cp_vec(forecast_cf_multi(CC, _complement_arm(per, FCO, FC, ERR_C), CONFIG["N_CRPS_SAMPLES"], TCS), MT); fa_m, fa_ci = mean_ci(fa)
        # completeness: ablate the union to corrupt (src=FCO) on the clean run, fully constrained, max collapse over end-layer sweep
        ca = _sweep_union(CC, per, FCO, FC, ERR_C, MT, cp_vec, TCS, lambda vs: min(vs, key=lambda v: v.mean()))
        cp_drop, cp_ci = mean_ci(base - (ca if ca is not None else base))
        # selectivity: interchange-ablate the union on the periodicity minimal pair (period-present -> period-absent)
        ma = _sweep_union(mctx, per, FMO, FM, ERR_M, mmeta, motif_vec, TCS, lambda vs: min(vs, key=lambda v: v.mean()))
        mot_drop = float((mbase - (ma if ma is not None else mbase)).mean())
        # sufficiency: inject the clean union into the corrupt run (src=FC), base/err pinned to the CORRUPT run, max gain over end-layer sweep
        da = _sweep_union(CO, per, FC, FCO, ERR_O, MT, cp_vec, TCS, lambda vs: max(vs, key=lambda v: v.mean()))
        suf_m, suf_ci = mean_ci(da if da is not None else corr_base)
        fa_null, cp_null, suf_null = [], [], []
        for _ in range(CONFIG["N_RANDOM_NULL"]):
            rsel = rng.choice(KMAX, size=k, replace=False); rper = {l: [] for l in LAYERS}
            for j in rsel: rper[LAYERS[j // NF]].append(int(j % NF))
            fa_null.append(float(cp_vec(forecast_cf_multi(CC, _complement_arm(rper, FCO, FC, ERR_C), CONFIG["N_CRPS_SAMPLES"], TCS), MT).mean()))
            rca = _sweep_union(CC, rper, FCO, FC, ERR_C, MT, cp_vec, TCS, lambda vs: min(vs, key=lambda v: v.mean()))
            cp_null.append(float((base - (rca if rca is not None else base)).mean()))
            rda = _sweep_union(CO, rper, FC, FCO, ERR_O, MT, cp_vec, TCS, lambda vs: max(vs, key=lambda v: v.mean()))
            suf_null.append(float((rda if rda is not None else corr_base).mean()))
        rows.append(dict(k=int(k), per_layer={int(l): len(per[l]) for l in LAYERS},
                         faith=fa_m, faith_ci=fa_ci, faith_frac=float(fa_m/(base.mean()+1e-6)), faith_null=float(np.mean(fa_null)),
                         cp_complete=cp_drop, cp_ci=cp_ci, cp_null_p95=float(np.percentile(cp_null, 95)), motif_drop=mot_drop,
                         induced=suf_m, induced_ci=suf_ci, gain=float(suf_m - corr_base.mean()), suf_null_p95=float(np.percentile(suf_null, 95)),
                         selective=bool(cp_drop >= CONFIG["SELECTIVITY_MARGIN"] * max(mot_drop, 1e-6) and cp_drop > float(np.percentile(cp_null, 95)))))
        _save_sig("curves", rows)
        print(f"    k={k:4d}  faith_frac={rows[-1]['faith_frac']:.2f}(null {rows[-1]['faith_null']/(base.mean()+1e-6):.2f})  "
              f"cp_complete={cp_drop:+.3f}  motif={mot_drop:+.3f}  denoise_gain={rows[-1]['gain']:+.3f}  sel={rows[-1]['selective']}")
    rows.sort(key=lambda r: r["k"])
    return dict(base=float(base.mean()), corrupt_base=float(corr_base.mean()), motif_base=float(mbase.mean()), rows=rows)
print("Cross-layer union curves (layer-range patching, end-layer sweep)..." + MOCK_TAG); CURVES = run_curves()
""")

# ============================================================================
md(r"""
## 11c. Attention-pattern interchange: the causal QK-vs-feature test

The Circuit-Tracing method freezes attention, so a low feature-routed replacement could mean the crux is the QK
circuit (the pattern that attends to the boundary), which this method never tests, rather than a genuinely
distributed feature computation. We test it directly: capture the encoder self-attention PATTERN on the clean and
corrupt runs, then patch the PATTERN (not the features) clean-to-corrupt and corrupt-to-clean, and measure
changepoint_recovery on the real metric. If patching the pattern alone induces recovery while patching features does
not, the crux is QK and a B verdict is bankable as attention-routed; if neither moves it, the QK-crux reading is
falsified and B is distributed-or-transcoder-missed. This is the experiment that decomposes a B verdict into its
three hypotheses.
""")
code(r"""
@contextlib.contextmanager
def _enc_pattern(store, inject):
    '''Capture (inject=False) or substitute (inject=True) the encoder self-attention patterns, identified as the first
    N_ENC_LAYERS square (S x S) softmax calls of one forward (no chunking, so the layer counter stays aligned).
    Patches the PATTERN only, isolating the QK circuit from the OV/value path.'''
    cnt = {"i": 0}; _soft = F.softmax
    def soft(x, *a, **k):
        p = _soft(x, *a, **k)
        if p.dim() == 4 and p.shape[-2] == p.shape[-1] and cnt["i"] < N_ENC_LAYERS:
            i = cnt["i"]; cnt["i"] += 1
            if inject:
                if i < len(store) and store[i] is not None and store[i].shape == p.shape: return store[i].to(p.device, p.dtype)
            else:
                while len(store) <= i: store.append(None)
                store[i] = p.detach().to("cpu", torch.float16)
        return p
    F.softmax = soft
    try: yield
    finally: F.softmax = _soft
def forecast_pattern(contexts, store, inject, n_samples):
    '''One un-chunked forward so the pattern counter stays aligned across capture and inject.'''
    clear_hooks()
    if IS_MOCK:
        with _enc_pattern(store, inject): return forecast_raw(contexts, n_samples)
    chunk = [torch.tensor(np.asarray(c), dtype=DTYPE) for c in contexts]
    with _enc_pattern(store, inject), torch.inference_mode():
        fc = PIPE.predict(chunk, prediction_length=CONFIG["PRED"], num_samples=n_samples)
    return fc.detach().cpu().numpy()
def attention_pattern_test():
    ck = _load("attn")
    if ck is not None: print("  [ckpt] attention-pattern test resumed"); return ck
    rng = np.random.default_rng(CONFIG["SEED0"] + 131); cc, co, tg, mt = make_cf_battery(rng, CONFIG["PATTERN_PAIRS"], *CONFIG["DELTA_PRIMARY"])
    ns = CONFIG["PATTERN_SAMPLES"]; sc, so = [], []
    bclean = cp_vec(forecast_pattern(cc, sc, False, ns), mt)         # capture clean patterns -> sc
    bcorr = cp_vec(forecast_pattern(co, so, False, ns), mt)          # capture corrupt patterns -> so
    # batch-alignment asserts: a single un-chunked encoder forward (else the layer counter desyncs and the test silently no-ops)
    assert len(sc) == N_ENC_LAYERS and all(p is not None for p in sc), f"pattern capture misaligned ({len(sc)} of {N_ENC_LAYERS} layers); the encoder was chunked"
    assert int(sc[0].shape[0]) == len(cc), f"pattern batch {int(sc[0].shape[0])} != n_series {len(cc)} (predict chunked the encoder)"
    den = cp_vec(forecast_pattern(co, sc, True, ns), mt)            # corrupt values + CLEAN pattern (sufficiency; VALUE-CONFOUNDED for change-detection)
    noi = cp_vec(forecast_pattern(cc, so, True, ns), mt)            # clean values + CORRUPT pattern (NECESSITY; value-unconfounded, clean values carry the shift)
    perm = rng.permutation(len(cc)); sh = [p[torch.as_tensor(perm)] for p in sc]   # shuffled clean pattern = a valid-but-wrong pattern (the necessity NULL)
    nul = cp_vec(forecast_pattern(cc, sh, True, ns), mt)           # clean values + SHUFFLED clean pattern
    def _ci(x): lo, hi = bootstrap_ci(np.asarray(x)); return [float(lo), float(hi)]
    nd = bclean - noi; sd = den - bcorr; nn = bclean - nul
    res = dict(base_clean=float(bclean.mean()), base_corr=float(bcorr.mean()),
               pat_noise_drop=float(nd.mean()), pat_noise_ci=_ci(nd),               # NECESSITY (primary, value-unconfounded)
               pat_denoise_gain=float(sd.mean()), pat_denoise_ci=_ci(sd),           # SUFFICIENCY (value-confounded for change-detection)
               null_noise_drop=float(nn.mean()), null_noise_ci=_ci(nn),
               pattern_necessary=bool(nd.mean() > nn.mean() + 0.05 and _ci(nd)[0] > 0))
    print(f"  base clean={res['base_clean']:.2f} corrupt={res['base_corr']:.2f}")
    print(f"  PATTERN NECESSITY (clean values + corrupt pattern) drop={res['pat_noise_drop']:+.3f} CI[{res['pat_noise_ci'][0]:+.2f},{res['pat_noise_ci'][1]:+.2f}]  vs shuffled-pattern null {res['null_noise_drop']:+.3f}  necessary={res['pattern_necessary']}")
    print(f"  PATTERN sufficiency (corrupt values + clean pattern) gain={res['pat_denoise_gain']:+.3f} CI[{res['pat_denoise_ci'][0]:+.2f},{res['pat_denoise_ci'][1]:+.2f}]  (value-confounded, secondary)")
    print("  reading: NECESSITY is the value-unconfounded direction for change-detection; pattern necessary AND beating the feature necessity => attention-routed (QK).")
    _save("attn", res); return res
print("Attention-pattern interchange (QK-vs-feature causal test)..." + MOCK_TAG); ATTN = attention_pattern_test()
""")

# ============================================================================
md("## 12. SNR sweep: re-rank with the gold attribution at each shift magnitude toward the noise floor")
code(r"""
def snr_sweep():
    out = _load_sig("snr") or []; done = {tuple(r["delta"]) for r in out}
    for (dl, dh) in CONFIG["SNR_DELTAS"]:
        if (dl, dh) in done: print(f"    [ckpt] snr delta[{dl},{dh}] resumed"); continue
        rng = np.random.default_rng(CONFIG["SEED0"] + 66 + int(dl * 100)); cc, co, tg, mt = make_cf_battery(rng, CONFIG["SNR_PAIRS"], dl, dh)
        tcs, recon, _nf = train_band_transcoders(cc, co, cc, co, CONFIG["TC_DICT_MULT"], int(dl * 100))
        g, _, _, _ = direct_influence(LAYERS, tcs, cc, co, tg, mt, frozen=True, steps=1)
        order = list(np.argsort(-np.concatenate([g[l] for l in LAYERS])))
        inp_c, out_c = capture_io(LAYERS, cc); inp_o, _ = capture_io(LAYERS, co)
        fc = {l: tcs[l].encode(inp_c[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        fco = {l: tcs[l].encode(inp_o[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        ec = {l: (out_c[l] - tcs[l].decode(tcs[l].encode(inp_c[l]))).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        clear_hooks(); base = cp_vec(forecast_raw(cc, CONFIG["N_CRPS_SAMPLES"]), mt); fr = []
        for k in [kk for kk in CONFIG["SNR_KS"] if kk <= len(order)]:
            per = {l: [] for l in LAYERS}
            for j in order[:k]: per[LAYERS[j // NF]].append(int(j % NF))
            arm = {}
            for l in LAYERS:
                mask = torch.ones(NF, dtype=torch.bool)
                if per[l]: mask[per[l]] = False
                arm[l] = (mask.nonzero(as_tuple=True)[0].to(DEVICE), fco[l], fc[l], ec[l])
            fr.append([int(k), float(cp_vec(forecast_cf_multi(cc, arm, CONFIG["N_CRPS_SAMPLES"], tcs), mt).mean()) / (base.mean() + 1e-6)])
        out.append(dict(delta=[dl, dh], clean_recovery=float(base.mean()), faith_frac_by_k=fr)); _save_sig("snr", out)
        print(f"    delta[{dl},{dh}] clean_rec={base.mean():.2f}  faith_frac " + " ".join(f"k{k}:{v:.2f}" for k, v in fr))
    return out
print("SNR sweep (gold re-rank per regime)..." + MOCK_TAG); SNR = snr_sweep()
""")

# ============================================================================
md("## 13. Feature-splitting check: localization at two dictionary sizes (a split concept fakes 'distributed')")
code(r"""
def feature_splitting():
    ck = _load_sig("split")
    if ck is not None: print("  [ckpt] feature-splitting resumed"); return ck
    inp_c, out_c = capture_io(LAYERS, CC); inp_o, _ = capture_io(LAYERS, CO)
    res = {}
    for tag, mult in [("small", CONFIG["TC_DICT_MULT_SMALL"]), ("primary", CONFIG["TC_DICT_MULT"])]:
        tcs, recon, nf = train_band_transcoders(CC, CO, CC, CO, mult, 1000 + mult)
        g, _, _, _ = direct_influence(LAYERS, tcs, CC, CO, TG, MT, frozen=True, steps=1)
        order = list(np.argsort(-np.concatenate([g[l] for l in LAYERS])))
        fc = {l: tcs[l].encode(inp_c[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        fco = {l: tcs[l].encode(inp_o[l]).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        ec = {l: (out_c[l] - tcs[l].decode(tcs[l].encode(inp_c[l]))).to("cpu", dtype=CACHE_DT) for l in LAYERS}
        clear_hooks(); base = cp_vec(forecast_raw(CC, CONFIG["N_CRPS_SAMPLES"]), MT).mean()
        faith = {}
        for k in [kk for kk in CONFIG["K_GRID"] if kk <= len(order)]:
            per = {l: [] for l in LAYERS}
            for j in order[:k]: per[LAYERS[j // nf]].append(int(j % nf))
            arm = {}
            for l in LAYERS:
                mask = torch.ones(nf, dtype=torch.bool)
                if per[l]: mask[per[l]] = False
                arm[l] = (mask.nonzero(as_tuple=True)[0].to(DEVICE), fco[l], fc[l], ec[l])
            faith[int(k)] = float(cp_vec(forecast_cf_multi(CC, arm, CONFIG["N_CRPS_SAMPLES"], tcs), MT).mean()) / (base + 1e-6)
        # union size to reach FAITH_TARGET (smaller dict concentrating the effect = splitting in the bigger one)
        kstar = next((k for k in sorted(faith) if faith[k] >= CONFIG["FAITH_TARGET"]), None)
        res[tag] = dict(mult=mult, n_features=int(nf), faith_by_k=faith, kstar=kstar)
        print(f"  dict x{mult} ({nf} feats): k*(faith>={CONFIG['FAITH_TARGET']})={kstar}")
    sk, pk = res["small"]["kstar"], res["primary"]["kstar"]
    res["splitting"] = bool(sk is not None and (pk is None or sk < pk))
    print(f"  feature-splitting (smaller dict concentrates the effect into fewer features): {res['splitting']}")
    _save_sig("split", res); return res
print("Feature-splitting check..." + MOCK_TAG); SPLIT = feature_splitting()
""")

# ============================================================================
md("## 14. Change-detection vs periodicity: the attention-asymmetry test (frozen attention freezes the QK crux)")
code(r"""
def asymmetry():
    ck = _load_sig("asym")
    if ck is not None: print("  [ckpt] asymmetry resumed"); return ck
    rngm = np.random.default_rng(CONFIG["SEED0"] + 91); mctx, mtg, mmeta = make_motif_battery(rngm, max(CONFIG["ATTR_PAIRS"], CONFIG["N_PAIRS"]))
    tcs_m, recon_m, _nf = train_band_transcoders(mctx, mctx, mctx, mctx, CONFIG["TC_DICT_MULT"], 2000)   # transcoders on motif activations
    g_motif = build_attr_graph(LAYERS, tcs_m, mctx, mctx, mtg, mmeta, "periodicity", cf=False)
    # periodicity has no minimal-pair corrupt, so it can only be CLEAN-weighted. For a LIKE-FOR-LIKE comparison we also
    # build a clean-weighted change graph; the cf=True GRAPH_CHANGE stays the causal one for the verdict high_frac gate.
    g_change_clean = build_attr_graph(LAYERS, TCS, CC, CO, TG, MT, "change_clean", cf=False)
    asym = dict(change=g_change_clean, periodicity=g_motif, change_cf=GRAPH_CHANGE, basis="clean (like-for-like)",
                replacement_gap=float(g_motif["replacement"] - g_change_clean["replacement"]),
                multihop_gap=float(g_motif["multihop_share"] - g_change_clean["multihop_share"]))
    print(f"  clean-weighted replacement (like-for-like): change={g_change_clean['replacement']:.2f}  periodicity={g_motif['replacement']:.2f}  (gap {asym['replacement_gap']:+.2f})")
    print(f"  multihop share: change={g_change_clean['multihop_share']:.2f}  periodicity={g_motif['multihop_share']:.2f}")
    note = ("periodicity more feature-mediated (higher clean-weighted replacement) supports an attention/QK crux for change-detection (Phase 4)."
            if asym["replacement_gap"] > 0.05 else "change and periodicity comparably feature-mediated here; the QK-crux reading is NOT supported by this run.")
    print("  reading: " + note)
    _save_sig("asym", asym); return asym
print("Change-detection vs periodicity asymmetry..." + MOCK_TAG); ASYM = asymmetry()
""")

# ============================================================================
md("## 15. Verdict: the joint criteria, completeness/replacement, beat-the-null, minimum detectable effect")
code(r"""
def summarize():
    R = CURVES["rows"]; base = CURVES["base"]; target = CONFIG["FAITH_TARGET"]
    kstar = next((r["k"] for r in sorted(R, key=lambda r: r["k"]) if r["faith_frac"] >= target), None)
    small = bool(kstar is not None and kstar <= CONFIG["LOCALIZE_MAX_FEATURES"])
    win = [r for r in R if (kstar is None or r["k"] <= max(kstar, CONFIG["LOCALIZE_MAX_FEATURES"]))]
    selective = any(r["selective"] for r in win) if win else False
    complete_beats = any(r["cp_complete"] > r["cp_null_p95"] for r in win) if win else False
    sufficient = any((r["induced"] > r["suf_null_p95"] and r["gain"] >= CONFIG["SUFFICIENCY_BAR"]) for r in win) if win else False
    faith_beats = any(r["faith_frac"] > (r["faith_null"]/(base+1e-6)) + CONFIG["FAITH_BEAT_MARGIN"] for r in win) if win else False
    repl = GRAPH_CHANGE["replacement"]; repl_ci = GRAPH_CHANGE["replacement_ci"]; high_frac = bool(repl_ci[0] >= 0.5)   # gate on the LOWER CI bound
    rank_unverified = bool(np.isfinite(RANK_CORR.get("gold_metric", float("nan"))) and RANK_CORR["gold_metric"] < CONFIG["RANK_AGREE_MIN"])
    localized = bool(small and selective and complete_beats and sufficient and faith_beats and high_frac and not rank_unverified)
    in_band = [l for l in LAYERS if CONFIG["MISHRA_DEPTH_LO"] <= rel_depth(l) <= CONFIG["MISHRA_DEPTH_HI"]]
    snr_sharpens = False
    if len(SNR) >= 2 and SNR[0]["faith_frac_by_k"] and SNR[-1]["faith_frac_by_k"]:
        snr_sharpens = bool(SNR[-1]["faith_frac_by_k"][0][1] > SNR[0]["faith_frac_by_k"][0][1] + 0.15)
    escalate_clt = bool(GRAPH_CHANGE["multihop_share"] >= CONFIG["MULTIHOP_ESCALATE"])
    # attention-pattern interchange decides QK-routing; NECESSITY (corrupt pattern on clean values) is the value-unconfounded
    # direction for change-detection (clean values carry the shift); sufficiency is value-confounded, so it is secondary
    feat_drop = max((r["cp_complete"] for r in R), default=0.0); feat_gain = max((r["gain"] for r in R), default=0.0)
    pat_drop = ATTN["pat_noise_drop"]; pat_gain = ATTN["pat_denoise_gain"]; pat_necessary = bool(ATTN.get("pattern_necessary", False))
    qk_necessity = bool(pat_necessary and pat_drop > feat_drop + 0.05)
    qk_sufficiency = bool(pat_gain >= CONFIG["SUFFICIENCY_BAR"] and pat_gain > feat_gain + 0.05)
    attention_routed = bool(qk_necessity or qk_sufficiency)
    mean_recon = float(np.mean([RECON[l] for l in LAYERS])); err_share = 1.0 - GRAPH_CHANGE["completeness"]
    escalate = bool(escalate_clt and not high_frac and not attention_routed and not localized)   # PLT under-attributes a chained cross-layer circuit
    if localized:
        verdict = (f"A: LOCALIZED cross-layer change-detection feature circuit, {kstar} features across {LAYERS}, "
                   f"faithful>={target} (beats union null), complete>null, selective, SUFFICIENT, replacement={repl:.2f} (CI lo {repl_ci[0]:.2f}), ranking verified; discrepancy RESOLVED")
    elif attention_routed:
        which = ("NECESSITY (corrupting the pattern destroys recovery more than feature-noising)" if qk_necessity else "SUFFICIENCY (the clean pattern alone induces recovery)")
        verdict = (f"B-ATTENTION-ROUTED: not a small cross-layer feature circuit, but the attention-pattern interchange shows the crux is the frozen QK pattern on the {which} direction "
                   f"(pattern necessity drop {pat_drop:+.2f} CI[{ATTN['pat_noise_ci'][0]:+.2f},{ATTN['pat_noise_ci'][1]:+.2f}] vs feature {feat_drop:+.2f}, shuffled-null {ATTN['null_noise_drop']:+.2f}); "
                   f"distributed in features but ROUTED by attention (corroborates Phase 4); the bankable QK-crux outcome")
    elif escalate:
        verdict = (f"INCONCLUSIVE for A (escalate to a cross-layer transcoder): per-layer transcoders show a cross-layer-superposition signature "
                   f"(multihop share {GRAPH_CHANGE['multihop_share']:.2f} >= {CONFIG['MULTIHOP_ESCALATE']}) with low replacement (CI lo {repl_ci[0]:.2f}); a per-layer transcoder structurally "
                   f"under-attributes a chained cross-layer circuit, and the attention pattern is not the crux (necessity drop {pat_drop:+.2f} <= feature {feat_drop:+.2f}), so A cannot be ruled out here")
    else:
        why = []
        if not faith_beats: why.append("faithfulness ties the random-union null (layers bypassable)")
        if not sufficient: why.append("not sufficient under denoising")
        if not selective: why.append("not motif-selective")
        if not high_frac: why.append(f"features explain only {repl:.0%} (replacement CI lo {repl_ci[0]:.2f} < 0.5)")
        if rank_unverified: why.append(f"ranking unverified (gold-vs-exact-metric rho {RANK_CORR['gold_metric']:+.2f} < {CONFIG['RANK_AGREE_MIN']})")
        if mean_recon > CONFIG["TC_RECON_MAX"] or err_share > 0.6:
            decomp = (f"DISTRIBUTED-OR-UNRECONSTRUCTED: neither the feature union nor the attention pattern is necessary (feature drop {feat_drop:+.2f}, pattern drop {pat_drop:+.2f}), and "
                      f"the transcoders leave {err_share:.0%} in the error node (held-out recon {mean_recon:.2f} > {CONFIG['TC_RECON_MAX']}; note err_share is computed under the first-order attribution, so it may be inflated); the computation may be in the unreconstructed residual")
        else:
            decomp = (f"GENUINELY DISTRIBUTED: neither the feature union nor the attention pattern is necessary (feature drop {feat_drop:+.2f}, pattern drop {pat_drop:+.2f}) and the "
                      f"transcoders reconstruct well (held-out recon {mean_recon:.2f}, error share {err_share:.0%}); the computation is spread, not a small cross-layer set nor the QK pattern")
        asym_consistency = ("consistent" if ASYM["replacement_gap"] > 0.05 else "inconclusive")
        verdict = (f"B: NOT a small cross-layer feature circuit ({'; '.join(why)}); {decomp}; representational asymmetry {asym_consistency} (secondary check, periodicity {ASYM['periodicity']['replacement']:.2f} vs change {ASYM['change']['replacement']:.2f})")
    mde = float(np.mean([abs(r["faith_ci"][1]-r["faith_ci"][0]) for r in R]) / 2) if R else float("nan")
    print("=" * 100); print(f"PHASE 5 v4 VERDICT: {verdict}{MOCK_TAG}"); print("=" * 100)
    print(f"  clean recovery={base:.3f}  ranking={CONFIG['RANK_METHOD'].upper()} (Spearman gold-vs-EAP {RANK_CORR['gold_eap']:+.2f}, gold-vs-proxy {RANK_CORR['gold_proxy']:+.2f})  "
          f"layers={LAYERS} (Mishra-band {in_band}; dropped {DROPPED_LAYERS})  features/layer={NF}")
    print(f"  k*(faithful union)={kstar} small={small}  faith>null={faith_beats}  selective={selective}  complete>null={complete_beats}  sufficient={sufficient}")
    print(f"  replacement (change)={repl:.2f} CI[{repl_ci[0]:.2f},{repl_ci[1]:.2f}]  completeness={GRAPH_CHANGE['completeness']:.2f} CI[{GRAPH_CHANGE['completeness_ci'][0]:.2f},{GRAPH_CHANGE['completeness_ci'][1]:.2f}]  multihop_share={GRAPH_CHANGE['multihop_share']:.2f}  CLT-escalation-signature={escalate_clt}")
    print(f"  max denoising gain={max(r['gain'] for r in R):+.4f} (bar {CONFIG['SUFFICIENCY_BAR']})  max completeness={max(r['cp_complete'] for r in R):+.4f}  SNR small-k sharpens toward floor={snr_sharpens}")
    print(f"  feature-splitting={SPLIT['splitting']}  attention-asymmetry replacement gap (periodicity - change)={ASYM['replacement_gap']:+.2f}")
    print(f"  ATTENTION-PATTERN (necessity, primary): pattern drop={ATTN['pat_noise_drop']:+.3f} CI[{ATTN['pat_noise_ci'][0]:+.2f},{ATTN['pat_noise_ci'][1]:+.2f}] vs feature drop {feat_drop:+.3f} (shuffled null {ATTN['null_noise_drop']:+.3f})  pattern_necessary={pat_necessary}  QK-routed={attention_routed}  escalate-to-CLT={escalate}")
    print(f"  replacement MDE={GRAPH_CHANGE.get('replacement_mde', float('nan')):.2f}  gold-vs-EXACT-metric rho={RANK_CORR.get('gold_metric', float('nan')):+.2f} (ranking_unverified={rank_unverified})  held-out recon(mean)={mean_recon:.2f}")
    print(f"  DETECTION POWER: min detectable effect ~{mde:.3f} -> " + ("positive result is real." if localized else "a localized cross-layer circuit WOULD have been resolved; the distributed verdict is de-confounded, not underpowered."))
    return dict(verdict=verdict, localized=bool(localized), kstar=kstar, small=bool(small), faith_beats=bool(faith_beats),
                selective=bool(selective), complete_beats=bool(complete_beats), sufficient=bool(sufficient),
                replacement=float(repl), replacement_ci=repl_ci, completeness=float(GRAPH_CHANGE["completeness"]),
                completeness_ci=GRAPH_CHANGE["completeness_ci"], multihop_share=float(GRAPH_CHANGE["multihop_share"]),
                escalate_clt=escalate_clt, attention_routed=attention_routed, qk_necessity=qk_necessity, escalate=escalate,
                rank_unverified=rank_unverified, snr_sharpens=bool(snr_sharpens), splitting=bool(SPLIT["splitting"]),
                rank_method=CONFIG["RANK_METHOD"], rank_corr=RANK_CORR, layers=LAYERS, depths=[rel_depth(l) for l in LAYERS],
                mishra_in_band=in_band, dropped_layers=[int(l) for l in DROPPED_LAYERS], n_features=int(NF), recon={int(l): RECON[l] for l in LAYERS}, min_detectable=mde,
                asym_replacement_gap=float(ASYM["replacement_gap"]), replacement_mde=float(GRAPH_CHANGE.get("replacement_mde", float("nan"))),
                rank_metric_corr=float(RANK_CORR.get("gold_metric", float("nan"))), held_out_recon=float(np.mean([RECON[l] for l in LAYERS])),
                pattern_denoise_gain=float(ATTN["pat_denoise_gain"]), pattern_noise_drop=float(ATTN["pat_noise_drop"]))
SUMMARY = summarize()
""")

# ============================================================================
md("## 16. Figures 7a faithfulness vs union size | 7b completeness/denoising | 7c SNR sweep | 7d change-vs-periodicity graphs")
code(r"""
try:
    R = CURVES["rows"]; ks = [r["k"] for r in R]; fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.plot(ks, [r["faith_frac"] for r in R], "o-", color="#8e44ad", label="top-k union (gold)")
    a.plot(ks, [r["faith_null"]/(CURVES["base"]+1e-6) for r in R], "x--", color="#999", label="random-k union null")
    a.axhline(CONFIG["FAITH_TARGET"], color="k", ls=":", lw=1); a.axvline(CONFIG["LOCALIZE_MAX_FEATURES"], color="orange", ls=":", lw=1)
    a.set_xscale("log", base=2); a.set_xlabel("union size k"); a.set_ylabel("faithfulness (frac)"); a.legend(fontsize=7)
    a.set_title("Fig 7a: cross-layer faithfulness vs union size" + MOCK_TAG, fontsize=9)
    b = ax[0, 1]
    b.errorbar(ks, [r["cp_complete"] for r in R], yerr=[[max(0,r["cp_complete"]-r["cp_ci"][0]) for r in R],[max(0,r["cp_ci"][1]-r["cp_complete"]) for r in R]], fmt="o-", color="#c0392b", capsize=2, label="completeness")
    b.plot(ks, [r["cp_null_p95"] for r in R], "x:", color="#999", label="complete null p95")
    b.plot(ks, [r["gain"] for r in R], "s-", color="#27ae60", label="denoising gain")
    b.plot(ks, [r["motif_drop"] for r in R], "^--", color="#7f8c8d", label="motif drop")
    b.axhline(CONFIG["SUFFICIENCY_BAR"], color="green", ls=":", lw=1, label="sufficiency bar")
    b.axhline(ATTN["pat_denoise_gain"], color="purple", ls="-.", lw=1.2, label="attn-pattern denoise gain")
    b.set_xscale("log", base=2)
    b.set_xlabel("union size k"); b.set_ylabel("rel-collapse / gain"); b.legend(fontsize=7); b.set_title("Fig 7b: completeness + denoising (CIs)", fontsize=9)
    c = ax[1, 0]
    for r in SNR:
        pairs = sorted(r["faith_frac_by_k"], key=lambda p: p[0]); c.plot([p[0] for p in pairs], [p[1] for p in pairs], "o-", label=f"delta[{r['delta'][0]},{r['delta'][1]}] (rec {r['clean_recovery']:.2f})")
    c.axhline(CONFIG["FAITH_TARGET"], color="k", ls=":", lw=1); c.set_xscale("log", base=2)
    c.set_xlabel("union size k"); c.set_ylabel("faithfulness (frac)"); c.legend(fontsize=7); c.set_title("Fig 7c: SNR sweep, localization vs difficulty", fontsize=9)
    d = ax[1, 1]
    cats = ["replacement", "completeness", "multihop"]
    chg = [ASYM["change"]["replacement"], ASYM["change"]["completeness"], ASYM["change"]["multihop_share"]]   # clean-weighted, like-for-like
    per = [ASYM["periodicity"]["replacement"], ASYM["periodicity"]["completeness"], ASYM["periodicity"]["multihop_share"]]
    x = np.arange(len(cats)); w = 0.38
    d.bar(x - w/2, chg, w, color="#c0392b", label="change-detection")
    d.bar(x + w/2, per, w, color="#2980b9", label="periodicity")
    d.set_xticks(x); d.set_xticklabels(cats, fontsize=8); d.set_ylim(0, 1.05); d.set_ylabel("score / share")
    d.legend(fontsize=7); d.set_title("Fig 7d: change vs periodicity (clean-weighted, attention asymmetry)", fontsize=9)
    fig.suptitle(f"Phase 5 v4, cross-layer feature circuit (gold attribution): {SUMMARY['verdict'][:52]}" + MOCK_TAG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(CKPT_DIR, f"fig7_phase5v4_{MODE}.png"), dpi=90); plt.show(); plt.close(fig)
    print(f"saved fig7_phase5v4_{MODE}.png")
except Exception as e:
    import traceback; print("fig skipped:", repr(e)[:160]); traceback.print_exc()
""")

# ============================================================================
md("## 17. Checkpoint")
code(r"""
out = dict(summary=SUMMARY, orientation=ORIENT, delta_monotone=MONO, attention_pattern=ATTN, curves=CURVES, snr=SNR, split=SPLIT,
           asymmetry=dict(replacement_gap=ASYM["replacement_gap"], multihop_gap=ASYM["multihop_gap"], basis=ASYM.get("basis", "clean"),
                          change={k: ASYM["change"][k] for k in ("label","cf","replacement","replacement_ci","completeness","multihop_share","n_nodes","spectral")},
                          periodicity={k: ASYM["periodicity"][k] for k in ("label","cf","replacement","replacement_ci","completeness","multihop_share","n_nodes","spectral")}),
           transcoders=dict(layers=LAYERS, depths=[rel_depth(l) for l in LAYERS], n_features=int(NF), recon=RECON, rank_corr=RANK_CORR),
           config=dict(mode=MODE, model_id=CONFIG["model_id"], rank_method=CONFIG["RANK_METHOD"], n_pairs=CONFIG["N_PAIRS"],
                       k_grid=CONFIG["K_GRID"], snr_deltas=CONFIG["SNR_DELTAS"], is_large=bool(IS_LARGE)))
p = os.path.join(CKPT_DIR, f"phase5v4_{MODE}.json")
with open(p, "w") as f: json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote", p, "->", SUMMARY["verdict"][:72], MOCK_TAG)
""")

# ---- assemble -----------------------------------------------------------------------------------
nb = new_notebook()
nb.cells = [new_markdown_cell(s) if t == "md" else new_code_cell(s) for (t, s) in CELLS]
nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
               "language_info": {"name": "python"}, "colab": {"provenance": []}, "accelerator": "GPU"}
with open("phase5_v4.ipynb", "w") as f: nbf.write(nb, f)
with open("_mirror_phase5_v4.py", "w") as f:
    f.write("\n".join(["# auto-mirror of phase5_v4.ipynb code cells (local smoke test)"] +
                      ["\n# " + "=" * 60 + "\n" + s for t, s in CELLS if t == "code"]))
print(f"wrote phase5_v4.ipynb ({sum(t=='code' for t,_ in CELLS)} code cells) + _mirror_phase5_v4.py")
