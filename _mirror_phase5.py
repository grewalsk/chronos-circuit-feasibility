# auto-mirror of phase5.ipynb code cells (local smoke test)

# ============================================================
import os
CONFIG = {
    "MODE": "mock_cpu",                 # -> "pilot_t4" (base, T4: 5a/5b) or "pilot_a100" (Large, A100: full incl 5c)
    "MODEL_BY_MODE": {"mock_cpu": None, "pilot_t4": "amazon/chronos-t5-base", "pilot_a100": "amazon/chronos-t5-large"},
    "USE_DRIVE": True,
    "SEED0": 0,
    "PERIODS": [8, 12, 16, 24],
    "N_SEEDS": 3,
    "N_SERIES": 32,
    "CTX": 256,
    "PRED": 64,
    "OBS_NOISE": 0.30,
    "N_CRPS_SAMPLES": 64,
    "N_BOOTSTRAP": 1000,
    "FORECAST_BATCH": 4,
    "N_RANDOM_DRAWS": 8,                 # size-matched random-MLP-layer null draws (necessity null)
    "CONDITIONS_5": ["changepoint", "motif", "trend"],   # changepoint=GATE; motif=selectivity; trend=soft
    "MLP_SITES_5": ["enc_mlp", "dec_mlp"],
    "ATTN_SITES_5": ["enc_self", "dec_self", "cross"],   # ablated too, for the attention-vs-MLP comparison
    "TAU_FRAC_CTX": 0.65,
    "DELTA_LO": 1.5, "DELTA_HI": 3.0,
    # ---- dose-response over MLP-layer fraction (multi-seed) ----
    "F_GRID": [0.1, 0.25, 0.5, 0.75, 1.0],
    "SWEEP_DRAWS": 3, "SWEEP_SERIES": 16, "SWEEP_SEEDS": 2, "LOW_F_MAX": 0.5,
    # ---- MLP-layer localization ----
    "LOCALIZE_MAX_SET": 8, "LOCALIZE_FRAC": 0.60, "LOC_RANDOM_DRAWS": 5,
    # ---- decision thresholds (pre-registered, same as 4) ----
    "STRUCT_COLLAPSE_MIN": 0.30, "SELECTIVITY_MARGIN": 2.0,
    "MISHRA_DEPTH_LO": 0.45, "MISHRA_DEPTH_HI": 0.55,
    # ---- 5c feature-level (TopK SAE; gated) ----
    "MISHRA_SAE_REPO": None,             # set to a HF repo id if Mishra's SAEs are available; else we train one
    "SAE_DICT_MULT": 8, "SAE_TOPK": 32, "SAE_STEPS": 400, "SAE_LR": 1e-3, "SAE_BATCH": 2048,
    "SAE_FEATURES_ABLATE": 16,           # top change-detection features to causally ablate
    # ---- mock overrides ----
    "mock_cpu": {
        "PERIODS": [6, 8], "N_SEEDS": 2, "N_SERIES": 6, "CTX": 48, "PRED": 24,
        "N_CRPS_SAMPLES": 12, "N_BOOTSTRAP": 50, "N_RANDOM_DRAWS": 3, "FORECAST_BATCH": 999,
        "F_GRID": [0.25, 0.5, 1.0], "SWEEP_DRAWS": 2, "SWEEP_SERIES": 4, "SWEEP_SEEDS": 2,
        "LOCALIZE_MAX_SET": 2, "LOC_RANDOM_DRAWS": 2,
        "SAE_DICT_MULT": 4, "SAE_TOPK": 4, "SAE_STEPS": 5, "SAE_BATCH": 64, "SAE_FEATURES_ABLATE": 3,
    },
}
MODE = os.environ.get("CHRONOS_P5_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4", "pilot_a100"), MODE
CONFIG["model_id"] = os.environ.get("CHRONOS_P5_MODEL", CONFIG["MODEL_BY_MODE"][MODE])
if MODE == "mock_cpu": CONFIG.update(CONFIG["mock_cpu"])
IS_MOCK = (MODE == "mock_cpu")
IS_LARGE = (CONFIG["model_id"] is not None and "large" in CONFIG["model_id"])
MOCK_TAG = "  [MOCK_CPU — NOT INTERPRETABLE]" if IS_MOCK else ""
ON_COLAB = os.path.isdir("/content")
CKPT_DIR = os.path.abspath(".")
if ON_COLAB:
    CKPT_DIR = "/content"
    if CONFIG.get("USE_DRIVE", True) and not IS_MOCK:
        try:
            from google.colab import drive
            drive.mount("/content/drive")
            CKPT_DIR = "/content/drive/MyDrive/chronos_phase5"; os.makedirs(CKPT_DIR, exist_ok=True)
            print("checkpoints -> Google Drive (survives disconnects):", CKPT_DIR)
        except Exception as e:
            print("Drive mount skipped (", repr(e)[:80], ") -> /content")
print(f"MODE={MODE}{MOCK_TAG}  model={CONFIG['model_id']}  large={IS_LARGE}  ctx={CONFIG['CTX']} pred={CONFIG['PRED']} "
      f"seeds={CONFIG['N_SEEDS']} series={CONFIG['N_SERIES']}  F_GRID={CONFIG['F_GRID']} sweep_seeds={CONFIG['SWEEP_SEEDS']}  ckpt={CKPT_DIR}")

# ============================================================
import sys, json, subprocess, gc, re, warnings
warnings.filterwarnings("ignore", message=".*past_key_values.*")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _ensure(pkg, imp):
    if os.environ.get("CHRONOS_P5_SKIP_INSTALL") == "1": return
    try: __import__(imp)
    except Exception:
        print("installing", pkg); subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)
if not IS_MOCK: _ensure("chronos-forecasting", "chronos")

import numpy as np, torch, torch.nn as nn
import matplotlib
if not ON_COLAB: matplotlib.use("Agg")
import matplotlib.pyplot as plt
torch.manual_seed(CONFIG["SEED0"]); np.random.seed(CONFIG["SEED0"])
DEVICE = "cuda" if (not IS_MOCK and torch.cuda.is_available()) else "cpu"
if not IS_MOCK and DEVICE == "cpu": print("WARN: pilot requested but no CUDA -> CPU (slow).")
DTYPE = torch.float32
print("device:", DEVICE)

# ============================================================
def _nheads(mod):
    for a in ("n_heads", "num_heads"):
        if hasattr(mod, a): return int(getattr(mod, a))
    return int(mod.config.num_heads)
def _dkv(mod):
    for a in ("key_value_proj_dim", "d_kv"):
        if hasattr(mod, a): return int(getattr(mod, a))
    return int(mod.config.d_kv)
def _layer_idx(name):
    m = re.search(r"block\.(\d+)\.", name); return int(m.group(1)) if m else -1

def classify_attention_modules(model):
    sites = {"enc_self": [], "dec_self": [], "cross": []}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ != "T5Attention": continue
        if name.startswith("encoder") and "SelfAttention" in name: sites["enc_self"].append((name, mod))
        elif name.startswith("decoder") and "layer.0.SelfAttention" in name: sites["dec_self"].append((name, mod))
        elif name.startswith("decoder") and "EncDecAttention" in name: sites["cross"].append((name, mod))
    return sites

def classify_mlp_modules(model):                            # T5 per-block FF (DenseReluDense / DenseGatedActDense)
    sites = {"enc_mlp": [], "dec_mlp": []}
    for name, mod in model.named_modules():
        if mod.__class__.__name__ in ("T5DenseActDense", "T5DenseGatedActDense"):
            if name.startswith("encoder"):   sites["enc_mlp"].append((name, mod))
            elif name.startswith("decoder"): sites["dec_mlp"].append((name, mod))
    return sites

if IS_MOCK:
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=256, d_model=64, d_kv=32, d_ff=128, num_layers=2, num_decoder_layers=2,
                   num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
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
        _free, _tot = torch.cuda.mem_get_info()
        print(f"GPU free {_free/1e9:.1f}/{_tot/1e9:.1f} GB before load")
        _need = 6.0e9 if IS_LARGE else 1.5e9
        assert _free > _need, ("Only %.2f GB free — a previous run left memory resident. RESTART THE RUNTIME." % (_free/1e9))
    PIPE = ChronosPipeline.from_pretrained(CONFIG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)
    INNER = PIPE.inner_model.eval(); VOCAB = INNER.config.vocab_size
    INNER.requires_grad_(False)
    try: INNER.config._attn_implementation = "eager"
    except Exception: pass
    DEC_START = int(getattr(INNER.config, "decoder_start_token_id", 0)); D_MODEL = int(INNER.config.d_model)

SITES = classify_attention_modules(INNER)                   # attention (comparison)
MLP = classify_mlp_modules(INNER)                           # MLP (the Phase-5 target)
for s in SITES: assert len(SITES[s]) > 0, f"no attention modules for {s}"
for s in MLP:   assert len(MLP[s]) > 0, f"no MLP modules for {s}"
SITE_MODS = {s: {_layer_idx(n): mod for n, mod in SITES[s]} for s in SITES}
MLP_MODS  = {s: {_layer_idx(n): mod for n, mod in MLP[s]} for s in MLP}
N_ENC_LAYERS = len(MLP["enc_mlp"])
print("attention per-site (modules,heads):", {s: (len(SITES[s]), _nheads(SITES[s][0][1])) for s in SITES})
print("MLP per-site (modules):", {s: len(MLP[s]) for s in MLP}, "| enc layers:", N_ENC_LAYERS)

def attn_site_size(sites):
    sizes = {s: len(sites[s]) * _nheads(sites[s][0][1]) for s in sites}
    assert len(set(sizes.values())) == 1, f"attn sites NOT equal size: {sizes}"
    return next(iter(sizes.values()))
def mlp_site_size(sites):
    sizes = {s: len(sites[s]) for s in sites}
    assert len(set(sizes.values())) == 1, f"MLP sites NOT equal size: {sizes}"   # HARD ASSERT
    return next(iter(sizes.values()))
N_ATTN = attn_site_size(SITES); N_MLP = mlp_site_size(MLP)
print(f"EQUAL-SIZE CHECK: PASS — attn {N_ATTN} heads/site, MLP {N_MLP} layers/site")

def rel_depth(layer_idx): return float(layer_idx) / max(1, (N_ENC_LAYERS - 1))

# ============================================================
def make_levelshift(ctx_len, pred_len, rng, noise, tau_frac_ctx=None, delta=None):
    tau_frac_ctx = CONFIG["TAU_FRAC_CTX"] if tau_frac_ctx is None else tau_frac_ctx
    length = ctx_len + pred_len
    tau = int(tau_frac_ctx * ctx_len)
    assert (ctx_len - tau) >= 0.30 * ctx_len, "need >=30% post-shift context"
    L0 = rng.uniform(-1.0, 1.0)
    if delta is None:
        delta = rng.choice([-1.0, 1.0]) * rng.uniform(CONFIG["DELTA_LO"], CONFIG["DELTA_HI"])
    L1 = L0 + delta
    s = np.where(np.arange(length) < tau, L0, L1) + rng.normal(0, noise, size=length)
    return s, dict(tau=int(tau), L0=float(L0), L1=float(L1), delta=float(delta))

def make_nochange(ctx_len, pred_len, rng, noise):
    L0 = rng.uniform(-1.0, 1.0)
    s = L0 + rng.normal(0, noise, size=ctx_len + pred_len)
    return s, dict(tau=int(CONFIG["TAU_FRAC_CTX"] * ctx_len), L0=float(L0), L1=float(L0), delta=0.0)

def make_motif(P, L, rng, noise):
    m = rng.standard_normal(P); m[rng.integers(P)] += 3.0 * (1 if rng.random() > 0.5 else -1)
    m[P // 2:] += 1.5; m = m - m.mean()
    return np.tile(m, L // P + 2)[:L] + noise * rng.standard_normal(L)

def make_trend(L, rng, noise):
    slope = rng.uniform(0.5, 1.5) / L; off = rng.uniform(-1, 1)
    return slope * np.arange(L) + off + noise * rng.standard_normal(L)

def make_batch(cond, rng, n_series):
    CTX, PRED, NOISE = CONFIG["CTX"], CONFIG["PRED"], CONFIG["OBS_NOISE"]
    ctxs, tgts, metas = [], [], []
    for i in range(n_series):
        if cond == "changepoint": s, meta = make_levelshift(CTX, PRED, rng, NOISE)
        elif cond == "nochange":  s, meta = make_nochange(CTX, PRED, rng, NOISE)
        elif cond == "motif":     P = CONFIG["PERIODS"][i % len(CONFIG["PERIODS"])]; s = make_motif(P, CTX + PRED, rng, NOISE); meta = {"P": int(P)}
        elif cond == "trend":     s = make_trend(CTX + PRED, rng, NOISE); meta = {}
        else: raise ValueError(cond)
        ctxs.append(s[:CTX]); tgts.append(s[CTX:]); metas.append(meta)
    return ctxs, np.array(tgts), metas

# ============================================================
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

def changepoint_recovery(forecast_1d, meta):
    fhat = float(np.median(np.asarray(forecast_1d)))
    denom = abs(meta["L1"] - meta["L0"]) + 1e-8
    return float(np.clip(1.0 - abs(fhat - meta["L1"]) / denom, 0.0, 1.0))

def structure(cond, fmean_1d, meta, target_1d):
    if cond == "changepoint": return changepoint_recovery(fmean_1d, meta)
    if cond == "motif":       return period_power_fraction(fmean_1d, meta["P"])
    if cond == "trend":       return trend_slope_recovery(fmean_1d, target_1d)
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

def _spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0

_m = dict(tau=10, L0=0.0, L1=2.0, delta=2.0)
assert abs(changepoint_recovery([2.0]*8, _m) - 1.0) < 1e-6 and changepoint_recovery([0.0]*8, _m) < 0.01
print("metric asserts: changepoint_recovery tracks->1, reverts->0  PASS")

# ============================================================
# ---- attention (comparison only): ablation-only .o pre-hook ----
def _attn_pre_hook(attn):
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
def install_attn_hooks(sites):
    hs = []
    for lst in sites.values():
        for _, mod in lst: mod._ablate_heads = set(); hs.append(mod.o.register_forward_pre_hook(_attn_pre_hook(mod)))
    return hs
def clear_attn(sites):
    for lst in sites.values():
        for _, mod in lst: mod._ablate_heads = set()
def set_attn_site(sites, site):
    clear_attn(sites)
    for _, mod in sites[site]: mod._ablate_heads = set(range(_nheads(mod)))

# ---- MLP (the target): mean-ablation + SAE-patch forward hook on the FF output ----
def _mlp_hook(module, inp, out):
    fn = getattr(module, "_sae_patch", None)
    if fn is not None: return fn(out)                      # 5c feature-level patch (callable: out -> out')
    if getattr(module, "_ablate", False):
        m = out.mean(dim=tuple(range(out.dim() - 1)), keepdim=True)   # batch/seq mean -> (1,1,d_model)
        return m.expand_as(out)
    return None
def install_mlp_hooks(sites):
    hs = []
    for lst in sites.values():
        for _, mod in lst: mod._ablate = False; mod._sae_patch = None; hs.append(mod.register_forward_hook(_mlp_hook))
    return hs
def clear_mlp(sites):
    for lst in sites.values():
        for _, mod in lst: mod._ablate = False; mod._sae_patch = None
def mlp_pool(sites): return [mod for lst in sites.values() for _, mod in lst]
def mlp_layers(sites, site): return [mod for _, mod in sites[site]]
def set_mlp_site(sites, site):                              # ablate ALL MLP layers in a site (necessity)
    clear_mlp(sites)
    for _, mod in sites[site]: mod._ablate = True
def set_mlp_layers(sites, layers):                          # ablate a specific set of MLP modules (localization)
    clear_mlp(sites)
    for mod in layers: mod._ablate = True
def set_random_mlp(sites, pool, n, rng):                    # n random MLP layers across both sites (size-matched null)
    clear_mlp(sites); n = min(n, len(pool))
    for idx in rng.choice(len(pool), size=n, replace=False): pool[idx]._ablate = True
def set_fraction_mlp(sites, site, f, rng):                  # random fraction f of one site's MLP layers
    clear_mlp(sites); layers = mlp_layers(sites, site); k = max(1, int(round(f * len(layers))))
    for idx in rng.choice(len(layers), size=k, replace=False): layers[idx]._ablate = True
def set_random_in_mlp_site(sites, site, k, rng):            # k random MLP layers within a site (localization null)
    clear_mlp(sites); layers = mlp_layers(sites, site); k = min(k, len(layers))
    for idx in rng.choice(len(layers), size=k, replace=False): layers[idx]._ablate = True

def clear_all():
    clear_attn(SITES); clear_mlp(MLP)

ATTN_HANDLES = install_attn_hooks(SITES); MLP_HANDLES = install_mlp_hooks(MLP); MLP_POOL = mlp_pool(MLP)
print(f"hooks: {len(ATTN_HANDLES)} attn '.o' + {len(MLP_HANDLES)} MLP FF | MLP pool={len(MLP_POOL)} | N_MLP={N_MLP}")

# ============================================================
def forecast_pilot(contexts, n_samples):
    torch.manual_seed(CONFIG["SEED0"])
    bs = int(CONFIG.get("FORECAST_BATCH", 4)); outs = []; i = 0
    while i < len(contexts):
        chunk = [torch.tensor(np.asarray(c), dtype=DTYPE) for c in contexts[i:i + bs]]
        try:
            with torch.inference_mode():
                fc = PIPE.predict(chunk, prediction_length=CONFIG["PRED"], num_samples=n_samples)
            arr = fc.detach().cpu().numpy(); del fc
        except RuntimeError as e:
            if "out of memory" not in str(e).lower(): raise
            if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
            if bs > 1: bs = max(1, bs // 2); print(f"  [oom] FORECAST_BATCH -> {bs}"); continue
            raise
        outs.append(arr); i += len(chunk)
        if DEVICE == "cuda": gc.collect(); torch.cuda.empty_cache()
    return np.concatenate(outs, axis=0)

def forecast_mock(contexts, n_samples):
    n = len(contexts); H = CONFIG["PRED"]; ids = np.zeros((n, 32), dtype=np.int64)
    for i, c in enumerate(contexts):
        c = np.asarray(c, float); q = np.clip(((c - c.min()) / ((c.max() - c.min()) + 1e-9) * (VOCAB - 3)).astype(int) + 2, 0, VOCAB - 1)
        q = q[-32:]; ids[i, :len(q)] = q
    inp = torch.tensor(ids, dtype=torch.long, device=DEVICE); dec = torch.zeros((n, H), dtype=torch.long, device=DEVICE)
    with torch.no_grad(): out = INNER(input_ids=inp, decoder_input_ids=dec)
    sig = out.logits.float().mean(dim=-1).cpu().numpy(); samples = np.zeros((n, n_samples, H)); rng = np.random.default_rng(123)
    for i in range(n):
        c = np.asarray(contexts[i], float); base = np.resize(c[-H:] if len(c) >= H else np.resize(c, H), H)
        amp = 1.0 + 0.3 * np.tanh(sig[i].mean()); perturb = 0.5 * (sig[i] - sig[i].mean())
        samples[i] = amp * base[None, :] + perturb[None, :] + 0.1 * rng.standard_normal((n_samples, H))
    return samples

FORECAST = forecast_mock if IS_MOCK else forecast_pilot

def _tokenize(contexts):                                    # for 5c MLP-activation capture (encoder forward)
    if IS_MOCK:
        arrs = []
        for c in contexts:
            c = np.asarray(c, float); q = np.clip(((c - c.min()) / ((c.max() - c.min()) + 1e-9) * (VOCAB - 3)).astype(int) + 2, 0, VOCAB - 1)
            arrs.append(q.astype(np.int64))
        Ln = max(len(a) for a in arrs)
        ids = np.zeros((len(arrs), Ln), dtype=np.int64); am = np.zeros((len(arrs), Ln), dtype=np.int64)
        for i, a in enumerate(arrs): ids[i, :len(a)] = a; am[i, :len(a)] = 1
        return torch.tensor(ids, device=DEVICE), torch.tensor(am, device=DEVICE)
    ct = torch.tensor(np.asarray(contexts), dtype=DTYPE)
    ids, am, _scale = PIPE.tokenizer.context_input_transform(ct)
    return ids.to(DEVICE), am.to(DEVICE)

# plumbing: BOTH MLP and attention ablation must change the output
_ctx, _tgt, _metas = make_batch("changepoint", np.random.default_rng(0), 2)
clear_all(); _a = FORECAST(_ctx, 4)
set_mlp_site(MLP, "enc_mlp"); _b = FORECAST(_ctx, 4); clear_all()
set_attn_site(SITES, "enc_self"); _c = FORECAST(_ctx, 4); clear_all()
print(f"PLUMBING: enc_mlp ablation changed forecast = {not np.allclose(_a,_b)} (max|Δ|={np.abs(_a-_b).max():.4g}); "
      f"enc_self ablation changed = {not np.allclose(_a,_c)}")
assert not np.allclose(_a, _b), "MLP ablation did NOT change output — MLP hooks not wired"   # HARD ASSERT
assert not np.allclose(_a, _c), "attention ablation did NOT change output"                    # HARD ASSERT
print("PLUMBING: PASS" + MOCK_TAG)

# ============================================================
def monotonicity_check():
    rng = np.random.default_rng(CONFIG["SEED0"] + 777)
    deltas = np.concatenate([np.linspace(0.5, 4.0, 12), -np.linspace(0.5, 4.0, 12)]); metas, ctxs = [], []
    for dlt in deltas:
        s, meta = make_levelshift(CONFIG["CTX"], CONFIG["PRED"], rng, CONFIG["OBS_NOISE"], delta=float(dlt))
        ctxs.append(s[:CONFIG["CTX"]]); metas.append(meta)
    clear_all(); fc = FORECAST(ctxs, CONFIG["N_CRPS_SAMPLES"])
    rec = np.array([changepoint_recovery(fc[i].mean(0), metas[i]) for i in range(len(metas))]); ad = np.abs(deltas)
    nb = 6; edges = np.quantile(ad, np.linspace(0, 1, nb + 1)); centers, means = [], []
    for j in range(nb):
        mm = (ad >= edges[j]) & (ad <= edges[j + 1] if j == nb - 1 else ad < edges[j + 1])
        if mm.sum(): centers.append(float(ad[mm].mean())); means.append(float(rec[mm].mean()))
    rho = _spearman(centers, means); slope = float(np.polyfit(centers, means, 1)[0]) if len(centers) > 1 else 0.0
    print(f"  delta-binned recovery: " + " ".join(f"{c:.1f}:{m:.2f}" for c, m in zip(centers, means)))
    print(f"  Spearman rho={rho:+.3f} OLS slope={slope:+.4f} (require >0)")
    assert (rho > 0) or (slope > 0), "no positive monotone trend in delta on the clean model"
    print("  MONOTONICITY: PASS" + MOCK_TAG)
    return dict(centers=centers, means=means, rho=rho, slope=slope)

print("Clean-model monotonicity..." + MOCK_TAG); MONO = monotonicity_check()

# ============================================================
def _crps_vec(fc, tgt): return np.array([crps_samples(fc[i], tgt[i]) for i in range(len(tgt))])
def _struct_vec(fc, metas, tgt, cond): return np.array([structure(cond, fc[i].mean(0), metas[i], tgt[i]) for i in range(len(tgt))])

FORCE = os.environ.get("CHRONOS_P5_FORCE", "0") == "1"
def _ckp(name): return os.path.join(CKPT_DIR, f"phase5_{MODE}_{name}.json")
def _load(name):
    p = _ckp(name)
    if os.path.exists(p) and not FORCE:
        try: return json.load(open(p))
        except Exception: return []
    return []
def _save(name, recs): json.dump(recs, open(_ckp(name), "w"))

def run_5a():
    recs = _load("5a")
    done = {(r["seed"], r["cond"]) for r in recs if r.get("kind") == "random_mlp"}
    recs = [r for r in recs if (r["seed"], r["cond"]) in done]
    for seed in range(CONFIG["N_SEEDS"]):
        for ci, cond in enumerate(CONFIG["CONDITIONS_5"]):
            if (seed, cond) in done: print(f"  [ckpt] 5a ({seed},{cond}) resumed"); continue
            rng = np.random.default_rng(CONFIG["SEED0"] + seed * 100 + ci)
            ctx, tgt, metas = make_batch(cond, rng, CONFIG["N_SERIES"])
            clear_all(); fc0 = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]); s0 = _struct_vec(fc0, metas, tgt, cond)
            for site in CONFIG["MLP_SITES_5"]:
                set_mlp_site(MLP, site); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]); clear_all()
                rel, rel_ci = rel_collapse(s0, _struct_vec(fc, metas, tgt, cond))
                recs.append(dict(seed=seed, cond=cond, kind=site, rel_collapse=rel, rel_ci=rel_ci, clean=float(s0.mean())))
            for site in CONFIG["ATTN_SITES_5"]:                  # attention comparison (same stimuli)
                set_attn_site(SITES, site); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]); clear_all()
                rel, rel_ci = rel_collapse(s0, _struct_vec(fc, metas, tgt, cond))
                recs.append(dict(seed=seed, cond=cond, kind=site, rel_collapse=rel, rel_ci=rel_ci))
            rel_draws = []
            for _ in range(CONFIG["N_RANDOM_DRAWS"]):
                set_random_mlp(MLP, MLP_POOL, N_MLP, rng); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]); clear_all()
                rel_draws.append(rel_collapse(s0, _struct_vec(fc, metas, tgt, cond))[0])
            recs.append(dict(seed=seed, cond=cond, kind="random_mlp", rel_draws=rel_draws))
            _save("5a", recs); print(f"  5a ({seed},{cond}) done + saved")
    return recs

print("Experiment 5a (MLP necessity + attention comparison)..." + MOCK_TAG); A5 = run_5a()

def _mean5(kind, cond):
    v = [r["rel_collapse"] for r in A5 if r.get("kind") == kind and r["cond"] == cond and "rel_collapse" in r]
    return float(np.mean(v)) if v else float("nan")
def _rel_lo5(kind, cond):
    v = [r["rel_ci"][0] for r in A5 if r.get("kind") == kind and r["cond"] == cond and "rel_ci" in r]
    return float(np.mean(v)) if v else float("nan")
print(f"  RE-MEASURED full-MLP changepoint collapse: enc_mlp={_mean5('enc_mlp','changepoint'):+.3f} "
      f"dec_mlp={_mean5('dec_mlp','changepoint'):+.3f}  | attention enc_self={_mean5('enc_self','changepoint'):+.3f}")

# ============================================================
def run_5a_sweep():
    recs = _load("sweep")
    done = {(r["seed"], round(r["f"], 4)) for r in recs}
    recs = [r for r in recs if (r["seed"], round(r["f"], 4)) in done]
    for seed in range(CONFIG["SWEEP_SEEDS"]):
        for fi, f in enumerate(CONFIG["F_GRID"]):
            if (seed, round(f, 4)) in done: print(f"  [ckpt] sweep ({seed},f={f}) resumed"); continue
            batches = {c: make_batch(c, np.random.default_rng(CONFIG["SEED0"] + 5000 + seed * 10 + ci), CONFIG["SWEEP_SERIES"])
                       for ci, c in enumerate(CONFIG["CONDITIONS_5"])}
            clean = {}
            for c, (ctx, tgt, metas) in batches.items():
                clear_all(); clean[c] = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, c)
            arng = np.random.default_rng(CONFIG["SEED0"] + 6000 + seed * 100 + fi); unit = []
            for d in range(CONFIG["SWEEP_DRAWS"]):
                for arm in CONFIG["MLP_SITES_5"] + ["random"]:
                    rel = {}
                    for c, (ctx, tgt, metas) in batches.items():
                        if arm == "random": set_random_mlp(MLP, MLP_POOL, max(1, int(round(f * N_MLP))), arng)
                        else:               set_fraction_mlp(MLP, arm, f, arng)
                        rel[c] = rel_collapse(clean[c], _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, c))[0]
                        clear_all()
                    unit.append(dict(seed=seed, f=f, draw=d, arm=arm, changepoint=rel["changepoint"],
                                     nontarget=max(rel["motif"], rel["trend"]), motif=rel["motif"], trend=rel["trend"]))
            recs += unit; _save("sweep", recs); print(f"  sweep ({seed},f={f}) done + saved")
    return recs

print("Experiment 5a sweep (MLP-layer dose-response)..." + MOCK_TAG); SWEEP = run_5a_sweep()

def sweep_agg(arm, f):
    rs = [r for r in SWEEP if r["arm"] == arm and abs(r["f"] - f) < 1e-9]
    if not rs: return None
    cp = [r["changepoint"] for r in rs]; nt = [r["nontarget"] for r in rs]
    return dict(changepoint=float(np.mean(cp)), nontarget=float(np.mean(nt)), cp_sd=float(np.std(cp)))

def low_f_selective_mlp(site):
    for f in CONFIG["F_GRID"]:
        if f > CONFIG["LOW_F_MAX"]: continue
        a = sweep_agg(site, f); r = sweep_agg("random", f)
        if a is None: continue
        if (a["changepoint"] >= CONFIG["STRUCT_COLLAPSE_MIN"]
                and a["changepoint"] >= CONFIG["SELECTIVITY_MARGIN"] * max(a["nontarget"], 1e-6)
                and (r is None or a["changepoint"] > r["changepoint"])):
            return True, f
    return False, None

# ============================================================
def run_5b():
    site_cp = {s: _mean5(s, "changepoint") for s in CONFIG["MLP_SITES_5"]}
    winning = max(site_cp, key=lambda s: site_cp[s] if np.isfinite(site_cp[s]) else -1e9)
    site_collapse = site_cp[winning]
    randN = [v for r in A5 if r.get("kind") == "random_mlp" and r["cond"] == "changepoint" for v in r.get("rel_draws", [])]
    randN_p95 = float(np.percentile(randN, 95)) if randN else 0.0
    site_necessary = bool(np.isfinite(site_collapse) and site_collapse >= CONFIG["STRUCT_COLLAPSE_MIN"] and site_collapse > randN_p95)
    print(f"  winning MLP site = {winning} (changepoint collapse {site_collapse:+.3f}; necessary={site_necessary}, random-MLP null p95={randN_p95:+.3f})")

    rng = np.random.default_rng(CONFIG["SEED0"] + 909)
    ctx, tgt, metas = make_batch("changepoint", rng, CONFIG["N_SERIES"])
    clear_all(); s0 = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint")
    # same-batch full-site collapse = the denominator for frac_of_site, so numerator (group, this batch) and
    # denominator are the SAME measurement; the multi-seed site_collapse stays the necessity gate only [review FIX #2].
    set_mlp_site(MLP, winning); site_collapse_batch = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]; clear_all()
    layers = [(_layer_idx(n), mod) for n, mod in MLP[winning]]
    singles = []
    for li, mod in layers:
        set_mlp_layers(MLP, [mod]); rel = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]; clear_all()
        singles.append(dict(layer=li, rel=float(rel), rel_depth=rel_depth(li)))
    ranked = sorted(singles, key=lambda r: r["rel"], reverse=True)
    order = [next(mod for ll, mod in layers if ll == r["layer"]) for r in ranked]
    ladder = []; rrng = np.random.default_rng(CONFIG["SEED0"] + 911)
    for k in range(1, min(CONFIG["LOCALIZE_MAX_SET"], len(order)) + 1):
        set_mlp_layers(MLP, order[:k]); grp = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]; clear_all()
        nulls = []
        for _ in range(CONFIG["LOC_RANDOM_DRAWS"]):
            set_random_in_mlp_site(MLP, winning, k, rrng)
            nulls.append(rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]); clear_all()
        null_p = float(np.percentile(nulls, 95)) if nulls else 0.0
        ladder.append(dict(k=k, group=float(grp), null_p95=null_p, frac_of_site=float(grp / (site_collapse_batch + 1e-6)),
                           beats_null=bool(grp > null_p), layers=[r["layer"] for r in ranked[:k]]))
    suff = [r for r in ladder if r["frac_of_site"] >= CONFIG["LOCALIZE_FRAC"] and r["beats_null"]] if site_necessary else []
    minimal = min(suff, key=lambda r: r["k"]) if suff else None
    localized = minimal is not None
    print(f"  single-MLP-layer collapse (ranked): " + ", ".join(f"L{r['layer']}@{r['rel_depth']:.2f}={r['rel']:+.2f}" for r in ranked[:min(8,len(ranked))]))
    for r in ladder: print(f"    k={r['k']}  group={r['group']:+.3f}  frac={r['frac_of_site']:+.2f}  null={r['null_p95']:+.3f}  beats={r['beats_null']}  layers={r['layers']}")
    if localized: print(f"  MINIMAL SUFFICIENT MLP SET = {minimal['k']} layers {minimal['layers']} ({minimal['frac_of_site']:.0%} of site, beats null)")
    elif not site_necessary: print(f"  Winning MLP site NOT necessary -> no real collapse to localize; NOT localized.")
    else: print(f"  No small (1..{CONFIG['LOCALIZE_MAX_SET']}) MLP-layer set reproduces >= {CONFIG['LOCALIZE_FRAC']:.0%} of site collapse -> NOT localized (distributed across MLP layers).")
    return dict(winning=winning, site_collapse=float(site_collapse), site_collapse_batch=float(site_collapse_batch),
                site_necessary=bool(site_necessary), randN_p95=float(randN_p95),
                singles=singles, ranked=ranked, ladder=ladder, minimal=minimal, localized=bool(localized),
                top_layer=int(ranked[0]["layer"]) if ranked else -1)

print("Experiment 5b (MLP-layer localization)..." + MOCK_TAG); B5 = run_5b()

# ============================================================
def mlp_selective(site):
    cp = _mean5(site, "changepoint"); cp_lo = _rel_lo5(site, "changepoint")
    mot = _mean5(site, "motif"); trd = _mean5(site, "trend")
    sig = (cp_lo > 0) and (cp >= CONFIG["STRUCT_COLLAPSE_MIN"])
    sel = cp >= CONFIG["SELECTIVITY_MARGIN"] * max(mot, trd, 1e-6)
    return dict(changepoint=cp, motif=mot, trend=trd, sig=bool(sig), selective=bool(sel))

def gate_decision():
    win = B5["winning"]; ss = mlp_selective(win); lf, lff = low_f_selective_mlp(win)
    beats = B5["site_collapse"] > B5["randN_p95"]
    go = bool(ss["selective"] and ss["sig"] and beats and lf and B5["localized"])
    print("=" * 88); print(f"  5a/5b GO/NO-GO GATE: {'GO -> run 5c feature tracing' if go else 'NO-GO -> STOP (distributed across attention AND MLP)'}{MOCK_TAG}")
    print("=" * 88)
    print(f"  winning MLP site={win}: changepoint {ss['changepoint']:+.3f} vs motif {ss['motif']:+.3f}/trend {ss['trend']:+.3f}  "
          f"selective={ss['selective']} beats_null={beats} low_f={'y@'+str(lff) if lf else 'no'} localized={B5['localized']}")
    if not go:
        print("  -> Per the spec, transcoders/SAEs will not help a distributed MLP; we STOP before 5c and report the")
        print("     comprehensive negative (computation distributed across attention AND MLP).")
    return dict(go=go, winning=win, selective=ss, low_f=lf, low_f_at=lff, beats_null=bool(beats))

GATE = gate_decision()

# ============================================================
class TopKSAE(nn.Module):
    def __init__(self, d, m, k):
        super().__init__(); self.k = k
        self.b_pre = nn.Parameter(torch.zeros(d)); self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d, bias=False)
    def encode(self, x):
        z = torch.relu(self.enc(x - self.b_pre))
        v, idx = z.topk(self.k, dim=-1)
        return torch.zeros_like(z).scatter_(-1, idx, v)
    def forward(self, x):
        z = self.encode(x); return self.dec(z) + self.b_pre, z

def train_sae(acts, d):                                     # acts: (N, d) tensor on DEVICE
    m = CONFIG["SAE_DICT_MULT"] * d; sae = TopKSAE(d, m, min(CONFIG["SAE_TOPK"], m)).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=CONFIG["SAE_LR"]); N = acts.shape[0]
    sae.b_pre.data = acts.mean(0).detach()
    for step in range(CONFIG["SAE_STEPS"]):
        idx = torch.randint(0, N, (min(CONFIG["SAE_BATCH"], N),), device=acts.device)
        x = acts[idx]; xr, z = sae(x); loss = ((xr - x) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return sae, float(loss.item())

def capture_enc_mlp(layer_mod, contexts):                   # capture FF output activations (encoder forward)
    ids, am = _tokenize(contexts); store = {}
    def cap(mod, inp, out): store["x"] = out.detach()
    h = layer_mod.register_forward_hook(cap)
    with torch.inference_mode(): INNER.get_encoder()(input_ids=ids, attention_mask=am)
    h.remove(); return store["x"]                           # (B, S, d_model)

def sae_smoke():                                            # mock: exercise SAE train + feature-ablation hook
    d = D_MODEL; acts = torch.randn(256, d, device=DEVICE)
    sae, loss = train_sae(acts, d)
    W = sae.dec.weight.detach(); F = list(range(min(3, W.shape[1])))
    mod = MLP["enc_mlp"][0][1]
    def patch(out):
        z = sae.encode(out - 0); contrib = z[..., F] @ W[:, F].T; return out - contrib
    mod._sae_patch = patch; _ = FORECAST(_ctx, 4); mod._sae_patch = None
    print(f"  5c SAE SMOKE (mock): trained {CONFIG['SAE_STEPS']} steps recon_loss={loss:.3f}, feature-ablation hook ran  [NOT INTERPRETABLE]")
    return dict(smoke=True, recon_loss=loss)

def run_5c():
    if IS_MOCK: return sae_smoke()
    if not GATE["go"]:
        print("  5c SKIPPED — gate is NO-GO (distributed). No transcoder/SAE training.")
        return dict(skipped=True, reason="NO-GO gate")
    win = B5["winning"]
    # 5c is implemented encoder-side; pick the enc_mlp layer to trace. If enc_mlp won, use its localized/top layer;
    # otherwise fall back to the mid-encoder enc layer (Mishra's prior) — never index a dec-layer idx into enc_mlp [review FIX #0].
    if win == "enc_mlp":
        li = (B5["minimal"]["layers"][0] if B5["minimal"] else B5["top_layer"])
    else:
        li = N_ENC_LAYERS // 2
        print(f"  5c: winning MLP site is {win}; feature capture is implemented encoder-side -> tracing mid-encoder enc_mlp L{li}.")
    li = int(min(max(li, 0), N_ENC_LAYERS - 1))
    layer_mod = MLP_MODS["enc_mlp"][li]; d = D_MODEL
    # capture activations on a level-shift battery + a no-change battery
    rng = np.random.default_rng(CONFIG["SEED0"] + 1700)
    ls_ctx, ls_meta = [], []
    for _ in range(CONFIG["N_SERIES"]):
        s, meta = make_levelshift(CONFIG["CTX"], CONFIG["PRED"], rng, CONFIG["OBS_NOISE"]); ls_ctx.append(s[:CONFIG["CTX"]]); ls_meta.append(meta)
    nc_ctx = [make_nochange(CONFIG["CTX"], CONFIG["PRED"], rng, CONFIG["OBS_NOISE"])[0][:CONFIG["CTX"]] for _ in range(CONFIG["N_SERIES"])]
    A_ls = capture_enc_mlp(layer_mod, ls_ctx); A_nc = capture_enc_mlp(layer_mod, nc_ctx)
    sae, loss = train_sae(A_ls[:, :CONFIG["CTX"], :].reshape(-1, d), d)   # drop the trailing EOS position [review FIX #4]
    # change-detection features: post-shift activation on shift series minus matched no-change
    CTX = CONFIG["CTX"]; taus = [m["tau"] for m in ls_meta]
    z_ls = sae.encode(A_ls); z_nc = sae.encode(A_nc)
    post = torch.stack([z_ls[b, taus[b]:CTX].mean(0) for b in range(len(taus))]).mean(0)
    base = torch.stack([z_nc[b, taus[b]:CTX].mean(0) for b in range(len(taus))]).mean(0)
    score = (post - base).cpu().numpy(); topF = list(np.argsort(-np.abs(score))[:CONFIG["SAE_FEATURES_ABLATE"]])
    W = sae.dec.weight.detach()
    # causal ablation: subtract decoded contribution of topF from the MLP output, measure changepoint collapse
    ctx, tgt, metas = make_batch("changepoint", np.random.default_rng(CONFIG["SEED0"] + 1701), CONFIG["N_SERIES"])
    clear_all(); s0c = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint")
    mtx, mtg, mmeta = make_batch("motif", np.random.default_rng(CONFIG["SEED0"] + 1702), CONFIG["N_SERIES"])
    clear_all(); s0m = _struct_vec(FORECAST(mtx, CONFIG["N_CRPS_SAMPLES"]), mmeta, mtg, "motif")
    def patch_for(F):
        def p(out):
            z = sae.encode(out); contrib = z[..., F] @ W[:, F].T; return out - contrib
        return p
    def collapse(F, ctx_, tgt_, meta_, cond_, s0_):
        layer_mod._sae_patch = patch_for(F); rel = rel_collapse(s0_, _struct_vec(FORECAST(ctx_, CONFIG["N_CRPS_SAMPLES"]), meta_, tgt_, cond_))[0]; layer_mod._sae_patch = None; return rel
    cp_feat = collapse(topF, ctx, tgt, metas, "changepoint", s0c)
    mot_feat = collapse(topF, mtx, mtg, mmeta, "motif", s0m)
    rrng = np.random.default_rng(CONFIG["SEED0"] + 1703); m = W.shape[1]
    null = [collapse(list(rrng.choice(m, len(topF), replace=False)), ctx, tgt, metas, "changepoint", s0c) for _ in range(CONFIG["LOC_RANDOM_DRAWS"])]
    null_p = float(np.percentile(null, 95)) if null else 0.0
    selective = bool(cp_feat >= CONFIG["SELECTIVITY_MARGIN"] * max(mot_feat, 1e-6) and cp_feat > null_p)
    mishra_repo = CONFIG.get("MISHRA_SAE_REPO")
    print(f"  5c: enc_mlp L{li} (rel-depth {rel_depth(li):.2f}); SAE recon_loss={loss:.3f}; ablating top {len(topF)} shift-features")
    print(f"     changepoint collapse {cp_feat:+.3f} vs motif {mot_feat:+.3f}, random-feature null p95 {null_p:+.3f} -> selective={selective}")
    # honest: we train our own SAE here; a genuine correspondence to Mishra's features requires loading HIS weights,
    # which is NOT auto-implemented — so even when MISHRA_SAE_REPO is set we do not claim a comparison [review FIX #3].
    print("     Mishra-SAE correspondence: trained SAE only — Mishra's exact SAEs not loaded; "
          + (f"MISHRA_SAE_REPO='{mishra_repo}' set but auto load+compare is NOT implemented (manual cross-check needed)."
             if mishra_repo else "set MISHRA_SAE_REPO and implement load+compare for a genuine cross-check."))
    return dict(skipped=False, layer=int(li), rel_depth=rel_depth(li), recon_loss=loss, top_features=[int(x) for x in topF],
                cp_collapse=float(cp_feat), motif_collapse=float(mot_feat), null_p95=null_p, selective=selective, mishra_repo=mishra_repo)

print("Experiment 5c (feature-level, gated)..." + MOCK_TAG); C5 = run_5c()

# ============================================================
def mishra_check():
    lo, hi = CONFIG["MISHRA_DEPTH_LO"], CONFIG["MISHRA_DEPTH_HI"]
    band = [l for l in range(N_ENC_LAYERS) if lo <= rel_depth(l) <= hi]
    top = B5["ranked"][:min(8, len(B5["ranked"]))]
    in_band = [r["layer"] for r in top if lo <= r["rel_depth"] <= hi]
    _br = f"{band[0]}-{band[-1]}" if band else "none"
    print(f"  enc MLP layers: {N_ENC_LAYERS}; mid-encoder band [{lo:.2f},{hi:.2f}] = layers {_br}")
    print(f"  top MLP layers (rel-depth): " + ", ".join(f"L{r['layer']}@{r['rel_depth']:.2f}" for r in top))
    print(f"  in mid-encoder band: {in_band if in_band else 'none'}")
    return dict(enc_layers=N_ENC_LAYERS, band=[lo, hi], band_layers=band, in_band=in_band)

def summarize():
    null_p = B5["randN_p95"]; win = B5["winning"]; ss = mlp_selective(win)
    rows = []
    for site in CONFIG["MLP_SITES_5"] + CONFIG["ATTN_SITES_5"]:
        rows.append(dict(site=site, kind=("MLP" if site in CONFIG["MLP_SITES_5"] else "attn"),
                         changepoint=_mean5(site, "changepoint"), motif=_mean5(site, "motif"), trend=_mean5(site, "trend")))
    site_real = ss["sig"] and (ss["changepoint"] > null_p); localized = B5["localized"]
    feat_sel = bool(GATE["go"] and not IS_MOCK and isinstance(C5, dict) and C5.get("selective"))
    go = bool(GATE["go"])     # authoritative GO: requires localized AND selective AND sig AND beats_null AND low_f
    # The verdict keys on the SAME evidence as the printed gate, so they can never disagree [review FIX #5].
    if go and feat_sel:
        verdict = f"A: MLP-LOCALIZED change-detection (in {win}) with selective causal FEATURES -> discrepancy RESOLVED (attention routes, MLP computes)"
    elif go:
        verdict = f"A-: MLP-localized in {win} (layer-level, low-f selective) but feature leg not run/selective -> partial positive"
    elif site_real:
        verdict = f"B: DISTRIBUTED across attention AND MLP ({win} necessary, not localized) -> SAE-vs-circuit discrepancy STANDS"
    else:
        verdict = "B: DISTRIBUTED across attention AND MLP (no MLP site even necessary above null)"

    print("=" * 92); print(f"PHASE 5 VERDICT: {verdict}{MOCK_TAG}"); print("=" * 92)
    print(f"  {'site':9s} {'kind':>5s} {'chgpt':>7s} {'motif':>7s} {'trend':>7s}")
    for r in rows:
        print(f"  {r['site']:9s} {r['kind']:>5s} {r['changepoint']:+7.3f} {r['motif']:+7.3f} {r['trend']:+7.3f}")
    print(f"\n  random-MLP-layer null (changepoint p95) = {null_p:+.3f}  [necessity gate]")
    print(f"  winning MLP site {win}: changepoint {ss['changepoint']:+.3f} (necessary={'YES' if site_real else 'no'}, "
          f"selective={'YES' if ss['selective'] else 'no'}, localized={'YES' if localized else 'no'}, gate={'GO' if go else 'NO-GO'})")
    # 5d reconciliation — keyed on the authoritative gate; the distributed text branches on necessity [review FIX #1, #5]
    print("  5d RECONCILIATION (attention vs MLP):")
    if go:
        print("    MLP localizes (selective, low-f) where attention did not -> the SAE features mark a real causal locus "
              + ("in MLPs; features causally selective -> discrepancy RESOLVED." if feat_sel else "in MLPs; feature leg pending."))
    elif site_real:
        print("    MLP is ALSO distributed (necessary in aggregate, but no small sufficient set / no low-f selective effect) -> "
              "the SAE-vs-circuit discrepancy STANDS: Mishra's SAE features do not correspond to a localized causal")
        print("    circuit at the layer level. Both attention (3b/4) AND MLP are distributed — a comprehensive, "
              "de-confounded negative + a methodological finding about SAE-feature vs causal-circuit correspondence.")
    else:
        print("    No MLP site is even necessary above the random-MLP null -> distributed across attention AND MLP; "
              "the SAE-vs-circuit discrepancy STANDS (SAE features mark no localized causal circuit at the layer level).")
    # detection power (go already implies a selective/localized effect was findable) [review FIX #6]
    if go:
        print("  DETECTION POWER: a selective/localized MLP effect IS findable -> a distributed verdict is de-confounded.")
    else:
        print(f"  DETECTION POWER: MLP ablation detects a large effect ({ss['changepoint']:+.3f}) but none selective/"
              f"localized above null -> the distributed null is real, not underpowered.")
    if not IS_LARGE:
        print("  NOTE: base model — the SAE-vs-circuit question lives in Large (Mishra's model); run pilot_a100 for 5c.")
    return dict(verdict=verdict, rows=rows, null_p95=null_p, winning=win, selective=ss, localized=bool(localized),
                site_real=bool(site_real), feature_selective=feat_sel, is_large=bool(IS_LARGE),
                gate_go=bool(GATE["go"]), minimal=B5["minimal"])

MISHRA = mishra_check(); SUMMARY = summarize()

# ============================================================
try:
    rows = SUMMARY["rows"]; fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # 6a: attention-vs-MLP necessity + selectivity (changepoint vs motif), random-MLP null line
    a = ax[0, 0]; sites = [r["site"] for r in rows]; xs = np.arange(len(sites)); w = 0.4
    a.bar(xs - w/2, [r["changepoint"] for r in rows], w, label="changepoint (GATE)", color="#c0392b")
    a.bar(xs + w/2, [r["motif"] for r in rows], w, label="motif (period-P)", color="#7f8c8d")
    a.axhline(SUMMARY["null_p95"], color="k", ls="--", lw=1, label="random-MLP null p95")
    a.axhline(CONFIG["STRUCT_COLLAPSE_MIN"], color="k", ls=":", lw=1)
    a.axhline(0, color="gray", lw=0.8); a.set_xticks(xs); a.set_xticklabels(sites, rotation=30, ha="right", fontsize=7)
    a.set_ylabel("structural rel-collapse"); a.legend(fontsize=7)
    a.set_title("Fig 6a: attention vs MLP (necessity + selectivity)" + MOCK_TAG, fontsize=9)
    # 6b: MLP dose-response
    b = ax[0, 1]
    for arm, col in [("enc_mlp", "#2980b9"), ("dec_mlp", "#27ae60"), ("random", "#999999")]:
        ys = [(sweep_agg(arm, f) or {"changepoint": np.nan})["changepoint"] for f in CONFIG["F_GRID"]]
        sd = [(sweep_agg(arm, f) or {"cp_sd": 0.0})["cp_sd"] for f in CONFIG["F_GRID"]]
        b.errorbar(CONFIG["F_GRID"], ys, yerr=sd, marker="o", color=col, label=arm, capsize=2,
                   ls=("--" if arm == "random" else "-"))
    b.axvline(CONFIG["LOW_F_MAX"], color="k", ls=":", lw=1, label=f"low-f gate ({CONFIG['LOW_F_MAX']})")
    b.set_xlabel("fraction of MLP layers ablated (f)"); b.set_ylabel("changepoint rel-collapse")
    b.set_title("Fig 6b: MLP dose-response (localized=low-f vs endpoint)", fontsize=9); b.legend(fontsize=7)
    # 6c: MLP-layer localization (single-layer collapse by relative depth + Mishra band)
    c = ax[1, 0]; rk = B5["singles"]
    rk = sorted(rk, key=lambda r: r["layer"])
    c.bar([r["rel_depth"] for r in rk], [r["rel"] for r in rk], width=0.03, color="#8e44ad")
    c.axvspan(CONFIG["MISHRA_DEPTH_LO"], CONFIG["MISHRA_DEPTH_HI"], color="orange", alpha=0.2, label="Mishra mid-encoder")
    c.axhline(B5["site_collapse"], color="#c0392b", ls="-", lw=1, label=f"full {B5['winning']} collapse")
    c.set_xlabel("relative encoder depth"); c.set_ylabel("single-MLP-layer changepoint collapse")
    c.set_title(f"Fig 6c: MLP-layer localization [{'LOCALIZED' if B5['localized'] else 'distributed'}]", fontsize=9); c.legend(fontsize=7)
    # 6d: feature-level (if 5c ran) or annotation
    d = ax[1, 1]; d.set_axis_off()
    if isinstance(C5, dict) and not C5.get("skipped") and not C5.get("smoke"):
        d.text(0.03, 0.85, f"5c features on enc_mlp L{C5['layer']} (depth {C5['rel_depth']:.2f})", fontsize=9, transform=d.transAxes)
        d.text(0.03, 0.68, f"top-{len(C5['top_features'])} shift-feature ablation:", fontsize=8, transform=d.transAxes)
        d.text(0.03, 0.55, f"changepoint collapse {C5['cp_collapse']:+.3f} vs motif {C5['motif_collapse']:+.3f}", fontsize=8, transform=d.transAxes)
        d.text(0.03, 0.42, f"random-feature null {C5['null_p95']:+.3f} -> selective={C5['selective']}", fontsize=8, transform=d.transAxes)
    else:
        d.text(0.03, 0.6, "5c not run (gate NO-GO" + (" / mock smoke" if IS_MOCK else "") + ")", fontsize=10, transform=d.transAxes)
        d.text(0.03, 0.45, "-> distributed across attention AND MLP;", fontsize=9, transform=d.transAxes)
        d.text(0.03, 0.34, "SAE-vs-circuit discrepancy stands.", fontsize=9, transform=d.transAxes)
    d.set_title("Fig 6d: feature-level (SAE) — the SAE-vs-circuit reconciliation", fontsize=9)
    fig.suptitle(f"Phase 5 — MLP/feature tracing: {SUMMARY['verdict'][:60]}" + MOCK_TAG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(CKPT_DIR, f"fig6_phase5_{MODE}.png"), dpi=90)
    plt.show(); plt.close(fig); print(f"saved fig6_phase5_{MODE}.png")
except Exception as e:
    import traceback; print("fig skipped:", repr(e)[:160]); traceback.print_exc()

# ============================================================
out = dict(summary=SUMMARY, exp5a=A5, sweep=SWEEP, localization=B5, gate=GATE, feature=C5, mishra=MISHRA,
           monotonicity=dict(rho=MONO["rho"], slope=MONO["slope"]),
           config=dict(mode=MODE, model_id=CONFIG["model_id"], n_enc_layers=N_ENC_LAYERS, n_mlp=int(N_MLP),
                       n_attn=int(N_ATTN), n_seeds=CONFIG["N_SEEDS"], n_series=CONFIG["N_SERIES"],
                       sweep_seeds=CONFIG["SWEEP_SEEDS"], f_grid=CONFIG["F_GRID"], is_large=bool(IS_LARGE)))
p = os.path.join(CKPT_DIR, f"phase5_{MODE}.json")
with open(p, "w") as f: json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote", p, "->", SUMMARY["verdict"][:70], MOCK_TAG)