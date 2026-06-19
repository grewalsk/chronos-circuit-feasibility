# auto-mirror of phase4.ipynb code cells (local smoke test)

# ============================================================
import os
CONFIG = {
    "MODE": "mock_cpu",                 # -> "pilot_t4" (base, T4) or "pilot_a100" (Large, A100)
    # model_id is derived from MODE below; override with CHRONOS_P4_MODEL to run Large on any big GPU.
    "MODEL_BY_MODE": {"mock_cpu": None, "pilot_t4": "amazon/chronos-t5-base", "pilot_a100": "amazon/chronos-t5-large"},
    "USE_DRIVE": True,                   # on Colab, persist checkpoints to Google Drive (survives a runtime reset)
    "SEED0": 0,
    "PERIODS": [8, 12, 16, 24],          # motif selectivity control
    "N_SEEDS": 3,
    "N_SERIES": 32,
    "CTX": 256,
    "PRED": 64,
    "OBS_NOISE": 0.30,
    "N_CRPS_SAMPLES": 64,                # dominant memory/time lever (gating metric is structural, not CRPS)
    "N_BOOTSTRAP": 1000,
    "FORECAST_BATCH": 4,                 # series per predict() call (T4-safe; auto-halves on OOM)
    "N_RANDOM_DRAWS": 8,                 # size-matched random@N (f=1) null draws
    "CONDITIONS_4": ["changepoint", "motif", "trend"],   # changepoint = GATE; motif/trend = selectivity controls
    "SITES_4": ["enc_self", "dec_self", "cross"],
    # ---- level-shift stimulus (CTX-relative; metadata threaded through make_batch) [FIX1,FIX2] ----
    "TAU_FRAC_CTX": 0.65,                # shift INSIDE context -> 35% post-shift context at 0.65 (>=30% asserted)
    "DELTA_LO": 1.5, "DELTA_HI": 3.0,    # |delta| >> noise; survives mean-scaling
    "BAND_W": 6,                         # boundary band half-width [tau-w, tau+w] for the attention scan
    # ---- fraction-sweep dose-response (multi-seed) [FIX7] ----
    "F_GRID": [0.1, 0.25, 0.5, 0.75, 1.0],
    "SWEEP_DRAWS": 3,
    "SWEEP_SERIES": 16,
    "SWEEP_SEEDS": 2,                    # >=2 (3 if affordable); the cross low-f severance rule is load-bearing [FIX7]
    "LOW_F_MAX": 0.5,                    # a cross locus must show its selective effect at f <= this
    # ---- behavioral scan + localization ----
    "SCAN_SERIES": 24,                   # series for the delta-response + boundary-attention battery
    "N_CANDIDATES": 8,                   # top heads carried into the head-level causal test
    "LOCALIZE_MAX_SET": 8,               # a localized set is 1..8 heads
    "LOCALIZE_FRAC": 0.60,              # a small set must reproduce >= 60% of the site's changepoint collapse
    "LOC_RANDOM_DRAWS": 5,               # random-k-within-site null draws
    # ---- decision thresholds (pre-registered) ----
    "STRUCT_COLLAPSE_MIN": 0.30,         # changepoint recovery must lose >= 30% of its structure
    "SELECTIVITY_MARGIN": 2.0,           # changepoint collapse must be >= 2x the non-target (motif/trend) collapse
    # ---- Mishra relative-depth band (mid-encoder ~46% depth) [FIX4] ----
    "MISHRA_DEPTH_LO": 0.45, "MISHRA_DEPTH_HI": 0.55,
    # ---- mock overrides (tiny random T5; NOT interpretable) ----
    "mock_cpu": {
        "PERIODS": [6, 8], "N_SEEDS": 2, "N_SERIES": 6, "CTX": 48, "PRED": 24,
        "N_CRPS_SAMPLES": 12, "N_BOOTSTRAP": 50, "N_RANDOM_DRAWS": 3, "FORECAST_BATCH": 999,
        "BAND_W": 3, "F_GRID": [0.25, 0.5, 1.0], "SWEEP_DRAWS": 2, "SWEEP_SERIES": 4, "SWEEP_SEEDS": 2,
        "SCAN_SERIES": 6, "N_CANDIDATES": 3, "LOCALIZE_MAX_SET": 3, "LOC_RANDOM_DRAWS": 2,
    },
}
MODE = os.environ.get("CHRONOS_P4_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4", "pilot_a100"), MODE
CONFIG["model_id"] = os.environ.get("CHRONOS_P4_MODEL", CONFIG["MODEL_BY_MODE"][MODE])
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
            CKPT_DIR = "/content/drive/MyDrive/chronos_phase4"; os.makedirs(CKPT_DIR, exist_ok=True)
            print("checkpoints -> Google Drive (survives disconnects):", CKPT_DIR)
        except Exception as e:
            print("Drive mount skipped (", repr(e)[:80], ") -> /content (lost on a full runtime reset)")
print(f"MODE={MODE}{MOCK_TAG}  model={CONFIG['model_id']}  large={IS_LARGE}  ctx={CONFIG['CTX']} pred={CONFIG['PRED']} "
      f"seeds={CONFIG['N_SEEDS']} series={CONFIG['N_SERIES']}  F_GRID={CONFIG['F_GRID']} sweep_seeds={CONFIG['SWEEP_SEEDS']}  ckpt={CKPT_DIR}")

# ============================================================
import sys, json, subprocess, gc, re, warnings
warnings.filterwarnings("ignore", message=".*past_key_values.*")    # harmless HF cache deprecation on manual forwards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _ensure(pkg, imp):
    if os.environ.get("CHRONOS_P4_SKIP_INSTALL") == "1": return
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

# ============================================================
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
def _layer_idx(name):
    m = re.search(r"block\.(\d+)\.", name); return int(m.group(1)) if m else -1

if IS_MOCK:
    from transformers import T5Config, T5ForConditionalGeneration
    cfg = T5Config(vocab_size=256, d_model=64, d_kv=32, d_ff=128, num_layers=2, num_decoder_layers=2,
                   num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    INNER = T5ForConditionalGeneration(cfg).eval(); VOCAB = cfg.vocab_size; PIPE = None
    DEC_START = int(cfg.decoder_start_token_id)
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
        _need = 6.0e9 if IS_LARGE else 1.5e9   # Large (710M) needs more headroom
        assert _free > _need, ("Only %.2f GB free — a previous run left memory resident. RESTART THE RUNTIME "
                               "(Runtime -> Restart session), then run again." % (_free/1e9))
    PIPE = ChronosPipeline.from_pretrained(CONFIG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)
    INNER = PIPE.inner_model.eval(); VOCAB = INNER.config.vocab_size
    INNER.requires_grad_(False)
    try: INNER.config._attn_implementation = "eager"      # eager so output_attentions surfaces (FIX5)
    except Exception: pass
    DEC_START = int(getattr(INNER.config, "decoder_start_token_id", 0))

SITES = classify_attention_modules(INNER)
for s in SITES: assert len(SITES[s]) > 0, f"no modules for {s}"
# per-head layer map (for relative depth + mapping candidate (site,layer,head) -> module)
SITE_MODS = {s: {_layer_idx(name): mod for name, mod in SITES[s]} for s in SITES}
N_ENC_LAYERS = len(SITES["enc_self"])
print("per-site (modules, heads):", {s: (len(SITES[s]), _nheads(SITES[s][0][1])) for s in SITES},
      "| enc layers:", N_ENC_LAYERS)

def site_size(sites):
    sizes = {s: len(sites[s]) * _nheads(sites[s][0][1]) for s in sites}
    assert len(set(sizes.values())) == 1, f"sites NOT equal size: {sizes}"   # HARD ASSERT: equal site size
    return next(iter(sizes.values()))
N = site_size(SITES)
print(f"EQUAL-SIZE CHECK: PASS — each site = {N} heads")

def rel_depth(layer_idx):                                 # fraction into the encoder (0=first, 1=last)
    return float(layer_idx) / max(1, (N_ENC_LAYERS - 1))

# ============================================================
def make_levelshift(ctx_len, pred_len, rng, noise, tau_frac_ctx=None, delta=None):
    tau_frac_ctx = CONFIG["TAU_FRAC_CTX"] if tau_frac_ctx is None else tau_frac_ctx
    length = ctx_len + pred_len
    tau = int(tau_frac_ctx * ctx_len)                      # shift INSIDE context; 35% post-shift context at 0.65
    assert (ctx_len - tau) >= 0.30 * ctx_len, "need >=30% post-shift context"
    L0 = rng.uniform(-1.0, 1.0)
    if delta is None:
        delta = rng.choice([-1.0, 1.0]) * rng.uniform(CONFIG["DELTA_LO"], CONFIG["DELTA_HI"])
    L1 = L0 + delta
    s = np.where(np.arange(length) < tau, L0, L1) + rng.normal(0, noise, size=length)
    return s, dict(tau=int(tau), L0=float(L0), L1=float(L1), delta=float(delta))

def make_nochange(ctx_len, pred_len, rng, noise):          # stationary content control for the behavioral scan
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

def make_batch(cond, rng, n_series):                       # MUST return metas now [FIX2]
    CTX, PRED, NOISE = CONFIG["CTX"], CONFIG["PRED"], CONFIG["OBS_NOISE"]
    ctxs, tgts, metas = [], [], []
    for i in range(n_series):
        if cond == "changepoint":
            s, meta = make_levelshift(CTX, PRED, rng, NOISE)
        elif cond == "nochange":
            s, meta = make_nochange(CTX, PRED, rng, NOISE)
        elif cond == "motif":
            P = CONFIG["PERIODS"][i % len(CONFIG["PERIODS"])]; s = make_motif(P, CTX + PRED, rng, NOISE); meta = {"P": int(P)}
        elif cond == "trend":
            s = make_trend(CTX + PRED, rng, NOISE); meta = {}
        else:
            raise ValueError(cond)
        ctxs.append(s[:CTX]); tgts.append(s[CTX:]); metas.append(meta)
    return ctxs, np.array(tgts), metas

# ============================================================
def crps_samples(samples, target):
    samples = np.asarray(samples, float); target = np.asarray(target, float)
    t1 = np.abs(samples - target[None, :]).mean(axis=0)
    pair = np.abs(samples[:, None, :] - samples[None, :, :]).mean(axis=(0, 1))
    return float((t1 - 0.5 * pair).mean())

def period_power_fraction(f1d, P):                         # motif structure (reused from 3b)
    x = np.asarray(f1d, float); x = x - x.mean(); H = len(x)
    power = np.abs(np.fft.rfft(x)) ** 2; freqs = np.fft.rfftfreq(H)
    if len(freqs) < 2: return 0.0
    df = freqs[1] - freqs[0]; total = power[1:].sum() + 1e-12; f0 = 1.0 / P
    band = (np.abs(freqs - f0) <= 1.5 * df) | (np.abs(freqs - 2 * f0) <= 1.5 * df); band[0] = False
    return float(power[band].sum() / total)

def trend_slope_recovery(f1d, target1d):                   # trend structure (reused from 3b; soft control)
    H = len(f1d); t = np.arange(H)
    sf = float(np.polyfit(t, f1d, 1)[0]); st = float(np.polyfit(t, target1d, 1)[0])
    return float(max(0.0, 1.0 - abs(sf - st) / (abs(st) + 1e-6)))

def changepoint_recovery(forecast_1d, meta):               # THE GATE [FIX1] — delta-normalized, median
    fhat = float(np.median(np.asarray(forecast_1d)))
    denom = abs(meta["L1"] - meta["L0"]) + 1e-8
    r = 1.0 - abs(fhat - meta["L1"]) / denom               # 1=tracks new level, 0=reverts to old level/mean
    return float(np.clip(r, 0.0, 1.0))

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

def rel_collapse(struct_clean, struct_abl):                # positive = structure destroyed (reused from 3b)
    sc = np.asarray(struct_clean, float); sa = np.asarray(struct_abl, float)
    d = sc - sa; cm = float(sc.mean()) + 1e-6
    lo, hi = bootstrap_ci(d)
    return float(d.mean() / cm), [lo / cm, hi / cm]

def _spearman(a, b):                                        # manual (no scipy dep): rank then Pearson
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = (np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else 0.0

# HARD ASSERT: changepoint_recovery in [0,1]
_m = dict(tau=10, L0=0.0, L1=2.0, delta=2.0)
assert 0.0 <= changepoint_recovery([2.0]*8, _m) <= 1.0 and 0.0 <= changepoint_recovery([0.0]*8, _m) <= 1.0
assert abs(changepoint_recovery([2.0]*8, _m) - 1.0) < 1e-6, "tracks new level -> 1"
assert changepoint_recovery([0.0]*8, _m) < 0.01, "reverts to old level -> ~0"
print("metric asserts: changepoint_recovery in [0,1], tracks->1, reverts->0  PASS")

# ============================================================
def _make_pre_hook(attn):
    d_kv = _dkv(attn); nh = _nheads(attn)
    def hook(o_module, args):
        x = args[0]
        if getattr(attn, "_record", False):                # RECORD-ONLY mode (read, do not modify) [FIX5]
            per = [x[..., h * d_kv:(h + 1) * d_kv].norm(dim=-1) for h in range(nh)]   # each (..., S)
            attn._rec = torch.stack(per, dim=-2).detach().float().cpu().numpy()       # (..., nh, S)
        cst = getattr(attn, "_const_patch", None)          # corrupt-mean patch (§13 resample cross-check)
        heads = getattr(attn, "_ablate_heads", None)
        if not heads and not cst: return None
        x = x.clone()
        if cst:
            for h, vec in cst.items():
                x[..., h * d_kv:(h + 1) * d_kv] = torch.as_tensor(vec, dtype=x.dtype, device=x.device)
        if heads:
            for h in heads:
                sl = slice(h * d_kv, (h + 1) * d_kv); seg = x[..., sl]
                x[..., sl] = seg.mean(dim=tuple(range(seg.dim() - 1)), keepdim=True)
        return (x,)
    return hook

def install_hooks(sites):
    handles = []
    for lst in sites.values():
        for _, mod in lst:
            mod._ablate_heads = set(); mod._record = False; mod._const_patch = None
            handles.append(mod.o.register_forward_pre_hook(_make_pre_hook(mod)))
    return handles

def clear_ablations(sites):
    for lst in sites.values():
        for _, mod in lst: mod._ablate_heads = set(); mod._const_patch = None

def build_head_pool(sites):
    return [(mod, h) for lst in sites.values() for _, mod in lst for h in range(_nheads(mod))]
def site_heads(sites, site):
    return [(mod, h) for _, mod in sites[site] for h in range(_nheads(mod))]

def set_site_ablation(sites, site):                        # f=1: all of one site
    clear_ablations(sites)
    for _, mod in sites[site]: mod._ablate_heads = set(range(_nheads(mod)))
def set_fraction_site(sites, site, f, rng):                # random fraction f WITHIN one site
    clear_ablations(sites); heads = site_heads(sites, site)
    k = max(1, int(round(f * len(heads))))
    for idx in rng.choice(len(heads), size=k, replace=False):
        mod, h = heads[idx]; mod._ablate_heads.add(h)
def set_random_pool(sites, pool, n, rng):                  # n heads across ALL sites (size matched)
    clear_ablations(sites); n = min(n, len(pool))
    for idx in rng.choice(len(pool), size=n, replace=False):
        mod, h = pool[idx]; mod._ablate_heads.add(h)
def set_explicit(sites, specs):                            # ablate a named (site,layer,head) set
    clear_ablations(sites)
    for s, l, h in specs: SITE_MODS[s][l]._ablate_heads.add(h)
def set_random_in_site(sites, site, k, rng):               # k random heads WITHIN one site (localization null)
    clear_ablations(sites); heads = site_heads(sites, site); k = min(k, len(heads))
    for idx in rng.choice(len(heads), size=k, replace=False):
        mod, h = heads[idx]; mod._ablate_heads.add(h)

def set_record(sites, on, which=("enc_self",)):
    for s, lst in sites.items():
        for _, mod in lst: mod._record = bool(on and s in which);
    if not on:
        for s, lst in sites.items():
            for _, mod in lst:
                if hasattr(mod, "_rec"): del mod._rec

HANDLES = install_hooks(SITES); HEAD_POOL = build_head_pool(SITES)
print(f"hooks on {len(HANDLES)} '.o' modules | pool={len(HEAD_POOL)} | N(site)={N}")

# ============================================================
def forecast_pilot(contexts, n_samples):
    torch.manual_seed(CONFIG["SEED0"])     # common random numbers (clean vs ablated share sampling noise)
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

# ---- tokenization for the manual attention/record forward (no left-pad: equal-length ctx -> token pos = series pos) ----
def _tokenize(contexts):
    if IS_MOCK:
        arrs = []
        for c in contexts:
            c = np.asarray(c, float)
            q = np.clip(((c - c.min()) / ((c.max() - c.min()) + 1e-9) * (VOCAB - 3)).astype(int) + 2, 0, VOCAB - 1)
            arrs.append(q.astype(np.int64))
        L = max(len(a) for a in arrs)
        ids = np.zeros((len(arrs), L), dtype=np.int64); am = np.zeros((len(arrs), L), dtype=np.int64)
        for i, a in enumerate(arrs): ids[i, :len(a)] = a; am[i, :len(a)] = 1
        return torch.tensor(ids, device=DEVICE), torch.tensor(am, device=DEVICE)
    ct = torch.tensor(np.asarray(contexts), dtype=DTYPE)
    ids, am, _scale = PIPE.tokenizer.context_input_transform(ct)   # appends EOS at end; tau (<CTX) -> token pos tau
    return ids.to(DEVICE), am.to(DEVICE)

def encoder_attentions(contexts):                          # NEW manual forward; VERIFY attentions surface [FIX5]
    ids, am = _tokenize(contexts)
    with torch.inference_mode():
        out = INNER.get_encoder()(input_ids=ids, attention_mask=am, output_attentions=True)
    assert out.attentions is not None, "encoder did not return attentions — output_attentions failed"
    assert len(out.attentions) == len(SITES["enc_self"]), "n attention layers != n enc_self modules"
    return [a.detach().float().cpu().numpy() for a in out.attentions]   # per layer (B, nh, Sq, Sk)

def encoder_head_norms(contexts):                          # delta-response primitive (record-mode hook reuse) [FIX5]
    ids, am = _tokenize(contexts)
    set_record(SITES, True, which=("enc_self",))
    with torch.inference_mode():
        INNER.get_encoder()(input_ids=ids, attention_mask=am)
    recs = []                                              # ordered list of (layer_idx, head, (B,S) norm array)
    for name, mod in SITES["enc_self"]:
        r = getattr(mod, "_rec", None)
        assert r is not None, "record hook did not fire"
        L = _layer_idx(name)
        for h in range(_nheads(mod)): recs.append((L, h, r[..., h, :]))
    set_record(SITES, False)
    return recs

# ---- plumbing: ablation changes output AND the attention-capture path is exercised (acceptance) ----
_ctx, _tgt, _metas = make_batch("changepoint", np.random.default_rng(0), 2)
clear_ablations(SITES); _a = FORECAST(_ctx, 4)
set_site_ablation(SITES, "dec_self"); _b = FORECAST(_ctx, 4); clear_ablations(SITES)
print(f"PLUMBING: ablating dec_self changed forecast = {not np.allclose(_a,_b)} (max|Δ|={np.abs(_a-_b).max():.4g})")
assert not np.allclose(_a, _b), "ablation did NOT change output — hooks not wired"            # HARD ASSERT
_att = encoder_attentions(_ctx); _nm = encoder_head_norms(_ctx)
print(f"ATTENTION CAPTURE: encoder attentions surfaced — {len(_att)} layers, head0 shape {_att[0].shape}; "
      f"record-mode norms captured for {len(_nm)} enc_self heads")
assert _att[0].ndim == 4 and len(_nm) == N, "attention/record capture path not wired"          # HARD ASSERT [FIX5]
print("PLUMBING + ATTENTION-CAPTURE: PASS" + MOCK_TAG)

# ============================================================
def monotonicity_check():
    rng = np.random.default_rng(CONFIG["SEED0"] + 777)
    deltas = np.concatenate([np.linspace(0.5, 4.0, 12), -np.linspace(0.5, 4.0, 12)])
    ctxs, recs = [], []
    metas = []
    for dlt in deltas:
        s, meta = make_levelshift(CONFIG["CTX"], CONFIG["PRED"], rng, CONFIG["OBS_NOISE"], delta=float(dlt))
        ctxs.append(s[:CONFIG["CTX"]]); metas.append(meta)
    clear_ablations(SITES); fc = FORECAST(ctxs, CONFIG["N_CRPS_SAMPLES"])
    rec = np.array([changepoint_recovery(fc[i].mean(0), metas[i]) for i in range(len(metas))])
    ad = np.abs(deltas)
    nb = 6; edges = np.quantile(ad, np.linspace(0, 1, nb + 1)); centers, means = [], []
    for j in range(nb):
        m = (ad >= edges[j]) & (ad <= edges[j + 1] if j == nb - 1 else ad < edges[j + 1])
        if m.sum() == 0: continue
        centers.append(float(ad[m].mean())); means.append(float(rec[m].mean()))
    centers, means = np.array(centers), np.array(means)
    rho = _spearman(centers, means)
    slope = float(np.polyfit(centers, means, 1)[0]) if len(centers) > 1 else 0.0
    print("  delta-binned changepoint_recovery (clean):")
    for c, mu in zip(centers, means): print(f"    |delta|~{c:4.2f}  recovery={mu:5.3f}")
    print(f"  Spearman rho={rho:+.3f}  OLS slope={slope:+.4f}  (require rho>0 OR slope>0)")
    assert (rho > 0) or (slope > 0), "no positive monotone trend in delta on the clean model — gate is broken"  # [FIX6]
    print("  MONOTONICITY (trend, binned): PASS" + MOCK_TAG)
    return dict(centers=centers.tolist(), means=means.tolist(), rho=rho, slope=slope)

print("Clean-model monotonicity sanity..." + MOCK_TAG); MONO = monotonicity_check()

# ============================================================
def run_behavioral_scan():
    rng = np.random.default_rng(CONFIG["SEED0"] + 321)
    nS = CONFIG["SCAN_SERIES"]; CTX = CONFIG["CTX"]; w = CONFIG["BAND_W"]
    ls_ctx, ls_meta = [], []
    for _ in range(nS):
        s, meta = make_levelshift(CTX, CONFIG["PRED"], rng, CONFIG["OBS_NOISE"]); ls_ctx.append(s[:CTX]); ls_meta.append(meta)
    nc_ctx = []
    for k in range(nS):                                    # matched no-change content control (same tau band)
        s, _ = make_nochange(CTX, CONFIG["PRED"], rng, CONFIG["OBS_NOISE"]); nc_ctx.append(s[:CTX])
    deltas = np.array([m["delta"] for m in ls_meta]); taus = [m["tau"] for m in ls_meta]

    # ---- delta-response: (post-shift - pre-shift) per-head norm, correlate with delta ----
    recs = encoder_head_norms(ls_ctx)                      # list of (layer, head, (B,S))
    heads_order = [(L, h) for (L, h, _) in recs]
    dr = np.zeros((nS, len(recs)))
    for j, (L, h, arr) in enumerate(recs):                 # arr: (B, S)
        for b in range(nS):
            tau = taus[b]; prof = arr[b]
            pre = prof[:tau]; post = prof[tau:CTX]          # exclude EOS (>=CTX); token pos = series pos
            dr[b, j] = (post.mean() if len(post) else 0.0) - (pre.mean() if len(pre) else 0.0)
    dr_corr = np.array([_pearson(dr[:, j], deltas) for j in range(len(recs))])
    dr_absc = np.array([_pearson(dr[:, j], np.abs(deltas)) for j in range(len(recs))])
    dr_score = np.maximum(np.abs(dr_corr), np.abs(dr_absc))

    # ---- boundary attention: mass to [tau-w, tau+w] from post-shift queries, minus no-change, vs random null ----
    att_ls = encoder_attentions(ls_ctx); att_nc = encoder_attentions(nc_ctx)
    def band_mass(att_layers, taus_b, centers=None):
        out = {}
        for L in range(len(att_layers)):
            A = att_layers[L]                              # (B, nh, Sq, Sk)
            nh = A.shape[1]
            for h in range(nh):
                vals = []
                for b in range(A.shape[0]):
                    tau = taus_b[b]; c = tau if centers is None else centers[b]
                    k0, k1 = max(0, c - w), min(CTX, c + w + 1)
                    q0, q1 = tau, CTX                       # post-shift queries (exclude EOS)
                    if q1 <= q0 or k1 <= k0: vals.append(0.0); continue
                    vals.append(float(A[b, h, q0:q1, k0:k1].sum(axis=1).mean()))
                out[(L, h)] = float(np.mean(vals))
        return out
    mass_ls = band_mass(att_ls, taus)
    mass_nc = band_mass(att_nc, taus)                      # content control: same band, stationary series
    rng2 = np.random.default_rng(CONFIG["SEED0"] + 654)
    rand_centers = [int(rng2.integers(2 * w + 2, CTX - 2 * w - 2)) for _ in range(nS)]
    mass_rand = band_mass(att_ls, taus, centers=rand_centers)   # random-position null
    boundary = {k: mass_ls[k] - mass_nc[k] for k in mass_ls}
    null_vals = list(mass_rand.values()); null_p95 = float(np.percentile(null_vals, 95)) if null_vals else 0.0

    rows = []
    for j, (L, h) in enumerate(heads_order):
        rows.append(dict(site="enc_self", layer=L, head=h, rel_depth=rel_depth(L),
                         dr_corr=float(dr_corr[j]), dr_absdelta=float(dr_absc[j]), dr_score=float(dr_score[j]),
                         boundary=float(boundary[(L, h)]), boundary_minus_null=float(boundary[(L, h)] - null_p95)))
    # combined rank: average of (delta-response rank, boundary rank)
    def _rank(key):
        order = sorted(range(len(rows)), key=lambda i: rows[i][key], reverse=True)
        rk = np.zeros(len(rows));  [rk.__setitem__(order[i], i) for i in range(len(rows))]; return rk
    comb = _rank("dr_score") + _rank("boundary")
    for i, r in enumerate(rows): r["combined_rank"] = float(comb[i])
    rows.sort(key=lambda r: r["combined_rank"])
    cand = rows[:CONFIG["N_CANDIDATES"]]
    print(f"  boundary-attention random-position null p95 = {null_p95:.4f}")
    print(f"  top {len(cand)} enc_self candidates (by delta-response + boundary attention):")
    print(f"    {'layer':>5s} {'head':>4s} {'reldepth':>8s} {'dr_corr':>8s} {'dr|Δ|':>7s} {'bound-null':>10s}")
    for r in cand:
        print(f"    {r['layer']:5d} {r['head']:4d} {r['rel_depth']:8.2f} {r['dr_corr']:+8.3f} "
              f"{r['dr_absdelta']:+7.3f} {r['boundary_minus_null']:+10.4f}")
    return dict(rows=rows, candidates=cand, boundary_null_p95=null_p95)

def _pearson(a, b):
    a = np.asarray(a, float) - np.mean(a); b = np.asarray(b, float) - np.mean(b)
    d = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0

print("Behavioral scan (delta-response + boundary attention)..." + MOCK_TAG); SCAN = run_behavioral_scan()

# ============================================================
def _crps_vec(fc, tgt): return np.array([crps_samples(fc[i], tgt[i]) for i in range(len(tgt))])
def _struct_vec(fc, metas, tgt, cond): return np.array([structure(cond, fc[i].mean(0), metas[i], tgt[i]) for i in range(len(tgt))])

FORCE = os.environ.get("CHRONOS_P4_FORCE", "0") == "1"
def _ckp(name): return os.path.join(CKPT_DIR, f"phase4_{MODE}_{name}.json")
def _load_recs(name):
    p = _ckp(name)
    if os.path.exists(p) and not FORCE:
        try: return json.load(open(p))
        except Exception: return []
    return []
def _save_recs(name, recs): json.dump(recs, open(_ckp(name), "w"))

def run_full_site():
    recs = _load_recs("full")
    done = {(r["seed"], r["cond"]) for r in recs if r.get("kind") == "random"}
    recs = [r for r in recs if (r["seed"], r["cond"]) in done]
    for seed in range(CONFIG["N_SEEDS"]):
        for ci, cond in enumerate(CONFIG["CONDITIONS_4"]):
            if (seed, cond) in done:
                print(f"  [ckpt] full-site ({seed},{cond}) resumed"); continue
            rng = np.random.default_rng(CONFIG["SEED0"] + seed * 100 + ci)
            ctx, tgt, metas = make_batch(cond, rng, CONFIG["N_SERIES"])
            clear_ablations(SITES); fc0 = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
            crps0 = _crps_vec(fc0, tgt); s0 = _struct_vec(fc0, metas, tgt, cond)
            for site in CONFIG["SITES_4"]:
                set_site_ablation(SITES, site); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                dcrps = _crps_vec(fc, tgt) - crps0; rel, rel_ci = rel_collapse(s0, _struct_vec(fc, metas, tgt, cond))
                recs.append(dict(seed=seed, cond=cond, kind=site, dcrps_mean=float(dcrps.mean()),
                                 dcrps_ci=bootstrap_ci(dcrps), rel_collapse=rel, rel_ci=rel_ci, clean_struct=float(s0.mean())))
            rel_draws, crps_draws = [], []
            for _ in range(CONFIG["N_RANDOM_DRAWS"]):
                set_random_pool(SITES, HEAD_POOL, N, rng); fc = FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"])
                rel_draws.append(rel_collapse(s0, _struct_vec(fc, metas, tgt, cond))[0])
                crps_draws.append(float((_crps_vec(fc, tgt) - crps0).mean()))
            recs.append(dict(seed=seed, cond=cond, kind="random", rel_draws=rel_draws, crps_draws=crps_draws))
            clear_ablations(SITES); _save_recs("full", recs)
            print(f"  full-site ({seed},{cond}) done + saved")
    return recs

print("Experiment A (full-site)..." + MOCK_TAG); FULL = run_full_site()

def _mean(kind, cond, field):
    v = [r[field] for r in FULL if r.get("kind") == kind and r["cond"] == cond and field in r]
    return float(np.mean(v)) if v else float("nan")
cp_enc = _mean("enc_self", "changepoint", "rel_collapse")
print(f"  RE-MEASURED full-site changepoint collapse (standardized metric): enc_self={cp_enc:+.3f} "
      f"dec_self={_mean('dec_self','changepoint','rel_collapse'):+.3f} cross={_mean('cross','changepoint','rel_collapse'):+.3f}")

# ============================================================
def run_sweep():
    recs = _load_recs("sweep")
    done = {(r["seed"], round(r["f"], 4)) for r in recs}
    recs = [r for r in recs if (r["seed"], round(r["f"], 4)) in done]
    for seed in range(CONFIG["SWEEP_SEEDS"]):
        for fi, f in enumerate(CONFIG["F_GRID"]):
            if (seed, round(f, 4)) in done:
                print(f"  [ckpt] sweep ({seed},f={f}) resumed"); continue
            batches = {c: make_batch(c, np.random.default_rng(CONFIG["SEED0"] + 5000 + seed * 10 + ci), CONFIG["SWEEP_SERIES"])
                       for ci, c in enumerate(CONFIG["CONDITIONS_4"])}
            clean = {}
            for c, (ctx, tgt, metas) in batches.items():
                clear_ablations(SITES); clean[c] = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, c)
            arng = np.random.default_rng(CONFIG["SEED0"] + 6000 + seed * 100 + fi)
            unit = []
            for d in range(CONFIG["SWEEP_DRAWS"]):
                for arm in CONFIG["SITES_4"] + ["random"]:
                    rel = {}
                    for c, (ctx, tgt, metas) in batches.items():
                        if arm == "random": set_random_pool(SITES, HEAD_POOL, max(1, int(round(f * N))), arng)
                        else:               set_fraction_site(SITES, arm, f, arng)
                        rel[c] = rel_collapse(clean[c], _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, c))[0]
                    unit.append(dict(seed=seed, f=f, draw=d, arm=arm, changepoint=rel["changepoint"],
                                     nontarget=max(rel["motif"], rel["trend"]), motif=rel["motif"], trend=rel["trend"]))
            recs += unit; clear_ablations(SITES); _save_recs("sweep", recs)
            print(f"  sweep ({seed},f={f}) done + saved")
    return recs

print("Experiment B (fraction sweep, multi-seed)..." + MOCK_TAG); SWEEP = run_sweep()

def sweep_agg(arm, f):
    rs = [r for r in SWEEP if r["arm"] == arm and abs(r["f"] - f) < 1e-9]
    if not rs: return None
    cp = [r["changepoint"] for r in rs]; nt = [r["nontarget"] for r in rs]
    return dict(changepoint=float(np.mean(cp)), nontarget=float(np.mean(nt)),
                cp_sd=float(np.std(cp)), nt_sd=float(np.std(nt)))

# ============================================================
def run_localization():
    site_cp = {s: _mean(s, "changepoint", "rel_collapse") for s in CONFIG["SITES_4"]}
    winning = max(site_cp, key=lambda s: site_cp[s] if np.isfinite(site_cp[s]) else -1e9)
    site_collapse = site_cp[winning]
    cands = [c for c in SCAN["candidates"] if c["site"] == winning]
    ranked = bool(cands)            # behavioral scan ranks enc_self only; a non-enc_self winner uses UNRANKED heads
    if not cands:                   # if another site wins, fall back to that site's heads in module order
        cands = [dict(site=winning, layer=_layer_idx(name), head=h)
                 for (name, mod) in SITES[winning] for h in range(_nheads(mod))][:CONFIG["N_CANDIDATES"]]
    # the winning site must itself be NECESSARY before any localization claim (else a small set would "reproduce
    # most of" a non-existent collapse — the spec's prohibited failure mode). Necessity = full-site collapse clears
    # STRUCT_COLLAPSE_MIN AND beats the size-matched random@N null from Experiment A. [review FIX #1]
    randN = [v for r in FULL if r.get("kind") == "random" and r["cond"] == "changepoint" for v in r.get("rel_draws", [])]
    randN_p95 = float(np.percentile(randN, 95)) if randN else 0.0
    site_necessary = bool(np.isfinite(site_collapse) and site_collapse >= CONFIG["STRUCT_COLLAPSE_MIN"] and site_collapse > randN_p95)
    print(f"  winning site = {winning} (full-site changepoint collapse {site_collapse:+.3f}; necessary={site_necessary}, "
          f"random@N null p95={randN_p95:+.3f}); {len(cands)} candidates {'(behaviorally ranked)' if ranked else '(UNRANKED — non-enc_self)'}")

    rng = np.random.default_rng(CONFIG["SEED0"] + 909)
    ctx, tgt, metas = make_batch("changepoint", rng, CONFIG["N_SERIES"])
    clear_ablations(SITES); s0 = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint")

    # single-head collapses (backup/Hydra)
    singles = []
    for c in cands:
        set_explicit(SITES, [(c["site"], c["layer"], c["head"])])
        rel = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]
        singles.append(dict(layer=c["layer"], head=c["head"], rel=float(rel)));
    # group ladder k=1..MAX_SET + random-k-within-site null
    ladder = []
    rrng = np.random.default_rng(CONFIG["SEED0"] + 911)
    for k in range(1, min(CONFIG["LOCALIZE_MAX_SET"], len(cands)) + 1):
        set_explicit(SITES, [(c["site"], c["layer"], c["head"]) for c in cands[:k]])
        grp = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]
        nulls = []
        for _ in range(CONFIG["LOC_RANDOM_DRAWS"]):
            set_random_in_site(SITES, winning, k, rrng)
            nulls.append(rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0])
        null_p = float(np.percentile(nulls, 95)) if nulls else 0.0
        frac = float(grp / (site_collapse + 1e-6))
        ladder.append(dict(k=k, group=float(grp), null_p95=null_p, frac_of_site=frac,
                           beats_null=bool(grp > null_p), heads=[(c["layer"], c["head"]) for c in cands[:k]]))
    clear_ablations(SITES)

    # minimal sufficient set: smallest k reproducing >= LOCALIZE_FRAC of the site collapse AND beating its
    # within-site random-k null — but ONLY if the winning site is itself necessary (else there is no real collapse
    # to reproduce). [review FIX #1]
    suff = [r for r in ladder if r["frac_of_site"] >= CONFIG["LOCALIZE_FRAC"] and r["beats_null"]] if site_necessary else []
    minimal = min(suff, key=lambda r: r["k"]) if suff else None
    localized = minimal is not None
    # cross gate: a cross locus also needs the low-f selective effect
    low_f_ok, low_f = low_f_selective(winning) if winning == "cross" else (True, None)
    if winning == "cross" and not low_f_ok: localized = False
    single_null_p95 = float(ladder[0]["null_p95"]) if ladder else 0.0   # within-site k=1 null (for the SPLIT test)

    print(f"  single-head changepoint collapse (backup/Hydra): "
          + ", ".join(f"L{s['layer']}H{s['head']}={s['rel']:+.2f}" for s in singles))
    print(f"  group ladder (k -> collapse / frac-of-site / >null):")
    for r in ladder:
        print(f"    k={r['k']}  group={r['group']:+.3f}  frac={r['frac_of_site']:+.2f}  "
              f"null_p95={r['null_p95']:+.3f}  beats_null={r['beats_null']}")
    if localized:
        print(f"  MINIMAL SUFFICIENT SET = {minimal['k']} heads {minimal['heads']} "
              f"(reproduces {minimal['frac_of_site']:.0%} of the {winning} collapse, beats within-site null"
              f"{'' if ranked else '; UNRANKED candidates -> exploratory'})")
    elif not site_necessary:
        print(f"  Winning site {winning} is NOT necessary (collapse {site_collapse:+.3f} < max(STRUCT_COLLAPSE_MIN, "
              f"random@N null {randN_p95:+.3f})) -> no real collapse to localize; NOT localized.")
    else:
        print(f"  No small (1..{CONFIG['LOCALIZE_MAX_SET']}) set reproduces >= {CONFIG['LOCALIZE_FRAC']:.0%} of the "
              f"{winning} collapse above the within-site null -> NOT localized (distributed within the site)")

    # resample (corrupt-mean) cross-check on the locus only
    resample = None
    if localized:
        corrupt_ctx, _, _ = make_batch("nochange", np.random.default_rng(CONFIG["SEED0"] + 913), CONFIG["N_SERIES"])
        const_map = capture_corrupt_means(minimal["heads"], winning, corrupt_ctx)
        set_const_patch(winning, const_map)
        rs = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]
        clear_ablations(SITES)
        resample = float(rs); print(f"  resample (corrupt-mean) cross-check on the locus: collapse={rs:+.3f} "
                                    f"(vs mean-ablation {minimal['group']:+.3f})")
    return dict(winning=winning, site_collapse=float(site_collapse), site_necessary=bool(site_necessary),
                randN_p95=float(randN_p95), candidates_ranked=bool(ranked), single_null_p95=single_null_p95,
                singles=singles, ladder=ladder, minimal=minimal, localized=bool(localized),
                low_f_ok=bool(low_f_ok), low_f=low_f, resample=resample)

def low_f_selective(site):
    for f in CONFIG["F_GRID"]:
        if f > CONFIG["LOW_F_MAX"]: continue
        a = sweep_agg(site, f); r = sweep_agg("random", f)
        if a is None: continue
        if (a["changepoint"] >= CONFIG["STRUCT_COLLAPSE_MIN"]
                and a["changepoint"] >= CONFIG["SELECTIVITY_MARGIN"] * max(a["nontarget"], 1e-6)
                and (r is None or a["changepoint"] > r["changepoint"])):
            return True, f
    return False, None

# corrupt-mean capture/patch: capture each targeted head's mean .o-input vector on a no-shift battery (one forward
# with temporary capture hooks), to inject as a constant on the clean run (the §13 resample cross-check on the locus).
def capture_corrupt_means(heads, site, corrupt_ctx):
    ids, am = _tokenize(corrupt_ctx)
    caps = {}
    def mk(layer, mod):
        def hook(o_mod, args):
            caps[layer] = args[0].detach().float().mean(dim=tuple(range(args[0].dim() - 1))).cpu().numpy()  # (nh*dk,)
        return hook
    hs = [SITE_MODS[site][l].o.register_forward_pre_hook(mk(l, SITE_MODS[site][l])) for l, h in heads]
    with torch.inference_mode():
        if site == "enc_self": INNER.get_encoder()(input_ids=ids, attention_mask=am)
        else:
            dec = torch.full((ids.shape[0], 1), DEC_START, dtype=torch.long, device=DEVICE)
            INNER(input_ids=ids, attention_mask=am, decoder_input_ids=dec)
    for hh in hs: hh.remove()
    out = {}
    for l, h in heads:
        dk = _dkv(SITE_MODS[site][l]); out.setdefault(l, {})[h] = caps[l][h * dk:(h + 1) * dk]
    return out

def set_const_patch(site, const_map):
    clear_ablations(SITES)
    for l, hd in const_map.items(): SITE_MODS[site][l]._const_patch = dict(hd)

print("Causal localization (single/group ablation)..." + MOCK_TAG); LOC = run_localization()

# ============================================================
def run_algorithm():
    # characterize the change-DETECTOR heads (an encoder-side measurement, Mishra's prior site): the localized
    # enc_self set if we localized in enc_self, else the top enc_self behavioral candidates (exploratory).
    if LOC["localized"] and LOC["winning"] == "enc_self":
        heads = LOC["minimal"]["heads"]
    else:
        heads = [(c["layer"], c["head"]) for c in SCAN["candidates"] if c["site"] == "enc_self"][:3]
    if not heads: print("  no enc_self candidates to characterize"); return dict(heads=[])
    site = "enc_self"; CTX = CONFIG["CTX"]; rng = np.random.default_rng(CONFIG["SEED0"] + 1001)

    def head_response(ctxs, metas):                        # (post-shift - pre-shift) norm summed over the head set
        recs = encoder_head_norms(ctxs)
        idx = {(L, h): arr for (L, h, arr) in recs}
        resp = np.zeros(len(ctxs))
        for b in range(len(ctxs)):
            tau = metas[b]["tau"]; tot = 0.0
            for (l, h) in heads:
                if (l, h) in idx:
                    prof = idx[(l, h)][b]; tot += (prof[tau:CTX].mean() - prof[:tau].mean())
            resp[b] = tot
        return resp

    # (a) delta-scaling
    ds_ctx, ds_meta, ds_delta = [], [], []
    for dlt in np.linspace(0.5, 4.0, 10):
        s, m = make_levelshift(CTX, CONFIG["PRED"], rng, CONFIG["OBS_NOISE"], delta=float(dlt))
        ds_ctx.append(s[:CTX]); ds_meta.append(m); ds_delta.append(dlt)
    resp = head_response(ds_ctx, ds_meta)
    slope, _b = np.polyfit(ds_delta, resp, 1); r2 = _pearson(ds_delta, resp) ** 2
    # (b) boundary-locality: same delta, add a global offset c -> response should be ~invariant to c
    bl_ctx, bl_meta, offs = [], [], []
    for c in np.linspace(-3, 3, 8):
        s, m = make_levelshift(CTX, CONFIG["PRED"], rng, CONFIG["OBS_NOISE"], delta=2.0)
        s = s + c; m = dict(m); m["L0"] += c; m["L1"] += c
        bl_ctx.append(s[:CTX]); bl_meta.append(m); offs.append(c)
    resp_bl = head_response(bl_ctx, bl_meta); off_corr = _pearson(offs, resp_bl)
    # (c) recency: two shifts; does the boundary response peak at the most recent boundary?
    recency = None
    try:
        att = encoder_attentions  # use attention mass at each boundary
        rc = []
        for _ in range(min(8, CONFIG["SCAN_SERIES"])):
            L0 = rng.uniform(-1, 1); d1 = rng.choice([-1, 1]) * rng.uniform(1.5, 3); d2 = rng.choice([-1, 1]) * rng.uniform(1.5, 3)
            t1 = int(0.4 * CTX); t2 = int(0.75 * CTX)
            s = np.full(CTX + CONFIG["PRED"], L0); s[t1:] += d1; s[t2:] += d2
            s = s + CONFIG["OBS_NOISE"] * rng.standard_normal(len(s)); rc.append((s[:CTX], t1, t2))
        A = att([x for x, _, _ in rc]); w = CONFIG["BAND_W"]
        m1 = m2 = 0.0
        for (l, h) in heads:
            for b, (_, t1, t2) in enumerate(rc):
                Al = A[l][b, h]
                m1 += float(Al[t2:CTX, max(0, t1 - w):t1 + w + 1].sum(1).mean())
                m2 += float(Al[t2:CTX, max(0, t2 - w):t2 + w + 1].sum(1).mean())
        recency = dict(mass_recent=m2, mass_older=m1, prefers_recent=bool(m2 > m1))
    except Exception as e:
        recency = dict(error=repr(e)[:80])

    # node-level corrupt patch (path-patch primitive) on the head set, if localized in the encoder
    node_patch = None
    if LOC["localized"] and LOC["winning"] == "enc_self":
        ctx, tgt, metas = make_batch("changepoint", np.random.default_rng(CONFIG["SEED0"] + 1003), max(8, CONFIG["SWEEP_SERIES"]))
        clear_ablations(SITES); s0 = _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint")
        corrupt_ctx, _, _ = make_batch("nochange", np.random.default_rng(CONFIG["SEED0"] + 1004), len(ctx))
        cmap = capture_corrupt_means(heads, site, corrupt_ctx); set_const_patch(site, cmap)
        npc = rel_collapse(s0, _struct_vec(FORECAST(ctx, CONFIG["N_CRPS_SAMPLES"]), metas, tgt, "changepoint"))[0]
        clear_ablations(SITES); node_patch = float(npc)

    print(f"  (a) delta-scaling: slope={slope:+.4f}  R^2={r2:.3f}  -> {'linear-ish delta detector' if r2>0.5 else 'weak/nonlinear'}")
    print(f"  (b) boundary-locality: corr(response, global offset)={off_corr:+.3f}  "
          f"-> {'invariant to absolute level (boundary-local)' if abs(off_corr)<0.3 else 'level-dependent'}")
    if recency and "prefers_recent" in recency:
        print(f"  (c) recency: mass@recent={recency['mass_recent']:.3f} vs @older={recency['mass_older']:.3f}  "
              f"-> {'tracks most recent boundary' if recency['prefers_recent'] else 'no recency preference'}")
    if node_patch is not None:
        print(f"  node corrupt-patch on the set: collapse={node_patch:+.3f}")
    return dict(heads=heads, delta_slope=float(slope), delta_r2=float(r2), offset_corr=float(off_corr),
                recency=recency, node_patch=node_patch)

print("Algorithm characterization..." + MOCK_TAG); ALG = run_algorithm()

# ============================================================
def mishra_check():
    lo, hi = CONFIG["MISHRA_DEPTH_LO"], CONFIG["MISHRA_DEPTH_HI"]
    # depth of the localized / top enc_self candidates
    if LOC["localized"] and LOC["winning"] == "enc_self":
        layers = sorted(set(l for (l, h) in LOC["minimal"]["heads"]))
        src = "localized set"
    else:
        layers = sorted(set(c["layer"] for c in SCAN["candidates"] if c["site"] == "enc_self"))[:CONFIG["N_CANDIDATES"]]
        src = "top behavioral candidates"
    depths = [(l, rel_depth(l)) for l in layers]
    in_band = [l for l, d in depths if lo <= d <= hi]
    band_layers = [l for l in range(N_ENC_LAYERS) if lo <= rel_depth(l) <= hi]   # exact in-band layers (no int() truncation) [review FIX #5]
    _br = f"{band_layers[0]}-{band_layers[-1]}" if band_layers else "none"
    print(f"  enc layers: {N_ENC_LAYERS}; mid-encoder band = [{lo:.2f},{hi:.2f}] rel-depth "
          f"(layers {_br} on this model)")
    print(f"  {src} enc_self layers + rel-depth: " + ", ".join(f"L{l}@{d:.2f}" for l, d in depths))
    print(f"  in mid-encoder band: {in_band if in_band else 'none'}")
    if not IS_LARGE:
        print("  AMBIGUITY (base): a DISTRIBUTED result here is ambiguous between 'distributed in Chronos' and "
              "'base too small to show what Large localizes' — run pilot_a100 (Large) to disambiguate.")
    return dict(enc_layers=N_ENC_LAYERS, band=[lo, hi], depths=depths, in_band=in_band, is_large=bool(IS_LARGE), source=src)

print("Mishra relative-depth cross-check..." + MOCK_TAG); MISHRA = mishra_check()

# ============================================================
def selective_change(site):
    cp = _mean(site, "changepoint", "rel_collapse"); cp_lo = float(np.mean(
        [r["rel_ci"][0] for r in FULL if r.get("kind") == site and r["cond"] == "changepoint" and "rel_ci" in r] or [float("nan")]))
    mot = _mean(site, "motif", "rel_collapse"); trd = _mean(site, "trend", "rel_collapse")
    sig = (cp_lo > 0) and (cp >= CONFIG["STRUCT_COLLAPSE_MIN"])
    sel = cp >= CONFIG["SELECTIVITY_MARGIN"] * max(mot, trd, 1e-6)
    return dict(changepoint=cp, cp_lo=cp_lo, motif=mot, trend=trd, sig=bool(sig), selective=bool(sel))

def summarize():
    null = [v for r in FULL if r["kind"] == "random" and r["cond"] == "changepoint" for v in r["rel_draws"]]
    null_p = float(np.percentile(null, 95)) if null else 0.0
    rows = []
    for site in CONFIG["SITES_4"]:
        ss = selective_change(site); beats = ss["changepoint"] > null_p
        rows.append(dict(site=site, **ss, beats_null=bool(beats),
                         dcrps=_mean(site, "changepoint", "dcrps_mean")))
    win = LOC["winning"]; necessity = next(r for r in rows if r["site"] == win)
    site_real = necessity["beats_null"]                     # change-detection is real & site-mediated (necessity)
    site_selective = necessity["selective"]
    localized = LOC["localized"]

    # `localized` already requires the winning site to be NECESSARY (>= STRUCT_COLLAPSE_MIN AND > random@N null),
    # so a LOCALIZED claim can never reproduce a non-existent collapse [review FIX #1]. A change-detection circuit
    # (A) additionally requires the site collapse to be changepoint-SELECTIVE; a localized-but-non-selective set is
    # general forecasting heads (A-), not change-detection-specific.
    expl = "" if LOC.get("candidates_ranked", True) else " [exploratory: UNRANKED candidates]"
    if localized and site_selective:
        verdict = f"A: LOCALIZED change-detection circuit in {win}{expl}"
    elif localized:
        verdict = (f"A-: LOCALIZED set in {win} reproduces the site collapse but is NOT changepoint-selective "
                   f"(general forecasting heads, not change-detection-specific){expl}")
    elif site_real:
        verdict = f"B: DISTRIBUTED (necessity in {win}, no small sufficient set)"
    else:
        verdict = "B: DISTRIBUTED (no site even necessary above null)"
    # SPLIT = partial localization: a single head clears the size-matched WITHIN-site k=1 null (not the N-head
    # random@N necessity null) but no small set fully localizes -> backup redundancy [review FIX #2].
    single_null = LOC.get("single_null_p95", null_p)
    singles_hit = sum(1 for s in LOC["singles"] if s["rel"] > single_null)
    if (not localized) and singles_hit > 0 and site_real:
        verdict = f"SPLIT: partial localization in {win} ({singles_hit} head(s) above within-site k=1 null) + backup redundancy"

    print("=" * 92); print(f"PHASE 4 VERDICT: {verdict}{MOCK_TAG}"); print("=" * 92)
    print(f"  {'site':9s} {'chgpt':>7s} {'motif':>7s} {'trend':>7s}  {'sel':>4s} {'>null':>6s}  {'ΔCRPS':>8s}")
    for r in rows:
        print(f"  {r['site']:9s} {r['changepoint']:+7.3f} {r['motif']:+7.3f} {r['trend']:+7.3f}  "
              f"{str(r['selective'])[:1]:>4s} {str(r['beats_null'])[:1]:>6s}  {r['dcrps']:+8.3f}")
    print(f"\n  size-matched random@N null (changepoint rel-collapse, p95) = {null_p:+.3f}  [necessity gate]")
    print(f"  winning site = {win}: full-site collapse {necessity['changepoint']:+.3f} "
          f"(necessity={'YES' if site_real else 'no'}, selective={'YES' if site_selective else 'no'})")
    if localized:
        m = LOC["minimal"]
        tag = ("circuit-level corroboration of Mishra" if (site_selective and LOC.get("candidates_ranked", True))
               else "general forecasting set, NOT change-detection-selective" if not site_selective
               else "exploratory — UNRANKED candidates (behavioral scan ranks enc_self only)")
        print(f"  LOCALIZATION: minimal sufficient set = {m['k']} heads {m['heads']} "
              f"({m['frac_of_site']:.0%} of site collapse, beats within-site null)  -> {tag}")
    else:
        print(f"  LOCALIZATION: no small (1..{CONFIG['LOCALIZE_MAX_SET']}) sufficient set -> change-detection is "
              f"DISTRIBUTED in {win} (necessary in aggregate, not localized) — symmetric to periodicity, still publishable.")
    if win == "cross":
        print(f"  CROSS low-f gate: {'PASS' if LOC['low_f_ok'] else 'FAIL'} "
              f"(low-f selective effect {'at f=' + str(LOC['low_f']) if LOC['low_f'] else 'absent -> severance, not a locus'})")
    # detection-power statement (reuse): is a selective, localized effect even findable here?
    if any(r["selective"] and r["beats_null"] for r in rows) or localized:
        print("  DETECTION POWER: a selective/localized change-detection effect IS findable here -> a distributed verdict is de-confounded.")
    else:
        best = max(rows, key=lambda r: r["changepoint"] if np.isfinite(r["changepoint"]) else -1e9)
        print(f"  DETECTION POWER: no site selectively collapses changepoint above the null "
              f"(best {best['changepoint']:+.3f} vs null {null_p:+.3f}) -> the DISTRIBUTED null is real, not underpowered.")
    if not IS_LARGE:
        print("  BASE AMBIGUITY: this is the base model; a DISTRIBUTED result is ambiguous vs 'base too small' — "
              "run pilot_a100 (Large) to disambiguate (Mishra's model is Large).")
    return dict(verdict=verdict, rows=rows, null_p95=null_p, winning=win, localized=bool(localized),
                site_real=bool(site_real), site_selective=bool(site_selective),
                site_necessary=bool(LOC.get("site_necessary", site_real)),
                candidates_ranked=bool(LOC.get("candidates_ranked", True)), is_large=bool(IS_LARGE),
                minimal=LOC["minimal"], thresholds={k: CONFIG[k] for k in
                ("STRUCT_COLLAPSE_MIN", "SELECTIVITY_MARGIN", "LOW_F_MAX", "LOCALIZE_FRAC", "LOCALIZE_MAX_SET")})

SUMMARY = summarize()

# ============================================================
try:
    rows = SUMMARY["rows"]; sites = [r["site"] for r in rows]; xs = np.arange(len(sites)); w = 0.25
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    # 5a: per-site structural selectivity (changepoint vs motif/trend) + random@N null line
    a = ax[0, 0]
    a.bar(xs - w, [r["changepoint"] for r in rows], w, label="changepoint (GATE)", color="#c0392b")
    a.bar(xs,     [r["motif"] for r in rows], w, label="motif (period-P)", color="#7f8c8d")
    a.bar(xs + w, [r["trend"] for r in rows], w, label="trend (slope)", color="#bdc3c7")
    a.axhline(SUMMARY["null_p95"], color="k", ls="--", lw=1, label="random@N null p95")
    a.axhline(CONFIG["STRUCT_COLLAPSE_MIN"], color="k", ls=":", lw=1, label="collapse min")
    a.axhline(0, color="gray", lw=0.8); a.set_xticks(xs); a.set_xticklabels(sites)
    a.set_ylabel("structural rel-collapse"); a.legend(fontsize=7)
    a.set_title("Fig 5a: per-site selectivity (changepoint gate)" + MOCK_TAG, fontsize=9)
    # 5b: dose-response changepoint collapse vs f per site (multi-seed bands) + cross low-f
    b = ax[0, 1]
    for arm, col in [("enc_self", "#2980b9"), ("dec_self", "#27ae60"), ("cross", "#c0392b"), ("random", "#999999")]:
        ys = [(sweep_agg(arm, f) or {"changepoint": np.nan})["changepoint"] for f in CONFIG["F_GRID"]]
        sd = [(sweep_agg(arm, f) or {"cp_sd": 0.0})["cp_sd"] for f in CONFIG["F_GRID"]]
        b.errorbar(CONFIG["F_GRID"], ys, yerr=sd, marker="o", color=col, label=arm, capsize=2,
                   lw=(2 if arm != "random" else 1), ls=("--" if arm == "random" else "-"))
    b.axvline(CONFIG["LOW_F_MAX"], color="k", ls=":", lw=1, label=f"low-f gate ({CONFIG['LOW_F_MAX']})")
    b.set_xlabel("fraction of site ablated (f)"); b.set_ylabel("changepoint rel-collapse")
    b.set_title("Fig 5b: dose-response (multi-seed)", fontsize=9); b.legend(fontsize=7)
    # 5c: head-level localization — group ladder vs site collapse + random-k null (Mishra depth is shown in 5d)
    c = ax[1, 0]
    lad = LOC["ladder"]
    if lad:
        ks = [r["k"] for r in lad]
        c.plot(ks, [r["group"] for r in lad], marker="o", color="#8e44ad", label="top-k group")
        c.plot(ks, [r["null_p95"] for r in lad], marker="x", color="#999", ls="--", label="random-k null p95")
        c.axhline(LOC["site_collapse"], color="#c0392b", ls="-", lw=1, label=f"full {LOC['winning']} collapse")
        c.axhline(CONFIG["LOCALIZE_FRAC"] * LOC["site_collapse"], color="k", ls=":", lw=1, label=f"{CONFIG['LOCALIZE_FRAC']:.0%} of site")
        c.set_xlabel("k (heads ablated, top behavioral candidates)"); c.set_ylabel("changepoint rel-collapse")
    c.set_title(f"Fig 5c: head-level localization [{'LOCALIZED' if LOC['localized'] else 'distributed'}]", fontsize=9)
    c.legend(fontsize=7)
    # 5d: mechanism — delta-scaling (if we have candidates), else annotate
    d = ax[1, 1]
    if ALG.get("heads"):
        d.text(0.03, 0.92, f"localized set: {ALG['heads']}", fontsize=8, transform=d.transAxes)
        d.text(0.03, 0.78, f"(a) delta-scaling slope={ALG['delta_slope']:+.3f}, R^2={ALG['delta_r2']:.2f}", fontsize=8, transform=d.transAxes)
        d.text(0.03, 0.64, f"(b) boundary-locality corr(offset)={ALG['offset_corr']:+.2f}", fontsize=8, transform=d.transAxes)
        rc = ALG.get("recency") or {}
        if "prefers_recent" in rc:
            d.text(0.03, 0.50, f"(c) recency: prefers most-recent boundary = {rc['prefers_recent']}", fontsize=8, transform=d.transAxes)
        # mishra band
        dep = MISHRA["depths"]
        if dep:
            d.text(0.03, 0.30, "enc-self rel-depth vs Mishra mid-encoder [%.2f,%.2f]:" % tuple(MISHRA["band"]), fontsize=8, transform=d.transAxes)
            d.text(0.03, 0.18, ", ".join(f"L{l}@{dd:.2f}" for l, dd in dep), fontsize=8, transform=d.transAxes)
    d.set_axis_off(); d.set_title("Fig 5d: mechanism + Mishra depth" + (" (localized)" if LOC["localized"] else " (exploratory)"), fontsize=9)
    fig.suptitle(f"Phase 4 — change detection: {SUMMARY['verdict']}" + MOCK_TAG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(CKPT_DIR, f"fig5_phase4_{MODE}.png"), dpi=90)
    plt.show(); plt.close(fig); print(f"saved fig5_phase4_{MODE}.png")
except Exception as e:
    import traceback; print("fig skipped:", repr(e)[:160]); traceback.print_exc()

# ============================================================
out = dict(summary=SUMMARY, full_site=FULL, sweep=SWEEP, scan={k: SCAN[k] for k in ("candidates", "boundary_null_p95")},
           localization=LOC, algorithm=ALG, mishra=MISHRA, monotonicity=MONO,
           config=dict(mode=MODE, model_id=CONFIG["model_id"], ctx=CONFIG["CTX"], pred=CONFIG["PRED"],
                       tau_frac_ctx=CONFIG["TAU_FRAC_CTX"], n_seeds=CONFIG["N_SEEDS"], n_series=CONFIG["N_SERIES"],
                       sweep_seeds=CONFIG["SWEEP_SEEDS"], f_grid=CONFIG["F_GRID"], is_large=bool(IS_LARGE)))
p = os.path.join(CKPT_DIR, f"phase4_{MODE}.json")
with open(p, "w") as f: json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote", p, "->", SUMMARY["verdict"], MOCK_TAG)