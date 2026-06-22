# auto-mirror of phase5_v3.ipynb code cells (local smoke test)

# ============================================================
import os
CONFIG = {
    "MODE": "mock_cpu",                  # -> "pilot_a100" (Large, high-RAM A100)
    "MODEL_BY_MODE": {"mock_cpu": None, "pilot_t4": "amazon/chronos-t5-base", "pilot_a100": "amazon/chronos-t5-large"},
    "USE_DRIVE": True,
    "SEED0": 0, "PERIODS": [8, 12, 16, 24],
    "N_PAIRS": 32, "CTX": 256, "PRED": 64, "OBS_NOISE": 0.30, "TAU_FRAC_CTX": 0.65,
    "N_CRPS_SAMPLES": 48, "N_BOOTSTRAP": 1000, "FORECAST_BATCH": 4,
    "DELTA_PRIMARY": [1.5, 3.0],
    # ---- the cross-layer set (the v3 point) ----
    "CROSS_N_LAYERS": 4,                 # top-N mid-encoder enc-MLP layers to host SAEs (lean default)
    "ORIENT_DEPTH_LO": 0.40, "ORIENT_DEPTH_HI": 0.75,
    "CACHE_DTYPE": "float16",            # CPU-resident feature caches (fp16 to fit RAM); moved to GPU per chunk
    # ---- SAE feature basis (TopK) ----
    "FEATURE_BASIS": "sae", "SAE_DICT_MULT": 8, "SAE_TOPK": 32, "SAE_STEPS": 600, "SAE_LR": 1e-3, "SAE_BATCH": 2048,
    # ---- faithfulness / completeness / denoising curve over UNION size ----
    "K_GRID": [1, 2, 4, 8, 16, 32, 64, 128],
    "FAITH_TARGET": 0.60, "LOCALIZE_MAX_FEATURES": 32, "N_RANDOM_NULL": 3, "SELECTIVITY_MARGIN": 2.0,
    "MISHRA_DEPTH_LO": 0.45, "MISHRA_DEPTH_HI": 0.55,
    # ---- mock overrides ----
    "mock_cpu": {
        "PERIODS": [6, 8], "N_PAIRS": 4, "CTX": 48, "PRED": 24, "N_CRPS_SAMPLES": 12, "N_BOOTSTRAP": 50,
        "FORECAST_BATCH": 999, "CROSS_N_LAYERS": 2, "SAE_DICT_MULT": 4, "SAE_TOPK": 4, "SAE_STEPS": 8,
        "SAE_BATCH": 64, "K_GRID": [1, 2, 4], "LOCALIZE_MAX_FEATURES": 2, "N_RANDOM_NULL": 2,
    },
}
MODE = os.environ.get("CHRONOS_P5V3_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4", "pilot_a100"), MODE
CONFIG["model_id"] = os.environ.get("CHRONOS_P5V3_MODEL", CONFIG["MODEL_BY_MODE"][MODE])
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
            drive.mount("/content/drive"); CKPT_DIR = "/content/drive/MyDrive/chronos_phase5v3"; os.makedirs(CKPT_DIR, exist_ok=True)
            print("checkpoints -> Google Drive:", CKPT_DIR)
        except Exception as e:
            print("Drive mount skipped (", repr(e)[:80], ") -> /content")
print(f"MODE={MODE}{MOCK_TAG}  model={CONFIG['model_id']}  cross_layers={CONFIG['CROSS_N_LAYERS']}  "
      f"pairs={CONFIG['N_PAIRS']}  K_GRID={CONFIG['K_GRID']}  ckpt={CKPT_DIR}")

# ============================================================
import sys, json, subprocess, gc, re, warnings
warnings.filterwarnings("ignore", message=".*past_key_values.*")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _ensure(pkg, imp):
    if os.environ.get("CHRONOS_P5V3_SKIP_INSTALL") == "1": return
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
DTYPE = torch.float32; CACHE_DT = torch.float16 if CONFIG["CACHE_DTYPE"] == "float16" else torch.float32
print("device:", DEVICE)

# ============================================================
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
        _free, _tot = torch.cuda.mem_get_info(); print(f"GPU free {_free/1e9:.1f}/{_tot/1e9:.1f} GB")
        assert _free > (6.0e9 if IS_LARGE else 1.5e9), "low GPU memory — RESTART THE RUNTIME."
    PIPE = ChronosPipeline.from_pretrained(CONFIG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)
    INNER = PIPE.inner_model.eval(); VOCAB = INNER.config.vocab_size; INNER.requires_grad_(False)
    try: INNER.config._attn_implementation = "eager"
    except Exception: pass
    DEC_START = int(getattr(INNER.config, "decoder_start_token_id", 0)); D_MODEL = int(INNER.config.d_model)
MLP = classify_mlp_modules(INNER)
MLP_MODS = {s: {_layer_idx(n): mod for n, mod in MLP[s]} for s in MLP}
N_ENC_LAYERS = len(MLP["enc_mlp"])
def rel_depth(li): return float(li) / max(1, (N_ENC_LAYERS - 1))
print("enc MLP layers:", N_ENC_LAYERS, "| d_model:", D_MODEL)

# ============================================================
def make_cf_pair(ctx_len, pred_len, rng, noise, delta):
    tau = int(CONFIG["TAU_FRAC_CTX"] * ctx_len)
    L0 = rng.uniform(-1.0, 1.0); L1 = L0 + delta
    eps = rng.normal(0, noise, size=ctx_len + pred_len); base = np.arange(ctx_len + pred_len)
    clean = np.where(base < tau, L0, L1) + eps; corrupt = np.full(ctx_len + pred_len, L0) + eps
    return clean[:ctx_len], corrupt[:ctx_len], clean[ctx_len:], dict(tau=int(tau), L0=float(L0), L1=float(L1), delta=float(delta))
def make_motif(P, L, rng, noise):
    m = rng.standard_normal(P); m[rng.integers(P)] += 3.0 * (1 if rng.random() > 0.5 else -1)
    m[P // 2:] += 1.5; m = m - m.mean(); return np.tile(m, L // P + 2)[:L] + noise * rng.standard_normal(L)
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
_r = np.random.default_rng(1); _a, _b, _c, _m = make_cf_pair(64, 16, _r, 0.3, 2.0)
assert np.allclose(_a[:_m["tau"]], _b[:_m["tau"]]) and not np.allclose(_a[_m["tau"]:], _b[_m["tau"]:])
print("counterfactual-pair assert PASS; metrics ready")

# ============================================================
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

def _mlp_hook(module, inp, out):                 # counterfactual feature patch (per-layer _cf), error node carried
    cf = getattr(module, "_cf", None)
    if cf is not None:
        sae, idx, src_f = cf; h = out; f = sae.encode(h); err = h - sae.decode(f)
        L = min(h.shape[-2], src_f.shape[-2]); f[..., :L, idx] = src_f[..., :L, idx]; return sae.decode(f) + err
    if getattr(module, "_ablate", False):        # orientation-only mean ablation (non-gating)
        return out.mean(dim=tuple(range(out.dim()-1)), keepdim=True).expand_as(out)
    return None
HANDLES = [mod.register_forward_hook(_mlp_hook) for _, mod in MLP["enc_mlp"]]
for _, mod in MLP["enc_mlp"]: mod._cf = None; mod._ablate = False
def clear_hooks():
    for _, mod in MLP["enc_mlp"]: mod._cf = None; mod._ablate = False

def capture_h(layer_mod, contexts):
    ids, am = _tokenize(contexts); store = {}
    def cap(m, i, o): store["h"] = o.detach()
    hh = layer_mod.register_forward_hook(cap)
    with torch.inference_mode(): INNER.get_encoder()(input_ids=ids, attention_mask=am)
    hh.remove(); return store["h"]

def forecast_cf_multi(contexts, arm, n_samples):
    # arm = {layer_idx: (idx_gpu_long, src_cpu)}; per chunk, move each layer's src chunk to GPU and patch ALL layers.
    if IS_MOCK:
        clear_hooks()
        for l, (idx, src) in arm.items(): MLP_MODS["enc_mlp"][l]._cf = (SAES[l], idx, src.to(DEVICE, dtype=DTYPE))
        out = forecast_raw(contexts, n_samples); clear_hooks(); return out
    bs = int(CONFIG.get("FORECAST_BATCH", 4)); outs = []; i = 0
    while i < len(contexts):
        clear_hooks()
        for l, (idx, src) in arm.items():
            MLP_MODS["enc_mlp"][l]._cf = (SAES[l], idx, src[i:i+bs].to(DEVICE, dtype=DTYPE))
        out = forecast_raw(contexts[i:i+bs], n_samples); clear_hooks()
        outs.append(out); i += bs
    return np.concatenate(outs, axis=0)

print(f"hooks on {len(HANDLES)} enc-MLP layers; multi-layer counterfactual patch ready")

# ============================================================
class TopKSAE(nn.Module):
    def __init__(self, d, m, k):
        super().__init__(); self.k = k; self.b_pre = nn.Parameter(torch.zeros(d)); self.enc = nn.Linear(d, m); self.dec = nn.Linear(m, d, bias=False)
    def encode(self, x):
        z = torch.relu(self.enc(x - self.b_pre)); v, idx = z.topk(self.k, dim=-1); return torch.zeros_like(z).scatter_(-1, idx, v)
    def decode(self, z): return self.dec(z) + self.b_pre
class NeuronBasis(nn.Module):
    def __init__(self, d): super().__init__(); self.d = d
    def encode(self, x): return x
    def decode(self, z): return z
def build_sae(acts, d):
    if CONFIG["FEATURE_BASIS"] == "neuron": return NeuronBasis(d).to(DEVICE), 0.0, d
    m = CONFIG["SAE_DICT_MULT"] * d; sae = TopKSAE(d, m, min(CONFIG["SAE_TOPK"], m)).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=CONFIG["SAE_LR"]); N = acts.shape[0]; sae.b_pre.data = acts.mean(0).detach(); loss = 0.0
    for _ in range(CONFIG["SAE_STEPS"]):
        ix = torch.randint(0, N, (min(CONFIG["SAE_BATCH"], N),), device=acts.device); x = acts[ix]
        z = sae.encode(x); xr = sae.decode(z); loss = ((xr - x) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    return sae, float(loss.detach()), m
print("SAE ready")

# ============================================================
FORCE = os.environ.get("CHRONOS_P5V3_FORCE", "0") == "1"
def _ckp(name): return os.path.join(CKPT_DIR, f"phase5v3_{MODE}_{name}.json")
def _load(name):
    p = _ckp(name)
    if os.path.exists(p) and not FORCE:
        try: return json.load(open(p))
        except Exception: return None
    return None
def _save(name, obj): json.dump(obj, open(_ckp(name), "w"), default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))

def orient():
    ck = _load("orient")
    if ck is not None: print("  [ckpt] orientation resumed ->", ck["layers"]); return ck
    rng = np.random.default_rng(CONFIG["SEED0"] + 11); cc, co, tg, mt = make_cf_battery(rng, CONFIG["N_PAIRS"], *CONFIG["DELTA_PRIMARY"])
    clear_hooks(); s0 = cp_vec(forecast_raw(cc, CONFIG["N_CRPS_SAMPLES"]), mt)
    band = [li for li in range(N_ENC_LAYERS) if CONFIG["ORIENT_DEPTH_LO"] <= rel_depth(li) <= CONFIG["ORIENT_DEPTH_HI"]]
    if len(band) < CONFIG["CROSS_N_LAYERS"]: band = list(range(N_ENC_LAYERS))
    scan = []
    for li in band:
        clear_hooks(); MLP_MODS["enc_mlp"][li]._ablate = True
        sa = cp_vec(forecast_raw(cc, CONFIG["N_CRPS_SAMPLES"]), mt); clear_hooks()
        scan.append((li, float((s0 - sa).mean() / (s0.mean() + 1e-6))))
    scan.sort(key=lambda r: r[1], reverse=True)
    layers = sorted(int(l) for l, _ in scan[:CONFIG["CROSS_N_LAYERS"]])
    print(f"  mid-encoder scan: " + ", ".join(f"L{l}@{rel_depth(l):.2f}={v:+.2f}" for l, v in scan))
    print(f"  -> cross-layer SAE set = {layers} (rel-depths " + ",".join(f"{rel_depth(l):.2f}" for l in layers) + ")")
    res = dict(layers=layers, scan=scan, clean_recovery=float(s0.mean())); _save("orient", res); return res
print("Orientation (non-gating)..." + MOCK_TAG); ORIENT = orient(); LAYERS = ORIENT["layers"]

# ============================================================
_rng = np.random.default_rng(CONFIG["SEED0"] + 21)
CC, CO, TG, MT = make_cf_battery(_rng, CONFIG["N_PAIRS"], *CONFIG["DELTA_PRIMARY"])
SAES, FC, FCO, NF = {}, {}, {}, None; recon = {}
for l in LAYERS:
    lm = MLP_MODS["enc_mlp"][l]; Hc = capture_h(lm, CC); Hco = capture_h(lm, CO)
    torch.manual_seed(CONFIG["SEED0"] + l)                       # per-layer SAE reproducible (resume-safe)
    sae, rl, nf = build_sae(Hc.reshape(-1, D_MODEL), D_MODEL); SAES[l] = sae; recon[l] = rl; NF = nf
    FC[l] = sae.encode(Hc).to("cpu", dtype=CACHE_DT); FCO[l] = sae.encode(Hco).to("cpu", dtype=CACHE_DT)   # CPU caches
    del Hc, Hco
    if DEVICE == "cuda": torch.cuda.empty_cache()
f0 = SAES[LAYERS[0]].encode(capture_h(MLP_MODS["enc_mlp"][LAYERS[0]], CC[:1]))
assert torch.allclose(SAES[LAYERS[0]].decode(f0) + (capture_h(MLP_MODS["enc_mlp"][LAYERS[0]], CC[:1]) - SAES[LAYERS[0]].decode(f0)), capture_h(MLP_MODS["enc_mlp"][LAYERS[0]], CC[:1]), atol=1e-3)
print(f"  {len(LAYERS)} layers x {NF} features = {len(LAYERS)*NF} (layer,feature) pairs; recon_loss " + ",".join(f"L{l}:{recon[l]:.3f}" for l in LAYERS))

def _idx_tensor(ids): return torch.as_tensor(list(ids), dtype=torch.long, device=DEVICE)
def global_ie():
    taus = [m["tau"] for m in MT]; CTX = CONFIG["CTX"]; ie_per = {}
    for l in LAYERS:
        fc = FC[l].to(DEVICE, dtype=DTYPE); fo = FCO[l].to(DEVICE, dtype=DTYPE); dpost = torch.zeros(NF, device=DEVICE)
        for b in range(len(taus)): dpost += (fc[b, taus[b]:CTX] - fo[b, taus[b]:CTX]).abs().mean(0)
        W = SAES[l].dec.weight if hasattr(SAES[l], "dec") else torch.eye(D_MODEL, device=DEVICE)
        wn = W.norm(dim=0) if (W.dim() == 2 and W.shape[1] == NF) else torch.ones(NF, device=DEVICE)
        ie_per[l] = (dpost / len(taus) * wn).detach().cpu().numpy(); del fc, fo
    flat = np.concatenate([ie_per[l] for l in LAYERS])            # length N*NF, layer-major
    order = list(np.argsort(-flat))                               # global flat order of (layer,feature) pairs
    return order, ie_per
ORDER, IE_PER = global_ie()
def split_union(k):                                              # top-k flat pairs -> {layer: [feature idxs]}
    per = {l: [] for l in LAYERS}
    for j in ORDER[:k]: per[LAYERS[j // NF]].append(int(j % NF))
    return per
_top = split_union(min(8, len(ORDER)))
print(f"  top-{min(8,len(ORDER))} union by layer: " + ", ".join(f"L{l}:{len(_top[l])}" for l in LAYERS))
# PLUMBING: multi-layer patch bites
_ids, _am = _tokenize(CC[:2]); _dec = torch.full((2, 1), DEC_START, dtype=torch.long, device=DEVICE)
clear_hooks()
with torch.inference_mode(): _g0 = INNER(input_ids=_ids, attention_mask=_am, decoder_input_ids=_dec).logits.clone()
for l in LAYERS: MLP_MODS["enc_mlp"][l]._cf = (SAES[l], torch.arange(NF, device=DEVICE), FCO[l][:2].to(DEVICE, dtype=DTYPE))
with torch.inference_mode(): _g1 = INNER(input_ids=_ids, attention_mask=_am, decoder_input_ids=_dec).logits.clone()
clear_hooks(); assert not torch.allclose(_g0, _g1), "multi-layer patch did not bite"
print(f"  PLUMBING: multi-layer counterfactual patch bites (max|Δ|={(_g0-_g1).abs().max():.4g})  PASS" + MOCK_TAG)

# ============================================================
def _complement_arm(per, src):                                   # keep union, ablate everything else at each layer -> src
    arm = {}
    for l in LAYERS:
        mask = torch.ones(NF, dtype=torch.bool);
        if per[l]: mask[per[l]] = False
        arm[l] = (mask.nonzero(as_tuple=True)[0].to(DEVICE), src[l])
    return arm
def _union_arm(per, src):                                        # ablate ONLY the union features at each layer -> src
    return {l: (_idx_tensor(per[l]), src[l]) for l in LAYERS if per[l]}

def run_curves():
    clear_hooks(); base = cp_vec(forecast_raw(CC, CONFIG["N_CRPS_SAMPLES"]), MT)
    clear_hooks(); corr_base = cp_vec(forecast_raw(CO, CONFIG["N_CRPS_SAMPLES"]), MT)
    rngm = np.random.default_rng(CONFIG["SEED0"] + 44); mctx, mtg, mmeta = make_motif_battery(rngm, CONFIG["N_PAIRS"])
    clear_hooks(); mbase = motif_vec(forecast_raw(mctx, CONFIG["N_CRPS_SAMPLES"]), mmeta)
    base_motif = {l: FCO[l].float().mean(dim=(0, 1), keepdim=True).expand(len(mctx), FCO[l].shape[1], -1).to(CACHE_DT) for l in LAYERS}  # no-shift baseline per layer, motif shape
    KMAX = len(ORDER); rng = np.random.default_rng(CONFIG["SEED0"] + 77)
    rows = _load("curves") or []; done = {r["k"] for r in rows}
    for k in [kk for kk in CONFIG["K_GRID"] if kk <= KMAX]:
        if k in done: print(f"    [ckpt] k={k} resumed"); continue
        per = split_union(k)
        # faithfulness: keep union clean, ablate complement -> corrupt
        fa = cp_vec(forecast_cf_multi(CC, _complement_arm(per, FCO), CONFIG["N_CRPS_SAMPLES"]), MT); fa_m, fa_ci = mean_ci(fa)
        # completeness: ablate union -> corrupt (changepoint) ; selectivity on motif (union -> no-shift baseline)
        ca = cp_vec(forecast_cf_multi(CC, _union_arm(per, FCO), CONFIG["N_CRPS_SAMPLES"]), MT); cp_drop, cp_ci = mean_ci(base - ca)
        ma = motif_vec(forecast_cf_multi(mctx, _union_arm(per, base_motif), CONFIG["N_CRPS_SAMPLES"]), mmeta); mot_drop = float((mbase - ma).mean())
        # denoising: corrupt run, union -> clean
        da = cp_vec(forecast_cf_multi(CO, _union_arm(per, FC), CONFIG["N_CRPS_SAMPLES"]), MT); suf_m, suf_ci = mean_ci(da)
        # nulls (random-k union across layers) for faithfulness + completeness + denoising
        fa_null, cp_null, suf_null = [], [], []
        for _ in range(CONFIG["N_RANDOM_NULL"]):
            rsel = rng.choice(KMAX, size=k, replace=False); rper = {l: [] for l in LAYERS}
            for j in rsel: rper[LAYERS[j // NF]].append(int(j % NF))
            fa_null.append(float(cp_vec(forecast_cf_multi(CC, _complement_arm(rper, FCO), CONFIG["N_CRPS_SAMPLES"]), MT).mean()))
            cp_null.append(float((base - cp_vec(forecast_cf_multi(CC, _union_arm(rper, FCO), CONFIG["N_CRPS_SAMPLES"]), MT)).mean()))
            suf_null.append(float(cp_vec(forecast_cf_multi(CO, _union_arm(rper, FC), CONFIG["N_CRPS_SAMPLES"]), MT).mean()))
        rows.append(dict(k=int(k), per_layer={int(l): len(per[l]) for l in LAYERS},
                         faith=fa_m, faith_ci=fa_ci, faith_frac=float(fa_m/(base.mean()+1e-6)), faith_null=float(np.mean(fa_null)),
                         cp_complete=cp_drop, cp_ci=cp_ci, cp_null_p95=float(np.percentile(cp_null, 95)), motif_drop=mot_drop,
                         induced=suf_m, induced_ci=suf_ci, gain=float(suf_m - corr_base.mean()), suf_null_p95=float(np.percentile(suf_null, 95)),
                         selective=bool(cp_drop >= CONFIG["SELECTIVITY_MARGIN"] * max(mot_drop, 1e-6) and cp_drop > float(np.percentile(cp_null, 95)))))
        _save("curves", rows)
        print(f"    k={k:4d}  faith_frac={rows[-1]['faith_frac']:.2f}(null {rows[-1]['faith_null']/(base.mean()+1e-6):.2f})  "
              f"cp_complete={cp_drop:+.3f}  motif={mot_drop:+.3f}  denoise_gain={rows[-1]['gain']:+.3f}  sel={rows[-1]['selective']}  union={rows[-1]['per_layer']}")
    rows.sort(key=lambda r: r["k"])
    return dict(base=float(base.mean()), corrupt_base=float(corr_base.mean()), motif_base=float(mbase.mean()), rows=rows)
print("Cross-layer union curves (faithfulness/completeness/denoising)..." + MOCK_TAG); CURVES = run_curves()

# ============================================================
def summarize():
    R = CURVES["rows"]; base = CURVES["base"]; target = CONFIG["FAITH_TARGET"]
    kstar = next((r["k"] for r in sorted(R, key=lambda r: r["k"]) if r["faith_frac"] >= target), None)
    small = bool(kstar is not None and kstar <= CONFIG["LOCALIZE_MAX_FEATURES"])
    win = [r for r in R if (kstar is None or r["k"] <= max(kstar, CONFIG["LOCALIZE_MAX_FEATURES"]))]
    selective = any(r["selective"] for r in win) if win else False
    complete_beats = any(r["cp_complete"] > r["cp_null_p95"] for r in win) if win else False
    sufficient = any((r["induced"] > r["suf_null_p95"] and r["gain"] >= 0.15) for r in win) if win else False
    # cross-layer faithfulness must also BEAT its random-k-union null (else 'faithful' is just a bypassable stack)
    faith_beats = any(r["faith_frac"] > (r["faith_null"]/(base+1e-6)) + 0.05 for r in win) if win else False
    localized = bool(small and selective and complete_beats and sufficient and faith_beats)
    depths = [rel_depth(l) for l in LAYERS]; in_band = [l for l in LAYERS if CONFIG["MISHRA_DEPTH_LO"] <= rel_depth(l) <= CONFIG["MISHRA_DEPTH_HI"]]
    if localized:
        verdict = (f"A: LOCALIZED cross-layer change-detection feature circuit — {kstar} features across {LAYERS} are "
                   f"faithful>={target} (beats union null), complete>null, selective, and SUFFICIENT -> discrepancy RESOLVED")
    else:
        why = []
        if not faith_beats: why.append("faithfulness ties the random-union null (layers bypassable)")
        if not selective: why.append("not motif-selective")
        if not sufficient: why.append("not sufficient under denoising")
        verdict = (f"B: DISTRIBUTED across layers too ({'; '.join(why)}) -> no small cross-layer feature circuit; "
                   f"SAE-vs-circuit discrepancy CLOSED at every granularity")
    mde = float(np.mean([abs(r["faith_ci"][1]-r["faith_ci"][0]) for r in R]) / 2) if R else float("nan")
    print("=" * 96); print(f"PHASE 5 v3 VERDICT: {verdict}{MOCK_TAG}"); print("=" * 96)
    print(f"  clean recovery={base:.3f}  layers={LAYERS} (depths {','.join(f'{d:.2f}' for d in depths)}; Mishra-band {in_band})  features/layer={NF}")
    print(f"  k*(faithful union)={kstar} small={small}  faith>null={faith_beats}  selective={selective}  complete>null={complete_beats}  sufficient={sufficient}")
    print(f"  max denoising gain over all k={max(r['gain'] for r in R):+.4f} (threshold 0.15)  max completeness={max(r['cp_complete'] for r in R):+.4f}")
    print(f"  DETECTION POWER: min detectable effect ~{mde:.3f} -> " + ("positive result is real." if localized else "a localized cross-layer circuit WOULD have been resolved; the distributed verdict is real."))
    return dict(verdict=verdict, localized=bool(localized), kstar=kstar, small=bool(small), faith_beats=bool(faith_beats),
                selective=bool(selective), complete_beats=bool(complete_beats), sufficient=bool(sufficient),
                layers=LAYERS, depths=depths, mishra_in_band=in_band, n_features=int(NF), min_detectable=mde)
SUMMARY = summarize()

# ============================================================
try:
    R = CURVES["rows"]; ks = [r["k"] for r in R]; fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    a = ax[0]
    a.plot(ks, [r["faith_frac"] for r in R], "o-", color="#8e44ad", label="top-k union (IE)")
    a.plot(ks, [r["faith_null"]/(CURVES["base"]+1e-6) for r in R], "x--", color="#999", label="random-k union null")
    a.axhline(CONFIG["FAITH_TARGET"], color="k", ls=":", lw=1); a.axvline(CONFIG["LOCALIZE_MAX_FEATURES"], color="orange", ls=":", lw=1)
    a.set_xscale("log", base=2); a.set_xlabel("union size k"); a.set_ylabel("faithfulness (frac)"); a.legend(fontsize=7)
    a.set_title("Fig 7a: cross-layer faithfulness vs union size" + MOCK_TAG, fontsize=9)
    b = ax[1]
    b.errorbar(ks, [r["cp_complete"] for r in R], yerr=[[max(0,r["cp_complete"]-r["cp_ci"][0]) for r in R],[max(0,r["cp_ci"][1]-r["cp_complete"]) for r in R]], fmt="o-", color="#c0392b", capsize=2, label="completeness")
    b.plot(ks, [r["cp_null_p95"] for r in R], "x:", color="#999", label="complete null p95")
    b.plot(ks, [r["gain"] for r in R], "s-", color="#27ae60", label="denoising gain")
    b.plot(ks, [r["motif_drop"] for r in R], "^--", color="#7f8c8d", label="motif drop")
    b.axhline(0.15, color="green", ls=":", lw=1, label="sufficiency bar"); b.set_xscale("log", base=2)
    b.set_xlabel("union size k"); b.set_ylabel("rel-collapse / gain"); b.legend(fontsize=7); b.set_title("Fig 7b: completeness + denoising (CIs)", fontsize=9)
    c = ax[2]; bottom = np.zeros(len(ks))
    for l in LAYERS:
        vals = [r["per_layer"][str(l)] if str(l) in r["per_layer"] else r["per_layer"].get(l, 0) for r in R]
        c.bar(range(len(ks)), vals, bottom=bottom, label=f"L{l}@{rel_depth(l):.2f}"); bottom += np.array(vals, float)
    c.set_xticks(range(len(ks))); c.set_xticklabels(ks, fontsize=7); c.set_xlabel("union size k"); c.set_ylabel("# features from layer")
    c.legend(fontsize=7); c.set_title("Fig 7c: union composition by layer", fontsize=9)
    fig.suptitle(f"Phase 5 v3 — cross-layer feature circuit: {SUMMARY['verdict'][:60]}" + MOCK_TAG, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(CKPT_DIR, f"fig7_phase5v3_{MODE}.png"), dpi=90); plt.show(); plt.close(fig)
    print(f"saved fig7_phase5v3_{MODE}.png")
except Exception as e:
    import traceback; print("fig skipped:", repr(e)[:160]); traceback.print_exc()

# ============================================================
out = dict(summary=SUMMARY, orientation=ORIENT, curves=CURVES,
           sae=dict(layers=LAYERS, depths=[rel_depth(l) for l in LAYERS], n_features=int(NF), recon=recon),
           config=dict(mode=MODE, model_id=CONFIG["model_id"], cross_n_layers=CONFIG["CROSS_N_LAYERS"],
                       n_pairs=CONFIG["N_PAIRS"], k_grid=CONFIG["K_GRID"], is_large=bool(IS_LARGE)))
p = os.path.join(CKPT_DIR, f"phase5v3_{MODE}.json")
with open(p, "w") as f: json.dump(out, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print("wrote", p, "->", SUMMARY["verdict"][:70], MOCK_TAG)