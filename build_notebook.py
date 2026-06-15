#!/usr/bin/env python3
# Single-source builder for the Chronos Selective Periodic-Induction feasibility notebook.
# Emits:  chronos_circuit_feasibility.ipynb   (the deliverable)
#         _mirror.py                           (concatenated code cells, for local smoke testing)
# Every notebook cell's source lives here exactly once, so testing the mirror tests the notebook.

import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

CELLS = []  # list of ("md"|"code", source)
def md(s):  CELLS.append(("md", s.strip("\n")))
def code(s): CELLS.append(("code", s.strip("\n")))

# ============================================================================
md(r"""
# Chronos Selective Periodic-Induction Circuit — Feasibility Notebook (Phases 0–3)

**What this is.** A single, self-contained feasibility pass of a circuit-level study of the original
**Chronos-T5** time-series foundation model (encoder–decoder, autoregressive, per-time-step tokenization
into a 4,096-bin vocabulary, T5 relative position bias). It tests whether Chronos learns a
**selective / multi-lag induction head** in the sense of d'Angelo, Croce & Flammarion, *Selective Induction
Heads* (ICLR 2025, arXiv:2509.08184): a circuit that infers a series' **own period P from content**, attends
to the same-phase positions one or more periods back (t−P, t−2P, …), and **copies** the bin value found there
into the forecast.

**Two hard gates decide the project**
- **Phase 1 gate** — does any head *track the correct lag as P varies across series* (the selective signature)?
- **Phase 3 gate** — is that head *causal and selective* (ablating it degrades periodic forecasting but not
  trend-only or changepoint-only)?

It is **win-either-way**: a clean null is informative and routes to the change-detection circuit
(Phase 1′, out of scope here — we only report the pivot). The headline causal metric is **selective ΔCRPS**.

**The central confound** is the T5 relative-position bias: a head can attend at a fixed offset P for purely
positional reasons. The defense is the **selective / multi-lag signature** — a genuine selective-induction
head must *track* the content-determined lag as P changes; a positional artifact attends at a constant offset
and cannot track. Identification therefore rests on **lag-tracking across P**, never a raw fixed-P lag score,
and every threshold is calibrated against a null distribution that includes a **fixed-offset reference**.

**Run order.** `CONFIG["MODE"] = "mock_cpu"` first (a few minutes on CPU, chronos-t5-tiny) — this is a
**pipeline smoke test only; its numbers are NOT scientifically interpretable**. Once it executes clean, flip
to `"pilot_t4"` and run on a free Colab T4 with chronos-t5-Base for the real feasibility verdict.

**Authoritative design.** The spec (`chronos_circuit_spec_v2.md`) and plan (`chronos_circuit_plan_v2.md`)
in `github.com/grewalsk/circuitTSFM` are the source of truth; Section 0 clones them into the session.

> Scope: Phases 0–3 only. Phases 4–7 (ETT, Large final numbers, cross-size universality, submission) come
> only **after** this feasibility gate passes.
""")

# ============================================================================
md(r"""
## Section 0 — Setup & repo context

**Maps to:** Phase 0 harness setup. **Pass condition:** CONFIG resolves, seeds/device/checkpoint dir are set,
and the spec + plan are cloned and their headers visible in-session.

The single `CONFIG` switch below is the *only* thing that changes between the CPU smoke test and the T4
pilot — the code path is identical.
""")

code(r"""
# --- CONFIG: the single switch (mock_cpu smoke test  ->  pilot_t4 real verdict) -------------------
CONFIG = {
    "MODE": "mock_cpu",            # flip to "pilot_t4" for the real feasibility run
    "null_percentile": 99,         # pre-registered threshold against the null distribution
    "selectivity_ratio_min": 5,    # periodic DeltaCRPS must exceed this x non-periodic DeltaCRPS
    "slope_tol": 0.35,             # pre-registered: |OLS slope of est_lag on P - 1| must be < this
    "copy_window": 8,              # +/- bins counted as a "copy" (soft/windowed copying score)
    "sample_seed": 1234,           # fixed seed for CRPS sampling (common random numbers: clean vs ablated)
    "mock_cpu": {
        "device": "cpu",
        "model_id": "amazon/chronos-t5-tiny",
        "periods": [6, 8],
        "n_seeds": 2,
        "n_series_per_condition": 4,
        "context_length": 64,
        "prediction_length": 16,
        "n_bootstrap": 10,
        "eap_top_edges": 3,
        "acdc_scope": "tiny",
        "crps_num_samples": 20,
        "crps_max_series": 999,        # logged cap on series/condition for Phase-3 CRPS ablation (no cap in mock)
        "staged_max": 999,             # logged cap on staged-structure exact-patch scan
    },
    "pilot_t4": {
        "device": "cuda",
        "model_id": "amazon/chronos-t5-base",
        "periods": [8, 12, 16, 24],
        "n_seeds": 3,
        "n_series_per_condition": 32,
        "context_length": 256,
        "prediction_length": 64,
        "n_bootstrap": 1000,
        "eap_top_edges": 75,
        "acdc_scope": "eap_region",
        "crps_num_samples": 100,
        "crps_max_series": 48,         # logged cap: fits a free-T4 budget (base generate is the dominant cost)
        "staged_max": 150,             # logged cap on the staged-structure exact-patch scan
    },
}

import os, sys, random, json, subprocess, textwrap
# MODE may be overridden by an env var for automated testing; default is the CONFIG value.
MODE = os.environ.get("CHRONOS_CIRCUIT_MODE", CONFIG["MODE"])
assert MODE in ("mock_cpu", "pilot_t4"), MODE
CFG = dict(CONFIG[MODE])                 # the active sub-config
CFG["null_percentile"]     = CONFIG["null_percentile"]
CFG["selectivity_ratio_min"] = CONFIG["selectivity_ratio_min"]
CFG["slope_tol"]           = CONFIG["slope_tol"]
CFG["copy_window"]         = CONFIG["copy_window"]
CFG["sample_seed"]         = CONFIG["sample_seed"]
IS_MOCK = (MODE == "mock_cpu")
# In mock mode we deliberately FORCE the downstream phases to run even if the (uninterpretable) Phase 1
# verdict is not GREEN, so the whole pipeline is exercised as a smoke test. In pilot mode we honor the gate.
FORCE_DOWNSTREAM = IS_MOCK
# Re-runnability (Hard Req. 9): each phase checkpoints scalars to disk and, on a fresh kernel, LOADS the
# checkpoint instead of recomputing — so a Colab disconnect never loses a completed phase. Set the env var
# CHRONOS_CIRCUIT_FORCE=1 (or delete the ckpt files) to force recomputation.
FORCE_RECOMPUTE = os.environ.get("CHRONOS_CIRCUIT_FORCE", "0") == "1"
def ckpt_path(name): return os.path.join(CKPT_DIR, f"{name}_{MODE}.json")
def load_ckpt(name):
    p = ckpt_path(name)
    if os.path.isfile(p) and not FORCE_RECOMPUTE:
        with open(p) as f:
            print(f"[ckpt] loaded {name} from {p} (set CHRONOS_CIRCUIT_FORCE=1 to recompute)")
            return json.load(f)
    return None

MOCK_TAG = "  [MOCK_CPU SMOKE TEST — NOT SCIENTIFICALLY INTERPRETABLE]" if IS_MOCK else ""

# --- determinism + fixed/versioned mean-scaling regime (Hard Req. 8) ----------------------------
SEED = 0
random.seed(SEED)
try:
    import numpy as _np; _np.random.seed(SEED)
except Exception:
    pass
# The scaling regime is Chronos' pretrained MeanScaleUniformBins (per-series mean-abs scaling) in float32.
# We stamp it into every checkpoint so results are reproducible and the regime is explicit, not implicit.
SCALING_REGIME = {"tokenizer": "MeanScaleUniformBins", "scale": "mean_abs_per_series", "dtype": "float32"}

# --- checkpoint dir + repo dir (Colab-aware) ----------------------------------------------------
ON_COLAB = os.path.isdir("/content")
CKPT_DIR = "/content/ckpt" if ON_COLAB else os.path.abspath("./ckpt_chronos_circuit")
REPO_DIR = "/content/circuitTSFM" if ON_COLAB else os.path.abspath("./circuitTSFM")
os.makedirs(CKPT_DIR, exist_ok=True)

print(f"MODE = {MODE}{MOCK_TAG}")
print(f"model_id = {CFG['model_id']}   device(req) = {CFG['device']}")
print(f"periods = {CFG['periods']}  n_seeds = {CFG['n_seeds']}  "
      f"n_series/cond = {CFG['n_series_per_condition']}  ctx = {CFG['context_length']}  pred = {CFG['prediction_length']}")
print(f"checkpoints -> {CKPT_DIR}")

# --- clone the authoritative spec + plan into the session ---------------------------------------
def _clone_repo():
    if os.path.isdir(os.path.join(REPO_DIR, ".git")) or os.path.isfile(
            os.path.join(REPO_DIR, "chronos_circuit_spec_v2.md")):
        print("repo already present:", REPO_DIR); return
    try:
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/grewalsk/circuitTSFM", REPO_DIR],
                       check=True, capture_output=True, text=True, timeout=120)
        print("cloned circuitTSFM ->", REPO_DIR)
    except Exception as e:
        print("WARN: could not clone repo (offline?). Design docs unavailable in-session.", repr(e)[:200])
_clone_repo()

for fn in ("chronos_circuit_spec_v2.md", "chronos_circuit_plan_v2.md"):
    p = os.path.join(REPO_DIR, fn)
    if os.path.isfile(p):
        with open(p) as f:
            head = "".join(f.readlines()[:14])
        print("\n" + "=" * 78 + f"\n{fn}  (header)\n" + "=" * 78)
        print(textwrap.shorten(head, width=1600, placeholder=" ..."))
    else:
        print(f"(missing {fn})")
""")

# ============================================================================
md(r"""
## Section 1 — Install & imports

**Maps to:** Phase 0 tooling. **Pass condition:** all libraries import and versions print.

**Tooling note (read this).** The spec calls for **nnsight primary, raw HF forward hooks as fallback**, and
explicitly says: *if anything is uncertain, implement the intervention with plain HF forward hooks.* nnsight's
tracing API has changed substantially across releases and cannot be smoke-tested on every Colab image. We
therefore make **validated HF forward hooks the intervention backend for every number in this notebook**
(they are stable torch primitives and are validated in the Section 3 plumbing gate), and we *probe* nnsight
so it is available for interactive exploration. This is the spec's sanctioned fallback and prioritizes
correctness over a fragile dependency. TransformerLens is **not** used (it is decoder-only; Chronos-T5 is
encoder–decoder).
""")

code(r"""
def _ensure(pkg, import_name=None, pip_spec=None):
    import importlib
    name = import_name or pkg
    if os.environ.get("CHRONOS_CIRCUIT_SKIP_INSTALL") == "1":
        try: importlib.import_module(name); return True
        except Exception: print(f"SKIP_INSTALL set; {name} not importable"); return False
    try:
        importlib.import_module(name); return True
    except Exception:
        spec = pip_spec or pkg
        print(f"installing {spec} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], check=False)
        try: importlib.import_module(name); return True
        except Exception as e: print(f"WARN: {name} still not importable: {repr(e)[:160]}"); return False

_ensure("chronos-forecasting", "chronos")
_ensure("nnsight", "nnsight")          # optional / interactive only
for p in ("scipy", "matplotlib"):
    _ensure(p, p)

import importlib.metadata as _md
import torch, numpy as np
torch.manual_seed(SEED)
try: torch.use_deterministic_algorithms(True, warn_only=True)   # surface any nondeterministic kernel
except Exception: pass
import matplotlib
if not ON_COLAB:
    matplotlib.use("Agg")              # headless-safe for local/nbconvert smoke testing
import matplotlib.pyplot as plt

def _ver(p):
    try: return _md.version(p)
    except Exception: return "MISSING"
print("versions:")
for p in ["torch", "transformers", "chronos-forecasting", "accelerate", "scipy", "numpy", "matplotlib", "nnsight"]:
    print(f"  {p:22s} {_ver(p)}")

HAVE_NNSIGHT = False
try:
    import nnsight
    HAVE_NNSIGHT = True
    print(f"nnsight available (v{_ver('nnsight')}) — used for interactive exploration only; "
          f"all scored interventions use validated HF forward hooks.")
except Exception:
    print("nnsight not importable — proceeding with validated HF forward hooks (spec-sanctioned fallback).")
""")

# ============================================================================
md(r"""
## Section 2 — Load Chronos & validate tokenization (Phase 0a)

**Maps to:** Phase 0a. **Pass condition:** the **original Chronos-T5** loads, one autoregressive forecast
runs, and the **per-step tokenization check passes** — token count equals series length up to the documented
special-token offset, so *lag-in-tokens = lag-in-time*. **No lag index is computed anywhere before this passes.**

We use `ChronosPipeline` (the original T5, autoregressive, per-step tokenization) — **never** Chronos-Bolt or
Chronos-2, which patch and/or use a different architecture for which the decoder-cross-attention copy
hypothesis has no referent.
""")

code(r"""
from chronos import ChronosPipeline
from chronos.chronos import ChronosModel  # noqa: used for the isinstance guard below

DEVICE = "cpu"
if CFG["device"] == "cuda":
    if torch.cuda.is_available(): DEVICE = "cuda"
    else: print("WARN: device 'cuda' requested but unavailable; falling back to CPU.")
DTYPE = torch.float32  # float32 everywhere for deterministic CPU/T4 hook arithmetic

pipe = ChronosPipeline.from_pretrained(CFG["model_id"], device_map=DEVICE, torch_dtype=DTYPE)

# Guard against accidentally loading a non-original Chronos (Bolt / Chronos-2).
assert isinstance(pipe.model, ChronosModel), f"Expected original ChronosModel, got {type(pipe.model)}"
assert pipe.model.config.model_type == "seq2seq", \
    f"Expected the original autoregressive seq2seq Chronos-T5, got model_type={pipe.model.config.model_type}"

inner = pipe.inner_model                # HF T5ForConditionalGeneration (the thing we hook)
tok   = pipe.tokenizer                  # MeanScaleUniformBins
inner.eval()
for p in inner.parameters():
    p.requires_grad_(True)              # frozen model, but grads must flow for the EAP proxy
# Defensive: force eager attention so output_attentions returns real weights.
try: inner.config._attn_implementation = "eager"
except Exception: pass

cc = pipe.model.config                  # ChronosConfig
ic = inner.config                       # T5Config
N_LAYERS, N_HEADS, D_KV, D_MODEL = ic.num_layers, ic.num_heads, ic.d_kv, ic.d_model
INNER_DIM = N_HEADS * D_KV
N_TOKENS, N_SPECIAL = cc.n_tokens, cc.n_special_tokens
PAD_ID, EOS_ID = cc.pad_token_id, cc.eos_token_id

print(f"loaded {CFG['model_id']} on {DEVICE}  |  T5: layers={N_LAYERS} heads={N_HEADS} "
      f"d_kv={D_KV} d_model={D_MODEL} inner_dim={INNER_DIM}")
print(f"vocab={ic.vocab_size}  n_tokens={N_TOKENS}  n_special={N_SPECIAL}  pad={PAD_ID} eos={EOS_ID}  "
      f"use_eos={cc.use_eos_token}")
print("--- architecture (top-level) ---")
print(inner)

# ---- one toy autoregressive forecast (confirms generation works) -------------------------------
_toy = torch.sin(torch.linspace(0, 8 * np.pi, 96))
_fc = pipe.predict(_toy, prediction_length=12, num_samples=8)
print("toy forecast tensor:", tuple(_fc.shape), "(B, num_samples, H) — generation OK")

# ---- PER-STEP TOKENIZATION CHECK (must pass before any lag index) -------------------------------
def tokenization_check():
    L = 24
    s = (torch.sin(torch.arange(L, dtype=DTYPE) * 0.6) + 3.0).unsqueeze(0)
    ids, mask, scale = tok.context_input_transform(s)
    ids0 = ids[0]
    # exactly one trailing EOS appended (use_eos_token & seq2seq); everything else is content.
    n_content = int((ids0 != EOS_ID).sum())
    assert ids0[-1].item() == EOS_ID,           "expected trailing EOS token"
    assert n_content == L,                       f"content tokens {n_content} != series length {L}"
    assert int(mask.sum()) == L + 1,             "attention mask should cover content + EOS"
    # special-token offset: a content value maps to token id = bucketize(value/scale)+N_SPECIAL,
    # so content token ids live in [N_SPECIAL, N_TOKENS-1] and bin index = token_id - N_SPECIAL.
    content = ids0[:-1]
    assert int(content.min()) >= N_SPECIAL,      "content token id below special-token offset"
    assert int(content.max()) <= N_TOKENS - 1,   "content token id above vocab"
    # token-index -> timestep map: content position i is timestep i (EOS is the only extra slot).
    # => lag of k tokens between two content positions == lag of k timesteps. Verified.
    # NB: content token id = bucketize(value/scale) + N_SPECIAL; the *center* index used by output_transform
    # to decode a token back to a value is token_id - N_SPECIAL - 1. The load-bearing fact here is only the
    # token-index<->timestep alignment (lag(tokens)==lag(time)); Phase 2 keeps source token id and lm_head
    # logits on the same token-id axis, so it never needs the center index.
    print(f"PER-STEP TOKENIZATION: PASS  | content_tokens={n_content}==L  EOS_offset=+1 at tail  "
          f"token_id = bucket + {N_SPECIAL}  | lag(tokens) == lag(time)")
    return True

TOKENIZATION_OK = tokenization_check()
""")

# ============================================================================
md(r"""
## Section 3 — Hook validation (Phase 0b — hard gate inside the notebook)

**Maps to:** Phase 0b. **Pass condition: `PLUMBING: PASS`** — we do not proceed otherwise. Cross-attention
path-patching is more custom than the decoder-only case; a flaky hook would silently contaminate every
downstream number, so we validate the plumbing on toy edits *before trusting any score*.

This section also defines the harness reused by every later phase:
- **Site discovery by walking the module tree** (no hardcoded names): encoder self-attn, decoder self-attn,
  decoder cross-attn, each attention output projection `.o` (the per-head intervention point), `lm_head`,
  and the T5 `relative_attention_bias`.
- **`OProjHooks`** — one forward-pre-hook per `.o` module; the per-head concatenated input is reshaped to
  `(B, S, n_heads, d_kv)` and can be *captured*, *mean-ablated*, *patched*, or *grad-captured*.
- **Encoder-position patcher** for cross-attention key/value provenance.

Four checks: (1) hooks fire with correct shapes at all sites; (2) **mean-ablation** positive control —
ablating the most-important head moves CRPS a lot, the least-important ~0; (3) **cross-attention position
patch** — patching the same-phase encoder source moves the forecast more than an off-phase source (the
patch is wired to the right key positions); (4) **gradient-flow** — the categorical-NLL proxy yields finite,
correctly-signed gradients on an internal activation.
""")

code(r"""
# ---------- site discovery --------------------------------------------------------------------
# We reach the attention modules via the standard HF-T5 sub-module layout (block[i].layer[0].SelfAttention,
# decoder block[i].layer[1].EncDecAttention, projection '.o'); we do NOT hardcode the full dotted *paths*
# (we walk enc/dec .block), and every assumption is then asserted below (.o in_features == n_heads*d_kv,
# len == num_layers, lm_head out_features, relative_attention_bias located by name-walk). If a future
# transformers renames these submodules, the asserts fail loudly rather than scoring silently-wrong numbers.
def discover_sites(inner):
    sites = {"enc_self": [], "dec_self": [], "cross": []}
    enc = inner.get_encoder(); dec = inner.get_decoder()
    for i, blk in enumerate(enc.block):
        attn = blk.layer[0].SelfAttention
        sites["enc_self"].append({"attn": attn, "o": attn.o, "block": blk})
    for i, blk in enumerate(dec.block):
        sattn = blk.layer[0].SelfAttention
        cattn = blk.layer[1].EncDecAttention
        sites["dec_self"].append({"attn": sattn, "o": sattn.o, "block": blk})
        sites["cross"].append({"attn": cattn, "o": cattn.o, "block": blk})
    # sanity: shapes & the relative_attention_bias location
    for s in sites:
        assert len(sites[s]) == N_LAYERS, (s, len(sites[s]))
        assert sites[s][0]["o"].in_features == INNER_DIM
    rab = [n for n, _ in inner.named_modules() if n.endswith("relative_attention_bias")]
    lm_head = inner.get_output_embeddings()
    assert lm_head.out_features == ic.vocab_size
    print("SITES discovered:", {k: len(v) for k, v in sites.items()},
          "| per-head point = '.o' input reshaped (B,S,%d,%d)" % (N_HEADS, D_KV))
    print("relative_attention_bias at:", rab)
    print("lm_head:", lm_head, "| tied:", lm_head.weight.data_ptr() == inner.shared.weight.data_ptr())
    return sites, lm_head

SITES, LM_HEAD = discover_sites(inner)
ATTN_FIELD = {"enc_self": "encoder_attentions", "dec_self": "decoder_attentions", "cross": "cross_attentions"}

# ---------- tokenization helpers (mirror Chronos exactly; see Section 2) -------------------------
def tokenize_context(series):
    if not torch.is_tensor(series): series = torch.tensor(series, dtype=DTYPE)
    ids, mask, scale = tok.context_input_transform(series.unsqueeze(0).to(DTYPE))
    return ids.to(DEVICE), mask.to(DEVICE), scale.to(DEVICE)

def tokenize_continuation(cont, scale):
    if not torch.is_tensor(cont): cont = torch.tensor(cont, dtype=DTYPE)
    # tok._input_transform mirrors label tokenization WITHOUT the length==prediction_length assertion.
    label_ids, _, _ = tok._input_transform(cont.unsqueeze(0).to(DTYPE), scale=scale.cpu())
    return label_ids.to(DEVICE)

def teacher_forced(series, cont):
    ids, mask, scale = tokenize_context(series)
    label_ids = tokenize_continuation(cont, scale)
    dst = ic.decoder_start_token_id
    dec_in = torch.cat([torch.full((1, 1), dst, dtype=torch.long, device=DEVICE), label_ids[:, :-1]], dim=1)
    return dict(ids=ids, mask=mask, scale=scale, label_ids=label_ids, dec_in=dec_in)

# ---------- the per-head .o-input hook manager --------------------------------------------------
class OProjHooks:
    # One forward_pre_hook per '.o'. Behavior is driven by instance state so a single registration
    # serves capture / mean-ablate / patch / grad-capture for every phase.
    def __init__(self, sites):
        self.sites = sites
        self.handles = []
        self.means = {}                    # (site,li) -> (n_heads, d_kv) ; persists across reset()
        self.reset()
        for s in sites:
            for li, d in enumerate(sites[s]):
                self.handles.append(d["o"].register_forward_pre_hook(self._mk(s, li)))
    def reset(self):
        # clears per-call intervention/capture state but PRESERVES precomputed means.
        self.mode = "off"                  # "off" | "capture_mean" | "capture_act" | "grad" | "intervene"
        self._sum = {}; self._cnt = {}     # capture_mean accumulators
        self.acts = {}                     # (site,li) -> last input tensor (capture_act / grad)
        self.ablate = set()                # {(site,li,h)} -> replace with mean
        self.patch  = {}                   # (site,li,h) -> tensor (B,S,d_kv) to write in
    def _mk(self, site, li):
        def hook(mod, args):
            x = args[0]
            B, S, _ = x.shape
            xh = x.view(B, S, N_HEADS, D_KV)
            if self.mode == "capture_mean":
                s = xh.detach().sum(dim=(0, 1))             # token-weighted: sum over (B,S)
                self._sum[(site, li)] = self._sum.get((site, li), 0) + s
                self._cnt[(site, li)] = self._cnt.get((site, li), 0) + (B * S)
                return None
            if self.mode in ("capture_act", "grad"):
                if self.mode == "grad":
                    x.retain_grad()
                self.acts[(site, li)] = x
                return None
            if self.mode == "intervene":
                xh = xh.clone()
                for h in range(N_HEADS):
                    if (site, li, h) in self.ablate:
                        xh[:, :, h, :] = self.means[(site, li)][h].to(x.dtype)
                    if (site, li, h) in self.patch:
                        xh[:, :, h, :] = self.patch[(site, li, h)].to(x.dtype)
                return (xh.view(B, S, INNER_DIM),)
            return None
        return hook
    def finalize_means(self):
        for k in self._sum: self.means[k] = (self._sum[k] / self._cnt[k]).detach()
    def remove(self):
        for h in self.handles: h.remove()
        self.handles = []

HOOKS = OProjHooks(SITES)

# ---------- encoder-position patcher (cross-attn key/value provenance) ---------------------------
class EncoderPatcher:
    # Replace encoder last_hidden_state at given positions with stored vectors. Cross-attn K,V read
    # from encoder hidden states, so this controls exactly what each cross-attn KEY position carries.
    def __init__(self, inner):
        self.enc = inner.get_encoder(); self.handle = None
        self.pos = None; self.vals = None
    def set(self, pos, vals):  # pos: list[int]; vals: (len(pos), d_model)
        self.pos, self.vals = pos, vals
    def __enter__(self):
        def hook(mod, args, out):
            if self.pos is None: return out
            hs = out[0] if isinstance(out, tuple) else out.last_hidden_state
            hs = hs.clone()
            for j, p in enumerate(self.pos):
                hs[:, p, :] = self.vals[j].to(device=hs.device, dtype=hs.dtype)   # device-safe (T4/cuda)
            if isinstance(out, tuple): return (hs,) + out[1:]
            out.last_hidden_state = hs; return out
        self.handle = self.enc.register_forward_hook(hook); return self
    def __exit__(self, *a):
        if self.handle: self.handle.remove()
        self.pos = self.vals = None

ENC_PATCH = EncoderPatcher(inner)

# ---------- forward / metric helpers ------------------------------------------------------------
def maybe_empty_cache():
    if DEVICE == "cuda":
        try: torch.cuda.empty_cache()
        except Exception: pass

def forward_logits(tf, output_attentions=False):
    return inner(input_ids=tf["ids"], attention_mask=tf["mask"], decoder_input_ids=tf["dec_in"],
                 output_attentions=output_attentions, use_cache=False)

def nll_continuation(series, cont):
    # categorical NLL of the realized continuation — the differentiable EAP proxy. No sampling.
    tf = teacher_forced(series, cont)
    out = forward_logits(tf)
    logits = out.logits.reshape(-1, out.logits.shape[-1])
    return torch.nn.functional.cross_entropy(logits, tf["label_ids"].reshape(-1))

def forecast_samples(series, H, num_samples):
    # gradient-free sampled forecast paths in real units; seeded for common random numbers.
    torch.manual_seed(CFG["sample_seed"])
    fc = pipe.predict(series if torch.is_tensor(series) else torch.tensor(series, dtype=DTYPE),
                      prediction_length=H, num_samples=num_samples)
    return fc[0].detach().cpu().numpy()    # (num_samples, H)

def crps(samples, y):
    # sample-based CRPS (energy form): E|X-y| - 0.5 E|X-X'|, averaged over horizon.
    samples = np.asarray(samples); y = np.asarray(y)
    out = 0.0
    for h in range(samples.shape[1]):
        s = samples[:, h]; yt = y[h]
        t1 = np.mean(np.abs(s - yt))
        t2 = 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))
        out += (t1 - t2)
    return float(out / samples.shape[1])

# ---------- precompute per-head means (for mean-ablation; never zero-ablation) -------------------
def compute_means(ref_stimuli):
    HOOKS.reset(); HOOKS.mode = "capture_mean"
    with torch.no_grad():
        for st in ref_stimuli:
            forward_logits(teacher_forced(st["series"], st["cont"]))
    HOOKS.finalize_means(); HOOKS.mode = "off"
    print(f"mean activations computed over {len(ref_stimuli)} reference stimuli "
          f"({len(HOOKS.means)} site-layers).")

def ablate_ctx(heads):           # heads: iterable of (site,li,h)
    HOOKS.reset(); HOOKS.mode = "intervene"; HOOKS.ablate = set(heads); HOOKS.means = HOOKS.means
    return HOOKS
""")

code(r"""
# ================= PLUMBING GATE (Phase 0b) — validate before trusting any number ================
def plumbing_gate():
    checks = {}
    H = min(CFG["prediction_length"], 16)
    P = CFG["periods"][0]
    L = CFG["context_length"]
    t = torch.arange(L + H, dtype=DTYPE)
    series = torch.sin(2 * np.pi * t[:L] / P)
    cont   = torch.sin(2 * np.pi * t[L:] / P)
    # repeated-motif clean series so next value EXACTLY equals value one period back (for the patch test)
    motif  = torch.sin(2 * np.pi * torch.arange(P, dtype=DTYPE) / P)
    rep    = motif.repeat((L + H) // P + 1)
    rseries, rcont = rep[:L], rep[L:L + H]

    # (1) hooks fire with correct shapes at all sites -------------------------------------------
    HOOKS.reset(); HOOKS.mode = "capture_act"
    with torch.no_grad(): out = forward_logits(teacher_forced(series, cont), output_attentions=True)
    HOOKS.mode = "off"
    shape_ok = True
    for s in SITES:
        for li in range(N_LAYERS):
            a = HOOKS.acts.get((s, li))
            if a is None or a.shape[-1] != INNER_DIM: shape_ok = False
    attn_ok = (out.encoder_attentions is not None and out.cross_attentions is not None and
               out.encoder_attentions[0].shape[1] == N_HEADS and
               out.cross_attentions[0].shape[1] == N_HEADS)
    checks["hooks_fire_all_sites"] = bool(shape_ok)
    checks["attentions_returned"]  = bool(attn_ok)
    print(f"  [1] hooks fire at {len(SITES)*N_LAYERS} site-layers, shapes OK = {shape_ok}; "
          f"enc/cross attentions returned = {attn_ok} "
          f"(enc {tuple(out.encoder_attentions[0].shape)}, cross {tuple(out.cross_attentions[0].shape)})")

    # (2) mean-ablation positive control: biggest single-head ablation >> 0, smallest ~ 0 --------
    compute_means([{"series": series, "cont": cont}, {"series": rseries, "cont": rcont}])
    base = crps(forecast_samples(series, H, CFG["crps_num_samples"]), cont.numpy())
    effects = []
    for li in range(N_LAYERS):
        for h in range(N_HEADS):
            HOOKS.reset(); HOOKS.mode = "intervene"; HOOKS.ablate = {("cross", li, h)}
            d = crps(forecast_samples(series, H, CFG["crps_num_samples"]), cont.numpy()) - base
            effects.append(abs(d)); HOOKS.reset()
    emax, emin = max(effects), min(effects)
    checks["ablation_moves_crps"] = bool(emax > 1e-3 and emax > 5 * (emin + 1e-9))
    print(f"  [2] mean-ablation positive control: max|DeltaCRPS|={emax:.4f} (>>0 = hooks bite), "
          f"min|DeltaCRPS|={emin:.4f} (~0 = targeted). base CRPS={base:.4f}")

    # (3) cross-attn position patch: same-phase source moves forecast > off-phase source ---------
    enc = inner.get_encoder()
    with torch.no_grad():
        ids, mask, scale = tokenize_context(rseries)
        hs_clean = enc(input_ids=ids, attention_mask=mask).last_hidden_state[0]   # (L+1, d_model)
    src_same = L - P            # same-phase source for the FIRST forecast step (time L -> L-P)
    src_off  = L - P - max(1, P // 2)
    if src_off < 0: src_off = max(0, L - 2 * P)
    noise = torch.randn(D_MODEL, dtype=DTYPE, device=hs_clean.device) * float(hs_clean.std()) * 3.0  # device-safe
    tf_r = teacher_forced(rseries, rcont)
    def step0_after_patch(pos):
        with ENC_PATCH:
            ENC_PATCH.set([pos], [(hs_clean[pos] + noise).detach()])
            with torch.no_grad(): lg = forward_logits(tf_r).logits[0, 0]
        return lg
    base_lg = forward_logits(tf_r).logits[0, 0].detach()
    d_same = float((step0_after_patch(src_same) - base_lg).abs().sum())
    d_off  = float((step0_after_patch(src_off)  - base_lg).abs().sum())
    checks["crossattn_wired_to_right_key"] = bool(d_same > d_off)
    print(f"  [3] cross-attn key provenance: patch same-phase src(pos {src_same}) -> |Dlogit|={d_same:.3f}  "
          f"vs off-phase src(pos {src_off}) -> |Dlogit|={d_off:.3f}  (same > off = wired correctly)")

    # (4) gradient flow + correct sign on the NLL proxy ------------------------------------------
    HOOKS.reset(); HOOKS.mode = "grad"
    tf = teacher_forced(series, cont)
    out = forward_logits(tf)
    nll = torch.nn.functional.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]),
                                            tf["label_ids"].reshape(-1))
    inner.zero_grad(set_to_none=True); nll.backward()
    key = ("cross", N_LAYERS // 2)
    a = HOOKS.acts[key]; g = a.grad
    grad_ok = (g is not None) and bool(torch.isfinite(g).all()) and float(g.norm()) > 0
    # finite-difference sign check: stepping the activation along -grad should DECREASE NLL.
    eps = 0.5 / (float(g.norm()) + 1e-9)
    HOOKS.reset(); HOOKS.mode = "intervene"
    B, S, _ = a.shape
    new = (a.detach() - eps * g).view(B, S, N_HEADS, D_KV)
    for h in range(N_HEADS): HOOKS.patch[(key[0], key[1], h)] = new[:, :, h, :]
    with torch.no_grad():
        out2 = forward_logits(tf)
        nll2 = torch.nn.functional.cross_entropy(out2.logits.reshape(-1, out2.logits.shape[-1]),
                                                 tf["label_ids"].reshape(-1))
    HOOKS.reset()
    sign_ok = float(nll2) <= float(nll) + 1e-6
    checks["grad_finite_and_signed"] = bool(grad_ok and sign_ok)
    print(f"  [4] NLL-proxy grad: finite={grad_ok}  |g|={float(g.norm()):.4f}  "
          f"NLL {float(nll):.4f} -> {float(nll2):.4f} along -grad (decrease = correct sign): {sign_ok}")

    passed = all(checks.values())
    print("\nPLUMBING: " + ("PASS" if passed else "FAIL") + MOCK_TAG)
    for k, v in checks.items(): print(f"    {'OK ' if v else 'XX '} {k}")
    return passed, checks

PLUMBING_OK, PLUMBING_CHECKS = plumbing_gate()
assert PLUMBING_OK or IS_MOCK, "Plumbing FAILED on a real run — do not proceed; fix hooks first."
if not PLUMBING_OK:
    print("\nWARN: plumbing imperfect but proceeding because this is the mock smoke test.")
""")

# ============================================================================
md(r"""
## Section 4 — Stimulus generators

**Maps to:** Phase 0 generator. **Pass condition:** all conditions generate at the configured periods/seeds
with known ground truth, under the fixed mean-scaling regime.

- **Periodic** — sinusoid + low-harmonic motif with known P and phase, controlled noise.
- **AR(1)** — variance-matched and **genuinely aperiodic** (monotone-decaying autocorrelation): the
  **primary "non-periodic collapse" control** for the lag-tracking gate. A lag-based head keys off the
  lag-P *autocorrelation*, so the control must destroy it.
- **Phase-scrambled** — same marginal & power spectrum, phase randomized. Reported as a spectral-amplitude
  control, **not** the collapse gate: by Wiener–Khinchin a 1-D phase-scramble preserves the autocorrelation,
  so it does **not** remove the lag-P repeat structure (a sharp subtlety — see the AR(1) note).
- **Period-altered** — same generator, **different period**: the Phase-3 *corrupt* input (the spec's
  "period-altered/aperiodic"), which genuinely changes the lag structure for path patching / EAP.
- **Trend-only** and **changepoint-only** — selectivity stimuli with known positions; these must *not* be
  hurt by ablating a genuine periodic head.

Each stimulus is a dict: `series` (context), `cont` (realized continuation), `P`, `phase`, `cond`, `seed`.
""")

code(r"""
def _rng(seed): return np.random.RandomState(seed)

def make_periodic(P, seed, L, H, noise=0.05):
    g = _rng(seed); phase = g.uniform(0, 2 * np.pi)
    t = np.arange(L + H)
    x = (np.sin(2 * np.pi * t / P + phase)
         + 0.35 * np.sin(2 * np.pi * 2 * t / P + phase * 1.3))   # low harmonic -> a richer motif
    x = x + noise * g.randn(L + H)
    return dict(series=torch.tensor(x[:L], dtype=DTYPE), cont=torch.tensor(x[L:], dtype=DTYPE),
                P=P, phase=float(phase), cond="periodic", seed=seed)

def make_scrambled(P, seed, L, H, noise=0.05):
    # phase-scramble a periodic series: identical power spectrum, randomized phases -> no coherent period.
    g = _rng(seed + 777)
    base = make_periodic(P, seed, L, H, noise=0.0)
    x = torch.cat([base["series"], base["cont"]]).numpy()
    Xf = np.fft.rfft(x); mag = np.abs(Xf)
    ph = np.exp(1j * g.uniform(0, 2 * np.pi, size=mag.shape)); ph[0] = 1.0
    xs = np.fft.irfft(mag * ph, n=len(x)) + noise * g.randn(len(x))
    return dict(series=torch.tensor(xs[:L], dtype=DTYPE), cont=torch.tensor(xs[L:], dtype=DTYPE),
                P=P, phase=0.0, cond="scrambled", seed=seed)

def make_ar1(P, seed, L, H, phi=0.8):
    # AR(1): genuinely aperiodic (monotone-decaying autocorrelation, NO lag-P repeat peak). This is the
    # honest "non-periodic" control for a lag-based head — unlike phase-scramble, which (Wiener-Khinchin)
    # preserves the power spectrum and therefore the lag-P autocorrelation of a narrowband signal.
    g = _rng(seed + 13); x = np.zeros(L + H)
    for i in range(1, L + H): x[i] = phi * x[i - 1] + g.randn()
    x = x / (x.std() + 1e-8)   # variance-match to unit-scale periodic
    return dict(series=torch.tensor(x[:L], dtype=DTYPE), cont=torch.tensor(x[L:], dtype=DTYPE),
                P=P, phase=0.0, cond="ar1", seed=seed)

def altered_period(P, periods):
    # a clearly DIFFERENT period from the configured set (for the period-altered corrupt in Phase 3)
    if len(periods) > 1:
        return periods[(periods.index(P) + len(periods) // 2) % len(periods)]
    return max(2, int(round(P * 1.5)))

def make_period_altered(P, seed, L, H):
    # same generator/seed but a different period -> destroys the original lag-P repeat structure while
    # staying on the periodic manifold. The spec's preferred Phase-3 corrupt ("period-altered/aperiodic").
    aP = altered_period(P, CFG["periods"])
    d = make_periodic(aP, seed + 999, L, H); d["cond"] = "period_altered"; d["orig_P"] = P; return d

def make_trend(seed, L, H):
    g = _rng(seed + 31); slope = g.uniform(0.5, 1.5) / (L); off = g.uniform(-1, 1)
    t = np.arange(L + H); x = slope * t + off + 0.05 * g.randn(L + H)
    return dict(series=torch.tensor(x[:L], dtype=DTYPE), cont=torch.tensor(x[L:], dtype=DTYPE),
                P=0, phase=0.0, cond="trend", seed=seed)

def make_changepoint(seed, L, H):
    g = _rng(seed + 57); cp = int(L * 0.6); jump = g.uniform(2, 4) * (1 if g.rand() > 0.5 else -1)
    x = 0.05 * g.randn(L + H); x[cp:] += jump
    return dict(series=torch.tensor(x[:L], dtype=DTYPE), cont=torch.tensor(x[L:], dtype=DTYPE),
                P=0, phase=0.0, cond="changepoint", cp=cp, seed=seed)

def build_stimuli():
    L, H = CFG["context_length"], CFG["prediction_length"]
    periods, nser, nseed = CFG["periods"], CFG["n_series_per_condition"], CFG["n_seeds"]
    S = {"periodic": [], "scrambled": [], "ar1": [], "period_altered": [], "trend": [], "changepoint": []}
    for sd in range(nseed):
        for P in periods:
            for k in range(nser):
                s = sd * 1000 + P * 10 + k
                S["periodic"].append(make_periodic(P, s, L, H))           # index-aligned across these 4 lists
                S["scrambled"].append(make_scrambled(P, s, L, H))
                S["ar1"].append(make_ar1(P, s, L, H))
                S["period_altered"].append(make_period_altered(P, s, L, H))
        for k in range(nser):
            S["trend"].append(make_trend(sd * 1000 + k, L, H))
            S["changepoint"].append(make_changepoint(sd * 1000 + k, L, H))
    print("stimuli:", {k: len(v) for k, v in S.items()}, f"(L={L}, H={H})" + MOCK_TAG)
    return S

STIM = build_stimuli()

# quick visual sanity (saved to ckpt; shown inline on Colab)
try:
    fig, ax = plt.subplots(1, 4, figsize=(15, 2.4))
    for a, (name, key) in zip(ax, [("periodic", "periodic"), ("scrambled", "scrambled"),
                                   ("trend", "trend"), ("changepoint", "changepoint")]):
        st = STIM[key][0]; xfull = torch.cat([st["series"], st["cont"]]).numpy()
        a.plot(xfull); a.axvline(len(st["series"]), color="r", ls=":", lw=1); a.set_title(name)
    fig.suptitle("Stimulus sanity (context | continuation)" + MOCK_TAG); fig.tight_layout()
    fig.savefig(os.path.join(CKPT_DIR, "stimuli.png"), dpi=80); plt.show(); plt.close(fig)
except Exception as e:
    print("plot skipped:", repr(e)[:120])
""")

# ============================================================================
md(r"""
## Section 5 — Phase 1: selective-lag scan (THE make-or-break gate; Fig 2)

**Maps to:** Phase 1 / Fig 2. **Gate output: `GREEN | PIVOT | AMBIGUOUS`.**

For every head × attention type we estimate the **peak-attended lag** per stimulus, then regress it on the
series' true period P across the multi-period set. A **selective** head has **slope ≈ 1** and **high
correlation** on periodic input, and **collapses on phase-scrambled** input.

- **Lag estimate** (per head, per stimulus): for each query, the past key with maximum attention gives
  `lag = query_time − key_time`; we take the attention-weighted median over queries (robust). Cross-attention
  query times are forecast steps `L+j`; key times are encoder/context positions; so a one-period-back copy
  sits at `lag = P`. EOS and out-of-range keys are masked. **Only per-head scalars are kept — raw attention
  tensors are never cached** (Hard Req. 6).
- **Tracking statistic** `T = Pearson r(est_lag, P)`, with the OLS slope reported alongside.
- **Nulls** (Hard Req. 7): **random-position**, **head-shuffled (label permutation)**, and a **fixed-offset
  reference** that attends at a constant offset and *cannot* track. The decision threshold is the
  `null_percentile`-th percentile of the pooled null `T` — never a hardcoded magic number.
- **Candidate** = `T_periodic > thresh` **and** `|slope−1| < slope_tol` **and** `T_scrambled < thresh`
  **and** sign-stable across seeds.

`GREEN` if ≥1 candidate; `PIVOT` if nothing beats the null; `AMBIGUOUS` otherwise.
""")

code(r"""
def site_time_maps(site, Lc, Hh):
    # returns (q_times, k_times, k_valid) for a stimulus with context length Lc, horizon Hh.
    # encoder length in attn = Lc+1 (EOS at index Lc); decoder length = Hh.
    if site == "enc_self":
        q = np.arange(Lc + 1).astype(float); k = q.copy()
        kv = np.ones(Lc + 1, bool); kv[Lc] = False                 # mask EOS
    elif site == "dec_self":
        q = (Lc + np.arange(Hh)).astype(float); k = q.copy(); kv = np.ones(Hh, bool)
    else:  # cross: query=decoder steps, key=encoder positions
        q = (Lc + np.arange(Hh)).astype(float); k = np.arange(Lc + 1).astype(float)
        kv = np.ones(Lc + 1, bool); kv[Lc] = False
    return q, k, kv

def estimate_peak_lag(attn, q_times, k_times, k_valid):
    # attn: (Q,K) numpy. Returns weighted-median past lag, or nan if too few valid queries.
    Q, K = attn.shape; lags = []; wts = []
    for qi in range(Q):
        mask = k_valid & (k_times < q_times[qi] - 1e-6)
        if not mask.any(): continue
        idx = np.where(mask)[0]; row = attn[qi, idx]
        j = int(np.argmax(row)); lags.append(q_times[qi] - k_times[idx[j]]); wts.append(float(row[j]))
    if len(lags) < 2: return np.nan
    lags = np.array(lags); wts = np.array(wts); order = np.argsort(lags)
    lags, wts = lags[order], wts[order]; c = np.cumsum(wts)
    return float(lags[np.searchsorted(c, 0.5 * c[-1])])   # weighted median

def per_head_lag_scalars(stim):
    # ONE forward per stimulus; compute a scalar est_lag per (site,layer,head); drop the attention tensors.
    Lc = len(stim["series"]); Hh = len(stim["cont"])
    HOOKS.reset()
    with torch.no_grad():
        out = forward_logits(teacher_forced(stim["series"], stim["cont"]), output_attentions=True)
    assert out.encoder_attentions is not None and out.cross_attentions is not None and \
           out.decoder_attentions is not None, "output_attentions returned None — force eager attention"
    fields = {"enc_self": out.encoder_attentions, "dec_self": out.decoder_attentions, "cross": out.cross_attentions}
    res = {}
    for site in SITES:
        q, k, kv = site_time_maps(site, Lc, Hh)
        for li in range(N_LAYERS):
            A = fields[site][li][0]   # (H, Q, K)
            for h in range(N_HEADS):
                res[(site, li, h)] = estimate_peak_lag(A[h].float().cpu().numpy(), q, k, kv)
    del out, fields
    return res

def _pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-9 or np.std(y[m]) < 1e-9: return 0.0
    return float(np.corrcoef(x[m], y[m])[0, 1])

def _ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.std(x[m]) < 1e-9: return 0.0
    return float(np.polyfit(x[m], y[m], 1)[0])

def run_phase1():
    cached = load_ckpt("phase1")
    if cached is not None: return cached
    ck = ckpt_path("phase1")
    Pser = [st["P"] for st in STIM["periodic"]]
    seeds = [st["seed"] // 1000 for st in STIM["periodic"]]
    # est_lag per head across periodic / AR(1)-aperiodic / scrambled sets (scalars only; raw attention dropped)
    keys = [(s, li, h) for s in SITES for li in range(N_LAYERS) for h in range(N_HEADS)]
    lag_per = {k: [] for k in keys}; lag_aper = {k: [] for k in keys}; lag_scr = {k: [] for k in keys}
    for st in STIM["periodic"]:
        for k, v in per_head_lag_scalars(st).items(): lag_per[k].append(v)
    for st in STIM["ar1"]:
        for k, v in per_head_lag_scalars(st).items(): lag_aper[k].append(v)
    for st in STIM["scrambled"]:
        for k, v in per_head_lag_scalars(st).items(): lag_scr[k].append(v)
    maybe_empty_cache()

    Pser  = np.array(Pser, float); seeds = np.array(seeds)
    Paper = np.array([st["P"] for st in STIM["ar1"]], float)
    Pscr  = np.array([st["P"] for st in STIM["scrambled"]], float)

    # ---- null distribution of the tracking statistic T (Hard Req. 7) ----------------------------
    rs = _rng(SEED); null_T = []
    lag_range = (1, max(CFG["periods"]) * 2)
    for key in lag_per:
        el = np.array(lag_per[key], float)
        if not np.isfinite(el).any(): continue
        for _ in range(20):                                    # head-shuffled / label-permutation null
            null_T.append(_pearson(rs.permutation(Pser), el))
        for _ in range(10):                                    # random-position null
            null_T.append(_pearson(Pser, rs.uniform(*lag_range, size=len(Pser))))
    # fixed-offset REFERENCE: a head attending at a CONSTANT offset, run through the SAME estimator.
    # Sweeping offsets with small jitter exercises the real estimator tail (not a degenerate point mass at 0),
    # which is the positional-artifact baseline that hardens the threshold (defeats the T5 rel-pos confound).
    for c in range(2, max(CFG["periods"]) * 2):
        for _ in range(3):
            null_T.append(_pearson(Pser, c + rs.normal(0, 0.25, size=len(Pser))))
    null_T = np.array(null_T)
    thresh = float(np.percentile(np.abs(null_T), CFG["null_percentile"]))

    def best_multiple(slope):
        # selective/multi-lag: a head may copy from t-P, t-2P, t-3P -> slope ~ m (m in {1,2,3}).
        ms = [1, 2, 3]; m = min(ms, key=lambda mm: abs(slope - mm)); return m, abs(slope - m)

    # per-head statistics + candidacy --------------------------------------------------------------
    rows = []
    for key in lag_per:
        el = np.array(lag_per[key], float)
        T = _pearson(Pser, el); slope = _ols_slope(Pser, el)
        Taper = _pearson(Paper, np.array(lag_aper[key], float))   # genuine aperiodic collapse control (gate)
        Tscr  = _pearson(Pscr,  np.array(lag_scr[key], float))    # spectral-amplitude control (reported only)
        m, dm = best_multiple(slope)
        seed_signs = []
        for sd in np.unique(seeds):
            mm = seeds == sd; seed_signs.append(np.sign(_pearson(Pser[mm], el[mm])))
        stable = len(set(s for s in seed_signs if s != 0)) <= 1 and T > 0
        # selective signature: tracks the content-determined lag (T high) at an INTEGER MULTIPLE of P
        # (slope ~ m, not forced to 1), collapses on the genuinely-aperiodic control, sign-stable across seeds.
        is_cand = (T > thresh) and (dm < CFG["slope_tol"]) and (Taper < thresh) and stable
        rows.append(dict(site=key[0], layer=key[1], head=key[2], T=float(T), slope=float(slope),
                         lag_multiple=int(m), T_aperiodic=float(Taper), T_scrambled=float(Tscr),
                         stable=bool(stable), candidate=bool(is_cand)))
    rows.sort(key=lambda r: -r["T"])
    cands = [r for r in rows if r["candidate"]]
    best = rows[0]

    if len(cands) >= 1:
        verdict = "GREEN"
    elif best["T"] <= thresh:
        verdict = "PIVOT"
    else:
        verdict = "AMBIGUOUS"

    out = dict(mode=MODE, mock=IS_MOCK, null_thresh=thresh, null_percentile=CFG["null_percentile"],
               slope_tol=CFG["slope_tol"], verdict=verdict, rows=rows, scaling_regime=SCALING_REGIME,
               candidates=cands, best=best, periods=CFG["periods"])
    with open(ck, "w") as f: json.dump(out, f, indent=2)
    return out

PHASE1 = run_phase1()

# ---- report + figures --------------------------------------------------------------------------
print(f"\nPhase 1 null threshold (|T| > {CFG['null_percentile']}th pct of nulls): "
      f"{PHASE1['null_thresh']:.3f}   slope_tol={CFG['slope_tol']}")
print("Top heads by lag-tracking T = corr(est_lag, P)  (mult = nearest integer period-multiple of the slope):")
print(f"  {'site':9s} {'L':>2s} {'H':>2s} {'T':>7s} {'slope':>7s} {'mult':>4s} {'T_aper':>7s} {'T_scr':>7s} {'stbl':>4s} cand")
for r in PHASE1["rows"][:10]:
    print(f"  {r['site']:9s} {r['layer']:2d} {r['head']:2d} {r['T']:7.3f} {r['slope']:7.3f} {r['lag_multiple']:4d} "
          f"{r['T_aperiodic']:7.3f} {r['T_scrambled']:7.3f} {str(r['stable'])[:1]:>4s} {'*' if r['candidate'] else ''}")
print(f"\nPHASE 1 VERDICT: {PHASE1['verdict']}{MOCK_TAG}")
if PHASE1["verdict"] == "PIVOT":
    print("  -> No head tracks a content-varying lag above the nulls. Periodicity is not a selective\n"
          "     attention circuit; this corroborates Mishra & Pandey. RECOMMENDATION: pivot to the\n"
          "     change-detection circuit (Phase 1', out of scope for this notebook).")

try:
    # Fig 2a: lag-tracking heatmap (heads x site), value = T
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    grid = np.full((N_HEADS * N_LAYERS, len(SITES)), np.nan)
    ylabels = [f"L{li}H{h}" for li in range(N_LAYERS) for h in range(N_HEADS)]
    sites_order = list(SITES)
    for r in PHASE1["rows"]:
        ri = r["layer"] * N_HEADS + r["head"]; ci = sites_order.index(r["site"]); grid[ri, ci] = r["T"]
    im = ax[0].imshow(grid, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
    ax[0].set_xticks(range(len(sites_order))); ax[0].set_xticklabels(sites_order, rotation=20)
    ax[0].set_yticks(range(len(ylabels))); ax[0].set_yticklabels(ylabels, fontsize=6)
    ax[0].set_title("Fig 2: lag-tracking T = corr(est_lag, P)" + MOCK_TAG); fig.colorbar(im, ax=ax[0])
    # Fig 2b: periodic vs aperiodic(AR1) collapse for top heads (the gate); scrambled shown for reference
    top = PHASE1["rows"][:min(12, len(PHASE1["rows"]))]
    xs = np.arange(len(top))
    ax[1].bar(xs - 0.27, [r["T"] for r in top], 0.27, label="periodic")
    ax[1].bar(xs,         [r["T_aperiodic"] for r in top], 0.27, label="AR(1) aperiodic (gate)")
    ax[1].bar(xs + 0.27, [r["T_scrambled"] for r in top], 0.27, label="scrambled (ref)")
    ax[1].axhline(PHASE1["null_thresh"], color="k", ls="--", lw=1, label=f"null p{CFG['null_percentile']}")
    ax[1].set_xticks(xs); ax[1].set_xticklabels([f"{r['site'][:4]}\nL{r['layer']}H{r['head']}" for r in top], fontsize=6)
    ax[1].set_title("selectivity: periodic vs aperiodic collapse"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(CKPT_DIR, "phase1_lagtracking.png"), dpi=80); plt.show(); plt.close(fig)
except Exception as e:
    print("phase1 plot skipped:", repr(e)[:120])
""")

# ============================================================================
md(r"""
## Section 6 — Phase 2: copying / OV leg (GREEN only; Fig 3)

**Maps to:** Phase 2 / Fig 3. Runs only when Phase 1 is GREEN — except in **mock mode**, where we force it
on the top-T head purely to smoke-test the code (labeled as such).

**Windowed copying score** via **direct logit attribution** through the head OV circuit and `lm_head`: for a
candidate head we form `W_O_head · W_V_head` (the head's value→output map, T5 has no biases here), apply it to
the hidden state at the **attended lag position**, unembed through `lm_head`, and ask whether the bin `b`
present at that position gets elevated logit mass in a **neighborhood** `[b−w, b+w]` (periodicity is noisy, so
copying is soft/windowed). The score is compared against a null built from non-candidate heads.

Classification: **tracks-lag + copies** ⇒ a *selective periodic-induction head*; **tracks-lag only** ⇒
phase-selection without the copy (a distinct, weaker finding).

The OV→`lm_head` path is only valid for heads that write into the **decoder** residual that `lm_head` reads:
**cross-attn** (value source = the final encoder hidden state — the spec's copy locus) and **decoder self-attn**
(value source = that block's decoder hidden state). For **encoder self-attn** the path is **not applicable**
(its output reaches `lm_head` only via cross-attention), so it is reported as `N/A`, not scored through the
wrong stream. The decoder's final RMS-LN and T5 tied-embedding logit scaling are applied for a calibrated score.
""")

code(r"""
def head_OV(site, li, h):
    attn = SITES[site][li]["attn"]
    Wv = attn.v.weight[h * D_KV:(h + 1) * D_KV, :]        # (d_kv, d_model)  (T5 has no bias here)
    Wo = attn.o.weight[:, h * D_KV:(h + 1) * D_KV]        # (d_model, d_kv)
    return Wv, Wo

def precompute_hidden(stimuli):
    # cache the residual streams the OV map reads, ONCE per stimulus (cross reads encoder_last_hidden_state;
    # dec_self reads the decoder hidden state INPUT to its block = decoder_hidden_states[li]).
    cache = []
    with torch.no_grad():
        for st in stimuli:
            tf = teacher_forced(st["series"], st["cont"])
            out = inner(input_ids=tf["ids"], attention_mask=tf["mask"], decoder_input_ids=tf["dec_in"],
                        output_hidden_states=True, use_cache=False)
            cache.append(dict(enc_last=out.encoder_last_hidden_state[0].detach(),
                              dec_h=[h[0].detach() for h in out.decoder_hidden_states],
                              ids=tf["ids"].detach(), label_ids=tf["label_ids"].detach(),
                              P=max(1, int(st["P"])), Lc=len(st["series"]), H=len(st["cont"])))
    maybe_empty_cache()
    return cache

def _copy_sources(site, li, c):
    # site-appropriate (value_vector, source_bin) pairs the head would copy from. Anchored at the FORECAST
    # reference (forecast step 0 same-phase source = Lc-P), matching the Phase-0 cross-attn provenance test.
    P, Lc, H = c["P"], c["Lc"], c["H"]; out = []
    if site == "cross":                                    # value source = final encoder hidden state
        for k in range(1, 4):
            p = Lc - k * P
            if p < 0: break
            b = int(c["ids"][0, p].item())
            if b >= N_SPECIAL: out.append((c["enc_last"][p], b))
    elif site == "dec_self":                               # value source = decoder hidden state at block li input
        for p in range(0, H):
            if p + P >= H: break                           # need a later query q=p+P that copies from p
            b = int(c["label_ids"][0, p].item())
            if b >= N_SPECIAL: out.append((c["dec_h"][li][p], b))
            if len(out) >= 3: break
    return out                                             # enc_self -> [] (OV->lm_head path not applicable)

def copying_score(site, li, h, cache_list, window):
    # Windowed direct-logit-attribution copying score through the head OV circuit -> decoder final-LN -> lm_head.
    # Returns (None, None) for enc_self: such heads write to the ENCODER residual and never reach lm_head
    # directly, so the OV->lm_head path is not a valid copy measurement for them.
    if site == "enc_self":
        return None, None
    Wv, Wo = head_OV(site, li, h); Wlm = LM_HEAD.weight; dec = inner.get_decoder()
    win_mass, rand_mass = [], []; rs = _rng(SEED + li * 7 + h)
    with torch.no_grad():                                   # frozen weights carry requires_grad for EAP
        for c in cache_list:
            for vvec, b in _copy_sources(site, li, c):
                ov = (vvec @ Wv.T) @ Wo.T                   # head OV map -> (d_model,)
                ov = dec.final_layer_norm(ov.unsqueeze(0)).squeeze(0)   # T5 RMSNorm precedes lm_head
                if ic.tie_word_embeddings: ov = ov * (D_MODEL ** -0.5)  # T5 tied-embedding logit scaling
                logit = (ov @ Wlm.T).float().cpu().numpy()  # (vocab,)
                prob = np.exp(logit - logit.max()); prob /= prob.sum()
                lo, hi = max(0, b - window), min(N_TOKENS, b + window + 1)
                win_mass.append(float(prob[lo:hi].sum()))
                rb = int(rs.randint(N_SPECIAL, N_TOKENS))    # null: window at a random bin
                rl, rh = max(0, rb - window), min(N_TOKENS, rb + window + 1)
                rand_mass.append(float(prob[rl:rh].sum()))
    if not win_mass: return 0.0, 0.0
    return float(np.mean(win_mass)), float(np.mean(rand_mass))

def run_phase2(candidates):
    cached = load_ckpt("phase2")
    if cached is not None: return cached
    ck = ckpt_path("phase2")
    per = STIM["periodic"][:min(8, len(STIM["periodic"]))]
    cache_list = precompute_hidden(per)
    # null from non-candidate DECODER-side heads (the OV->lm_head path is valid only there)
    cand_keys = {(c["site"], c["layer"], c["head"]) for c in candidates}
    rs = _rng(SEED); null = []
    allk = [(s, li, h) for s in ("cross", "dec_self") for li in range(N_LAYERS) for h in range(N_HEADS)
            if (s, li, h) not in cand_keys]
    for s, li, h in [allk[i] for i in rs.choice(len(allk), size=min(12, len(allk)), replace=False)]:
        w, _ = copying_score(s, li, h, cache_list[:3], CFG["copy_window"])
        if w is not None: null.append(w)
    null_thresh = float(np.percentile(null, CFG["null_percentile"])) if null else 0.0
    rows = []
    for c in candidates:
        w, rmass = copying_score(c["site"], c["layer"], c["head"], cache_list, CFG["copy_window"])
        if w is None:
            rows.append(dict(site=c["site"], layer=c["layer"], head=c["head"], T=c.get("T", 0.0),
                             slope=c.get("slope", 0.0), copy_mass=None, rand_mass=None, copies=False,
                             classification="tracks_lag_only (enc_self: OV->lm_head path N/A)"))
        else:
            copies = (w > null_thresh) and (w > rmass)
            rows.append(dict(site=c["site"], layer=c["layer"], head=c["head"], T=c.get("T", 0.0),
                             slope=c.get("slope", 0.0), copy_mass=w, rand_mass=rmass, copies=bool(copies),
                             classification="selective_periodic_induction" if copies else "tracks_lag_only"))
    out = dict(mode=MODE, mock=IS_MOCK, null_thresh=null_thresh, rows=rows,
               minimal_candidate_set=[r for r in rows if r["copies"]] or rows[:1])
    with open(ck, "w") as f: json.dump(out, f, indent=2)
    return out

RUN_DOWNSTREAM = (PHASE1["verdict"] == "GREEN") or FORCE_DOWNSTREAM
PHASE2 = None
if RUN_DOWNSTREAM:
    cands = PHASE1["candidates"] if PHASE1["candidates"] else [PHASE1["best"]]
    if not PHASE1["candidates"]:
        print("MOCK: no real candidate — forcing Phase 2 on top-T head as a SMOKE TEST only." if IS_MOCK
              else "Proceeding on best head.")
    PHASE2 = run_phase2(cands)
    print(f"\nPhase 2 copying-score null threshold (p{CFG['null_percentile']}): {PHASE2['null_thresh']:.4f}")
    print(f"  {'site':9s} {'L':>2s} {'H':>2s} {'copy_mass':>9s} {'rand':>6s}  classification")
    for r in PHASE2["rows"]:
        cm = "   N/A  " if r["copy_mass"] is None else f"{r['copy_mass']:9.4f}"
        rm = "  N/A " if r["rand_mass"] is None else f"{r['rand_mass']:6.4f}"
        print(f"  {r['site']:9s} {r['layer']:2d} {r['head']:2d} {cm} {rm}  {r['classification']}")
    print("PHASE 2 classification done." + MOCK_TAG)
    try:
        plot_rows = [r for r in PHASE2["rows"] if r["copy_mass"] is not None]
        fig, ax = plt.subplots(figsize=(7, 3.2)); xs = np.arange(len(plot_rows))
        ax.bar(xs - 0.2, [r["copy_mass"] for r in plot_rows], 0.4, label="windowed copy mass")
        ax.bar(xs + 0.2, [r["rand_mass"] for r in plot_rows], 0.4, label="random-bin null")
        ax.axhline(PHASE2["null_thresh"], color="k", ls="--", lw=1, label="non-cand-head null")
        ax.set_xticks(xs); ax.set_xticklabels([f"{r['site'][:4]}\nL{r['layer']}H{r['head']}" for r in plot_rows], fontsize=7)
        ax.set_title("Fig 3: windowed copying score (DLA through OV->lm_head)" + MOCK_TAG); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(CKPT_DIR, "phase2_copying.png"), dpi=80); plt.show(); plt.close(fig)
    except Exception as e:
        print("phase2 plot skipped:", repr(e)[:120])
else:
    print("Phase 1 did not green -> skipping Phases 2-3 (honest branching). See PIVOT recommendation above.")
""")

# ============================================================================
md(r"""
## Section 7 — Phase 3: causal validation (THE pillar; Fig 4)

**Maps to:** Phase 3 / Fig 4. **Gate output: `PHASE 3: PASS | FAIL`.**

1. **Selective mean-ablation** (never zero): mean-ablate the candidate set, measure **sampled ΔCRPS** on
   **periodic vs trend-only vs changepoint-only** with **bootstrap CIs** and a **selectivity ratio**. PASS
   requires the **periodic-side CI excludes zero** and the non-periodic side either includes zero or is
   `selectivity_ratio_min`× smaller.
2. **Path patching** between clean (periodic) and corrupted (**period-altered**) inputs via per-head
   activation patching, localizing the effect to **encoder-self vs decoder-cross**. We corrupt with a
   *period-altered* series (not phase-scramble, which preserves the lag-P autocorrelation).
3. **EAP sweep** with the **categorical-NLL proxy** (one backward pass) ranks every head edge; we then do
   **exact path-patch verification** of the top `eap_top_edges` (catches false positives) **and a low-EAP-rank
   sample** (catches the AtP* false negatives that motivate exact verification), and **ACDC scoped to the
   EAP-surfaced region only** (never the whole model).
4. **Staged-structure test**: head→head patching asks whether upstream heads feed the selecting head
   (estimate → aggregate → select). Reported as a proxy in this feasibility pass.

Headline metric: **selective ΔCRPS** with CIs.
""")

code(r"""
F = torch.nn.functional

def crps_condition(cond_stim, ablate_heads, H, ns, max_series=None):
    # paired clean/ablated CRPS per series under common random numbers -> per-series DeltaCRPS
    stim = cond_stim if (max_series is None) else cond_stim[:max_series]
    if max_series is not None and len(cond_stim) > len(stim):
        print(f"  [phase3] CRPS ablation capped to {len(stim)}/{len(cond_stim)} series this condition "
              f"(CFG['crps_max_series']={max_series}; logged, not silent)")
    deltas = []
    for st in stim:
        HOOKS.reset()
        clean = crps(forecast_samples(st["series"], H, ns), st["cont"].numpy())
        HOOKS.reset(); HOOKS.mode = "intervene"; HOOKS.ablate = set(ablate_heads)
        abl = crps(forecast_samples(st["series"], H, ns), st["cont"].numpy()); HOOKS.reset()
        deltas.append(abl - clean)
    return np.array(deltas)

def boot_ci(x, nboot, seed=0):
    rs = _rng(seed); x = np.asarray(x); means = [rs.choice(x, len(x), replace=True).mean() for _ in range(nboot)]
    return float(np.mean(x)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def capture_all_acts(tf):
    # capture every site-layer '.o' input ONCE (the corrupt side is constant across all heads patched)
    HOOKS.reset(); HOOKS.mode = "capture_act"
    with torch.no_grad(): forward_logits(tf)
    a = {k: v.detach().clone() for k, v in HOOKS.acts.items()}
    HOOKS.reset(); return a

def base_nll(tf_c):
    HOOKS.reset()
    with torch.no_grad():
        return float(F.cross_entropy(forward_logits(tf_c).logits.reshape(-1, ic.vocab_size), tf_c["label_ids"].reshape(-1)))

def head_nll_attr(clean_st, a_corr_all):
    # EAP: score(head) = sum((a_corr - a_clean) * grad_clean) over dims & positions (corrupt->clean attribution)
    tf_c = teacher_forced(clean_st["series"], clean_st["cont"])
    HOOKS.reset(); HOOKS.mode = "grad"
    out = forward_logits(tf_c)
    nll = F.cross_entropy(out.logits.reshape(-1, out.logits.shape[-1]), tf_c["label_ids"].reshape(-1))
    inner.zero_grad(set_to_none=True); nll.backward()
    scores = {}
    for key, a in HOOKS.acts.items():
        if a.grad is None or key not in a_corr_all: continue
        B, S, _ = a.shape; cc = a_corr_all[key]
        if cc.shape != a.shape: continue
        ac = a.detach().view(B, S, N_HEADS, D_KV); cc = cc.view(B, S, N_HEADS, D_KV); g = a.grad.view(B, S, N_HEADS, D_KV)
        per_head = ((cc - ac) * g).sum(dim=(0, 1, 3))   # (n_heads,)
        for h in range(N_HEADS): scores[(key[0], key[1], h)] = float(per_head[h])
    HOOKS.reset(); return scores

def exact_patch_effect(tf_c, head, a_corr_all, base):
    # exact activation patch: write the (precomputed) corrupt head activation into the clean run -> true DeltaNLL.
    a_corr = a_corr_all[(head[0], head[1])]
    B, S, _ = a_corr.shape; ah = a_corr.view(B, S, N_HEADS, D_KV)[:, :, head[2], :]
    HOOKS.reset(); HOOKS.mode = "intervene"; HOOKS.patch[(head[0], head[1], head[2])] = ah
    with torch.no_grad():
        patched = float(F.cross_entropy(forward_logits(tf_c).logits.reshape(-1, ic.vocab_size), tf_c["label_ids"].reshape(-1)))
    HOOKS.reset()
    return patched - base

def run_phase3(cand_rows):
    cached = load_ckpt("phase3")
    if cached is not None: return cached
    ck = ckpt_path("phase3")
    H, ns, nb = CFG["prediction_length"], CFG["crps_num_samples"], CFG["n_bootstrap"]
    cap = CFG.get("crps_max_series", 10**9)
    heads = [(c["site"], c["layer"], c["head"]) for c in cand_rows]
    ref = STIM["periodic"] + STIM["ar1"] + STIM["trend"] + STIM["changepoint"]   # mean over a genuine mix
    compute_means(ref)

    # (1) selective mean-ablation ----------------------------------------------------------------
    res = {}
    for cond in ("periodic", "trend", "changepoint"):
        d = crps_condition(STIM[cond], heads, H, ns, max_series=cap)
        m, lo, hi = boot_ci(d, nb, seed=SEED)
        res[cond] = dict(mean=m, lo=lo, hi=hi, n=len(d))
    nonper = max(abs(res["trend"]["mean"]), abs(res["changepoint"]["mean"]), 1e-9)
    ratio = res["periodic"]["mean"] / nonper
    periodic_excludes_zero = (res["periodic"]["lo"] > 0)
    nonper_ok = (res["trend"]["lo"] <= 0 <= res["trend"]["hi"]) and (res["changepoint"]["lo"] <= 0 <= res["changepoint"]["hi"])
    selective_pass = periodic_excludes_zero and (nonper_ok or ratio >= CFG["selectivity_ratio_min"])
    maybe_empty_cache()

    # clean (periodic) vs corrupt (PERIOD-ALTERED) for path patching / EAP. Period-altered genuinely
    # changes the lag structure (unlike phase-scramble, which preserves the lag-P autocorrelation).
    clean_st, corr_st = STIM["periodic"][0], STIM["period_altered"][0]
    tf_c   = teacher_forced(clean_st["series"], clean_st["cont"])
    A_CORR = capture_all_acts(teacher_forced(corr_st["series"], corr_st["cont"]))   # corrupt acts, captured ONCE
    base   = base_nll(tf_c)

    # (3) EAP sweep (one backward pass) -> rank ----------------------------------------------------
    eap = head_nll_attr(clean_st, A_CORR)
    ranked = sorted(eap.items(), key=lambda kv: -abs(kv[1]))
    topk = ranked[:CFG["eap_top_edges"]]
    verified = [dict(head=list(k), eap=float(sc), exact_dNLL=exact_patch_effect(tf_c, k, A_CORR, base)) for k, sc in topk]
    # false-NEGATIVE probe (the reason exact verification exists, per AtP*): exact-patch a sample of LOW-EAP
    # heads and flag any with a large true effect that EAP under-ranked.
    rs = _rng(SEED); low = ranked[CFG["eap_top_edges"]:]
    fn_idx = rs.choice(len(low), size=min(len(low), max(3, CFG["eap_top_edges"])), replace=False) if low else []
    fn_checked = [dict(head=list(low[i][0]), eap=float(low[i][1]),
                       exact_dNLL=exact_patch_effect(tf_c, low[i][0], A_CORR, base)) for i in fn_idx]
    big = float(np.percentile([abs(v["exact_dNLL"]) for v in verified], 75)) if verified else 0.0
    false_negs = [v for v in fn_checked if abs(v["exact_dNLL"]) >= big > 0]
    union = verified + fn_checked
    eap_corr = _pearson([v["eap"] for v in union], [v["exact_dNLL"] for v in union]) if len(union) >= 3 else float("nan")

    # (2) localization encoder-self vs decoder-cross: EAP per-site mass (adjudicates all three loci) ---
    loc = {s: 0.0 for s in SITES}
    for (s, li, h), sc in eap.items(): loc[s] += abs(float(sc))
    dominant_site = max(loc, key=loc.get) if loc else None

    # (4) ACDC scoped to the EAP region (greedy keep-if-removal-hurts) ----------------------------
    region = [tuple(v["head"]) for v in verified]
    acdc_thresh = float(np.percentile([abs(v["exact_dNLL"]) for v in verified], 50)) if verified else 0.0
    kept = [list(hk) for hk in region if abs(exact_patch_effect(tf_c, hk, A_CORR, base)) >= acdc_thresh]

    # (5) staged-structure: which upstream (strictly-earlier-layer) heads feed the selecting head? ----
    top_cand = (cand_rows[0]["site"], cand_rows[0]["layer"], cand_rows[0]["head"]); cand_set = set(heads)
    up_pool = [(s, li, h) for s in SITES for li in range(N_LAYERS) for h in range(N_HEADS)
               if li < top_cand[1] and (s, li, h) not in cand_set]
    smax = CFG.get("staged_max", 10**9)
    if len(up_pool) > smax:
        print(f"  [phase3] staged scan capped to {smax}/{len(up_pool)} upstream heads (CFG['staged_max']; logged)")
        up_pool = up_pool[:smax]
    upstream = [((s, li, h), abs(exact_patch_effect(tf_c, (s, li, h), A_CORR, base))) for (s, li, h) in up_pool]
    upstream.sort(key=lambda x: -x[1])
    up_thresh = (np.percentile([e for _, e in upstream], CFG["null_percentile"]) if upstream else 0.0)
    stages = [list(k) for k, e in upstream if e >= up_thresh][:5]
    staged = len(stages) >= 2
    maybe_empty_cache()

    out = dict(mode=MODE, mock=IS_MOCK, selective=res, selectivity_ratio=float(ratio), scaling_regime=SCALING_REGIME,
               periodic_excludes_zero=bool(periodic_excludes_zero), nonper_includes_zero=bool(nonper_ok),
               corrupt="period_altered", localization=loc, dominant_site=dominant_site, eap_verified=verified,
               eap_false_neg_checked=fn_checked, eap_false_negatives=false_negs, eap_exact_corr=eap_corr,
               acdc_kept=kept, staged_stages=stages, staged=bool(staged),
               candidate_heads=[list(h) for h in heads], verdict=("PASS" if bool(selective_pass) else "FAIL"))
    with open(ck, "w") as f: json.dump(out, f, indent=2)
    return out

PHASE3 = None
if RUN_DOWNSTREAM:
    cand_rows = (PHASE2["minimal_candidate_set"] if PHASE2 else None) or PHASE1["candidates"] or [PHASE1["best"]]
    # normalize keys (Phase2 rows carry the same site/layer/head fields)
    cand_rows = [{"site": c["site"], "layer": c["layer"], "head": c["head"],
                  "T": c.get("T", 0.0), "slope": c.get("slope", 0.0)} for c in cand_rows]
    PHASE3 = run_phase3(cand_rows)
    s = PHASE3["selective"]
    print("\nSelective mean-ablation DeltaCRPS (mean [95% CI]):")
    for cond in ("periodic", "trend", "changepoint"):
        c = s[cond]; print(f"  {cond:12s} {c['mean']:+.4f}  [{c['lo']:+.4f}, {c['hi']:+.4f}]  (n={c['n']})")
    print(f"  selectivity ratio (periodic / max non-periodic) = {PHASE3['selectivity_ratio']:.2f}  "
          f"(need >= {CFG['selectivity_ratio_min']} or non-periodic CIs include 0)")
    print(f"  periodic CI excludes 0: {PHASE3['periodic_excludes_zero']}   "
          f"non-periodic CIs include 0: {PHASE3['nonper_includes_zero']}")
    print(f"  corrupt input for patching/EAP: {PHASE3['corrupt']} (period-altered; genuinely breaks lag-P structure)")
    print(f"  localization (EAP |attr| mass by site): "
          f"{ {k: round(v,4) for k,v in PHASE3['localization'].items()} }  -> dominant: {PHASE3['dominant_site']}")
    nfn = len(PHASE3["eap_false_negatives"]); ncheck = len(PHASE3["eap_false_neg_checked"])
    print(f"  EAP vs exact corr over top-{CFG['eap_top_edges']} + {ncheck} low-rank probes: {PHASE3['eap_exact_corr']:.3f}  "
          f"| false negatives (large exact effect, low EAP rank): {nfn}")
    print(f"  ACDC-kept heads (scoped to EAP region): {PHASE3['acdc_kept']}")
    print(f"  staged structure present (>=2 upstream stages): {PHASE3['staged']}  stages={PHASE3['staged_stages']}")
    print(f"\nPHASE 3: {PHASE3['verdict']}{MOCK_TAG}")

    try:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        conds = ["periodic", "trend", "changepoint"]
        means = [PHASE3["selective"][c]["mean"] for c in conds]
        los = [PHASE3["selective"][c]["mean"] - PHASE3["selective"][c]["lo"] for c in conds]
        his = [PHASE3["selective"][c]["hi"] - PHASE3["selective"][c]["mean"] for c in conds]
        ax[0].bar(conds, means, yerr=[los, his], capsize=5, color=["#c0392b", "#7f8c8d", "#7f8c8d"])
        ax[0].axhline(0, color="k", lw=1); ax[0].set_ylabel("DeltaCRPS under ablation")
        ax[0].set_title("Fig 4: selective ablation" + MOCK_TAG)
        # circuit diagram (text-on-axes): candidate + dominant site + stages
        ax[1].axis("off"); y = 0.9
        ax[1].text(0.02, y, "Path-patched circuit sketch:", fontsize=10, weight="bold"); y -= 0.12
        ax[1].text(0.04, y, f"candidate(s): {PHASE3['candidate_heads']}", fontsize=8); y -= 0.1
        ax[1].text(0.04, y, f"dominant locus: {PHASE3['dominant_site']}", fontsize=8); y -= 0.1
        ax[1].text(0.04, y, f"upstream stages -> selector: {PHASE3['staged_stages']}", fontsize=8); y -= 0.1
        ax[1].text(0.04, y, f"ACDC-kept: {PHASE3['acdc_kept']}", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(CKPT_DIR, "phase3_causal.png"), dpi=80); plt.show(); plt.close(fig)
    except Exception as e:
        print("phase3 plot skipped:", repr(e)[:120])
""")

# ============================================================================
md(r"""
## Section 8 — Feasibility report

**Maps to:** the feasibility gate. Collects every verdict and prints an overall
**`GO | NO-GO | PIVOT`** recommendation for committing to the A100 Large run, stating plainly whether this
was a **mock_cpu smoke test (not interpretable)** or a **pilot_t4 real verdict**.
""")

code(r"""
def feasibility_report():
    print("=" * 78)
    print(f"FEASIBILITY REPORT  |  run = {MODE}  " + ("(MOCK SMOKE TEST — NOT INTERPRETABLE)" if IS_MOCK else "(REAL PILOT VERDICT)"))
    print("=" * 78)
    print(f"Plumbing (Phase 0b):       {'PASS' if PLUMBING_OK else 'FAIL'}")
    print(f"Tokenization (Phase 0a):   {'PASS' if TOKENIZATION_OK else 'FAIL'}  (lag-in-tokens == lag-in-time)")
    print(f"Phase 1 (selective-lag):   {PHASE1['verdict']}   "
          f"best head {PHASE1['best']['site']} L{PHASE1['best']['layer']}H{PHASE1['best']['head']}  "
          f"T={PHASE1['best']['T']:.3f} slope={PHASE1['best']['slope']:.3f} (null thr {PHASE1['null_thresh']:.3f})")
    if PHASE2:
        cls = ", ".join(sorted(set(r["classification"] for r in PHASE2["rows"])))
        print(f"Phase 2 (copying/OV):      {cls}")
    else:
        print(f"Phase 2 (copying/OV):      skipped (Phase 1 not GREEN)")
    if PHASE3:
        s = PHASE3["selective"]["periodic"]
        print(f"Phase 3 (causal):          {PHASE3['verdict']}   selective DeltaCRPS(periodic)="
              f"{s['mean']:+.4f} [{s['lo']:+.4f},{s['hi']:+.4f}]  ratio={PHASE3['selectivity_ratio']:.2f}  "
              f"dominant={PHASE3['dominant_site']}  staged={PHASE3['staged']}")
    else:
        print(f"Phase 3 (causal):          skipped (Phase 1 not GREEN)")

    # overall recommendation
    if PHASE1["verdict"] == "PIVOT":
        rec = "PIVOT — periodicity is not a selective attention circuit; trace the change-detection circuit (Phase 1')."
    elif PHASE1["verdict"] == "GREEN" and PHASE3 and PHASE3["verdict"] == "PASS":
        rec = "GO — candidate selective periodic-induction head is causal & selective; commit to the A100 Large run."
    elif PHASE1["verdict"] == "GREEN":
        rec = "NO-GO (yet) — Phase 1 green but causal selectivity not established; tighten Phase 3 before scaling."
    else:
        rec = "AMBIGUOUS — add periods/series/seeds and re-decide before committing compute."
    if IS_MOCK:
        rec = "MOCK SMOKE TEST PASSED — pipeline executes end-to-end. Flip CONFIG['MODE']='pilot_t4' for the real verdict. " \
              "(The recommendation logic above is exercised but the numbers are NOT interpretable.)"
    print("-" * 78)
    print("OVERALL:", rec)
    print("=" * 78)

feasibility_report()
""")

# ============================================================================
# ---- assemble notebook + mirror ----------------------------------------------------------------
nb = new_notebook()
nb.cells = [new_markdown_cell(s) if t == "md" else new_code_cell(s) for (t, s) in CELLS]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}
OUT_IPYNB = "chronos_circuit_feasibility.ipynb"
with open(OUT_IPYNB, "w") as f:
    nbf.write(nb, f)

# runnable mirror: concatenate code cells (for local smoke testing of the exact source)
mirror = ["# AUTO-GENERATED mirror of the notebook's code cells (for local smoke testing only).\n"]
for t, s in CELLS:
    if t == "code":
        mirror.append("\n# " + "=" * 70 + "\n" + s + "\n")
with open("_mirror.py", "w") as f:
    f.write("\n".join(mirror))

n_code = sum(1 for t, _ in CELLS if t == "code")
n_md = sum(1 for t, _ in CELLS if t == "md")
print(f"wrote {OUT_IPYNB}  ({len(CELLS)} cells: {n_md} markdown, {n_code} code)")
print("wrote _mirror.py")
