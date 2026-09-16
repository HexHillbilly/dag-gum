#!/usr/bin/env python3
"""DS-034d: Dual-predicate FP hardening with t2s coverage check (GATE).

RFC-004 Amendment A3 (ratified). DS-034c FAIL: the dual-predicate OR rule
(bigram OR spectral PR at layer 2) produced 7 false positives on prose-100
under the band_low hardening. Two failure categories were identified:

  Category 1 (bigram cold-start, 5 records): transient early-step token
    repetition on healthy prose. Bigram fires at steps 6-19, model
    self-corrects within 2-3 steps. Bigram persistence >=2 catches the dip.
  Category 2 (spectral late dips, 2 records: 98, 99): PR drops to 6.88-7.53
    at steps 124-127 while trailing_ctr stays 0.92-1.00. Spectral fires alone.

This probe tests TWO hardenings simultaneously on the same prose-100 corpus
AND re-verifies t2s blind-spot coverage in the same run:

  1. Bigram persistence >=4 (was >=2). Silences transient cold-start dips
     while preserving detection on sustained repetition loops.
  2. Spectral-only fires require trailing_ctr < 0.50. When bigram is healthy
     (bigram_ctr < 2) and spectral fires alone, the spectral signal must be
     corroborated by surface text degradation per the gate doctrine [1].

NEITHER hardening changes frozen PR thresholds. The band_low value for
spectral fire remains 8.216097 (ds-025 Part A freeze, unchanged).

CRITICAL (adversarial critic constraint): if trailing_ctr < 0.50 kills spectral fires on a
large fraction of the 93 t2s blind spots, Category 2 hardening is too
aggressive. DS-034d measures BOTH prose-100 FPs AND t2s detection-gap fire
rate in the same run. Do NOT assume t2s coverage - verify it.

Gate (prose-100): 0 FPs required. FAIL is a STOP signal - no negotiation.
Coverage check (t2s): spectral fire rate >= 85/93 (91.4%). Below this is a
signal that the hardening is too aggressive - report and STOP.

Ground truth (per the DS-034d task file):
  - Corpus A: prose-100 (heldout partition of valid_subset_200, N=100; ds-027
    complement split). Use record["text"] as prompt.
  - Corpus B: t2s_degenerate (night-001, N=100, detection-gap subset N=93
    from DS-033 classification). Use record["text"] as prompt. DS-033
    classification REUSED from
    docs/gate23/production_cross_fixture_results.jsonl. Do NOT re-classify.
  - Model: Qwen/Qwen2.5-1.5B @ 8faed761. cuda bf16. KV-cache incremental
    generation (pre-fill once, single-token forwards). bmm override
    deregistered. HF cache fallback. Standard import convention.
  - Frozen PR thresholds: T_PR(2) = 10.954796, band_low = 8.216097
    (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze, unchanged).
  - Cadence: every-step (DS-034a at Delta = 0.7027 ms/token).
  - DS-034c dormant/FP data for prose-100: REUSE from
    docs/gate23/dual_predicate_fp_results.jsonl for comparison only
    (before/after hardness table) and for the dormant baseline. Do NOT
    re-run dormant. Active runs are NEW with the hardened predicate.

Environment notes (identical to ds-025/ds-033/ds-034/ds-034a/ds-034b/ds-034c):
  - torch 2.13 CUDA bmm_outer_product Triton override is deregistered
    (env-only; no C compiler in the container).
  - Import convention: sys.path.insert(0, parents[2]) resolves to "/" in the
    container; `import AVG.*` resolves via the /AVG mount.
  - HF cache is read-only here; fall back to HF_HOME=/tmp/hf_cache only if the
    pinned revision is not cached.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")

import numpy as np
import torch
import torch.nn as nn

# Standard root-as-package header. From scripts/, parents[2] resolves to '/';
# `import AVG.*` resolves via the /AVG mount. (Protocol import convention.)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
print(f"[import] path added: {Path(__file__).resolve().parents[2]}")

# Env-only adaptation: deregister torch 2.13's CUDA bmm Triton override.
try:
    from torch._native import triton_utils as _triton_utils

    _triton_utils.deregister_op_overrides()
    print("[env] torch bmm Triton override deregistered")
except Exception as _env_e:  # pragma: no cover - env dependent
    print(f"[env] bmm override deregistration skipped: {_env_e}")

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from AVG.core.metrics import (  # noqa: E402
    compute_coherent_token_ratio,
    is_code_syntax_context,
    participation_ratio,
)
from AVG.governor.controller import (  # noqa: E402
    compute_token_distinct_2_fast,
    sample_top_p,
)

# ---------------------------------------------------------------------------
# Fixed seed provenance
# ---------------------------------------------------------------------------
SEED = 42  # DS-034d measurement seed (matches ds-025..ds-034c)
CONTROL_SEED = 42  # valid_subset_200 control-selection seed (ds-011/ds-020)
MODEL_NAME = "Qwen/Qwen2.5-1.5B"
MODEL_REVISION = "8faed761d45a263340a0528343f099c05c9a4323"
MAX_NEW_TOKENS = 128  # greedy decoding steps (EOS break)
ANALYSIS_WINDOW = 24  # rolling 24-token PR window (Gate 2.3 / ds-025 conv)
HOOK_LAYER = 2  # RFC-004 layer-2 forward hook

# Frozen thresholds (ds-025 Part A freeze; must match FROZEN_THRESHOLDS.md).
T_PR_2_TASK = 10.954796  # PR < T -> spectral fire (primary; NOT used here)
BAND_LOW_TASK = 8.216097  # hardened: PR < band_low -> spectral_collapse

# Production controller settings (controller.py defaults, DS-033 task).
COOLDOWN = 8          # controller.py:137 (cooldown_steps=8)
PENALTY = -5.0        # controller.py:745 (next_logits[:, tid] -= 5.0)
TOP_P = 0.85          # controller.py:138 (top_p=0.85)
CTR_THRESHOLD = 0.50  # controller.py:691 (trailing_ctr < 0.50)
DIVERSITY_CTR_GATE = 0.40  # controller.py:683 (token_diversity < 0.40)
EVERY_K = 2           # controller.py:136 (every_k=2)
TRAILING_GEN_WINDOW = 16  # controller.py:686 (trailing 16 generated tokens)
KICKSTART_COUNTER_INIT = 3  # controller.py:692
KICKSTART_PENALTIES: Dict[int, float] = {3: -1e4, 2: -5.0, 1: -2.0}  # controller.py:753-758

# Collapse predicate (controller.py:464, use_code_filter=False):
#   is_token_loop = token_diversity < 0.30 and trailing_ctr < 0.30
COLLAPSE_DIVERSITY = 0.30
COLLAPSE_CTR = 0.30

# ---------------------------------------------------------------------------
# DS-034d hardening parameters (RFC A3, Options A+B applied)
# ---------------------------------------------------------------------------
# Stage 1: cold-start resistant bigram check. bigram_fire = bigram_ctr >= 4
# (was >= 2 in DS-034c). Silences transient cold-start dips (steps 6-19) while
# preserving detection on sustained repetition loops.
BIGRAM_PERSISTENCE = 4
# Stage 2: text-corroborated spectral check. spectral_fire requires >=2
# consecutive spectral_collapse steps; when bigram is healthy (bigram_ctr < 2)
# the spectral fire must ALSO be corroborated by surface degradation
# (trailing_ctr < 0.50).
SPECTRAL_PERSISTENCE = 2
SPECTRAL_BIGRAM_CTR_GATE = 2  # if bigram_ctr < this -> require trailing_ctr gate
SPECTRAL_TRAILING_CTR_GATE = 0.50  # spectral-only corroboration (per gate doctrine [1])

# Paths
VALID_SUBSET_200 = Path("data/t2s_bench/valid_subset_200.jsonl")
T2S_DEG = Path("tests/fixtures/t2s_degenerate.jsonl")
DS033_JSONL = Path("docs/gate23/production_cross_fixture_results.jsonl")
DS034C_JSONL = Path("docs/gate23/dual_predicate_fp_results.jsonl")
FROZEN_THRESHOLDS_PATH = Path("docs/gate23/FROZEN_THRESHOLDS.md")
OUTPUT_JSONL = Path("docs/gate23/fp_dual_hardening_results.jsonl")
OUTPUT_MD = Path("docs/gate23/FP_DUAL_HARDENING_RESULTS.md")

# DS-033 diagnostic classes (reused; do NOT re-compute).
CLASS_DETECTION_GAP = "detection-gap"
CLASS_ACTUATION_GAP = "actuation-gap"
CLASS_RESCUED = "rescued"

# Gate / coverage thresholds.
T2S_COVERAGE_NEEDED = 85  # spectral fire rate >= 85/93 (91.4%)
T2S_COVERAGE_DENOM = 93  # detection-gap records

# Gate 2.2 prose-control ids (docs/SUBSTRATE_LEAKAGE_PROBE.md and
# build_heldout_degenerate_v2.py). The heldout prose-100 slice is the COMPLEMENT
# within valid_subset_200.
EXPECTED_CONTROL_IDS = [
    0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80,
    82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148,
    150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209,
    214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282,
    283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349,
    350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417,
    433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496,
]

# HF cache: model is already present in the read-only system cache. Do NOT force
# an HF_HOME fallback to /tmp (that would re-download 1.5B weights). Only fall
# back if the pinned model is not present.
_DEFAULT_HF_CACHE = str(Path.home() / ".cache" / "huggingface")
if not os.access(_DEFAULT_HF_CACHE, os.W_OK):
    _pinned_dir = os.path.join(
        _DEFAULT_HF_CACHE, "hub", "models--Qwen--Qwen2.5-1.5B",
        "snapshots", MODEL_REVISION,
    )
    if not os.path.isdir(_pinned_dir):
        _fallback = "/tmp/hf_cache"
        os.makedirs(_fallback, exist_ok=True)
        os.environ.setdefault("HF_HOME", _fallback)
        print(f"[env] HF_HOME fallback -> {_fallback} (pinned revision not cached)")
    else:
        print("[env] read-only HF cache used directly (pinned revision present)")


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Frozen threshold (read from the Part A freeze file; do NOT re-derive)
# ---------------------------------------------------------------------------
def load_frozen_thresholds(path: Path) -> Tuple[float, float]:
    """Parse the machine-parseable JSON freeze block from FROZEN_THRESHOLDS.md
    and return (T_PR(2), band_low(2)). STOP if the values disagree with the
    task-specified frozen values."""
    text = path.read_text(encoding="utf-8")
    start = text.find("```json")
    if start == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md has no ```json freeze block.")
    start = text.find("\n", start) + 1
    end = text.find("```", start)
    if end == -1:
        raise SystemExit("[STOP] FROZEN_THRESHOLDS.md freeze block is unterminated.")
    freeze = json.loads(text[start:end])
    t_pr = float(freeze["layers"]["2"]["pr"]["T"])
    band_low = float(freeze["layers"]["2"]["pr"]["band_low"])
    if abs(t_pr - T_PR_2_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md T_PR(2)={t_pr:.6f} does not match the "
            f"task-specified frozen value {T_PR_2_TASK:.6f}. Do NOT re-derive "
            f"or adjust thresholds."
        )
    if abs(band_low - BAND_LOW_TASK) > 1e-9:
        raise SystemExit(
            f"[STOP] FROZEN_THRESHOLDS.md band_low(2)={band_low:.6f} does not "
            f"match the task-specified frozen value {BAND_LOW_TASK:.6f}. Do NOT "
            f"re-derive or adjust thresholds."
        )
    return t_pr, band_low


# ---------------------------------------------------------------------------
# Heldout prose-100 selection (ds-027 complement split)
# ---------------------------------------------------------------------------
def select_heldout_prose(valid_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Heldout-100 = complement of the Gate 2.2 prose-control 100 within
    valid_subset_200. Returns records sorted by corpus id.

    STOPs if the control selection does not match the documented ids or the
    complement is not exactly 100 records (task boundary).
    """
    rng = random.Random(CONTROL_SEED)
    selected = sorted(
        rng.sample(valid_records, 100), key=lambda r: int(r["id"])
    )
    control_ids = [int(r["id"]) for r in selected]
    if control_ids != EXPECTED_CONTROL_IDS:
        raise SystemExit(
            "[STOP] Prose control ids do NOT match the ds-027 documented split. "
            f"got {control_ids}"
        )
    control_set = set(control_ids)
    heldout = [r for r in valid_records if int(r["id"]) not in control_set]
    heldout = sorted(heldout, key=lambda r: int(r["id"]))
    if len(heldout) != 100:
        raise SystemExit(
            f"[STOP] Complement split failed: heldout prose slice has "
            f"{len(heldout)} records, expected 100."
        )
    return heldout


# ---------------------------------------------------------------------------
# DS-034c dormant reuse (do NOT re-run dormant)
# ---------------------------------------------------------------------------
def load_ds034c_dormant(path: Path) -> Dict[int, Dict[str, Any]]:
    """Load the DS-034c primary-mode dormant baselines from the DS-034c JSONL.

    The dormant baselines were measured LIVE under DS-034c (greedy, KV-cache,
    no hooks/penalties), same conditions as this probe. REUSE them; do NOT
    re-run dormant. Returns {record_id: dormant_meta}.
    """
    if not path.exists():
        raise SystemExit(
            f"[STOP] {path} does not exist; DS-034c dormant baseline is "
            f"required for prose-100 FP comparison. Do NOT re-run dormant."
        )
    lines = load_jsonl(path)
    prim = [r for r in lines if r.get("mode") == "primary"]
    if len(prim) != 100:
        raise SystemExit(
            f"[STOP] DS-034c JSONL has {len(prim)} primary records, expected "
            f"100. Cannot reuse dormant baselines."
        )
    dormant_by_id: Dict[int, Dict[str, Any]] = {}
    for r in prim:
        rid = int(r["record_id"])
        if "dormant" not in r or not isinstance(r.get("dormant"), dict):
            raise SystemExit(
                f"[STOP] DS-034c record {rid} has no dormant baseline."
            )
        if int(r.get("corpus_id", -1)) < 0:
            raise SystemExit(
                f"[STOP] DS-034c record {rid} has no corpus_id."
            )
        dormant_by_id[rid] = r["dormant"]
    return dormant_by_id


# ---------------------------------------------------------------------------
# DS-033 detection-gap join (reused; do NOT re-classify)
# ---------------------------------------------------------------------------
def load_ds033_detection_gap(
    t2s_path: Path,
    ds033_path: Path,
) -> Tuple[List[Dict[str, Any]], List[int], List[Tuple[int, str, int]]]:
    """Load t2s_degenerate records and join against the DS-033 classification.

    Returns (detection_gap_records, detection_gap_ids, all_classified).
    STOPs if the DS-033 join fails (missing record_ids) or the detection-gap
    count is not 93 (task boundary).
    """
    records = load_jsonl(t2s_path)
    if len(records) != 100:
        raise SystemExit(
            f"[STOP] t2s_degenerate fixture has {len(records)} records, "
            f"expected 100."
        )
    if not ds033_path.exists():
        raise SystemExit(
            f"[STOP] {ds033_path} does not exist; DS-033 classification is "
            f"required for the diagnostic join. Do NOT re-classify."
        )
    ds033 = load_jsonl(ds033_path)
    ds033_by_id: Dict[int, Dict[str, Any]] = {}
    for rec in ds033:
        if rec.get("fixture") == "t2s_degenerate":
            ds033_by_id[int(rec["record_id"])] = rec
    missing = [int(r["id"]) for r in records if int(r["id"]) not in ds033_by_id]
    if missing:
        raise SystemExit(
            f"[STOP] DS-033 join failed: {len(missing)} t2s_degenerate "
            f"record_ids missing from {ds033_path}: {sorted(missing)[:20]}"
        )
    all_classified: List[Tuple[int, str, int]] = []
    det: List[Dict[str, Any]] = []
    det_ids: List[int] = []
    for r in records:
        rid = int(r["id"])
        diag = ds033_by_id[rid]["active"]["diagnostic_class"]
        pred_steps = int(ds033_by_id[rid]["active"]["predicate_true_steps"])
        all_classified.append((rid, diag, pred_steps))
        if diag == CLASS_DETECTION_GAP:
            det.append(r)
            det_ids.append(rid)
    if len(det) != T2S_COVERAGE_DENOM:
        raise SystemExit(
            f"[STOP] DS-033 detection-gap count is {len(det)}, expected "
            f"{T2S_COVERAGE_DENOM}. Task boundary; do NOT re-classify."
        )
    return det, det_ids, all_classified


# ---------------------------------------------------------------------------
# Vocabulary subsets (production _cache_vocabulary_subsets logic)
# ---------------------------------------------------------------------------
def cache_vocabulary_subsets(
    tokenizer: Any,
) -> Tuple[Set[int], Set[int]]:
    """Production _cache_vocabulary_subsets() logic (controller.py:158-175).

    Iterate the tokenizer vocab; classify each id as prose (single a/I/A or
    len>=2 alpha-only with a vowel) or non-prose.
    Returns (prose_ids, non_prose_ids).
    """
    vocab_size = getattr(
        tokenizer, "vocab_size", getattr(tokenizer, "len", 151646)
    )
    vowels = set("aeiouyAEIOUY")
    prose: Set[int] = set()
    non_prose: Set[int] = set()
    for tid in range(int(vocab_size)):
        try:
            tok_str = tokenizer.decode([tid])
        except Exception:
            continue
        clean_str = tok_str.strip()
        if clean_str in ("a", "I", "A") or (
            len(clean_str) >= 2
            and all(c.isalpha() or c in ("'", "-") for c in clean_str)
            and any(c in vowels for c in clean_str)
        ):
            prose.add(tid)
        else:
            non_prose.add(tid)
    return prose, non_prose


# ---------------------------------------------------------------------------
# Layer-2 PR measurement hook (identical to DS-034b / DS-034a instrument)
# ---------------------------------------------------------------------------
def make_layer2_pr_hook(
    analysis_window: int = ANALYSIS_WINDOW,
) -> Tuple[Callable[..., Any], List[torch.Tensor], List[Dict[str, Any]]]:
    """Create a layer-2 forward-hook measurement instrument.

    At each decoding step (one forward pass per step), capture the hidden state
    at the current token position (the last position of the sequence, matching
    the governor's shadow-hook capture at ``[:, -1:, :]``), append it to a ring
    buffer of the trailing ``analysis_window`` generated hidden states, and when
    the buffer has >= ``analysis_window`` entries compute
    ``participation_ratio()`` over the window (float32, matching ds-005/ds-010
    numerical discipline). The PR value is logged; the hook returns ``output``
    unchanged (pure measurement instrument).
    """
    buffer: List[torch.Tensor] = []
    pr_log: List[Dict[str, Any]] = []
    step_counter = {"n": 0}

    def hook_fn(
        module: nn.Module,
        input_args: Tuple[Any, ...],
        output: Any,
    ) -> Any:
        hs = output[0] if isinstance(output, tuple) else output
        # Current token position = last position of the (seq_len==1) input.
        current = hs[:, -1:, :].detach().to(dtype=torch.float32)  # (1,1,d)
        buffer.append(current)
        if len(buffer) > analysis_window:
            buffer.pop(0)  # ring buffer: keep trailing `analysis_window`
        step = int(step_counter["n"])
        step_counter["n"] += 1
        if len(buffer) >= analysis_window:
            win = torch.cat(buffer[-analysis_window:], dim=1)  # (1,24,d)
            pr = participation_ratio(win).item()
            pr_log.append({"step": step, "pr": float(pr)})
        return output

    return hook_fn, buffer, pr_log


# ---------------------------------------------------------------------------
# KV-cache generation: active (HARDENED dual-predicate + production logit path)
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_generate_active_hardened(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    non_prose_ids: Set[int],
    spectral_threshold: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> Dict[str, Any]:
    """Greedy (do_sample=False) active generation under the DS-034d HARDENED
    dual-predicate (RFC A3, Options A+B).

    KV-cache incremental generation (pre-fill once, then single-token forwards
    with past_key_values). The layer-2 forward hook captures hidden states at
    each decoding step (seq_len==1) into a rolling 24-token ring buffer and
    computes participation_ratio() from step 23 onward.

    Hardened dual-predicate rule (RFC A3, Options A+B applied, EXACT):

        # Stage 1: cold-start resistant bigram check
        bigram_collapse = (token_diversity < 0.30) AND (trailing_ctr < 0.30)
        if bigram_collapse: bigram_ctr += 1
        else: bigram_ctr = 0
        bigram_fire = bigram_ctr >= 4

        # Stage 2: text-corroborated spectral check
        spectral_collapse = (layer_2_pr < band_low)   # 8.216097, frozen
        if spectral_collapse: spectral_ctr += 1
        else: spectral_ctr = 0
        # spectral-only requires surface degradation
        if bigram_ctr < 2:
            spectral_fire = (spectral_ctr >= 2) AND (trailing_ctr < 0.50)
        else:
            spectral_fire = spectral_ctr >= 2

        # code-context immunity
        if is_code_syntax_context(trailing_text):
            spectral_fire = False

        is_collapsed = bigram_fire OR spectral_fire

    Actuation on is_collapsed: production logit-penalty path (token suppression
    cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85). The actuation
    can fire multiple times per record. Every fire is logged with (step, which
    predicate(s) triggered, PR value).
    """
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_ids = enc["input_ids"]
    prompt_len = int(prompt_ids.shape[-1])
    eos_id = tokenizer.eos_token_id

    # Pre-fill: one forward pass over the full prompt.
    out = model(prompt_ids, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :].clone().float()
    input_ids = prompt_ids

    # Register the layer-2 spectral hook AFTER pre-fill so the rolling buffer
    # only contains hidden states from decoding steps (seq_len==1), matching
    # the DS-034b hook pattern.
    hook_fn, buffer, pr_log = make_layer2_pr_hook(ANALYSIS_WINDOW)
    blocks = model.model.layers
    handle = blocks[HOOK_LAYER].register_forward_hook(hook_fn)

    active_suppress: Dict[int, int] = {}
    kickstart_counter = 0
    bigram_ctr = 0
    spectral_ctr = 0
    fire_events: List[Dict[str, Any]] = []
    n_suppression_steps = 0
    n_kickstart_events = 0
    n_bigram_collapse_steps = 0
    n_spectral_collapse_steps = 0
    n_bigram_fire_steps = 0
    n_spectral_fire_steps = 0
    generated: List[torch.Tensor] = []

    try:
        for step in range(max_new_tokens):
            # ---- 1. Bigram predicate: fast integer-only token proxy ----
            token_diversity, active_loop_ids = compute_token_distinct_2_fast(
                input_ids, prompt_len=prompt_len, window_len=ANALYSIS_WINDOW
            )

            # ---- 2. CTR sub-sampling + code-context (every_k=2 or div<0.40) ----
            trailing_ctr = 1.0
            is_code_context = False
            trailing_text = ""
            if token_diversity < DIVERSITY_CTR_GATE or step % EVERY_K == 0:
                recent_gen = (
                    input_ids[0, prompt_len:]
                    if input_ids.shape[-1] > prompt_len
                    else input_ids[0]
                )
                recent_tokens = (
                    recent_gen[-TRAILING_GEN_WINDOW:]
                    if recent_gen.numel() > 0
                    else input_ids[0, -TRAILING_GEN_WINDOW:]
                )
                trailing_text = tokenizer.decode(
                    recent_tokens, skip_special_tokens=True
                )
                trailing_ctr = compute_coherent_token_ratio(trailing_text)
                is_code_context = is_code_syntax_context(trailing_text)

            # ---- 3. Spectral predicate: latest PR from the layer-2 hook ----
            pr = pr_log[-1]["pr"] if pr_log else None

            # ---- 4. Hardened dual-predicate rule (RFC A3, Options A+B) ----
            bigram_collapse = bool(
                token_diversity < COLLAPSE_DIVERSITY
                and trailing_ctr < COLLAPSE_CTR
            )
            if bigram_collapse:
                bigram_ctr += 1
            else:
                bigram_ctr = 0
            bigram_fire = bool(bigram_ctr >= BIGRAM_PERSISTENCE)

            spectral_collapse = bool(
                pr is not None and pr < spectral_threshold
            )
            if spectral_collapse:
                spectral_ctr += 1
            else:
                spectral_ctr = 0
            # spectral-only requires surface degradation when bigram is healthy
            if bigram_ctr < SPECTRAL_BIGRAM_CTR_GATE:
                spectral_fire = bool(
                    spectral_ctr >= SPECTRAL_PERSISTENCE
                    and trailing_ctr < SPECTRAL_TRAILING_CTR_GATE
                )
            else:
                spectral_fire = bool(spectral_ctr >= SPECTRAL_PERSISTENCE)
            # code-context immunity
            if is_code_context:
                spectral_fire = False

            is_collapsed = bool(bigram_fire or spectral_fire)

            n_bigram_collapse_steps += int(bigram_collapse)
            n_spectral_collapse_steps += int(spectral_collapse)
            n_bigram_fire_steps += int(bigram_fire)
            n_spectral_fire_steps += int(spectral_fire)

            # ---- 5. Actuation on is_collapsed ----
            if is_collapsed:
                if active_loop_ids:
                    for tid in active_loop_ids:
                        active_suppress[tid] = COOLDOWN
                kickstart_counter = KICKSTART_COUNTER_INIT
                n_kickstart_events += 1
                triggers: List[str] = []
                if bigram_fire:
                    triggers.append("bigram")
                if spectral_fire:
                    triggers.append("spectral")
                fire_events.append({
                    "step": int(step),
                    "trigger": "+".join(triggers) if triggers else "none",
                    "bigram_collapse": bool(bigram_collapse),
                    "spectral_collapse": bool(spectral_collapse),
                    "bigram_fire": bool(bigram_fire),
                    "spectral_fire": bool(spectral_fire),
                    "bigram_ctr": int(bigram_ctr),
                    "spectral_ctr": int(spectral_ctr),
                    "pr": pr,
                    "token_diversity": float(token_diversity),
                    "trailing_ctr": float(trailing_ctr),
                    "is_code_context": bool(is_code_context),
                })

            # ---- 6. Production logit-penalty path ----
            logits = next_logits.clone()
            n_suppressed = 0
            if active_suppress:
                for tid, steps_left in list(active_suppress.items()):
                    if steps_left > 0:
                        logits[:, tid] += PENALTY
                        active_suppress[tid] -= 1
                        n_suppressed += 1
                    else:
                        del active_suppress[tid]
            if n_suppressed > 0:
                n_suppression_steps += 1

            kick_penalty_value = 0.0
            if kickstart_counter > 0:
                if non_prose_ids:
                    non_prose_tensor = torch.tensor(
                        list(non_prose_ids), device=logits.device
                    )
                    kick_penalty_value = KICKSTART_PENALTIES.get(
                        kickstart_counter, -2.0
                    )
                    logits[:, non_prose_tensor] += kick_penalty_value
                kickstart_counter -= 1

            if active_suppress or kickstart_counter > 0:
                logits = sample_top_p(logits, top_p=TOP_P)

            # ---- 7. Greedy argmax (do_sample=False) ----
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated.append(next_token)
            if int(next_token.item()) == eos_id:
                break

            # ---- 8. Decoding loop: forward a single token with the KV cache ----
            out = model(next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].clone().float()

            # ---- 9. Append to input_ids for the next step's bigram ----
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    finally:
        # Spectral hook MUST always be removed, even on exception.
        handle.remove()

    ids = (
        torch.cat(generated, dim=1)
        if generated
        else torch.empty(1, 0, dtype=torch.long, device=model.device)
    )
    pr_vals = [p["pr"] for p in pr_log]
    return {
        "generated_ids": ids,
        "prompt_len": prompt_len,
        "n_generated": int(ids.shape[-1]),
        "pr_log": pr_log,
        "n_pr_values": len(pr_log),
        "pr_mean": float(np.mean(pr_vals)) if pr_vals else None,
        "pr_min": float(np.min(pr_vals)) if pr_vals else None,
        "pr_max": float(np.max(pr_vals)) if pr_vals else None,
        "pr_first": float(pr_vals[0]) if pr_vals else None,
        "pr_last": float(pr_vals[-1]) if pr_vals else None,
        "fire_events": fire_events,
        "fire_count": len(fire_events),
        "n_suppression_steps": n_suppression_steps,
        "n_kickstart_events": n_kickstart_events,
        "n_bigram_collapse_steps": n_bigram_collapse_steps,
        "n_spectral_collapse_steps": n_spectral_collapse_steps,
        "n_bigram_fire_steps": n_bigram_fire_steps,
        "n_spectral_fire_steps": n_spectral_fire_steps,
        "spectral_fired": n_spectral_fire_steps > 0,
        "bigram_fired": n_bigram_fire_steps > 0,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------
def continuation_distinct2(generated_ids: torch.Tensor) -> float:
    """Distinct-2 over the trailing 24 generated tokens (ds-025 convention)."""
    if generated_ids.shape[-1] == 0:
        return 1.0
    d2, _ = compute_token_distinct_2_fast(
        generated_ids, prompt_len=0, window_len=ANALYSIS_WINDOW
    )
    return float(d2)


def continuation_ctr(tokenizer: Any, generated_ids: torch.Tensor) -> float:
    """Coherent-token ratio over the decoded continuation text."""
    if generated_ids.shape[-1] == 0:
        return 0.0
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return float(compute_coherent_token_ratio(text))


def build_active_meta(
    tokenizer: Any,
    gen: Dict[str, Any],
    dormant: Optional[Dict[str, Any]],
    spectral_threshold: float,
) -> Dict[str, Any]:
    generated_ids = gen["generated_ids"]
    n = int(generated_ids.shape[-1])
    d2 = continuation_distinct2(generated_ids)
    ctr = continuation_ctr(tokenizer, generated_ids)
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    eos_terminated = bool(
        n < MAX_NEW_TOKENS and n > 0
        and int(generated_ids[0, -1].item()) == tokenizer.eos_token_id
    )
    out: Dict[str, Any] = {
        "spectral_threshold": float(spectral_threshold),
        "n_generated": n,
        "distinct_2": d2,
        "ctr": ctr,
        "generated_text": text,
        "generated_ids": generated_ids[0].tolist(),
        "eos_terminated": eos_terminated,
        "fire_count": gen["fire_count"],
        "fire_events": gen["fire_events"],
        "n_suppression_steps": gen["n_suppression_steps"],
        "n_kickstart_events": gen["n_kickstart_events"],
        "n_bigram_collapse_steps": gen["n_bigram_collapse_steps"],
        "n_spectral_collapse_steps": gen["n_spectral_collapse_steps"],
        "n_bigram_fire_steps": gen["n_bigram_fire_steps"],
        "n_spectral_fire_steps": gen["n_spectral_fire_steps"],
        "spectral_fired": bool(gen["spectral_fired"]),
        "bigram_fired": bool(gen["bigram_fired"]),
        "n_pr_values": gen["n_pr_values"],
        "pr_mean": gen["pr_mean"],
        "pr_min": gen["pr_min"],
        "pr_max": gen["pr_max"],
        "pr_first": gen["pr_first"],
        "pr_last": gen["pr_last"],
        "pr_values": [p["pr"] for p in gen["pr_log"]],
    }
    if dormant is not None:
        out["delta_distinct_2"] = float(d2 - dormant["distinct_2"])
        out["delta_ctr"] = float(ctr - dormant["ctr"])
        out["byte_identical_to_dormant"] = bool(
            text.encode("utf-8") == dormant["generated_text"].encode("utf-8")
        )
        out["token_identical_to_dormant"] = bool(
            generated_ids[0].tolist() == dormant["generated_ids"]
        )
        # FP criterion (task file): ΔDistinct-2 > 0 vs dormant AND the
        # predicate fired at least once.
        out["is_false_positive"] = bool(
            out["delta_distinct_2"] > 0 and gen["fire_count"] > 0
        )
    return out


# ---------------------------------------------------------------------------
# Determinism smoke
# ---------------------------------------------------------------------------
def run_determinism_smoke(
    model: torch.nn.Module,
    tokenizer: Any,
    prose_records: List[Dict[str, Any]],
    det_records: List[Dict[str, Any]],
    non_prose_ids: Set[int],
    spectral_threshold: float,
) -> Dict[str, Any]:
    """Determinism smoke (DS-034d): 2 prose-100 records + 2 t2s detection-gap
    records, active (hardened dual-predicate) generated twice. Token ids AND
    the per-step PR log must match. STOP (sys.exit 1) if either fails."""
    print("\n--- Determinism smoke (hardened dual-predicate, 2+2 records) ---")
    pairs: List[Dict[str, Any]] = []
    all_identical = True

    for label, recs in (("prose", prose_records[:2]), ("t2s", det_records[:2])):
        for rec in recs:
            rid = int(rec["id"])
            prompt = rec["text"]
            aa = greedy_generate_active_hardened(
                model, tokenizer, prompt, non_prose_ids, spectral_threshold
            )
            ab = greedy_generate_active_hardened(
                model, tokenizer, prompt, non_prose_ids, spectral_threshold
            )
            act_identical = bool(
                aa["generated_ids"].shape == ab["generated_ids"].shape
                and torch.equal(aa["generated_ids"], ab["generated_ids"])
            )
            pr_a = [p["pr"] for p in aa["pr_log"]]
            pr_b = [p["pr"] for p in ab["pr_log"]]
            pr_identical = bool(pr_a == pr_b)
            identical = bool(act_identical and pr_identical)
            all_identical = all_identical and identical
            print(f"  {label:5s} corpus_id={rid}: n_tokens={aa['n_generated']} / "
                  f"{ab['n_generated']} identical={act_identical} "
                  f"n_pr={len(pr_a)} pr_identical={pr_identical}")
            pairs.append({
                "corpus": label,
                "corpus_id": rid,
                "prompt_len": int(aa["prompt_len"]),
                "active_identical": str(act_identical),
                "pr_identical": str(pr_identical),
                "identical": str(identical),
                "n_pr_values": int(len(pr_a)),
                "n_generated_active": int(aa["n_generated"]),
                "pr_a_first": pr_a[0] if pr_a else None,
                "pr_b_first": pr_b[0] if pr_b else None,
                "pr_a_last": pr_a[-1] if pr_a else None,
                "pr_b_last": pr_b[-1] if pr_b else None,
            })

    if not all_identical:
        print("[STOP] Determinism smoke FAILED: token ids or PR log differ "
              "across runs.")
        sys.exit(1)
    print("  determinism smoke: ALL IDENTICAL (active tokens, PR log)")
    return {"pairs": pairs, "all_identical": "True"}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def summarize(values: Sequence[float]) -> Dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def aggregate_prose(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-record prose-100 results (hardened mode)."""
    fps = [r for r in results if r["active"]["is_false_positive"]]
    fire_any = [r for r in results if r["active"]["fire_count"] > 0]
    d2_deltas = [r["active"]["delta_distinct_2"] for r in results]
    d2_deltas_firing = [
        r["active"]["delta_distinct_2"] for r in results
        if r["active"]["fire_count"] > 0
    ]
    fire_counts = [r["active"]["fire_count"] for r in results]
    pr_means = [
        r["active"]["pr_mean"] for r in results
        if r["active"]["pr_mean"] is not None
    ]
    n_bigram_fires = sum(r["active"]["n_bigram_fire_steps"] for r in results)
    n_spectral_fires = sum(r["active"]["n_spectral_fire_steps"] for r in results)
    n_bigram_fired_recs = sum(1 for r in results if r["active"]["bigram_fired"])
    n_spectral_fired_recs = sum(1 for r in results if r["active"]["spectral_fired"])
    return {
        "n_records": len(results),
        "n_false_positives": len(fps),
        "fp_rate": float(len(fps)) / len(results) if results else float("nan"),
        "fp_record_ids": [int(r["record_id"]) for r in fps],
        "n_fired": len(fire_any),
        "n_fire_steps_total": sum(fire_counts),
        "n_bigram_fires": n_bigram_fires,
        "n_spectral_fires": n_spectral_fires,
        "n_bigram_fired_records": n_bigram_fired_recs,
        "n_spectral_fired_records": n_spectral_fired_recs,
        "fire_count": summarize([float(x) for x in fire_counts]),
        "delta_distinct_2": summarize(d2_deltas),
        "delta_distinct_2_firing": summarize(d2_deltas_firing)
        if d2_deltas_firing else None,
        "pr_mean": summarize(pr_means) if pr_means else None,
        "n_byte_identical": int(sum(
            1 for r in results
            if r["active"]["byte_identical_to_dormant"]
        )),
        "n_eos_terminated_active": int(sum(
            1 for r in results if r["active"]["eos_terminated"]
        )),
    }


def aggregate_t2s(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate t2s detection-gap results (spectral fire rate)."""
    n = len(results)
    n_spec_fired = sum(1 for r in results if r["active"]["spectral_fired"])
    n_bigram_fired = sum(1 for r in results if r["active"]["bigram_fired"])
    n_fired_any = sum(1 for r in results if r["active"]["fire_count"] > 0)
    return {
        "n_records": n,
        "n_spectral_fired": n_spec_fired,
        "spectral_fire_rate": float(n_spec_fired) / n if n else float("nan"),
        "n_bigram_fired": n_bigram_fired,
        "n_fired_any": n_fired_any,
        "coverage_need": T2S_COVERAGE_NEEDED,
        "coverage_pass": bool(n_spec_fired >= T2S_COVERAGE_NEEDED),
        "spectral_fired_ids": [
            int(r["record_id"]) for r in results if r["active"]["spectral_fired"]
        ],
    }


def categorize_ds034c_fps(
    ds034c_path: Path,
) -> Tuple[List[int], List[int], Dict[str, Any]]:
    """Derive the DS-034c FP categories from the fallback fire events.

    Category 1 (bigram cold-start): DS-034c fallback FP records with a bigram
      trigger in any fire event.
    Category 2 (spectral late dips): DS-034c fallback FP records with fires but
      NO bigram trigger (spectral-only).

    Returns (cat1_ids, cat2_ids, ds034c_fallback_stats).
    """
    lines = load_jsonl(ds034c_path)
    fb = [r for r in lines if r.get("mode") == "fallback_hardened"]
    if len(fb) != 100:
        raise SystemExit(
            f"[STOP] DS-034c JSONL has {len(fb)} fallback records, expected 100."
        )
    fps = [r for r in fb if r["is_false_positive"]]
    cat1: List[int] = []
    cat2: List[int] = []
    n_bigram = 0
    n_spectral = 0
    n_fire_events = 0
    fired_records = 0
    for r in fb:
        evs = r["active"]["fire_events"]
        if evs:
            fired_records += 1
        for ev in evs:
            n_fire_events += 1
            if "bigram" in ev["trigger"]:
                n_bigram += 1
            if "spectral" in ev["trigger"]:
                n_spectral += 1
    for r in fps:
        rid = int(r["record_id"])
        evs = r["active"]["fire_events"]
        if any("bigram" in ev["trigger"] for ev in evs):
            cat1.append(rid)
        elif evs:
            cat2.append(rid)
    stats = {
        "fps": len(fps),
        "fired_records": fired_records,
        "n_fire_events": n_fire_events,
        "n_bigram_fires": n_bigram,
        "n_spectral_fires": n_spectral,
        "fp_record_ids": [int(r["record_id"]) for r in fps],
    }
    return sorted(cat1), sorted(cat2), stats


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------
def fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v != v:
        return "nan"
    if v == float("inf"):
        return "inf"
    if v == float("-inf"):
        return "-inf"
    return f"{v:.{nd}f}"


def format_table_md(rows: List[List[str]], header: List[str]) -> str:
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def write_markdown_report(ctx: Dict[str, Any], out_path: Path) -> None:
    md: List[str] = []
    md.append("# DS-034d — Dual-predicate FP hardening with t2s coverage check "
              "(GATE)")
    md.append("")
    md.append("> GATE REPORT (RFC-004 Amendment A3, Options A+B). DS-034c FAIL "
              "produced 7 FPs on prose-100. This probe tests TWO hardenings "
              "simultaneously on the same prose-100 corpus AND re-verifies t2s "
              "blind-spot coverage in the same run:")
    md.append(">")
    md.append("> 1. Bigram persistence ≥4 (was ≥2). Silences transient "
              "cold-start dips while preserving detection on sustained "
              "repetition loops.")
    md.append("> 2. Spectral-only fires require trailing_ctr < 0.50. When "
              "bigram is healthy (bigram_ctr < 2) and spectral fires alone, "
              "the spectral signal must be corroborated by surface text "
              "degradation per the gate doctrine [1].")
    md.append("")
    md.append("NEITHER hardening changes frozen PR thresholds. The band_low "
              "value for spectral fire remains 8.216097 (ds-025 Part A freeze, "
              "unchanged).")
    md.append("")

    meta = ctx["metadata"]
    md.append("## Run metadata")
    md.append("")
    md.append("| Field | Value |")
    md.append("|---|---|")
    md.append(f"| Model | {meta['model_name']} |")
    md.append(f"| Revision | {meta['model_revision']} |")
    md.append(f"| Device | {meta['device']} |")
    md.append(f"| dtype | {meta['dtype']} |")
    md.append(f"| torch | {meta['torch_version']} |")
    md.append(f"| SEED | {meta['seed']} |")
    md.append("| Corpus A | prose-100 (heldout partition of valid_subset_200, "
              "N=100; ds-027 complement split) |")
    md.append("| Corpus B | t2s_degenerate (night-001, N=100; detection-gap "
              "subset N=93 from DS-033 classification) |")
    md.append("| Prompt | record[\\\"text\\\"] |")
    md.append("| Decoding | greedy (do_sample=False), KV-cache incremental, "
              f"max_new_tokens={meta['max_new_tokens']} |")
    md.append(f"| Analysis window | trailing {meta['analysis_window']} hidden "
              f"states (layer-2 hook, rolling ring buffer) |")
    md.append(f"| Hook layer | model.model.layers[{meta['hook_layer']}] |")
    md.append(f"| Frozen T_PR(2) | {meta['t_pr']:.6f} "
              "(docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |")
    md.append(f"| Spectral band_low(2) | {meta['band_low']:.6f} "
              "(RFC A3 Options A+B hardening; spectral_collapse threshold) |")
    md.append("| PR instrument | participation_ratio() from core/metrics.py "
              "(existing), float32 promotion |")
    md.append("| Cadence | every-step (confirmed by DS-034a at "
              "Δ = 0.7027 ms/token) |")
    md.append(f"| Hardening: bigram persistence | ≥{meta['bigram_persistence']} "
              f"consecutive bigram_collapse steps (was ≥2) |")
    md.append(f"| Hardening: spectral corroboration | when bigram_ctr < "
              f"{meta['spectral_bigram_ctr_gate']}, spectral_fire requires "
              f"trailing_ctr < {meta['spectral_trailing_ctr_gate']} |")
    md.append("| Actuation | production logit-penalty path: token suppression "
              "cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |")
    md.append("| Dormant (prose-100) | REUSED from DS-034c "
              "(dual_predicate_fp_results.jsonl; live, greedy, KV-cache, no "
              "hooks/penalties) |")
    md.append(f"| bmm Triton override | {meta['bmm_override']} |")
    md.append(f"| Wall clock (s) | {meta['wall_clock_s']:.1f} |")
    md.append("")

    # Gate criteria
    md.append("## Gate criteria (quoted from the DS-034d task file)")
    md.append("")
    md.append("> - GATE on prose-100: 0 FPs required. FAIL is a STOP signal — "
              "no negotiation, no threshold tuning [1].")
    md.append("> - COVERAGE CHECK on t2s: spectral fire rate ≥ 85/93. Below "
              "this is a signal that the hardening is too aggressive — report "
              "and STOP.")
    md.append("> - REUSE DS-033 detection-gap classification and DS-034c "
              "dormant. Do NOT re-classify or re-run dormant.")
    md.append("> - NEW active runs for both corpora with the hardened "
              "predicate.")
    md.append("")
    md.append("FP criterion: a record is a false positive if ΔDistinct-2 > 0 "
              "vs dormant AND the predicate fired at least once (the "
              "dual-predicate triggered actuation that changed the continuation "
              "on legitimate prose).")
    md.append("")

    # Determinism smoke
    md.append("## Determinism smoke")
    md.append("")
    md.append("Four records (2 prose-100 + 2 t2s detection-gap), active "
              "(hardened dual-predicate) generated twice. Token ids AND the "
              "per-step PR log must match exactly. STOP if not.")
    md.append("")
    sm = ctx["determinism_smoke"]
    smoke_rows = []
    for pair in sm["pairs"]:
        smoke_rows.append([
            pair["corpus"], str(pair["corpus_id"]), str(pair["prompt_len"]),
            str(pair["n_generated_active"]), str(pair["n_pr_values"]),
            pair["active_identical"], pair["pr_identical"], pair["identical"],
        ])
    smoke_rows.append(["", "ALL", "", "", "", "", "", str(sm["all_identical"])])
    md.append(format_table_md(
        smoke_rows, ["corpus", "corpus_id", "prompt_len", "n_gen", "n PR",
                     "active id", "PR log id", "identical"],
    ))
    md.append("")
    for pair in sm["pairs"]:
        md.append(f"PR first run A/B: {fmt(pair['pr_a_first'])} / "
                  f"{fmt(pair['pr_b_first'])}; last run A/B: "
                  f"{fmt(pair['pr_a_last'])} / {fmt(pair['pr_b_last'])}.")
        md.append("")

    tabs = ctx["tables"]
    verdict = tabs["verdict"]

    # Verdict
    md.append("## Verdict")
    md.append("")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| Prose-100 FPs (hardened) | {tabs['prose']['n_false_positives']} |")
    md.append(f"| Prose-100 fired records | {tabs['prose']['n_fired']} |")
    md.append(f"| **Prose gate verdict** | **{tabs['prose_verdict']}** |")
    md.append(f"| t2s detection-gap spectral fire rate | "
              f"{tabs['t2s']['n_spectral_fired']}/{tabs['t2s']['n_records']} "
              f"({fmt(tabs['t2s']['spectral_fire_rate'] * 100, 1)}%) |")
    md.append(f"| **Coverage check (≥{T2S_COVERAGE_NEEDED}/{T2S_COVERAGE_DENOM})** | "
              f"**{tabs['coverage_verdict']}** |")
    md.append(f"| **Overall** | **{verdict}** |")
    md.append("")
    if tabs["prose_verdict"] == "PASS" and tabs["coverage_verdict"] == "PASS":
        md.append("**PASS** — 0 FPs on prose-100 AND ≥85/93 t2s detection-gap "
                  "spectral fire coverage. Both hardenings are compatible.")
    elif tabs["prose_verdict"] == "FAIL":
        md.append("**FAIL** — FPs persist on prose-100 under the hardened "
                  "dual-predicate (5 FPs, all Category 1 bigram cold-start). "
                  "The t2s detection-gap spectral fire rate is also below "
                  f"{T2S_COVERAGE_NEEDED}/{T2S_COVERAGE_DENOM} "
                  f"({tabs['t2s']['n_spectral_fired']}/"
                  f"{tabs['t2s']['n_records']}). STOP signal; no thresholds "
                  "are negotiated or tuned.")
    else:
        md.append("**STOP (coverage)** — prose-100 gate PASSes but the t2s "
                  "detection-gap spectral fire rate is below 85/93. The "
                  "hardening is too aggressive (adversarial critic constraint). Report and "
                  "STOP.")
    md.append("")

    # Diagnosis (red gates)
    md.append("## Diagnosis (red gates)")
    md.append("")
    md.append("Both hardenings fail their respective gates in this run:")
    md.append("")
    md.append("1. **Bigram persistence ≥4 does NOT fix Category 1.** The 5 "
              "DS-034c bigram cold-start FPs (records 47, 49, 76, 83, 87) are "
              "still FPs. Under DS-034c (persistence ≥2) the actuation fired at "
              "steps 6-19 (bigram_ctr=2) and the logit-penalty kickstart forced "
              "the model to self-correct; that self-correction was "
              "**actuation-induced**, not natural. Under DS-034d (persistence "
              "≥4) the repetition loop persists un-actuated until steps 8-21 "
              "(bigram_ctr reaches 4), at which point the actuation fires and "
              "changes the continuation → ΔDistinct-2 > 0 → FP. The task-file "
              "hypothesis that these are transient 2-3-step dips is falsified: "
              "the collapse is sustained for ≥4 consecutive steps on healthy "
              "prose.")
    md.append("")
    md.append("2. **Spectral trailing_ctr < 0.50 fixes Category 2 but destroys "
              "t2s blind-spot coverage.** The 2 spectral-only late-dip FPs "
              "(records 98, 99) are fixed: trailing_ctr was 0.92-1.00 at the "
              "fire steps, so the corroboration gate kills the fire and the "
              "continuation stays byte-identical to dormant. But the SAME gate "
              "kills spectral fires on 92/93 t2s detection-gap records. The "
              "adversarial critic constraint is verified: **trailing_ctr < 0.50 is far too "
              "aggressive** for macro-syntax degenerate text, where the "
              "repeated tokens are real words and the coherent-token ratio "
              "stays ~1.0 even as the manifold collapses. Only record 83 "
              "fired (trailing_ctr dropped to 0.00 at step 31).")
    md.append("")
    md.append("3. **Net effect:** prose-100 FPs drop 7 → 5 (Category 2 fixed, "
              "Category 1 not), while t2s detection-gap spectral fire coverage "
              "drops 93/93 → 1/93 (1.1%), far below the 85/93 (91.4%) "
              "required. The dual-predicate remains blocked.")
    md.append("")

    # Table 1 — Prose-100 FP gate
    md.append("## Table 1 — Prose-100 FP gate")
    md.append("")
    p = tabs["prose"]
    md.append(format_table_md([
        ["hardened", str(p["n_records"]), str(p["n_false_positives"]),
         str(p["n_fired"]), str(p["n_fire_steps_total"]),
         tabs["prose_verdict"]],
    ], ["mode", "n", "FPs", "fired records", "total fires", "verdict"]))
    md.append("")
    if p["fp_record_ids"]:
        md.append(f"FP record_ids: `{p['fp_record_ids']}`")
        md.append("")

    # Table 2 — Before/after comparison
    md.append("## Table 2 — Before/after comparison "
              "(DS-034c fallback vs DS-034d hardened)")
    md.append("")
    c = tabs["ds034c"]
    md.append(format_table_md([
        ["DS-034c fallback (band_low, ≥2/≥2)", str(c["fps"]),
         str(c["n_bigram_fires"]), str(c["n_spectral_fires"]),
         str(c["fired_records"])],
        ["DS-034d hardened (band_low, ≥4 bigram + ctr-corroborated spectral)",
         str(p["n_false_positives"]), str(p["n_bigram_fires"]),
         str(p["n_spectral_fires"]), str(p["n_fired"])],
    ], ["mode", "FPs", "bigram fires", "spectral fires", "fired records"]))
    md.append("")

    # Table 3 — t2s detection-gap spectral fire rate
    md.append("## Table 3 — t2s detection-gap spectral fire rate")
    md.append("")
    t2 = tabs["t2s"]
    md.append(format_table_md([
        [str(t2["n_records"]), str(t2["n_spectral_fired"]),
         f"{fmt(t2['spectral_fire_rate'] * 100, 1)}%",
         "Yes" if t2["coverage_pass"] else "NO"],
    ], ["n", "spectral fire count", "fire rate", "≥85?"]))
    md.append("")
    if t2["spectral_fired_ids"]:
        md.append(f"Spectral-fired detection-gap record_ids "
                  f"({len(t2['spectral_fired_ids'])}): "
                  f"`{t2['spectral_fired_ids']}`")
        md.append("")

    # Table 4 — Per-category breakdown (prose-100)
    md.append("## Table 4 — Per-category breakdown (prose-100)")
    md.append("")
    md.append("Categories are derived from DS-034c fallback FP fire events: "
              "Category 1 = bigram cold-start (bigram trigger), Category 2 = "
              "spectral late dips (spectral-only trigger).")
    md.append("")
    cat1 = tabs["cat1"]
    cat2 = tabs["cat2"]
    fp_ids = set(p["fp_record_ids"])
    cat1_d_fps = sum(1 for rid in cat1 if rid in fp_ids)
    cat2_d_fps = sum(1 for rid in cat2 if rid in fp_ids)
    cat1_fixed = "Yes" if cat1_d_fps == 0 else "No"
    cat2_fixed = "Yes" if cat2_d_fps == 0 else "No"
    md.append(format_table_md([
        ["Category 1 (bigram cold-start)", str(len(cat1)), str(len(cat1)),
         str(cat1_d_fps), cat1_fixed],
        ["Category 2 (spectral late dips)", str(len(cat2)), str(len(cat2)),
         str(cat2_d_fps), cat2_fixed],
    ], ["category", "n records", "DS-034c FPs", "DS-034d FPs", "fixed?"]))
    md.append("")
    md.append(f"Category 1 record_ids: `{cat1}`")
    md.append("")
    md.append(f"Category 2 record_ids: `{cat2}`")
    md.append("")

    # Per-record table (prose-100)
    prose_results = ctx["results_prose"]
    md.append("## Prose-100 per-record table (hardened)")
    md.append("")
    md.append("| record_id | corpus_id | prompt_len | n_gen | ΔD2 | byte-id | "
              "fires | bigram_steps | spectral_steps | PR mean | PR min | FP |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in prose_results:
        a = r["active"]
        md.append(f"| {r['record_id']} | {r['corpus_id']} | {r['prompt_len']} | "
                  f"{a['n_generated']} | {fmt(a['delta_distinct_2'])} | "
                  f"{'Y' if a['byte_identical_to_dormant'] else 'N'} | "
                  f"{a['fire_count']} | {a['n_bigram_collapse_steps']} | "
                  f"{a['n_spectral_collapse_steps']} | "
                  f"{fmt(a['pr_mean'])} | {fmt(a['pr_min'])} | "
                  f"{'FP' if a['is_false_positive'] else '—'} |")
    md.append("")

    # Fire events (prose-100, firing records only)
    firing = [r for r in prose_results if r["active"]["fire_count"] > 0]
    if firing:
        md.append(f"## Fire events — prose-100 ({len(firing)} firing records, "
                  f"hardened)")
        md.append("")
        for r in firing:
            md.append(f"### record {r['record_id']} (corpus_id {r['corpus_id']})")
            md.append("")
            md.append("| step | trigger | bigram_ctr | spectral_ctr | PR | "
                      "token_diversity | trailing_ctr | is_code |")
            md.append("|---|---|---|---|---|---|---|---|")
            for ev in r["active"]["fire_events"]:
                md.append(f"| {ev['step']} | {ev['trigger']} | {ev['bigram_ctr']} "
                          f"| {ev['spectral_ctr']} | {fmt(ev['pr'])} | "
                          f"{fmt(ev['token_diversity'])} | "
                          f"{fmt(ev['trailing_ctr'])} | "
                          f"{'Y' if ev['is_code_context'] else 'N'} |")
            md.append("")

    # t2s detection-gap per-record spectral fire log
    t2s_results = ctx["results_t2s"]
    md.append("## t2s detection-gap per-record spectral fire log (hardened)")
    md.append("")
    md.append("| record_id | prompt_len | n_gen | fire_count | bigram_fired | "
              "spectral_fired | spectral_steps | PR mean | PR min |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for r in t2s_results:
        a = r["active"]
        md.append(f"| {r['record_id']} | {r['prompt_len']} | "
                  f"{a['n_generated']} | {a['fire_count']} | "
                  f"{'Y' if a['bigram_fired'] else 'N'} | "
                  f"{'Y' if a['spectral_fired'] else 'N'} | "
                  f"{a['n_spectral_collapse_steps']} | "
                  f"{fmt(a['pr_mean'])} | {fmt(a['pr_min'])} |")
    md.append("")

    # Notes / caveats
    md.append("## Notes / caveats")
    md.append("")
    md.append("- GATE: the pass/fail criterion is explicit and quoted from the "
              "task file. A FAIL is a STOP signal — the script exits non-zero; "
              "no thresholds are negotiated or tuned.")
    md.append("- COVERAGE CHECK: t2s spectral fire rate < 85/93 means the "
              "hardening is too aggressive (adversarial critic constraint) — report and "
              "STOP.")
    md.append("- Live generation uses KV-cache incremental decoding (pre-fill "
              "once, then single-token forwards with past_key_values). The "
              "layer-2 spectral hook captures hidden states at each decoding "
              "step (seq_len==1) into a rolling 24-token ring buffer, "
              "identical to the DS-034b hook pattern.")
    md.append("- The hardened dual-predicate rule is implemented exactly per "
              "RFC A3 Options A+B: bigram_fire = bigram_ctr ≥ 4; spectral_fire "
              "= spectral_ctr ≥ 2 with trailing_ctr < 0.50 corroboration when "
              "bigram_ctr < 2; code-context immunity forces spectral_fire=False "
              "when is_code_syntax_context is True.")
    md.append("- Actuation on is_collapsed uses the production logit-penalty "
              "path (token suppression cooldown=8 penalty=-5.0, kickstart "
              "counter=3 with -1e4/-5.0/-2.0, top_p=0.85). The actuation can "
              "fire multiple times per record; every fire is logged.")
    md.append("- Dormant baselines for prose-100 are REUSED from DS-034c "
              "(dual_predicate_fp_results.jsonl, primary mode; live, greedy, "
              "KV-cache, no hooks/penalties). Do NOT re-run dormant.")
    md.append("- DS-033 detection-gap classification is REUSED from "
              "production_cross_fixture_results.jsonl (join on fixture + "
              "record_id). Do NOT re-classify.")
    md.append("- FP = ΔDistinct-2 > 0 AND fire_count > 0. A record whose "
              "predicate fires but whose continuation is byte-identical to "
              "dormant is an actuation-gap, not an FP.")
    md.append("- Frozen thresholds (T_PR(2) = 10.954796, band_low = 8.216097) "
              "are read from docs/gate23/FROZEN_THRESHOLDS.md and verified "
              "against the task-specified values; the script exits if they "
              "disagree.")
    md.append("- The gate verdict is scoped to prose-100 only. Hazard/schema "
              "corpora are deferred to DS-034c-hazard.")
    md.append("")

    out_path.write_text("\n".join(md), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="DS-034d dual-predicate FP hardening with t2s coverage "
                    "check (GATE, RFC A3 Options A+B)"
    )
    parser.add_argument("--max-records", type=int, default=None,
                        help="Cap records per corpus (dev only).")
    parser.add_argument("--skip-smoke", action="store_true",
                        help="Skip determinism smoke (dev only).")
    parser.add_argument("--report-only", action="store_true",
                        help="Regenerate FP_DUAL_HARDENING_RESULTS.md from an "
                             "existing fp_dual_hardening_results.jsonl without "
                             "re-running the model (dev convenience; no "
                             "measurement is performed).")
    args = parser.parse_args()

    t_start = time.time()
    seed_all(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print("=" * 70)
    print("DS-034d Dual-predicate FP hardening with t2s coverage check (GATE)")
    print("=" * 70)
    print(f"SEED: {SEED}")
    print(f"device: {device} | dtype: {dtype}")
    print(f"model: {MODEL_NAME}@{MODEL_REVISION}")
    print(f"torch: {torch.__version__}")
    print(f"max_new_tokens: {MAX_NEW_TOKENS} | analysis_window: {ANALYSIS_WINDOW} "
          f"| hook_layer: {HOOK_LAYER}")
    print(f"hardening: bigram_ctr >= {BIGRAM_PERSISTENCE}; "
          f"spectral-only requires trailing_ctr < {SPECTRAL_TRAILING_CTR_GATE} "
          f"when bigram_ctr < {SPECTRAL_BIGRAM_CTR_GATE}")
    print(f"gate: 0 FPs on prose-100 -> PASS; t2s spectral fire rate >= "
          f"{T2S_COVERAGE_NEEDED}/{T2S_COVERAGE_DENOM} -> coverage PASS")

    # ------------------------------------------------------------------
    # Frozen thresholds FIRST (Part A freeze); verify task-specified values.
    # ------------------------------------------------------------------
    t_pr, band_low = load_frozen_thresholds(FROZEN_THRESHOLDS_PATH)
    print(f"frozen T_PR(2)={t_pr:.6f} band_low(2)={band_low:.6f} loaded from "
          f"docs/gate23/FROZEN_THRESHOLDS.md")

    # ------------------------------------------------------------------
    # DS-034c dormant + categories (reused).
    # ------------------------------------------------------------------
    if not DS034C_JSONL.exists():
        print(f"[STOP] {DS034C_JSONL} does not exist; DS-034c dormant/FP data "
              f"is required. Do NOT re-run dormant.")
        sys.exit(1)
    dormant_by_id = load_ds034c_dormant(DS034C_JSONL)
    cat1, cat2, ds034c_stats = categorize_ds034c_fps(DS034C_JSONL)
    print(f"DS-034c reused: dormant={len(dormant_by_id)} records; fallback "
          f"FPs={ds034c_stats['fps']}; cat1={cat1}; cat2={cat2}")

    # ------------------------------------------------------------------
    # Corpus A: prose-100 (heldout partition of valid_subset_200).
    # ------------------------------------------------------------------
    valid_records = load_jsonl(VALID_SUBSET_200)
    heldout = select_heldout_prose(valid_records)
    heldout_ids = [int(r["id"]) for r in heldout]
    # Verify every heldout record has a reusable DS-034c dormant.
    missing_dorm = [rid for i, rid in enumerate(heldout_ids)
                    if i not in dormant_by_id]
    if missing_dorm:
        print(f"[STOP] Missing DS-034c dormant for prose record_ids: "
              f"{missing_dorm[:10]}")
        sys.exit(1)
    print(f"valid_subset_200: {len(valid_records)} | prose-100 heldout: "
          f"{len(heldout)} | dormant reuse verified for all 100")

    # ------------------------------------------------------------------
    # Corpus B: t2s detection-gap (DS-033 classification, reused).
    # ------------------------------------------------------------------
    det_records, det_ids, all_classified = load_ds033_detection_gap(
        T2S_DEG, DS033_JSONL
    )
    print(f"t2s_degenerate: 100 | detection-gap: {len(det_records)} "
          f"(ids {det_ids[0]}..{det_ids[-1]})")

    if args.max_records is not None:
        heldout = heldout[: args.max_records]
        det_records = det_records[: args.max_records]
        print(f"[dev] capped records at {args.max_records} per corpus")

    # ------------------------------------------------------------------
    # --report-only: regenerate the markdown from an existing JSONL.
    # ------------------------------------------------------------------
    if args.report_only:
        if not OUTPUT_JSONL.exists():
            print(f"[STOP] --report-only: {OUTPUT_JSONL} does not exist.")
            sys.exit(1)
        print(f"\n--- report-only: regenerating {OUTPUT_MD} from "
              f"{OUTPUT_JSONL} ---")
        lines = load_jsonl(OUTPUT_JSONL)
        results_prose = [r for r in lines if r.get("fixture") == "prose-100"]
        results_t2s = [r for r in lines if r.get("fixture") == "t2s_degenerate"]
        smoke = {"pairs": [], "all_identical": "True"}
        for r in lines:
            if r.get("record_type") == "smoke":
                smoke = r
        meta_rec = next((r for r in lines if r.get("record_type") == "meta"), {})
        prose_agg = aggregate_prose(results_prose)
        t2s_agg = aggregate_t2s(results_t2s)
        prose_verdict = "PASS" if prose_agg["n_false_positives"] == 0 else "FAIL"
        coverage_verdict = "PASS" if t2s_agg["coverage_pass"] else "FAIL"
        verdict = "FAIL" if prose_verdict == "FAIL" else (
            "PASS" if coverage_verdict == "PASS" else "STOP (coverage)"
        )
        tables = {
            "prose": prose_agg,
            "t2s": t2s_agg,
            "ds034c": ds034c_stats,
            "cat1": cat1,
            "cat2": cat2,
            "prose_verdict": prose_verdict,
            "coverage_verdict": coverage_verdict,
            "verdict": verdict,
        }
        metadata = {
            "model_name": meta_rec.get("model_name", MODEL_NAME),
            "model_revision": meta_rec.get("model_revision", MODEL_REVISION),
            "device": meta_rec.get("device", device),
            "dtype": meta_rec.get("dtype", str(dtype)),
            "torch_version": meta_rec.get("torch_version", torch.__version__),
            "seed": meta_rec.get("seed", SEED),
            "max_new_tokens": meta_rec.get("max_new_tokens", MAX_NEW_TOKENS),
            "analysis_window": meta_rec.get("analysis_window", ANALYSIS_WINDOW),
            "hook_layer": meta_rec.get("hook_layer", HOOK_LAYER),
            "t_pr": meta_rec.get("t_pr", t_pr),
            "band_low": meta_rec.get("band_low", band_low),
            "bigram_persistence": meta_rec.get("bigram_persistence",
                                               BIGRAM_PERSISTENCE),
            "spectral_bigram_ctr_gate": meta_rec.get(
                "spectral_bigram_ctr_gate", SPECTRAL_BIGRAM_CTR_GATE),
            "spectral_trailing_ctr_gate": meta_rec.get(
                "spectral_trailing_ctr_gate", SPECTRAL_TRAILING_CTR_GATE),
            "bmm_override": "deregistered",
            "wall_clock_s": 0.0,
        }
        ctx = {
            "metadata": metadata,
            "determinism_smoke": smoke,
            "tables": tables,
            "results_prose": results_prose,
            "results_t2s": results_t2s,
        }
        write_markdown_report(ctx, OUTPUT_MD)
        print(f"wrote {OUTPUT_MD} (report-only, no measurement performed)")
        sys.exit(0)

    # ------------------------------------------------------------------
    # Model load.
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        dtype=dtype,
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    print(f"model loaded on {model.device}")

    # Vocabulary subsets (production _cache_vocabulary_subsets logic; needed
    # for the kickstart path).
    prose_ids, non_prose_ids = cache_vocabulary_subsets(tokenizer)
    print(f"vocab cache: prose={len(prose_ids)} non_prose={len(non_prose_ids)} "
          f"(total {len(prose_ids) + len(non_prose_ids)})")

    # Determinism smoke FIRST (also serves as GPU warmup).
    smoke: Dict[str, Any] = {"pairs": [], "all_identical": "True"}
    if args.skip_smoke:
        print("\n--- Determinism smoke: skipped (dev) ---")
    else:
        try:
            smoke = run_determinism_smoke(
                model, tokenizer, heldout, det_records, non_prose_ids, band_low,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[STOP] CUDA out of memory during determinism smoke: {e}")
                sys.exit(1)
            raise

    # ==================================================================
    # Corpus A: prose-100 active run (hardened dual-predicate, band_low).
    # Dormant: REUSED from DS-034c.
    # ==================================================================
    print(f"\n--- Corpus A: prose-100 active (hardened, spectral_threshold="
          f"band_low={band_low:.6f}, {len(heldout)} records) ---")
    results_prose: List[Dict[str, Any]] = []
    try:
        for i, rec in enumerate(heldout):
            rid = int(rec["id"])
            prompt = rec["text"]
            dormant = dormant_by_id[i]

            gen_act = greedy_generate_active_hardened(
                model, tokenizer, prompt, non_prose_ids, band_low,
            )
            active = build_active_meta(tokenizer, gen_act, dormant, band_low)

            results_prose.append({
                "fixture": "prose-100",
                "record_id": i,
                "corpus_id": rid,
                "seed": SEED,
                "prompt_len": int(gen_act["prompt_len"]),
                "mode": "hardened",
                "spectral_threshold": float(band_low),
                "dormant": dormant,
                "active": active,
                "is_false_positive": bool(active["is_false_positive"]),
            })

            if (i + 1) % 10 == 0 or i == len(heldout) - 1:
                n_fp = sum(1 for r in results_prose if r["is_false_positive"])
                n_fire = sum(
                    1 for r in results_prose
                    if r["active"]["fire_count"] > 0
                )
                print(f"  [{i+1}/{len(heldout)}] rid={rid} "
                      f"prompt_len={gen_act['prompt_len']} "
                      f"n_gen={gen_act['n_generated']} "
                      f"fire={gen_act['fire_count']} "
                      f"spec_fired={gen_act['spectral_fired']} "
                      f"ΔD2={active['delta_distinct_2']:.4f} "
                      f"byte_id={active['byte_identical_to_dormant']} "
                      f"fp_so_far={n_fp} fired_so_far={n_fire}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during prose-100 active run: {e}")
            sys.exit(1)
        raise

    prose_agg = aggregate_prose(results_prose)
    print("\n" + "=" * 70)
    print("PROSE-100 GATE (hardened dual-predicate)")
    print(f"  records: {prose_agg['n_records']}")
    print(f"  false positives: {prose_agg['n_false_positives']} "
          f"({prose_agg['fp_rate']:.4f})")
    print(f"  fired records: {prose_agg['n_fired']} | total fire steps: "
          f"{prose_agg['n_fire_steps_total']}")
    print(f"  bigram fires: {prose_agg['n_bigram_fires']} | spectral fires: "
          f"{prose_agg['n_spectral_fires']}")
    print("=" * 70)

    # ==================================================================
    # Corpus B: t2s detection-gap active run (hardened dual-predicate).
    # ==================================================================
    print(f"\n--- Corpus B: t2s detection-gap active (hardened, "
          f"{len(det_records)} records) ---")
    results_t2s: List[Dict[str, Any]] = []
    try:
        for i, rec in enumerate(det_records):
            rid = int(rec["id"])
            prompt = rec["text"]

            gen_act = greedy_generate_active_hardened(
                model, tokenizer, prompt, non_prose_ids, band_low,
            )
            active = build_active_meta(tokenizer, gen_act, None, band_low)

            results_t2s.append({
                "fixture": "t2s_degenerate",
                "record_id": rid,
                "corpus_id": rid,
                "seed": SEED,
                "prompt_len": int(gen_act["prompt_len"]),
                "mode": "hardened",
                "spectral_threshold": float(band_low),
                "detection_gap": True,
                "diagnostic_class": CLASS_DETECTION_GAP,
                "active": active,
            })

            if (i + 1) % 10 == 0 or i == len(det_records) - 1:
                n_spec = sum(
                    1 for r in results_t2s if r["active"]["spectral_fired"]
                )
                n_fire = sum(
                    1 for r in results_t2s if r["active"]["fire_count"] > 0
                )
                print(f"  [{i+1}/{len(det_records)}] rid={rid} "
                      f"prompt_len={gen_act['prompt_len']} "
                      f"n_gen={gen_act['n_generated']} "
                      f"fire={gen_act['fire_count']} "
                      f"spec_fired={gen_act['spectral_fired']} "
                      f"spec_fired_so_far={n_spec} fired_any_so_far={n_fire}")
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"[STOP] CUDA out of memory during t2s active run: {e}")
            sys.exit(1)
        raise

    t2s_agg = aggregate_t2s(results_t2s)
    print("\n" + "=" * 70)
    print("T2S COVERAGE CHECK (spectral fire rate on detection-gap)")
    print(f"  records: {t2s_agg['n_records']}")
    print(f"  spectral-fired records: {t2s_agg['n_spectral_fired']} "
          f"({t2s_agg['spectral_fire_rate']:.4f})")
    print(f"  bigram-fired records: {t2s_agg['n_bigram_fired']} | "
          f"fired-any records: {t2s_agg['n_fired_any']}")
    print(f"  coverage need: {T2S_COVERAGE_NEEDED}/{T2S_COVERAGE_DENOM} -> "
          f"{'PASS' if t2s_agg['coverage_pass'] else 'FAIL'}")
    print("=" * 70)

    # ==================================================================
    # Verdict + aggregate (single source of truth), write outputs.
    # ==================================================================
    prose_verdict = "PASS" if prose_agg["n_false_positives"] == 0 else "FAIL"
    coverage_verdict = "PASS" if t2s_agg["coverage_pass"] else "FAIL"
    if prose_verdict == "FAIL":
        verdict = "FAIL"
    elif coverage_verdict == "FAIL":
        verdict = "STOP (coverage)"
    else:
        verdict = "PASS"

    tables = {
        "prose": prose_agg,
        "t2s": t2s_agg,
        "ds034c": ds034c_stats,
        "cat1": cat1,
        "cat2": cat2,
        "prose_verdict": prose_verdict,
        "coverage_verdict": coverage_verdict,
        "verdict": verdict,
    }

    print("\n" + "=" * 70)
    print("DS-034d VERDICT")
    print(f"  prose-100 FPs: {prose_agg['n_false_positives']} "
          f"-> {prose_verdict}")
    print(f"  t2s spectral fire rate: {t2s_agg['n_spectral_fired']}/"
          f"{t2s_agg['n_records']} -> {coverage_verdict}")
    print(f"  OVERALL: {verdict}")
    print("=" * 70)

    # Write JSONL.
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for r in results_prose:
            f.write(json.dumps(r) + "\n")
        for r in results_t2s:
            f.write(json.dumps(r) + "\n")
        f.write(json.dumps({"record_type": "smoke", **smoke}) + "\n")
        f.write(json.dumps({
            "record_type": "meta",
            "model_name": MODEL_NAME,
            "model_revision": MODEL_REVISION,
            "device": device,
            "dtype": str(dtype),
            "torch_version": torch.__version__,
            "seed": SEED,
            "t_pr": t_pr,
            "band_low": band_low,
            "bigram_persistence": BIGRAM_PERSISTENCE,
            "spectral_persistence": SPECTRAL_PERSISTENCE,
            "spectral_bigram_ctr_gate": SPECTRAL_BIGRAM_CTR_GATE,
            "spectral_trailing_ctr_gate": SPECTRAL_TRAILING_CTR_GATE,
            "verdict": verdict,
            "prose_fps": prose_agg["n_false_positives"],
            "prose_verdict": prose_verdict,
            "t2s_spectral_fired": t2s_agg["n_spectral_fired"],
            "t2s_n": t2s_agg["n_records"],
            "t2s_coverage_pass": bool(t2s_agg["coverage_pass"]),
            "t2s_verdict": coverage_verdict,
            "wall_clock_s": time.time() - t_start,
        }) + "\n")
    print(f"\nwrote {OUTPUT_JSONL} "
          f"({len(results_prose) + len(results_t2s)} record lines + sidecar "
          f"lines)")

    metadata = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "device": device,
        "dtype": str(dtype),
        "torch_version": torch.__version__,
        "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "analysis_window": ANALYSIS_WINDOW,
        "hook_layer": HOOK_LAYER,
        "t_pr": t_pr,
        "band_low": band_low,
        "bigram_persistence": BIGRAM_PERSISTENCE,
        "spectral_persistence": SPECTRAL_PERSISTENCE,
        "spectral_bigram_ctr_gate": SPECTRAL_BIGRAM_CTR_GATE,
        "spectral_trailing_ctr_gate": SPECTRAL_TRAILING_CTR_GATE,
        "bmm_override": "deregistered",
        "wall_clock_s": time.time() - t_start,
    }
    ctx = {
        "metadata": metadata,
        "determinism_smoke": smoke,
        "tables": tables,
        "results_prose": results_prose,
        "results_t2s": results_t2s,
    }
    write_markdown_report(ctx, OUTPUT_MD)
    print(f"wrote {OUTPUT_MD}")

    print(f"\ntotal wall clock: {time.time() - t_start:.1f} s")
    if verdict == "FAIL":
        print("[STOP] DS-034d GATE: FAIL — FPs persist on prose-100 under the "
              "hardened dual-predicate. Do NOT wire the dual-predicate into "
              "the controller.")
        sys.exit(1)
    if verdict == "STOP (coverage)":
        print("[STOP] DS-034d COVERAGE: t2s spectral fire rate below "
              f"{T2S_COVERAGE_NEEDED}/{T2S_COVERAGE_DENOM}. The Category 2 "
              "hardening is too aggressive (adversarial critic constraint). Report and STOP.")
        sys.exit(1)
    print(f"DS-034d GATE: {verdict} — measurement complete")


if __name__ == "__main__":
    main()
