# DS-022 — Probe B Reconstruction Report (Part 1)

> Part 1 of DS-022. Strictly factual; every element is cited to `file:line`.
> Anything not recorded in the v2.4 Probe B record is marked **NOT IN RECORD**.
> No guessing: where an element is absent from the record but recoverable from
> the entry-point script, that cross-reference is stated explicitly as such.

## 1. Source record

- **File**: `the source smoke record`
- **Branch/date header**: `night/night-000-smoke`, 2026-08-04T11:32:33Z (`the source smoke record:3-4`)
- **Gate**: `make test` (`the source smoke record:5`)
- **Probe B block**: `the source smoke record:102-109`
- **Verbatim** (`the source smoke record:104-109`):
  ```text
  [Probe B] Word Loop
    Prompt: 'word word word word word word word word word word'
    Fired: 14 | Mean Δres: 0.6265 L2
    First Δres: 0.4638 L2 | Max Δres: 0.7861 L2
    Δres values: [0.4638, 0.4556, ... 0.7656]
    ✅ PASS: Word loop rescued with bounded residual force.
  ```

## 2. Element-by-element reconstruction

### 2.1 Model + revision

- **Model id**: **NOT IN RECORD**. `the source smoke record` never names the model.
- **Cross-referenced from the entry-point script**: `scripts/test_path2_code_filter.py:77` sets
  `model_name: str = "Qwen/Qwen2.5-1.5B"` (argparse default at `:213-214`).
- **Weight-count evidence**: every gate block in the record shows `Loading weights: .../338`
  (`the source smoke record:27-31`, `:55-59`, `:76-80`, `:118-122`). The
  `Qwen/Qwen2.5-1.5B` checkpoint cached at
  `~/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323/model.safetensors`
  contains exactly **338** weight tensors (verified by `safetensors.safe_open` key count). This is
  consistent with the model behind the record being Qwen2.5-1.5B.
- **Model revision**: **NOT IN RECORD**. The cached snapshot (HF default `main` ref) is
  `8faed761d45a263340a0528343f099c05c9a4323`.

### 2.2 Decoding params (do_sample / temperature / top_p)

- **NOT IN RECORD**. `the source smoke record` does not state decoding parameters.
- **Cross-referenced from the entry-point script** (`scripts/test_path2_code_filter.py:167-173`),
  the Probe B `governor.generate(...)` call passes:
  - `max_new_tokens=64`
  - `temperature=0.7`
  - `top_p=0.9`
  - `intervene=True`
  - `do_sample` not passed → defaults to `True` (`governor/controller.py:632`)

### 2.3 max_new_tokens / rollout length

- **NOT IN RECORD**. Cross-referenced: `max_new_tokens=64` (`scripts/test_path2_code_filter.py:169`).
- The record's Δres list contains **43** entries (`the source smoke record:108`), i.e. 43 residual
  injections were logged over the 64-token rollout.

### 2.4 Prompts / corpus

- **Prompt (Probe B)**: `'word word word word word word word word word word'`
  - Record: `the source smoke record:105`
  - Script literal: `scripts/test_path2_code_filter.py:27-30` (`WORD_LOOP_PROBE`)
- **Probe A (code-context) prompts** also run by the same gate script before Probe B:
  `scripts/test_path2_code_filter.py:21-25` (`CODE_CONTEXT_PROBES`).
- **Does the corpus/script still exist in the tree?** YES. `scripts/test_path2_code_filter.py` is
  present and unchanged since commit `a433d69` ("feat(track-b): Path 2 live validation + flag-on
  safety gate (phase 4)", 2026-08-03). The word-loop prompt is a hardcoded literal (not a separate
  corpus file), so it cannot go stale.

### 2.5 Governor configuration

- **NOT IN RECORD** (the record does not state governor arguments).
- **Cross-referenced from the entry-point script** (`scripts/test_path2_code_filter.py:35-42`):
  - `ActiveVarietyGovernor(model, tokenizer=tokenizer, use_code_filter=True)`
  - All other constructor args use defaults (`governor/controller.py:120-146`): `base_strength=0.45`,
    `max_strength=0.75`, `every_k=2`, `cooldown_steps=8`, `history_len=16`, `top_p=0.85`, etc.
- **Calibration text**: the trusted calibration sequence is
  `tokenizer("The quick brown fox jumps over the lazy dog.")` then `governor.calibrate(trusted)`
  (`scripts/test_path2_code_filter.py:40-41`).

### 2.6 Engage criteria / trigger code path

- **NOT IN RECORD** (the record prints results only). Cross-referenced from current
  `governor/controller.py` (the diagnosis/trigger path used by the entry-point script):
  - Dual-gate immunity reset: `trailing_ctr >= 0.75 and token_diversity >= 0.35` →
    clear state, no decision (`governor/controller.py:449-452`)
  - Path 2 structural code exclusion: `use_code_filter and is_code_context` → clear state, no
    decision (`governor/controller.py:455-458`)
  - Collapse predicate with code filter enabled: `is_token_loop = token_diversity < 0.25 or
    trailing_ctr < 0.20` (`governor/controller.py:461-464`)
  - Persistence hysteresis: fires only when `_collapse_persistence_counter >= 2`
    (`governor/controller.py:473-475`)
  - Layer selection: candidate layers at ≥55% depth; deepest selected
    (`governor/controller.py:486-490`)
  - Impulse strength: `min(max_strength, base_strength + prev_steps*0.05)`
    (`governor/controller.py:493`)
  - Injection clamps the applied L2 force to `[0.35, 0.75]` (`governor/controller.py:566`)
  - `generate()` triggers diagnosis on profile steps (`step % every_k == 0`) when
    `token_diversity < 0.35 or trailing_ctr < 0.35` (`governor/controller.py:704-705`), then calls
    `diagnose(...)` (`governor/controller.py:718-724`)

### 2.7 Seed

- **NOT IN RECORD**. Cross-referenced from the entry-point script:
  `set_seed(19)`, `torch.manual_seed(19)`, and `torch.cuda.manual_seed_all(19)`
  (`scripts/test_path2_code_filter.py:87-90`).
- The module docstring at `scripts/test_path2_code_filter.py:8` says "seed=1", but the executed
  code path uses **19** (the docstring is stale). The comment at `:82-86` explains 19 was chosen by
  sweeping seeds 0–49 on Qwen2.5-1.5B.

### 2.8 Device / dtype

- **NOT IN RECORD**. Cross-referenced: `device = "cuda" if torch.cuda.is_available() else "cpu"`;
  model dtype is `torch.bfloat16` on cuda, `torch.float32` on cpu
  (`scripts/test_path2_code_filter.py:92-98`).

### 2.9 Entry point

- **`scripts/test_path2_code_filter.py`** → `run_path2_validation()`:
  - Model load: `:77-98`
  - Probe A loop: `:105-150`
  - Probe B: `:155-200`
  - Exit code gate: `:202-208`
- The record's "Path 2 Structural Code Exclusion Filter Validation" header
  (`the source smoke record:74`) matches this script's printed banner
  (`scripts/test_path2_code_filter.py:78-80`).

### 2.10 Record shape (Δres list count)

- The task brief states "44 Δres values". The actual record
  (`the source smoke record:108`) contains **43** values. Direct parse of the record:
  count = 43, first = 0.4638, max = 0.7861, mean = 0.626453… → printed "0.6265".
  The record is self-consistent at 43 values.
- The record prints `Fired: 14` (the *decision* count, `len(out["decisions"])`) while the Δres list
  has 43 entries (the *intervention log* records). This 14→43 ratio is consistent with an
  intervention decision registering a residual hook that fires on multiple subsequent forward
  passes until cleared (`governor/controller.py:727-729`, `:731`).

## 3. Existence check — does each element still exist in current main?

| Element | Still exists? | Evidence |
|---|---|---|
| `scripts/test_path2_code_filter.py` (entry point) | **YES** | present; unchanged since `a433d69` (2026-08-03) |
| `WORD_LOOP_PROBE` prompt | **YES** | `scripts/test_path2_code_filter.py:27-30` |
| Probe A code-context prompts | **YES** | `scripts/test_path2_code_filter.py:21-25` |
| `ActiveVarietyGovernor` with `use_code_filter` | **YES** | `governor/controller.py:119-207`, `:139` |
| `diagnose()` gate logic (dual-gate, Path-2, hysteresis) | **YES** | `governor/controller.py:434-519` |
| `generate()` (autoregressive loop, intervention path) | **YES** | `governor/controller.py:628-798` |
| `_make_sae_guided_reset_fn` (L2 clamp [0.35,0.75]) | **YES** | `governor/controller.py:521-590`, `:566` |
| `is_code_syntax_context` (Path-2 detector) | **YES** | `core/metrics.py:33-40` |
| `compute_coherent_token_ratio` (CTR) | **YES** | `core/metrics.py:43-79` |
| `Qwen/Qwen2.5-1.5B` weights | **YES** | HF cache snapshot `8faed761...` |

The controller last changed by commit `2b49e4e` (DS-005, 2026-08-08), which is a **docstring/comment
only** change (`governor/controller.py` `_remove_shadow_hooks` docstring and a `finally`-block
comment); no functional change since the v2.4 record.

## 4. NOT IN RECORD items (never guessed)

The v2.4 record (`the source smoke record`) does **not** state any of:
- model id / revision
- seed
- temperature / top_p / max_new_tokens / do_sample
- device / dtype
- governor constructor arguments / calibration text
- engage criteria

All of the above were recovered by cross-referencing the entry-point script
(`scripts/test_path2_code_filter.py`), which is the exact script whose banner the record prints.
Where the record and script disagree (e.g., docstring says seed=1 vs code uses seed=19), the
executed code path is cited as authoritative.

## 5. Reconstruction sufficiency

All four required elements — **model** (§2.1), **prompts** (§2.4), **governor config** (§2.5), and
**entry point** (§2.9) — are identified with file:line citations. The re-run therefore proceeded
to Part 2 (positive control), documented in `docs/PROBE_B_POSITIVE_CONTROL.md`.

## 6. Environment notes for the re-run

- The re-run used `torch 2.13.0+cu130`, `transformers 4.57.6`, NVIDIA RTX 3060 (cuda).
- torch 2.13's native `bmm_outer_product` override for `aten::bmm` on CUDA requires a Triton
  JIT C compiler that is absent from this container. The override was **deregistered** for the
  re-run (falling back to eager `torch.bmm`). This is an environment-only adaptation; no protected
  file was modified.
- HF hub emitted a benign read-only warning ("Could not cache non-existence of file ... Read-only
  file system") during model load; it did not affect the run.
