# DS-022 — Probe B Positive Control (Part 2)

> MEASUREMENT ONLY — no thresholds changed, no gate logic touched, no
> `governor/controller.py` edits, no asserts on outcomes, **NO verdict**.
> Interpretation (fires = intervention path live; zero = instrument changed)
> is for human review and humans.

## 1. Purpose

Positive control: re-run Probe B's reconstructed v2.4 conditions
(`docs/PROBE_B_RECONSTRUCTION.md`) in the current build, UNCHANGED, to
measure whether the governor intervention path can still fire. The v2.4
record (`the source smoke record:106-108`) reported `Fired: 14 | Mean Δres:
0.6265 L2`, `First Δres: 0.4638 L2`, `Max Δres: 0.7861 L2`.

## 2. Run metadata

| Field | Run 1 (faithful v2.4) | Run 2 (task-designated) |
|-------|------------------------|--------------------------|
| Model | `Qwen/Qwen2.5-1.5B` | `Qwen/Qwen2.5-1.5B` |
| Model revision | `8faed761d45a263340a0528343f099c05c9a4323` | `8faed761d45a263340a0528343f099c05c9a4323` |
| Device | `cuda` (RTX 3060) | `cuda` (RTX 3060) |
| dtype | `torch.bfloat16` | `torch.float16` |
| SEED | `19` (gate script's hardcoded seed) | `42` (task: "SEED 42 where seedable") |
| Prompt | `'word word word word word word word word word word'` | same |
| Governor | `ActiveVarietyGovernor(use_code_filter=True)`, calibrated on fox text | same |
| Decoding | `max_new_tokens=64, temperature=0.7, top_p=0.9, do_sample=True, intervene=True` | same |
| Probe-A RNG prefix | yes (3 generations before Probe B, replicating gate script) | yes (same structure) |
| Entry point | `scripts/test_path2_code_filter.py` run as-is (env-only bmm fix) | measurement harness (same Probe-B conditions) |

## 3. Results

### 3.1 Run 1 — seed 19, bfloat16 (faithful v2.4 reproduction)

- **Fired: 14** | **Mean Δres: 0.6265 L2**
- **First Δres: 0.4638 L2** | **Max Δres: 0.7861 L2**
- Δres count: **43**
- **Match vs v2.4 record**: **EXACT** — all 43 Δres values identical
  (byte-for-byte) to `the source smoke record:108`; `Fired`, `Mean`, `First`,
  `Max` all match the record.

### 3.2 Run 2 — seed 42, fp16 (task-designated control)

- **Fired: 1** | **Mean Δres: 0.4514 L2**
- **First Δres: 0.4513 L2** | **Max Δres: 0.4544 L2**
- Δres count: **62** (1 intervention decision → 62 residual injections
  logged; the residual hook persists across steps — see
  `governor/controller.py:727-729`, `:731`)
- All 62 values lie in `[0.4492, 0.4544]` L2.

## 4. Gate output (verbatim)

The faithful v2.4 entry point (`scripts/test_path2_code_filter.py`, run via a
wrapper that only deregisters torch 2.13's native `bmm` Triton override —
no protected file touched) produced:

```text
======================================================================
AVG Path 2 Structural Code Exclusion Filter Validation
======================================================================

--- Probe A: Code-Context Repetition Immunity ---

[Probe A] C++ Comment Header
  Prompt: '// using\n// array\n// '
  Fired: 0 | Mean Δres: 0.0000 L2
  Code-context steps: 3
  ✅ PASS: Code context suppressed residual steering.

[Probe A] Markdown Fence
  Prompt: '```python\nx = 1\n```\n'
  Fired: 0 | Mean Δres: 0.0000 L2
  Code-context steps: 1
  ✅ PASS: Code context suppressed residual steering.

[Probe A] Jupyter Cell Break
  Prompt: '# %%\n# %%\n# %%'
  Fired: 0 | Mean Δres: 0.0000 L2
  Code-context steps: 4
  ✅ PASS: Code context suppressed residual steering.

--- Probe B: Natural-Language Word-Loop Rescue ---

[Probe B] Word Loop
  Prompt: 'word word word word word word word word word word'
  Fired: 14 | Mean Δres: 0.6265 L2
  First Δres: 0.4638 L2 | Max Δres: 0.7861 L2
  Δres values: [0.4638, 0.4556, 0.4722, 0.5171, 0.5141, 0.5173, 0.5762, 0.5698, 0.5704, 0.6294, 0.6232, 0.6369, 0.668, 0.664, 0.6647, 0.7252, 0.7336, 0.7237, 0.4656, 0.4613, 0.4666, 0.5289, 0.5254, 0.5262, 0.5706, 0.5719, 0.5736, 0.6168, 0.6247, 0.6282, 0.6784, 0.6876, 0.6791, 0.7346, 0.7298, 0.7311, 0.7861, 0.7665, 0.7769, 0.7759, 0.7655, 0.7754, 0.7656]
  ✅ PASS: Word loop rescued with bounded residual force.

======================================================================
✅ [CI PASS] Path 2 validation passed.
```

The full Δres list is byte-identical to the v2.4 record (`the source smoke record:108`).
A benign HF read-only-cache warning preceded the run and is reproduced in the
run log; it did not affect the measurement.

## 5. Strictly factual observations

The following statements report measured values only. No verdict is offered;
human review and humans interpret.

- The intervention path **fired** in both runs: 14 decisions / 43 residual
  injections (seed 19, bfloat16) and 1 decision / 62 residual injections
  (seed 42, fp16).
- The seed-19 / bfloat16 run reproduces the v2.4 Probe B record exactly,
  including all 43 Δres values.
- The seed-42 / fp16 run also fires, with a single intervention decision
  whose 62 residual injections all lie within `[0.4492, 0.4544]` L2
  (within the surgical band `[0.35, 0.75]`).
- Probe A (code-context immunity) measured 0 fires / 0.00 L2 on all three
  code prompts in the faithful run, matching the v2.4 record.

## 6. Data

- `docs/probe_b_positive_control_results.jsonl` — 2 records (seed 19 bf16,
  seed 42 fp16), each with the full Δres list and provenance.

## 7. Environment / provenance

- torch `2.13.0+cu130`; transformers `4.57.6`; NVIDIA RTX 3060 (cuda).
- torch 2.13 native `bmm_outer_product` Triton override deregistered for the
  run (C compiler absent); falls back to eager `torch.bmm`. No protected file
  modified; `git status` contains only the DS-022 deliverables.
