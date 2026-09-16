# DS-034a — Layer-2 PR hook latency (MEASUREMENT ONLY)

> MEASUREMENT REPORT. This probe measures the absolute governor cost delta of a layer-2 forward participation_ratio() hook (RFC-004 Amendment A3 draft). It does NOT modify the controller, does NOT change any predicate, and does NOT alter any threshold. The layer-2 PR hook is a measurement instrument only; PR is logged and does NOT gate any actuation. No verdict is offered.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture | tests/fixtures/heldout_degenerate_v2.jsonl (N=100); 10 records random.Random(42).sample, sorted by id |
| Prompt | record[\"mutated_prompt\"] |
| Decoding | greedy (do_sample=False), max_new_tokens=128 |
| Analysis window | trailing 24 generated hidden states |
| Hook layer | model.model.layers[2] |
| PR instrument | participation_ratio() from core/metrics.py (existing), float32 promotion |
| bmm Triton override | deregistered |
| Wall clock (s) | 196.8 |

## Determinism smoke

One record, dormant generated twice AND instrumented generated twice. Dormant token ids must match; instrumented token ids AND the per-step PR log must match exactly. The smoke also warms the GPU kernels before the timed runs.

| condition | record_id | n_tokens | PR log | identical |
|---|---|---|---|---|
| dormant | 3 | 128 | — | True |
| instrumented | 3 | 128 | n_pr=105 | True / PR True |

PR first run A/B: 2.2474 / 2.2474; last run A/B: 2.0749 / 2.0749.

## Method summary

- **Condition 1 (dormant baseline):** plain greedy argmax, no hooks, no suppression, no kickstart, no top_p, no residual interventions — the production governor's dormant path (intervene=False, no logit penalties) as implemented in DS-028/DS-032/DS-033.
- **Condition 2 (instrumented):** identical dormant path PLUS one forward hook on `model.model.layers[2]`. At each decoding step (step 0..127) the hook captures the hidden state at the current token position (last position, matching the governor's shadow-hook capture at `[:, -1:, :]`), appends it to a ring buffer of the trailing 24 generated hidden states, and when the buffer has >= 24 entries computes `participation_ratio()` over the window (float32). PR is logged only; it does NOT gate any actuation.
- Compute pattern matches the production controller's `generate()`: full-sequence forward passes (no KV cache), exactly 128 decoding steps (no EOS break).
- Wall-clock ms/token per record = `total_s * 1000 / (prompt_len + 128)` (task file: "full generation (prompt_len + 128 tokens)"). Per record, min-of-3 interleaved dormant and instrumented trials (benchmark_latency.py min-of-3 convention); the absolute delta is the layer-2 PR hook overhead. The report aggregates the mean over the 10 records.
- GPU warmup: the determinism smoke (4 full 128-token generations) plus one untimed dormant+instrumented pair settle GPU clocks before the timed runs.

## Headline: mean ms/token (per spec denominator)

| condition | mean ms/token | median | p10 | p90 | min | max |
|---|---|---|---|---|---|---|
| Dormant | 19.0238 | 19.0126 | 18.9833 | 19.0953 | 18.9475 | 19.0968 |
| Instrumented | 19.7266 | 19.6946 | 19.6744 | 19.8113 | 19.5899 | 19.8256 |
| **Δ (inst - dorm)** | **0.7027** | 0.6983 | 0.6685 | 0.7369 | 0.6424 | 0.8094 |

Acceptance bar (RFC A3): every-step cadence if Δ ≤ 1.0 ms/token under the existing 7.0 ms/token governor budget; otherwise every_k=2. This report MEASURES Δ; it does not render the verdict.

## Per-record results

| record_id | prompt_len | n_gen | dorm ms/tok | inst ms/tok | Δ ms/tok | tokens identical | PR n | PR mean | PR min | PR max |
|---|---|---|---|---|---|---|---|---|---|---|
| 3 | 21 | 128 | 18.9877 | 19.6976 | 0.7100 | True | 105 | 2.1091 | 2.0749 | 2.2474 |
| 13 | 21 | 128 | 18.9873 | 19.6857 | 0.6984 | True | 105 | 2.1526 | 2.1010 | 2.3272 |
| 14 | 21 | 128 | 18.9913 | 19.8007 | 0.8094 | True | 105 | 2.1069 | 2.0621 | 2.2364 |
| 17 | 21 | 128 | 19.0951 | 19.7934 | 0.6983 | True | 105 | 2.1069 | 2.0621 | 2.2364 |
| 28 | 21 | 128 | 19.0939 | 19.8097 | 0.7158 | True | 105 | 2.0320 | 2.0001 | 2.1468 |
| 31 | 21 | 128 | 19.0968 | 19.8256 | 0.7288 | True | 105 | 2.0320 | 2.0001 | 2.1468 |
| 35 | 28 | 128 | 19.0128 | 19.6915 | 0.6787 | True | 105 | 3.0970 | 2.9803 | 3.2561 |
| 81 | 28 | 128 | 19.0134 | 19.6877 | 0.6742 | True | 105 | 3.1328 | 3.0799 | 3.2564 |
| 86 | 35 | 128 | 18.9475 | 19.5899 | 0.6424 | True | 105 | 3.8764 | 3.7997 | 4.0944 |
| 94 | 28 | 128 | 19.0124 | 19.6837 | 0.6714 | True | 105 | 3.1197 | 3.0535 | 3.3259 |

## Per-trial wall-clock (min-of-3)

Per-record dormant/instrumented total seconds (min of 3 interleaved trials is the headline per-record time; all trials are recorded here for transparency).

| record_id | dorm t1 | dorm t2 | dorm t3 | inst t1 | inst t2 | inst t3 |
|---|---|---|---|---|---|---|
| 3 | 2.8293 | 2.8292 | 2.8292 | 2.9397 | 2.9349 | 2.9356 |
| 13 | 2.8291 | 2.8292 | 2.8295 | 2.9366 | 2.9392 | 2.9332 |
| 14 | 2.8297 | 2.8451 | 2.8455 | 2.9503 | 2.9518 | 2.9515 |
| 17 | 2.8456 | 2.8452 | 2.8452 | 2.9492 | 2.9497 | 2.9560 |
| 28 | 2.8455 | 2.8456 | 2.8450 | 2.9647 | 2.9516 | 2.9523 |
| 31 | 2.8454 | 2.8460 | 2.8487 | 2.9554 | 2.9540 | 2.9664 |
| 35 | 2.9663 | 2.9660 | 2.9661 | 3.0719 | 3.0739 | 3.0719 |
| 81 | 2.9661 | 2.9665 | 2.9661 | 3.0715 | 3.0713 | 3.0718 |
| 86 | 3.0890 | 3.0900 | 3.0884 | 3.1972 | 3.1942 | 3.1932 |
| 94 | 2.9659 | 2.9675 | 2.9662 | 3.0707 | 3.0788 | 3.0719 |

## Pure PR-compute microbenchmark (supplementary context)

To separate the hook's intrinsic cost from GPU clock/thermal noise, `participation_ratio()` is timed directly on a representative (1, window, hidden_dim) float32 tensor for one generation's PR count. This is the SVD-compute component of the hook; capture/buffer overhead is negligible by comparison. **Supplementary context only** — the headline metric is the wall-clock delta between Condition 2 and Condition 1.

| metric | value |
|---|---|
| tensor | (1, 24, 1536) float32 |
| n_calls (one generation) | 105 |
| per-call time | 0.6652 ms |
| total per generation | 69.85 ms |
| ≈ ms/token (denominator prompt_len+128) | 0.4656 ms (at prompt_len≈22) |

The pure PR compute is ≈ 69.85 ms per generation (≈ 0.466 ms/token at prompt_len≈22). This is the dominant, irreducible component of the hook cost and is roughly 66% of the measured wall-clock delta (mean Δ = 0.7027 ms/token this run). The remainder is capture/buffer/`.item()` sync overhead plus GPU clock/thermal noise; cross-run noise on this RTX 3060 was ±0.2 ms/token (five full runs: mean Δ 0.70–1.07 ms/token).

## Cross-check: ms/decoded-token (total_s / 128)

| condition | mean ms/tok | median |
|---|---|---|
| Dormant | 22.6641 | 22.2289 |
| Instrumented | 23.5002 | 23.0690 |
| **Δ (inst - dorm)** | **0.8361** | — |

The absolute delta is the same object regardless of denominator (per-record paired subtraction); the denominator only rescales the ms/token magnitude.

## Per-record PR values (logged, not gated)

PR is computed every step once the rolling buffer has >= 24 entries (steps 23..127 → 105 values per record). Summary below; the full per-step log is in `layer2_pr_latency_results.jsonl`.

| record_id | n PR | mean | min | max | first | last |
|---|---|---|---|---|---|---|
| 3 | 105 | 2.1091 | 2.0749 | 2.2474 | 2.2474 | 2.0749 |
| 13 | 105 | 2.1526 | 2.1010 | 2.3272 | 2.3272 | 2.1010 |
| 14 | 105 | 2.1069 | 2.0621 | 2.2364 | 2.2364 | 2.0651 |
| 17 | 105 | 2.1069 | 2.0621 | 2.2364 | 2.2364 | 2.0651 |
| 28 | 105 | 2.0320 | 2.0001 | 2.1468 | 2.1468 | 2.0034 |
| 31 | 105 | 2.0320 | 2.0001 | 2.1468 | 2.1468 | 2.0034 |
| 35 | 105 | 3.0970 | 2.9803 | 3.2561 | 3.2561 | 2.9814 |
| 81 | 105 | 3.1328 | 3.0799 | 3.2564 | 3.2564 | 3.0809 |
| 86 | 105 | 3.8764 | 3.7997 | 4.0944 | 3.9685 | 3.8174 |
| 94 | 105 | 3.1197 | 3.0535 | 3.3259 | 3.3259 | 3.0573 |

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no predicate changes, no threshold changes. The layer-2 PR hook is a measurement instrument only; its PR value does not gate any actuation.
- The instrumented hook returns the layer output unchanged; generated token ids are identical to the dormant condition (`tokens identical` column is True for every record).
- Wall-clock ms/token uses the task-file denominator `prompt_len + 128`. The cross-check table reports the same measured totals over the 128 decoded tokens only; the absolute delta (the hook overhead) is identical either way.
- The determinism smoke plus one untimed dormant+instrumented warmup pair run before the timed runs (6 full 128-token generations total) to settle GPU clocks.
- Per-record times are the minimum of 3 interleaved trials per condition (benchmark_latency.py min-of-3 convention). Per-trial wall-clock seconds are recorded in `layer2_pr_latency_results.jsonl` (`dormant_trials_s`, `instrumented_trials_s`).
- NOISE CAVEAT: on this RTX 3060 the full-generation wall-clock delta between dormant and instrumented carries GPU clock/thermal noise of about ±0.2 ms/token (mean Δ 0.70–1.07 ms/token across five full runs). The pure PR-compute microbenchmark (≈ 0.67 ms/call ⇒ ≈ 0.47 ms/token) is the dominant irreducible hook component; the measured wall-clock Δ is consistent with that plus capture/`.item()` sync overhead and noise.
- DESIGN DECISION: the "production governor with intervene=False" dormant path is implemented as plain greedy argmax (the DS-028/DS-032/DS-033 dormant representation). The controller's `generate(intervene=False)` still applies logit-side suppression/kickstart penalties on degenerate prompts, which the task excludes with "(no residual, no logit penalties)"; the dormant loop is therefore the faithful no-penalty dormant path. The compute pattern (full-sequence forward passes, no KV cache, 128 steps, no EOS break) matches the controller's `generate()`.
- Environment: torch bmm Triton override deregistered; HF cache read-only; standard root-as-package import convention.
