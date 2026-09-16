# DS-036 — Temporal Hardening Sweep (Option B) — MEASUREMENT REPORT

> MEASUREMENT REPORT (sampling-relative recalibration, owner-directed). Not a
> ratification. Measures whether temporal hardening (spectral persistence >= 4,
> or step >= 30) suppresses prose FPs without clipping recall, under the
> sampling regime.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B @ 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda (RTX 3060 12GB) |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Regime | sampling (do_sample=True, temp=0.8, top_p=0.85) |
| Arms | baseline (DS-034e), P4 (spectral persistence >= 4), S30 (step >= 30) |
| Fixtures | prose-100 (heldout partition), t2s detection-gap 93 (recall) |
| band_low | 8.216097 (frozen) |

## Dispatch gate

- DS-035 (post-B2) sampling rescue: t2s 97/100, qwen 97/100 (bar 85/100 MET) -> DS-036 dispatchable.
- DS-033 detection-gap classification: reproducible from `docs/gate23/production_cross_fixture_results.jsonl` (`diagnostic_class == "detection-gap"`, 93 records).

## Determinism + liveness

- Determinism smoke (baseline, greedy, t2s record 0, 2 runs): token-identical **True**, `_layer2_pr` identical **True**.
- Spectral liveness (`_layer2_pr` non-None): **True**.

## Results (sampling regime)

| arm | prose spectral FP | prose bigram FP | t2s recall (93) |
|---|---|---|---|
| baseline (DS-034e) | 0/100 | 0/100 | 39/93 (0.4194) |
| P4 (persistence >= 4) | 0/100 | 0/100 | 38/93 (0.4086) |
| S30 (step >= 30) | 0/100 | 0/100 | 33/93 (0.3548) |

## Verdict (recalibrated to sampling-relative acceptance)

- **5a (FP):** sampling baseline is already **0/100** — the "2 late-dip FPs"
  (DS-034c/d/e records 98, 99) do NOT occur under sampling. Neither arm
  changes FP (0 -> 0). No FP benefit available.
- **5b (recall):** baseline 39/93. P4 clips 1 (-> 38/93, -2.6% relative);
  S30 clips 6 (-> 33/93, -15.4% relative).

**Neither arm is a ratification candidate.** Temporal hardening under sampling
is pure cost: it cannot reduce FPs (already 0) and it clips real recall
(S30 materially, P4 marginally). S30's step floor suppresses early spectral
fires on the loops that DO still collapse under sampling — exactly the
"short macro-loop" recall tax independent review's critique warned about.

## Key finding

The two spectral late-dip FPs that motivated DS-036 are a **greedy-only
artifact**. Under sampling, healthy prose never drives the layer-2 PR below
band_low (the stochastic regime keeps the representation diverse at late
steps), so the spectral arm fires on 0/100 prose records. Temporal hardening
is therefore a solution to a problem that does not exist in the product
(sampling) regime — and it costs recall. Recommend: leave the DS-034e rule
unhardened under sampling; revisit temporal hardening only if a greedy-deploy
FP profile is ever targeted.

## Notes

- prose-100 prompts truncated to 512 tokens (full `text` fields run to 7,265
  tokens and OOM the 12GB GPU); the spectral fire signal is within the first
  ~24 generated tokens, so truncation does not mask a fire.
- Step tracking is over non-numeric `_evaluate_collapse` calls (exact for t2s,
  approximate for prose; the 30/120 step thresholds are coarse).
- No controller diff landed; no threshold changed. Measurement harness:
  `scripts/measure_ds036_temporal_hardening.py` (subclass override of
  `_evaluate_collapse`).
