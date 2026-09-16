# night-010 — Live-generation ECS on t2s detection-gap records (MEASUREMENT ONLY)

> MEASUREMENT REPORT. This probe measures per-head ECS during LIVE autoregressive generation (KV-cache, `output_attentions=True`) on the 10 t2s_degenerate detection-gap records. It compares the live per-head ECS to the DS-038 teacher-forced ECS reference (`docs/gate23/ecs_pks_results.jsonl`). PURELY DIAGNOSTIC: no thresholds, no actuation, no controller change. No verdict is offered.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture | tests/fixtures/t2s_degenerate.jsonl (night-001; N=100), first 10 detection-gap records sorted by record_id from the DS-033 classification (docs/gate23/production_cross_fixture_results.jsonl); record[\"text\"] as prompt |
| Records | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (detection-gap; predicate_true_steps == 0) |
| Decoding | greedy (do_sample=False), KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 generated positions (PR / distinct-2) |
| Hook layer | model.model.layers[2] (layer-2 PR, rolling 24-token ring buffer) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| Dual-predicate | DS-034e frozen config (night-008): bigram OR spectral, band_low, >=2 consecutive, token_diversity < 0.40 corroboration on spectral-only fires, code-context immunity |
| ECS candidate layers | {2, 6, 14, 21, 27} |
| Attention heads / layer | 12 (real Qwen2.5-1.5B config; DS-038 confirmed) |
| ECS prompt boundary | prompt_len = tokenized length of record[\"text\"] (genuine prompt/generation split tracked by the controller; NO heuristic) |
| DS-038 reference | REUSED from docs/gate23/ecs_pks_results.jsonl (teacher-forced; prompt_len = seq_len - 24). Do NOT re-run. |
| Attention capture | `output_attentions=True` at every decoding step; per-step tensor is (num_heads x current_seq_len) at ~6KB per layer (128 tokens x 12 heads). |
| bmm Triton override | deregistered |
| Wall clock (s) | 35.4 |

## Determinism smoke (FIRST)

One detection-gap record (record_id 0), generated twice. Token ids must match. Per-step per-head ECS values (5 layers x 12 heads x 128 steps = 7680 values per run) must match to 6 decimal places. STOP if not.

| check | value |
|---|---|
| record_id | 0 |
| prompt_len | 199 |
| n_generated run A / B | 128 / 128 |
| token_ids_identical | True |
| per-step per-head ECS checked | 7680 |
| ECS values differing >= 1e-6 | 0 |
| identical (STOP gate) | True |

First-step ECS L21H3 run A/B: 0.993862 / 0.993862; last-step A/B: 0.622452 / 0.622452.

## Primary diagnostic table — live ECS vs DS-038 teacher-forced ECS (per record, per identified head)

Live mean = mean of per-step ECS across the decoding steps. DS-038 tf = teacher-forced ECS reference from `ecs_pks_results.jsonl` (prompt_len = seq_len - 24). delta = live - tf.

### ECS_L21H3 (layer/head 21/3)

| record_id | ECS live mean | ECS DS-038 tf | delta |
|---|---|---|---|
| 0 | 0.634749 | 0.758377 | -0.123627 |
| 1 | 0.733022 | 0.840400 | -0.107378 |
| 2 | 0.826938 | 0.943695 | -0.116757 |
| 3 | 0.770074 | 0.854563 | -0.084489 |
| 4 | 0.683704 | 0.790803 | -0.107099 |
| 5 | 0.637280 | 0.758348 | -0.121067 |
| 6 | 0.731344 | 0.850016 | -0.118672 |
| 7 | 0.714351 | 0.843338 | -0.128987 |
| 8 | 0.713992 | 0.818559 | -0.104567 |
| 9 | 0.777452 | 0.872746 | -0.095294 |
| mean | 0.722291 | 0.833084 | -0.110794 |

Pearson r(live, tf) = 0.9756; mean |delta| = 0.110794.

### ECS_L6H6 (layer/head 6/6)

| record_id | ECS live mean | ECS DS-038 tf | delta |
|---|---|---|---|
| 0 | 0.175484 | 0.525986 | -0.350502 |
| 1 | 0.207098 | 0.519624 | -0.312526 |
| 2 | 0.131028 | 0.350510 | -0.219482 |
| 3 | 0.099219 | 0.369996 | -0.270777 |
| 4 | 0.286630 | 0.598260 | -0.311630 |
| 5 | 0.169241 | 0.508826 | -0.339585 |
| 6 | 0.175069 | 0.521357 | -0.346287 |
| 7 | 0.150092 | 0.474314 | -0.324221 |
| 8 | 0.177237 | 0.477974 | -0.300737 |
| 9 | 0.166225 | 0.417192 | -0.250967 |
| mean | 0.173732 | 0.476404 | -0.302671 |

Pearson r(live, tf) = 0.8522; mean |delta| = 0.302671.

### ECS_L2H1 (layer/head 2/1)

| record_id | ECS live mean | ECS DS-038 tf | delta |
|---|---|---|---|
| 0 | 0.827719 | 0.898694 | -0.070975 |
| 1 | 0.902977 | 0.935904 | -0.032927 |
| 2 | 0.798936 | 0.913675 | -0.114738 |
| 3 | 0.783099 | 0.827591 | -0.044492 |
| 4 | 0.923080 | 0.956711 | -0.033631 |
| 5 | 0.852539 | 0.905930 | -0.053391 |
| 6 | 0.876062 | 0.919970 | -0.043908 |
| 7 | 0.921340 | 0.938304 | -0.016963 |
| 8 | 0.909834 | 0.935094 | -0.025260 |
| 9 | 0.910523 | 0.931782 | -0.021259 |
| mean | 0.870611 | 0.916365 | -0.045754 |

Pearson r(live, tf) = 0.8461; mean |delta| = 0.045754.

## Per-head ECS stability (10 records x 128 steps)

Distribution of per-step per-head ECS across ALL decoding steps of all 10 records (up to 1280 values per head).

| layer | head | mean | median | p10 | p90 | std | n |
|---|---|---|---|---|---|---|---|
| 2 | 0 | 0.6881 | 0.7275 | 0.4456 | 0.8776 | 0.1680 | 1280 |
| 2 | 1 | 0.8706 | 0.8906 | 0.7681 | 0.9447 | 0.0792 | 1280 |
| 2 | 2 | 0.7822 | 0.7867 | 0.6303 | 0.9379 | 0.1137 | 1280 |
| 2 | 3 | 0.6927 | 0.6687 | 0.5013 | 0.9342 | 0.1591 | 1280 |
| 2 | 4 | 0.5585 | 0.5543 | 0.3266 | 0.7866 | 0.1754 | 1280 |
| 2 | 5 | 0.6096 | 0.5833 | 0.3626 | 0.9180 | 0.1979 | 1280 |
| 2 | 6 | 0.6884 | 0.6965 | 0.5076 | 0.8555 | 0.1251 | 1280 |
| 2 | 7 | 0.4860 | 0.5067 | 0.3127 | 0.6583 | 0.1498 | 1280 |
| 2 | 8 | 0.3044 | 0.2759 | 0.1252 | 0.4867 | 0.1667 | 1280 |
| 2 | 9 | 0.5405 | 0.5243 | 0.3245 | 0.7776 | 0.1690 | 1280 |
| 2 | 10 | 0.4527 | 0.4363 | 0.2211 | 0.6795 | 0.1701 | 1280 |
| 2 | 11 | 0.6840 | 0.7259 | 0.4178 | 0.8678 | 0.1766 | 1280 |
| 6 | 0 | 0.6206 | 0.6121 | 0.4319 | 0.8155 | 0.1436 | 1280 |
| 6 | 1 | 0.8996 | 0.9290 | 0.7608 | 0.9917 | 0.0930 | 1280 |
| 6 | 2 | 0.5573 | 0.5277 | 0.4198 | 0.7349 | 0.1265 | 1280 |
| 6 | 3 | 0.9284 | 0.9397 | 0.8592 | 0.9856 | 0.0558 | 1280 |
| 6 | 4 | 0.5870 | 0.6006 | 0.4532 | 0.7005 | 0.1032 | 1280 |
| 6 | 5 | 0.5697 | 0.5802 | 0.3867 | 0.7208 | 0.1271 | 1280 |
| 6 | 6 | 0.1737 | 0.1004 | 0.0306 | 0.3979 | 0.2151 | 1280 |
| 6 | 7 | 0.9321 | 0.9633 | 0.8134 | 0.9996 | 0.0768 | 1280 |
| 6 | 8 | 0.5849 | 0.6128 | 0.3283 | 0.7951 | 0.1808 | 1280 |
| 6 | 9 | 0.9958 | 0.9975 | 0.9899 | 0.9996 | 0.0049 | 1280 |
| 6 | 10 | 0.3101 | 0.2916 | 0.1808 | 0.4715 | 0.1198 | 1280 |
| 6 | 11 | 0.7374 | 0.7722 | 0.4968 | 0.9144 | 0.1560 | 1280 |
| 14 | 0 | 0.2813 | 0.2097 | 0.1020 | 0.5897 | 0.2167 | 1280 |
| 14 | 1 | 0.3789 | 0.3316 | 0.1387 | 0.7280 | 0.2293 | 1280 |
| 14 | 2 | 0.7108 | 0.6987 | 0.5267 | 0.9261 | 0.1454 | 1280 |
| 14 | 3 | 0.8716 | 0.8995 | 0.7062 | 0.9946 | 0.1162 | 1280 |
| 14 | 4 | 0.8595 | 0.8760 | 0.6956 | 0.9935 | 0.1142 | 1280 |
| 14 | 5 | 0.9877 | 0.9920 | 0.9708 | 0.9988 | 0.0127 | 1280 |
| 14 | 6 | 0.7647 | 0.7707 | 0.5766 | 0.9492 | 0.1415 | 1280 |
| 14 | 7 | 0.5206 | 0.5245 | 0.3426 | 0.6642 | 0.1202 | 1280 |
| 14 | 8 | 0.4823 | 0.4798 | 0.3180 | 0.6165 | 0.1234 | 1280 |
| 14 | 9 | 0.1485 | 0.1389 | 0.0910 | 0.2012 | 0.0613 | 1280 |
| 14 | 10 | 0.5822 | 0.5528 | 0.4321 | 0.7948 | 0.1430 | 1280 |
| 14 | 11 | 0.7086 | 0.7086 | 0.5167 | 0.8980 | 0.1403 | 1280 |
| 21 | 0 | 0.9618 | 0.9636 | 0.9349 | 0.9862 | 0.0207 | 1280 |
| 21 | 1 | 0.8258 | 0.8266 | 0.7492 | 0.9046 | 0.0614 | 1280 |
| 21 | 2 | 0.7916 | 0.7937 | 0.6628 | 0.9198 | 0.1002 | 1280 |
| 21 | 3 | 0.7223 | 0.7200 | 0.5637 | 0.8814 | 0.1174 | 1280 |
| 21 | 4 | 0.8658 | 0.8696 | 0.7659 | 0.9589 | 0.0729 | 1280 |
| 21 | 5 | 0.9530 | 0.9570 | 0.9147 | 0.9875 | 0.0275 | 1280 |
| 21 | 6 | 0.8969 | 0.8964 | 0.8366 | 0.9695 | 0.0528 | 1280 |
| 21 | 7 | 0.8441 | 0.8594 | 0.7442 | 0.9234 | 0.0712 | 1280 |
| 21 | 8 | 0.8311 | 0.8379 | 0.7363 | 0.9223 | 0.0751 | 1280 |
| 21 | 9 | 0.7834 | 0.7878 | 0.6691 | 0.9005 | 0.0901 | 1280 |
| 21 | 10 | 0.7281 | 0.7274 | 0.6437 | 0.8133 | 0.0764 | 1280 |
| 21 | 11 | 0.9050 | 0.9051 | 0.8516 | 0.9724 | 0.0516 | 1280 |
| 27 | 0 | 0.7592 | 0.7576 | 0.6043 | 0.9269 | 0.1228 | 1280 |
| 27 | 1 | 0.7917 | 0.7879 | 0.6634 | 0.9324 | 0.0990 | 1280 |
| 27 | 2 | 0.8204 | 0.8214 | 0.7073 | 0.9423 | 0.0863 | 1280 |
| 27 | 3 | 0.7800 | 0.7766 | 0.6395 | 0.9313 | 0.1052 | 1280 |
| 27 | 4 | 0.7900 | 0.7958 | 0.6413 | 0.9364 | 0.1071 | 1280 |
| 27 | 5 | 0.7644 | 0.7761 | 0.5915 | 0.9311 | 0.1269 | 1280 |
| 27 | 6 | 0.7404 | 0.7349 | 0.5931 | 0.9224 | 0.1231 | 1280 |
| 27 | 7 | 0.7726 | 0.7765 | 0.6040 | 0.9411 | 0.1263 | 1280 |
| 27 | 8 | 0.7320 | 0.7215 | 0.5773 | 0.9162 | 0.1254 | 1280 |
| 27 | 9 | 0.5865 | 0.5819 | 0.4345 | 0.7400 | 0.1214 | 1280 |
| 27 | 10 | 0.0958 | 0.0829 | 0.0310 | 0.1728 | 0.0574 | 1280 |
| 27 | 11 | 0.7833 | 0.7795 | 0.6422 | 0.9369 | 0.1068 | 1280 |

## Comparison with DS-038 per-head separation (scatter)

Per-record live mean ECS vs DS-038 teacher-forced ECS for the three identified heads. If the live values cluster near the teacher-forced values, the artificial boundary wasn't producing artifacts and the ECS signal is real. If they diverge systematically, the teacher-forced ECS was measuring a different phenomenon.

### L21H3 (layer/head 21/3) — scatter coordinates

| record_id | live x | DS-038 tf y | delta |
|---|---|---|---|
| 0 | 0.634749 | 0.758377 | -0.123627 |
| 1 | 0.733022 | 0.840400 | -0.107378 |
| 2 | 0.826938 | 0.943695 | -0.116757 |
| 3 | 0.770074 | 0.854563 | -0.084489 |
| 4 | 0.683704 | 0.790803 | -0.107099 |
| 5 | 0.637280 | 0.758348 | -0.121067 |
| 6 | 0.731344 | 0.850016 | -0.118672 |
| 7 | 0.714351 | 0.843338 | -0.128987 |
| 8 | 0.713992 | 0.818559 | -0.104567 |
| 9 | 0.777452 | 0.872746 | -0.095294 |

Pearson r = 0.9756; mean |delta| = 0.110794; mean delta = -0.110794.

### L6H6 (layer/head 6/6) — scatter coordinates

| record_id | live x | DS-038 tf y | delta |
|---|---|---|---|
| 0 | 0.175484 | 0.525986 | -0.350502 |
| 1 | 0.207098 | 0.519624 | -0.312526 |
| 2 | 0.131028 | 0.350510 | -0.219482 |
| 3 | 0.099219 | 0.369996 | -0.270777 |
| 4 | 0.286630 | 0.598260 | -0.311630 |
| 5 | 0.169241 | 0.508826 | -0.339585 |
| 6 | 0.175069 | 0.521357 | -0.346287 |
| 7 | 0.150092 | 0.474314 | -0.324221 |
| 8 | 0.177237 | 0.477974 | -0.300737 |
| 9 | 0.166225 | 0.417192 | -0.250967 |

Pearson r = 0.8522; mean |delta| = 0.302671; mean delta = -0.302671.

### L2H1 (layer/head 2/1) — scatter coordinates

| record_id | live x | DS-038 tf y | delta |
|---|---|---|---|
| 0 | 0.827719 | 0.898694 | -0.070975 |
| 1 | 0.902977 | 0.935904 | -0.032927 |
| 2 | 0.798936 | 0.913675 | -0.114738 |
| 3 | 0.783099 | 0.827591 | -0.044492 |
| 4 | 0.923080 | 0.956711 | -0.033631 |
| 5 | 0.852539 | 0.905930 | -0.053391 |
| 6 | 0.876062 | 0.919970 | -0.043908 |
| 7 | 0.921340 | 0.938304 | -0.016963 |
| 8 | 0.909834 | 0.935094 | -0.025260 |
| 9 | 0.910523 | 0.931782 | -0.021259 |

Pearson r = 0.8461; mean |delta| = 0.045754; mean delta = -0.045754.

## Per-record time series (summary)

Per-step snapshots are in `live_ecs_results.jsonl` (`time_series`). The table below summarizes the per-record live ECS means at the three identified heads plus the dual-predicate fire-step counts.

| record_id | steps | mean ECS_L21H3 | mean ECS_L6H6 | mean ECS_L2H1 | bigram-fire steps | spectral-fire steps |
|---|---|---|---|---|---|---|
| 0 | 128 | 0.6347 | 0.1755 | 0.8277 | 0 | 103 |
| 1 | 128 | 0.7330 | 0.2071 | 0.9030 | 0 | 103 |
| 2 | 128 | 0.8269 | 0.1310 | 0.7989 | 0 | 103 |
| 3 | 128 | 0.7701 | 0.0992 | 0.7831 | 0 | 103 |
| 4 | 128 | 0.6837 | 0.2866 | 0.9231 | 0 | 103 |
| 5 | 128 | 0.6373 | 0.1692 | 0.8525 | 0 | 103 |
| 6 | 128 | 0.7313 | 0.1751 | 0.8761 | 0 | 103 |
| 7 | 128 | 0.7144 | 0.1501 | 0.9213 | 0 | 103 |
| 8 | 128 | 0.7140 | 0.1772 | 0.9098 | 0 | 103 |
| 9 | 128 | 0.7775 | 0.1662 | 0.9105 | 0 | 103 |

## Example per-step time series (record 0, first 12 steps)

Full per-step snapshots for every record are in `live_ecs_results.jsonl` (`time_series` / `steps`).

| step | ECS_L21H3 | ECS_L6H6 | ECS_L2H1 | bigram_fire | spectral_fire | token_diversity | trailing_ctr |
|---|---|---|---|---|---|---|---|
| 0 | 0.9939 | 0.9992 | 0.9840 | N | N | 0.3478 | 1.0000 |
| 1 | 0.9578 | 0.9985 | 0.9852 | N | N | 0.3478 | 1.0000 |
| 2 | 0.8238 | 0.9998 | 0.9765 | N | N | 0.3478 | 1.0000 |
| 3 | 0.6457 | 0.9991 | 0.9388 | N | N | 0.3478 | 1.0000 |
| 4 | 0.7220 | 0.9874 | 0.9733 | N | N | 1.0000 | 1.0000 |
| 5 | 0.7294 | 0.9920 | 0.8850 | N | N | 1.0000 | 1.0000 |
| 6 | 0.8227 | 0.9975 | 0.9527 | N | N | 1.0000 | 1.0000 |
| 7 | 0.8187 | 0.9884 | 0.9498 | N | N | 1.0000 | 1.0000 |
| 8 | 0.9261 | 0.3612 | 0.6650 | N | N | 1.0000 | 1.0000 |
| 9 | 0.9042 | 0.3879 | 0.9191 | N | N | 1.0000 | 1.0000 |
| 10 | 0.7768 | 0.3463 | 0.9421 | N | N | 0.8889 | 1.0000 |
| 11 | 0.6157 | 0.2862 | 0.8815 | N | N | 0.8000 | 1.0000 |

## Key observations (diagnostic; no verdict)

The per-record ordering of live ECS is strongly correlated with the DS-038 teacher-forced ECS on all three identified heads (Pearson r: L21H3 = 0.9756, L6H6 = 0.8522, L2H1 = 0.8461). The artificial teacher-forced boundary is therefore NOT producing a per-record ordering that is absent in live generation.

There is, however, a systematic level shift: every live mean is LOWER than the corresponding teacher-forced value (all deltas negative). The shift magnitude is head-dependent:

| head | mean live | mean tf | mean delta | mean \|delta\| |
|---|---|---|---|---|
| L21H3 | 0.7223 | 0.8331 | -0.1108 | 0.1108 |
| L6H6 | 0.1737 | 0.4764 | -0.3027 | 0.3027 |
| L2H1 | 0.8706 | 0.9164 | -0.0458 | 0.0458 |

The largest shift is at L6H6 (mean delta -0.303): the live generation drives this head's prompt-attention from ~0.48 (teacher-forced) down to ~0.17. The small L2H1 shift (mean delta -0.046) indicates that the early-layer head's prompt-attention is largely preserved under live generation. This head-dependent divergence is the measurement's primary finding: the live generation does not uniformly rescale the teacher-forced ECS — it disproportionately reduces prompt-attention at mid/late heads while the DS-038 separation ordering is retained.

## Notes / caveats

- MEASUREMENT ONLY: no thresholds, no actuation, no controller change [1]. This is a diagnostic probe; it offers no verdict.
- The prompt/generation boundary is genuine: the controller tracks `prompt_len` = tokenized length of record[\"text\"]. Attention from generated tokens back to the actual prompt is measured without the DS-038 heuristic (prompt_len = seq_len - 24).
- Attention weights are captured with `output_attentions=True` at every decoding step (KV-cache; eager attention). The per-step tensor is (num_heads x current_seq_len) ~6KB per layer at 128 tokens x 12 heads — no OOM expected. If CUDA OOMs, the documented fallback is to reduce to layers {2, 14, 27}.
- ECS = sum(attention to prompt positions) / sum(attention to all positions) for the current generated token at each step. Range [0, 1].
- Per-step dual-predicate diagnostics (token_diversity, trailing_ctr, layer2_pr, bigram_fire, spectral_fire) are computed with the exact night-008 configuration (frozen band_low).
- DS-038 teacher-forced ECS values are REUSED verbatim from `ecs_pks_results.jsonl` (do NOT re-run DS-038).
- All ECS sums are computed in float32 (per the DS-036 bf16 dead-zone finding: bf16 precision step ~0.03 at late-layer residual norms; the attention weights are upcast before summing).
- Per-record per-step per-head values are in `live_ecs_results.jsonl`.
