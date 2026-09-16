# DS-034b — Online PR parity: teacher-forced vs hook-based (HARD GATE)

> HARD GATE REPORT (RFC-004 Amendment A3). The dual-predicate architecture requires a live hook-based PR from a forward hook at layer 2 capturing hidden states step-by-step into a rolling 24-token ring buffer during autoregressive generation. DS-034 measured spectral PR via teacher-forced replay (single forward pass over the full sequence). These are different measurement geometries. This probe quantifies fire/no-fire disagreement between the two and applies the RFC A3 gate. A FAIL is a STOP signal — no negotiation, no threshold tuning.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture | tests/fixtures/t2s_degenerate.jsonl (night-001; N=100), record[\"text\"] input |
| Records measured | 93 detection-gap (DS-033 classification REUSED, do NOT re-compute) |
| Teacher-forced PR | REUSED from DS-034 spectral_pr_t2s_results.jsonl (do NOT re-measure) |
| Decoding | greedy (do_sample=False), max_new_tokens=128 |
| Analysis window | trailing 24 hidden states (hook-based); last 24 token positions (teacher-forced) |
| Hook layer | model.model.layers[2] |
| Frozen T_PR(2) | 10.954796 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| PR instrument | participation_ratio() from core/metrics.py (existing), float32 promotion |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| bmm Triton override | deregistered |
| Wall clock (s) | 613.2 |

## Gate criterion (RFC A3, quoted from the task file)

> HARD GATE per RFC A3: if fire/no-fire disagreement between teacher-forced and hook-based PR exceeds 5/93 records, the hook implementation must be revised before wiring the dual-predicate into the controller.

| Criterion | Value |
|---|---|
| Disagreement count ≤ 5/93 records | PASS — hook implementation is safe |
| Disagreement count > 5/93 records | FAIL — hook must be revised |

Fire criterion (both methods): `PR < T_PR(2)`. Teacher-forced fires on a single windowed PR (last 24 token positions of the full text). Hook-based fires if PR drops below T at ANY decoding step (rolling 24-token buffer, from step 23 onward).

## Determinism smoke

Two detection-gap records, hook-based generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

| record_id | prompt_len | n PR | tokens identical | PR log identical | identical |
|---|---|---|---|---|---|
| 0 | 199 | 105 | True | True | True |
| 1 | 160 | 105 | True | True | True |
|  | ALL |  |  |  | True |

PR first run A/B: 7.0831 / 7.0831; last run A/B: 7.0808 / 7.0808.

PR first run A/B: 4.0484 / 4.0484; last run A/B: 4.0632 / 4.0632.

## Verdict (RFC A3 HARD GATE)

| metric | value |
|---|---|
| Records compared | 93 |
| Agreement | 93 |
| **Disagreement** | **0** |
| Gate max disagreements | 5 |
| **Gate verdict** | **PASS** |

**PASS** — 0 ≤ 5: the hook implementation is safe. Fire/no-fire agreement between teacher-forced and hook-based PR holds within the RFC A3 tolerance.

## PR delta distribution (|PR_hook_mean − PR_teacher_forced|)

| stat | value |
|---|---|
| mean | 0.1389 |
| median | 0.0379 |
| p10 | 0.0049 |
| p90 | 0.1879 |

| measure | mean | median | p10 | p90 |
|---|---|---|---|---|
| Hook-based PR (per-record mean of per-step PR) | 4.3796 | 4.9866 | 2.0746 | 6.9314 |
| Teacher-forced PR (DS-034, layer 2) | 4.2875 | 4.9477 | 1.5590 | 6.9640 |

## Disagreement records

Total disagreements: 0. Near-boundary (hook PR min within 0.5 of T): 0; systematic (large PR delta / hook PR far from T): 0.

None — all 93 records agree (both fire).

## Per-record comparison (all 93 detection-gap records)

| record_id | teacher PR | teacher fire | hook PR mean | hook PR min | hook PR max | hook fire | PR delta | agreement |
|---|---|---|---|---|---|---|---|---|
| 0 | 7.1053 | True | 7.0852 | 7.0760 | 7.0954 | True | 0.0201 | True |
| 1 | 4.0927 | True | 4.0657 | 4.0364 | 4.0935 | True | 0.0269 | True |
| 2 | 1.5763 | True | 1.9220 | 1.5755 | 2.1738 | True | 0.3457 | True |
| 3 | 2.1130 | True | 2.1148 | 2.1048 | 2.1285 | True | 0.0017 | True |
| 4 | 6.9946 | True | 6.9530 | 6.9403 | 6.9639 | True | 0.0415 | True |
| 5 | 5.0750 | True | 5.1202 | 5.1082 | 5.1329 | True | 0.0452 | True |
| 6 | 5.9105 | True | 6.0558 | 5.9984 | 6.1170 | True | 0.1453 | True |
| 7 | 5.0733 | True | 5.0477 | 5.0354 | 5.0645 | True | 0.0256 | True |
| 8 | 3.1641 | True | 3.1132 | 3.0988 | 3.1266 | True | 0.0508 | True |
| 9 | 3.1106 | True | 3.1138 | 3.1033 | 3.1236 | True | 0.0032 | True |
| 10 | 3.0814 | True | 3.1307 | 3.1221 | 3.1442 | True | 0.0493 | True |
| 11 | 3.0861 | True | 3.0821 | 3.0740 | 3.0890 | True | 0.0039 | True |
| 12 | 6.9303 | True | 7.0568 | 7.0405 | 7.0755 | True | 0.1265 | True |
| 13 | 4.0792 | True | 4.0574 | 4.0190 | 4.0981 | True | 0.0218 | True |
| 14 | 5.0929 | True | 5.1434 | 5.1271 | 5.1541 | True | 0.0505 | True |
| 15 | 5.1086 | True | 5.1135 | 5.1032 | 5.1306 | True | 0.0049 | True |
| 16 | 3.1089 | True | 3.1441 | 3.1306 | 3.1645 | True | 0.0352 | True |
| 17 | 7.0048 | True | 6.9996 | 6.9763 | 7.0295 | True | 0.0053 | True |
| 18 | 3.1038 | True | 3.1119 | 3.1030 | 3.1198 | True | 0.0081 | True |
| 19 | 6.9968 | True | 7.0964 | 7.0712 | 7.1205 | True | 0.0996 | True |
| 20 | 5.5699 | True | 5.5168 | 5.4646 | 5.5786 | True | 0.0531 | True |
| 21 | 6.0010 | True | 6.0290 | 5.9809 | 6.0737 | True | 0.0281 | True |
| 22 | 6.4507 | True | 6.4664 | 6.4451 | 6.4852 | True | 0.0157 | True |
| 23 | 3.1085 | True | 3.2682 | 3.0916 | 3.3774 | True | 0.1598 | True |
| 24 | 3.0914 | True | 3.0686 | 3.0583 | 3.0783 | True | 0.0227 | True |
| 25 | 5.1126 | True | 5.1460 | 5.1314 | 5.1713 | True | 0.0333 | True |
| 26 | 5.9967 | True | 5.9840 | 5.9269 | 6.0287 | True | 0.0126 | True |
| 27 | 5.0431 | True | 5.0725 | 5.0628 | 5.0900 | True | 0.0293 | True |
| 29 | 5.0635 | True | 5.0844 | 5.0749 | 5.0993 | True | 0.0209 | True |
| 30 | 1.0711 | True | 1.0998 | 1.0944 | 1.1049 | True | 0.0286 | True |
| 31 | 2.0494 | True | 2.0426 | 2.0376 | 2.0519 | True | 0.0067 | True |
| 32 | 6.9869 | True | 6.9252 | 6.9122 | 6.9363 | True | 0.0617 | True |
| 33 | 2.1173 | True | 2.1493 | 2.1004 | 2.4229 | True | 0.0320 | True |
| 35 | 5.8555 | True | 5.6661 | 5.6516 | 5.6820 | True | 0.1894 | True |
| 36 | 5.0632 | True | 5.0569 | 5.0435 | 5.0712 | True | 0.0063 | True |
| 37 | 1.1565 | True | 3.3016 | 1.0707 | 13.3622 | True | 2.1451 | True |
| 38 | 4.1096 | True | 4.1276 | 4.1133 | 4.1488 | True | 0.0180 | True |
| 39 | 5.0987 | True | 5.0659 | 5.0482 | 5.0757 | True | 0.0328 | True |
| 40 | 1.5547 | True | 2.1524 | 1.7492 | 2.5378 | True | 0.5977 | True |
| 41 | 7.0057 | True | 6.9315 | 6.9179 | 6.9434 | True | 0.0742 | True |
| 43 | 2.1741 | True | 2.1594 | 2.1405 | 2.1795 | True | 0.0146 | True |
| 44 | 1.8421 | True | 2.6653 | 2.1537 | 3.1193 | True | 0.8232 | True |
| 45 | 6.0886 | True | 5.9771 | 5.9584 | 5.9956 | True | 0.1115 | True |
| 46 | 2.1295 | True | 2.1423 | 2.1294 | 2.1544 | True | 0.0128 | True |
| 47 | 4.0584 | True | 4.1094 | 4.0872 | 4.1423 | True | 0.0510 | True |
| 49 | 6.9096 | True | 6.9739 | 6.9591 | 6.9869 | True | 0.0643 | True |
| 50 | 4.1348 | True | 4.1111 | 4.0847 | 4.1458 | True | 0.0237 | True |
| 51 | 5.1135 | True | 5.0792 | 5.0698 | 5.0868 | True | 0.0343 | True |
| 52 | 5.0500 | True | 5.1246 | 5.1116 | 5.1344 | True | 0.0745 | True |
| 53 | 7.0210 | True | 6.9946 | 6.9576 | 7.1263 | True | 0.0264 | True |
| 54 | 3.1118 | True | 3.1085 | 3.1017 | 3.1161 | True | 0.0033 | True |
| 55 | 2.0922 | True | 2.0648 | 2.0605 | 2.0707 | True | 0.0274 | True |
| 56 | 5.9116 | True | 5.9806 | 5.9198 | 6.0258 | True | 0.0690 | True |
| 58 | 3.0061 | True | 3.1133 | 3.1059 | 3.1254 | True | 0.1071 | True |
| 59 | 3.9852 | True | 4.0810 | 4.0654 | 4.1047 | True | 0.0959 | True |
| 60 | 3.0901 | True | 3.1163 | 3.0916 | 3.1549 | True | 0.0262 | True |
| 61 | 5.0659 | True | 5.0713 | 5.0644 | 5.0810 | True | 0.0054 | True |
| 62 | 6.0170 | True | 6.0309 | 5.9612 | 6.0933 | True | 0.0139 | True |
| 63 | 4.9477 | True | 5.0594 | 5.0456 | 5.0750 | True | 0.1118 | True |
| 65 | 5.9975 | True | 6.0243 | 5.9602 | 6.0968 | True | 0.0268 | True |
| 66 | 3.1132 | True | 3.1759 | 3.1547 | 3.1931 | True | 0.0627 | True |
| 67 | 2.1138 | True | 2.1278 | 2.1191 | 2.1431 | True | 0.0140 | True |
| 68 | 2.1466 | True | 2.1422 | 2.1297 | 2.1659 | True | 0.0044 | True |
| 69 | 6.9169 | True | 6.9308 | 6.9188 | 6.9385 | True | 0.0139 | True |
| 70 | 1.2394 | True | 1.7255 | 1.4361 | 3.5034 | True | 0.4861 | True |
| 71 | 5.8946 | True | 5.9768 | 5.9070 | 6.0424 | True | 0.0823 | True |
| 72 | 1.4968 | True | 1.1467 | 1.1068 | 1.2410 | True | 0.3501 | True |
| 73 | 2.1097 | True | 2.1139 | 2.1055 | 2.1235 | True | 0.0043 | True |
| 74 | 3.0915 | True | 3.1469 | 3.1284 | 3.1696 | True | 0.0554 | True |
| 75 | 7.0266 | True | 7.0705 | 7.0626 | 7.0795 | True | 0.0439 | True |
| 76 | 6.7524 | True | 6.8568 | 6.8480 | 6.8665 | True | 0.1044 | True |
| 77 | 3.0917 | True | 3.0957 | 3.0817 | 3.1114 | True | 0.0040 | True |
| 78 | 4.0444 | True | 4.0907 | 4.0712 | 4.1107 | True | 0.0463 | True |
| 79 | 1.1218 | True | 1.1597 | 1.1439 | 1.1715 | True | 0.0379 | True |
| 80 | 1.0396 | True | 1.3440 | 1.0432 | 1.8195 | True | 0.3044 | True |
| 81 | 2.1069 | True | 2.1465 | 2.1297 | 2.1673 | True | 0.0396 | True |
| 82 | 1.1907 | True | 1.2980 | 1.2023 | 1.4511 | True | 0.1072 | True |
| 83 | 1.0611 | True | 4.1965 | 1.0438 | 7.5826 | True | 3.1354 | True |
| 84 | 2.1340 | True | 2.1288 | 2.1169 | 2.1442 | True | 0.0051 | True |
| 85 | 5.0898 | True | 5.1311 | 5.1194 | 5.1418 | True | 0.0413 | True |
| 87 | 1.4000 | True | 1.4726 | 1.3132 | 1.6216 | True | 0.0726 | True |
| 88 | 5.9791 | True | 6.0411 | 5.9901 | 6.1112 | True | 0.0620 | True |
| 89 | 4.0471 | True | 4.0491 | 4.0265 | 4.0842 | True | 0.0020 | True |
| 90 | 5.0693 | True | 5.1018 | 5.0920 | 5.1111 | True | 0.0325 | True |
| 91 | 5.1088 | True | 5.2621 | 5.2257 | 5.2966 | True | 0.1532 | True |
| 92 | 4.9605 | True | 4.9866 | 4.9775 | 4.9964 | True | 0.0261 | True |
| 93 | 5.9409 | True | 6.0410 | 5.9645 | 6.1030 | True | 0.1000 | True |
| 94 | 5.0852 | True | 5.1280 | 5.1151 | 5.1375 | True | 0.0427 | True |
| 95 | 6.8891 | True | 6.9080 | 6.8986 | 6.9209 | True | 0.0189 | True |
| 96 | 7.0018 | True | 7.0050 | 6.9887 | 7.0157 | True | 0.0031 | True |
| 97 | 6.9724 | True | 6.2320 | 6.0379 | 6.4461 | True | 0.7405 | True |
| 98 | 3.9325 | True | 4.0730 | 4.0399 | 4.1568 | True | 0.1405 | True |
| 99 | 6.5129 | True | 6.3310 | 6.3083 | 6.3546 | True | 0.1818 | True |

## Notes / caveats

- HARD GATE: the pass/fail criterion is explicit and quoted from RFC A3. A FAIL is a STOP signal — the script exits non-zero; no thresholds are negotiated or tuned.
- Teacher-forced PR values are REUSED from docs/gate23/spectral_pr_t2s_results.jsonl (DS-034). They are NOT re-measured. The teacher-forced PR is the layer-2 participation ratio over the last 24 token positions of the full input text in a single forward pass.
- DS-033 detection-gap classification is REUSED from docs/gate23/production_cross_fixture_results.jsonl (join on fixture + record_id, class == 'detection-gap'). Not re-classified.
- Hook-based PR is measured live: greedy decoding (do_sample=False), max_new_tokens=128, prompt = record[\"text\"]. Forward hook at model.model.layers[2] captures the hidden state at every decoding step; a rolling ring buffer keeps the trailing 24 generated hidden states (float32 promotion); participation_ratio() is computed from step 23 onward (105 values per record).
- Fire criterion is identical for both methods: PR < T_PR(2) = 10.954796. Hook-based fire = any step below T; teacher-forced fire = the single windowed PR below T.
- bf16 accumulation across the 128 separate forward passes (hook-based) vs the single forward pass (teacher-forced) can produce numerical drift; this probe quantifies the resulting fire/no-fire disagreement.
- Per-record per-step PR values are in `online_pr_parity_results.jsonl`.
