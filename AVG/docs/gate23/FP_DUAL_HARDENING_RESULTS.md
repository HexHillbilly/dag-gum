# DS-034d — Dual-predicate FP hardening with t2s coverage check (GATE)

> GATE REPORT (RFC-004 Amendment A3, Options A+B). DS-034c FAIL produced 7 FPs on prose-100. This probe tests TWO hardenings simultaneously on the same prose-100 corpus AND re-verifies t2s blind-spot coverage in the same run:
>
> 1. Bigram persistence ≥4 (was ≥2). Silences transient cold-start dips while preserving detection on sustained repetition loops.
> 2. Spectral-only fires require trailing_ctr < 0.50. When bigram is healthy (bigram_ctr < 2) and spectral fires alone, the spectral signal must be corroborated by surface text degradation per the gate doctrine [1].

NEITHER hardening changes frozen PR thresholds. The band_low value for spectral fire remains 8.216097 (ds-025 Part A freeze, unchanged).

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Corpus A | prose-100 (heldout partition of valid_subset_200, N=100; ds-027 complement split) |
| Corpus B | t2s_degenerate (night-001, N=100; detection-gap subset N=93 from DS-033 classification) |
| Prompt | record[\"text\"] |
| Decoding | greedy (do_sample=False), KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 hidden states (layer-2 hook, rolling ring buffer) |
| Hook layer | model.model.layers[2] |
| Frozen T_PR(2) | 10.954796 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| Spectral band_low(2) | 8.216097 (RFC A3 Options A+B hardening; spectral_collapse threshold) |
| PR instrument | participation_ratio() from core/metrics.py (existing), float32 promotion |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Hardening: bigram persistence | ≥4 consecutive bigram_collapse steps (was ≥2) |
| Hardening: spectral corroboration | when bigram_ctr < 2, spectral_fire requires trailing_ctr < 0.5 |
| Actuation | production logit-penalty path: token suppression cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |
| Dormant (prose-100) | REUSED from DS-034c (dual_predicate_fp_results.jsonl; live, greedy, KV-cache, no hooks/penalties) |
| bmm Triton override | deregistered |
| Wall clock (s) | 0.0 |

## Gate criteria (quoted from the DS-034d task file)

> - GATE on prose-100: 0 FPs required. FAIL is a STOP signal — no negotiation, no threshold tuning [1].
> - COVERAGE CHECK on t2s: spectral fire rate ≥ 85/93. Below this is a signal that the hardening is too aggressive — report and STOP.
> - REUSE DS-033 detection-gap classification and DS-034c dormant. Do NOT re-classify or re-run dormant.
> - NEW active runs for both corpora with the hardened predicate.

FP criterion: a record is a false positive if ΔDistinct-2 > 0 vs dormant AND the predicate fired at least once (the dual-predicate triggered actuation that changed the continuation on legitimate prose).

## Determinism smoke

Four records (2 prose-100 + 2 t2s detection-gap), active (hardened dual-predicate) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

| corpus | corpus_id | prompt_len | n_gen | n PR | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|
| prose | 3 | 248 | 128 | 105 | True | True | True |
| prose | 5 | 206 | 128 | 105 | True | True | True |
| t2s | 0 | 199 | 128 | 105 | True | True | True |
| t2s | 1 | 160 | 128 | 105 | True | True | True |
|  | ALL |  |  |  |  |  | True |

PR first run A/B: 19.0569 / 19.0569; last run A/B: 15.2324 / 15.2324.

PR first run A/B: 20.1514 / 20.1514; last run A/B: 19.7161 / 19.7161.

PR first run A/B: 7.0874 / 7.0874; last run A/B: 7.0750 / 7.0750.

PR first run A/B: 4.0550 / 4.0550; last run A/B: 4.0379 / 4.0379.

## Verdict

| metric | value |
|---|---|
| Prose-100 FPs (hardened) | 5 |
| Prose-100 fired records | 5 |
| **Prose gate verdict** | **FAIL** |
| t2s detection-gap spectral fire rate | 1/93 (1.1%) |
| **Coverage check (≥85/93)** | **FAIL** |
| **Overall** | **FAIL** |

**FAIL** — FPs persist on prose-100 under the hardened dual-predicate (5 FPs, all Category 1 bigram cold-start). The t2s detection-gap spectral fire rate is also below 85/93 (1/93). STOP signal; no thresholds are negotiated or tuned.

## Diagnosis (red gates)

Both hardenings fail their respective gates in this run:

1. **Bigram persistence ≥4 does NOT fix Category 1.** The 5 DS-034c bigram cold-start FPs (records 47, 49, 76, 83, 87) are still FPs. Under DS-034c (persistence ≥2) the actuation fired at steps 6-19 (bigram_ctr=2) and the logit-penalty kickstart forced the model to self-correct; that self-correction was **actuation-induced**, not natural. Under DS-034d (persistence ≥4) the repetition loop persists un-actuated until steps 8-21 (bigram_ctr reaches 4), at which point the actuation fires and changes the continuation → ΔDistinct-2 > 0 → FP. The task-file hypothesis that these are transient 2-3-step dips is falsified: the collapse is sustained for ≥4 consecutive steps on healthy prose.

2. **Spectral trailing_ctr < 0.50 fixes Category 2 but destroys t2s blind-spot coverage.** The 2 spectral-only late-dip FPs (records 98, 99) are fixed: trailing_ctr was 0.92-1.00 at the fire steps, so the corroboration gate kills the fire and the continuation stays byte-identical to dormant. But the SAME gate kills spectral fires on 92/93 t2s detection-gap records. The adversarial critic constraint is verified: **trailing_ctr < 0.50 is far too aggressive** for macro-syntax degenerate text, where the repeated tokens are real words and the coherent-token ratio stays ~1.0 even as the manifold collapses. Only record 83 fired (trailing_ctr dropped to 0.00 at step 31).

3. **Net effect:** prose-100 FPs drop 7 → 5 (Category 2 fixed, Category 1 not), while t2s detection-gap spectral fire coverage drops 93/93 → 1/93 (1.1%), far below the 85/93 (91.4%) required. The dual-predicate remains blocked.

## Table 1 — Prose-100 FP gate

| mode | n | FPs | fired records | total fires | verdict |
|---|---|---|---|---|---|
| hardened | 100 | 5 | 5 | 5 | FAIL |

FP record_ids: `[47, 49, 76, 83, 87]`

## Table 2 — Before/after comparison (DS-034c fallback vs DS-034d hardened)

| mode | FPs | bigram fires | spectral fires | fired records |
|---|---|---|---|---|
| DS-034c fallback (band_low, ≥2/≥2) | 7 | 5 | 9 | 7 |
| DS-034d hardened (band_low, ≥4 bigram + ctr-corroborated spectral) | 5 | 5 | 0 | 5 |

## Table 3 — t2s detection-gap spectral fire rate

| n | spectral fire count | fire rate | ≥85? |
|---|---|---|---|
| 93 | 1 | 1.1% | NO |

Spectral-fired detection-gap record_ids (1): `[83]`

## Table 4 — Per-category breakdown (prose-100)

Categories are derived from DS-034c fallback FP fire events: Category 1 = bigram cold-start (bigram trigger), Category 2 = spectral late dips (spectral-only trigger).

| category | n records | DS-034c FPs | DS-034d FPs | fixed? |
|---|---|---|---|---|
| Category 1 (bigram cold-start) | 5 | 5 | 5 | No |
| Category 2 (spectral late dips) | 2 | 2 | 0 | Yes |

Category 1 record_ids: `[47, 49, 76, 83, 87]`

Category 2 record_ids: `[98, 99]`

## Prose-100 per-record table (hardened)

| record_id | corpus_id | prompt_len | n_gen | ΔD2 | byte-id | fires | bigram_steps | spectral_steps | PR mean | PR min | FP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 3 | 248 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.9784 | 15.1766 | — |
| 1 | 5 | 206 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.2180 | 18.4682 | — |
| 2 | 7 | 164 | 128 | 0.0000 | Y | 0 | 0 | 0 | 15.4372 | 13.4152 | — |
| 3 | 9 | 294 | 128 | 0.0000 | Y | 0 | 0 | 0 | 21.0703 | 19.2462 | — |
| 4 | 12 | 1426 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.9567 | 14.5563 | — |
| 5 | 22 | 1029 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.1103 | 17.1037 | — |
| 6 | 33 | 290 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.7108 | 15.8902 | — |
| 7 | 35 | 1020 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.3773 | 16.2227 | — |
| 8 | 40 | 818 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.0536 | 12.3496 | — |
| 9 | 44 | 670 | 128 | 0.0000 | Y | 0 | 0 | 0 | 15.2847 | 11.6570 | — |
| 10 | 47 | 316 | 128 | 0.0000 | Y | 0 | 0 | 0 | 14.8966 | 10.7838 | — |
| 11 | 51 | 149 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.4432 | 17.1633 | — |
| 12 | 61 | 416 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.8038 | 17.0522 | — |
| 13 | 62 | 538 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.2768 | 13.7813 | — |
| 14 | 63 | 245 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.7546 | 17.4116 | — |
| 15 | 70 | 402 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.6357 | 14.1075 | — |
| 16 | 73 | 1394 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.5617 | 19.8774 | — |
| 17 | 74 | 783 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.1731 | 17.1023 | — |
| 18 | 91 | 753 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.2220 | 12.8119 | — |
| 19 | 109 | 465 | 128 | 0.0000 | Y | 0 | 0 | 0 | 15.4872 | 14.0917 | — |
| 20 | 110 | 289 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.0759 | 15.7644 | — |
| 21 | 112 | 829 | 10 | 0.0000 | Y | 0 | 0 | 0 | — | — | — |
| 22 | 116 | 1058 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.9340 | 11.2172 | — |
| 23 | 118 | 584 | 128 | 0.0000 | Y | 0 | 0 | 0 | 12.4607 | 10.1370 | — |
| 24 | 125 | 941 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.3004 | 14.5534 | — |
| 25 | 126 | 70 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.6698 | 14.1081 | — |
| 26 | 136 | 203 | 128 | 0.0000 | Y | 0 | 0 | 0 | 16.2346 | 11.6754 | — |
| 27 | 137 | 203 | 128 | 0.0000 | Y | 0 | 0 | 0 | 16.2346 | 11.6754 | — |
| 28 | 138 | 683 | 128 | 0.0000 | Y | 0 | 0 | 0 | 14.0538 | 13.3357 | — |
| 29 | 165 | 1197 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.4374 | 19.5913 | — |
| 30 | 166 | 153 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.8873 | 17.4929 | — |
| 31 | 174 | 395 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.3439 | 17.3800 | — |
| 32 | 183 | 2768 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.2969 | 14.8457 | — |
| 33 | 185 | 2768 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.2969 | 14.8457 | — |
| 34 | 193 | 1300 | 128 | 0.0000 | Y | 0 | 0 | 0 | 10.4929 | 9.7810 | — |
| 35 | 194 | 1300 | 128 | 0.0000 | Y | 0 | 0 | 0 | 10.4929 | 9.7810 | — |
| 36 | 197 | 2343 | 128 | 0.0000 | Y | 0 | 0 | 0 | 16.6282 | 13.6372 | — |
| 37 | 210 | 514 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.2808 | 16.3986 | — |
| 38 | 211 | 1811 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.6556 | 19.2756 | — |
| 39 | 216 | 1091 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.3016 | 16.9371 | — |
| 40 | 237 | 987 | 128 | 0.0000 | Y | 0 | 0 | 0 | 16.9711 | 13.7363 | — |
| 41 | 250 | 1040 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.3926 | 12.3659 | — |
| 42 | 252 | 1063 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.5025 | 15.2566 | — |
| 43 | 270 | 819 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.5803 | 17.2957 | — |
| 44 | 273 | 923 | 128 | 0.0000 | Y | 0 | 0 | 0 | 15.5458 | 9.3815 | — |
| 45 | 274 | 923 | 128 | 0.0000 | Y | 0 | 0 | 0 | 15.5458 | 9.3815 | — |
| 46 | 278 | 2195 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.4624 | 17.8729 | — |
| 47 | 279 | 316 | 9 | 0.2065 | N | 1 | 4 | 0 | — | — | FP |
| 48 | 285 | 1840 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.2436 | 18.4762 | — |
| 49 | 298 | 1942 | 19 | 0.1473 | N | 1 | 4 | 0 | — | — | FP |
| 50 | 299 | 416 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.4045 | 15.2909 | — |
| 51 | 302 | 617 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.9380 | 13.6047 | — |
| 52 | 306 | 657 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.4419 | 17.9900 | — |
| 53 | 311 | 2608 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.9099 | 15.6653 | — |
| 54 | 313 | 1014 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.0866 | 14.6596 | — |
| 55 | 327 | 370 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.7662 | 15.4465 | — |
| 56 | 330 | 707 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.9504 | 17.8307 | — |
| 57 | 335 | 637 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.0993 | 19.8199 | — |
| 58 | 338 | 689 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.4799 | 14.9016 | — |
| 59 | 346 | 1397 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.2130 | 12.8876 | — |
| 60 | 347 | 601 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.1933 | 16.5702 | — |
| 61 | 352 | 1445 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.8568 | 15.3878 | — |
| 62 | 357 | 948 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.7469 | 13.3215 | — |
| 63 | 362 | 3621 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.8956 | 16.5150 | — |
| 64 | 366 | 610 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.0309 | 16.9470 | — |
| 65 | 367 | 5786 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.5778 | 16.2614 | — |
| 66 | 373 | 7265 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.9963 | 16.3147 | — |
| 67 | 374 | 7265 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.9963 | 16.3147 | — |
| 68 | 379 | 2665 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.0737 | 13.4383 | — |
| 69 | 380 | 1635 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.8557 | 12.6529 | — |
| 70 | 386 | 6084 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.1746 | 15.8532 | — |
| 71 | 395 | 710 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.8089 | 16.2791 | — |
| 72 | 405 | 487 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.2013 | 11.4312 | — |
| 73 | 409 | 393 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.5290 | 11.4099 | — |
| 74 | 411 | 2297 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.7254 | 13.4113 | — |
| 75 | 414 | 554 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.0014 | 16.8889 | — |
| 76 | 424 | 3229 | 128 | 0.8261 | N | 1 | 4 | 0 | 17.0040 | 9.6312 | FP |
| 77 | 427 | 533 | 128 | 0.0000 | Y | 0 | 0 | 0 | 10.0775 | 9.2733 | — |
| 78 | 429 | 612 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.7272 | 17.3216 | — |
| 79 | 430 | 468 | 128 | 0.0000 | Y | 0 | 0 | 0 | 14.1680 | 9.6566 | — |
| 80 | 436 | 1059 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.4686 | 18.1576 | — |
| 81 | 438 | 343 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.2501 | 12.3261 | — |
| 82 | 445 | 687 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.7875 | 10.6101 | — |
| 83 | 455 | 1078 | 16 | 0.2232 | N | 1 | 4 | 0 | — | — | FP |
| 84 | 456 | 698 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.5158 | 17.2997 | — |
| 85 | 464 | 1425 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.6264 | 19.3082 | — |
| 86 | 467 | 894 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.6165 | 16.0344 | — |
| 87 | 478 | 468 | 9 | 0.2065 | N | 1 | 4 | 0 | — | — | FP |
| 88 | 480 | 1539 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.0592 | 14.6372 | — |
| 89 | 481 | 2305 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.0714 | 19.4285 | — |
| 90 | 482 | 2305 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.0714 | 19.4285 | — |
| 91 | 483 | 711 | 128 | 0.0000 | Y | 0 | 0 | 0 | 15.7084 | 14.5924 | — |
| 92 | 484 | 494 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.5697 | 14.1448 | — |
| 93 | 485 | 1802 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.0504 | 17.5494 | — |
| 94 | 486 | 1802 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.0504 | 17.5494 | — |
| 95 | 487 | 1802 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.0504 | 17.5494 | — |
| 96 | 492 | 2317 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.0391 | 19.4563 | — |
| 97 | 494 | 420 | 128 | 0.0000 | Y | 0 | 0 | 0 | 18.4627 | 16.5492 | — |
| 98 | 498 | 1794 | 128 | 0.0000 | Y | 0 | 0 | 5 | 18.1681 | 5.8950 | — |
| 99 | 499 | 1794 | 128 | 0.0000 | Y | 0 | 0 | 5 | 18.1681 | 5.8950 | — |

## Fire events — prose-100 (5 firing records, hardened)

### record 47 (corpus_id 279)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 8 | bigram | 4 | 0 | — | 0.1429 | 0.0000 | N |

### record 49 (corpus_id 298)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 18 | bigram | 4 | 0 | — | 0.2353 | 0.0000 | N |

### record 76 (corpus_id 424)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 21 | bigram | 4 | 0 | — | 0.2500 | 0.0000 | N |

### record 83 (corpus_id 455)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 15 | bigram | 4 | 0 | — | 0.2143 | 0.0000 | N |

### record 87 (corpus_id 478)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 8 | bigram | 4 | 0 | — | 0.1429 | 0.0000 | N |

## t2s detection-gap per-record spectral fire log (hardened)

| record_id | prompt_len | n_gen | fire_count | bigram_fired | spectral_fired | spectral_steps | PR mean | PR min |
|---|---|---|---|---|---|---|---|---|
| 0 | 199 | 128 | 0 | N | N | 104 | 7.0846 | 7.0729 |
| 1 | 160 | 128 | 0 | N | N | 104 | 4.0666 | 4.0379 |
| 2 | 187 | 128 | 0 | N | N | 104 | 1.8896 | 1.5886 |
| 3 | 172 | 128 | 0 | N | N | 104 | 2.1152 | 2.0971 |
| 4 | 181 | 128 | 0 | N | N | 104 | 6.9501 | 6.9285 |
| 5 | 168 | 128 | 0 | N | N | 104 | 5.1173 | 5.1058 |
| 6 | 180 | 128 | 0 | N | N | 104 | 6.0542 | 5.9913 |
| 7 | 191 | 128 | 0 | N | N | 104 | 5.0446 | 5.0208 |
| 8 | 174 | 128 | 0 | N | N | 104 | 3.1122 | 3.0937 |
| 9 | 190 | 128 | 0 | N | N | 104 | 3.1121 | 3.0978 |
| 10 | 160 | 128 | 0 | N | N | 104 | 3.1327 | 3.1202 |
| 11 | 179 | 128 | 0 | N | N | 104 | 3.0792 | 3.0709 |
| 12 | 177 | 128 | 0 | N | N | 104 | 7.0576 | 7.0345 |
| 13 | 200 | 128 | 0 | N | N | 104 | 4.0560 | 4.0178 |
| 14 | 183 | 128 | 0 | N | N | 104 | 5.1405 | 5.1236 |
| 15 | 181 | 128 | 0 | N | N | 104 | 5.1135 | 5.0971 |
| 16 | 163 | 128 | 0 | N | N | 104 | 3.1425 | 3.1290 |
| 17 | 197 | 128 | 0 | N | N | 104 | 6.9968 | 6.9698 |
| 18 | 171 | 128 | 0 | N | N | 104 | 3.1102 | 3.0998 |
| 19 | 163 | 128 | 0 | N | N | 104 | 7.0928 | 7.0648 |
| 20 | 182 | 128 | 0 | N | N | 104 | 5.5162 | 5.4650 |
| 21 | 188 | 128 | 0 | N | N | 104 | 6.0269 | 5.9788 |
| 22 | 183 | 128 | 0 | N | N | 104 | 6.4666 | 6.4381 |
| 23 | 186 | 128 | 0 | N | N | 104 | 3.2711 | 3.0920 |
| 24 | 175 | 128 | 0 | N | N | 104 | 3.0659 | 3.0523 |
| 25 | 169 | 128 | 0 | N | N | 104 | 5.1433 | 5.1309 |
| 26 | 176 | 128 | 0 | N | N | 104 | 5.9836 | 5.9231 |
| 27 | 172 | 128 | 0 | N | N | 104 | 5.0712 | 5.0543 |
| 29 | 195 | 128 | 0 | N | N | 104 | 5.0831 | 5.0674 |
| 30 | 198 | 128 | 0 | N | N | 104 | 1.0969 | 1.0896 |
| 31 | 167 | 128 | 0 | N | N | 104 | 2.0434 | 2.0361 |
| 32 | 192 | 128 | 0 | N | N | 104 | 6.9231 | 6.9106 |
| 33 | 180 | 128 | 0 | N | N | 104 | 2.1464 | 2.0935 |
| 35 | 182 | 128 | 0 | N | N | 104 | 5.6644 | 5.6515 |
| 36 | 175 | 128 | 0 | N | N | 104 | 5.0535 | 5.0372 |
| 37 | 175 | 128 | 0 | N | N | 0 | 18.4335 | 12.7104 |
| 38 | 185 | 128 | 0 | N | N | 104 | 4.1241 | 4.1062 |
| 39 | 167 | 128 | 0 | N | N | 104 | 5.0661 | 5.0528 |
| 40 | 168 | 128 | 0 | N | N | 104 | 2.1522 | 1.7347 |
| 41 | 172 | 128 | 0 | N | N | 104 | 6.9324 | 6.9228 |
| 43 | 188 | 128 | 0 | N | N | 104 | 2.1582 | 2.1315 |
| 44 | 184 | 128 | 0 | N | N | 104 | 2.6414 | 2.0917 |
| 45 | 181 | 128 | 0 | N | N | 104 | 5.9772 | 5.9621 |
| 46 | 172 | 128 | 0 | N | N | 104 | 2.1402 | 2.1227 |
| 47 | 164 | 128 | 0 | N | N | 104 | 4.1079 | 4.0795 |
| 49 | 181 | 128 | 0 | N | N | 104 | 6.9756 | 6.9598 |
| 50 | 189 | 128 | 0 | N | N | 104 | 4.1082 | 4.0819 |
| 51 | 171 | 128 | 0 | N | N | 104 | 5.0763 | 5.0629 |
| 52 | 174 | 128 | 0 | N | N | 104 | 5.1215 | 5.1091 |
| 53 | 191 | 128 | 0 | N | N | 104 | 6.9953 | 6.9589 |
| 54 | 176 | 128 | 0 | N | N | 104 | 3.1068 | 3.0996 |
| 55 | 163 | 128 | 0 | N | N | 104 | 2.0660 | 2.0621 |
| 56 | 196 | 128 | 0 | N | N | 104 | 5.9788 | 5.9201 |
| 58 | 200 | 128 | 0 | N | N | 104 | 3.1132 | 3.1055 |
| 59 | 169 | 128 | 0 | N | N | 104 | 4.0809 | 4.0647 |
| 60 | 184 | 128 | 0 | N | N | 104 | 3.1147 | 3.0854 |
| 61 | 185 | 128 | 0 | N | N | 104 | 5.0700 | 5.0546 |
| 62 | 186 | 128 | 0 | N | N | 104 | 6.0289 | 5.9573 |
| 63 | 176 | 128 | 0 | N | N | 104 | 5.0577 | 5.0375 |
| 65 | 181 | 128 | 0 | N | N | 104 | 6.0221 | 5.9570 |
| 66 | 197 | 128 | 0 | N | N | 104 | 3.1699 | 3.1465 |
| 67 | 197 | 128 | 0 | N | N | 104 | 2.1242 | 2.1141 |
| 68 | 171 | 128 | 0 | N | N | 104 | 2.1413 | 2.1212 |
| 69 | 177 | 128 | 0 | N | N | 104 | 6.9325 | 6.9195 |
| 70 | 189 | 128 | 0 | N | N | 104 | 2.5886 | 2.1692 |
| 71 | 171 | 128 | 0 | N | N | 104 | 5.9773 | 5.9053 |
| 72 | 189 | 128 | 0 | N | N | 104 | 1.1428 | 1.1006 |
| 73 | 185 | 128 | 0 | N | N | 104 | 2.1149 | 2.1090 |
| 74 | 177 | 128 | 0 | N | N | 104 | 3.1443 | 3.1236 |
| 75 | 196 | 128 | 0 | N | N | 104 | 7.0714 | 7.0568 |
| 76 | 173 | 128 | 0 | N | N | 104 | 6.8557 | 6.8460 |
| 77 | 160 | 128 | 0 | N | N | 104 | 3.0960 | 3.0827 |
| 78 | 189 | 128 | 0 | N | N | 104 | 4.0886 | 4.0583 |
| 79 | 174 | 128 | 0 | N | N | 104 | 1.1612 | 1.1466 |
| 80 | 175 | 128 | 0 | N | N | 104 | 1.3666 | 1.0457 |
| 81 | 196 | 128 | 0 | N | N | 104 | 2.1416 | 2.1203 |
| 82 | 164 | 128 | 0 | N | N | 104 | 1.2982 | 1.2094 |
| 83 | 171 | 128 | 1 | N | Y | 104 | 4.7412 | 1.0560 |
| 84 | 200 | 128 | 0 | N | N | 104 | 2.1257 | 2.1069 |
| 85 | 165 | 128 | 0 | N | N | 104 | 5.1294 | 5.1146 |
| 87 | 180 | 128 | 0 | N | N | 104 | 1.4777 | 1.3153 |
| 88 | 160 | 128 | 0 | N | N | 104 | 6.0433 | 5.9884 |
| 89 | 168 | 128 | 0 | N | N | 104 | 4.0483 | 4.0203 |
| 90 | 163 | 128 | 0 | N | N | 104 | 5.1031 | 5.0951 |
| 91 | 183 | 128 | 0 | N | N | 104 | 5.2568 | 5.2146 |
| 92 | 191 | 128 | 0 | N | N | 104 | 4.9854 | 4.9745 |
| 93 | 184 | 128 | 0 | N | N | 104 | 6.0381 | 5.9613 |
| 94 | 177 | 128 | 0 | N | N | 104 | 5.1281 | 5.1127 |
| 95 | 199 | 128 | 0 | N | N | 104 | 6.9052 | 6.8916 |
| 96 | 176 | 128 | 0 | N | N | 104 | 7.0038 | 6.9885 |
| 97 | 173 | 128 | 0 | N | N | 104 | 6.2300 | 6.0429 |
| 98 | 162 | 128 | 0 | N | N | 104 | 4.0740 | 4.0404 |
| 99 | 179 | 128 | 0 | N | N | 104 | 6.3286 | 6.3003 |

## Notes / caveats

- GATE: the pass/fail criterion is explicit and quoted from the task file. A FAIL is a STOP signal — the script exits non-zero; no thresholds are negotiated or tuned.
- COVERAGE CHECK: t2s spectral fire rate < 85/93 means the hardening is too aggressive (adversarial critic constraint) — report and STOP.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer, identical to the DS-034b hook pattern.
- The hardened dual-predicate rule is implemented exactly per RFC A3 Options A+B: bigram_fire = bigram_ctr ≥ 4; spectral_fire = spectral_ctr ≥ 2 with trailing_ctr < 0.50 corroboration when bigram_ctr < 2; code-context immunity forces spectral_fire=False when is_code_syntax_context is True.
- Actuation on is_collapsed uses the production logit-penalty path (token suppression cooldown=8 penalty=-5.0, kickstart counter=3 with -1e4/-5.0/-2.0, top_p=0.85). The actuation can fire multiple times per record; every fire is logged.
- Dormant baselines for prose-100 are REUSED from DS-034c (dual_predicate_fp_results.jsonl, primary mode; live, greedy, KV-cache, no hooks/penalties). Do NOT re-run dormant.
- DS-033 detection-gap classification is REUSED from production_cross_fixture_results.jsonl (join on fixture + record_id). Do NOT re-classify.
- FP = ΔDistinct-2 > 0 AND fire_count > 0. A record whose predicate fires but whose continuation is byte-identical to dormant is an actuation-gap, not an FP.
- Frozen thresholds (T_PR(2) = 10.954796, band_low = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified values; the script exits if they disagree.
- The gate verdict is scoped to prose-100 only. Hazard/schema corpora are deferred to DS-034c-hazard.
