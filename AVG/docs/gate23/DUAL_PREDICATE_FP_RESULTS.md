# DS-034c — Dual-predicate FP re-verification on prose-100 (GATE)

> GATE REPORT (RFC-004 Amendment A3). Dual-predicate OR rule — bigram OR spectral PR at layer 2 — re-verified under LIVE generation with the production cadence and hysteresis per RFC A3. A FAIL is a STOP signal — no negotiation, no threshold tuning.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Corpus | prose-100 (heldout partition of valid_subset_200, N=100; ds-027 complement split) |
| Prompt | record[\"text\"] |
| Decoding | greedy (do_sample=False), KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 hidden states (layer-2 hook, rolling ring buffer) |
| Hook layer | model.model.layers[2] |
| Frozen T_PR(2) | 10.954796 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| Fallback band_low(2) | 8.216097 (RFC A3 Options A+B hardening) |
| PR instrument | participation_ratio() from core/metrics.py (existing), float32 promotion |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Persistence hysteresis | ≥2 consecutive is_collapsed steps (RFC A3 tightening, applied to the OR output) |
| Actuation | production logit-penalty path: token suppression cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |
| Dormant | LIVE (greedy, KV-cache, no hooks/penalties) |
| bmm Triton override | deregistered |
| Wall clock (s) | 715.1 |

## Gate criteria (quoted from the DS-034c task file)

> - Primary: 0 FPs on prose-100 under simple OR (PR < T, ≥2 consecutive steps) → PASS.
> - Fallback (if FPs > 0): re-test with spectral fire at band_low (PR < 8.22, ≥2 consecutive steps) on the same corpus. 0 FPs → PASS with hardening applied. FPs persist → FAIL.
> - A FAIL is a STOP signal — do NOT negotiate, do NOT tune thresholds [1].

FP criterion: a record is a false positive if ΔDistinct-2 > 0 vs dormant AND the predicate fired at least once (the dual-predicate triggered actuation that changed the continuation on legitimate prose).

## Determinism smoke

Two prose-100 records: dormant generated twice AND active (dual-predicate, primary threshold) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not.

| corpus_id | prompt_len | n_gen | n PR | dormant id | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|
| 3 | 248 | 128 | 105 | True | True | True | True |
| 5 | 206 | 128 | 105 | True | True | True | True |
|  | ALL |  |  |  |  |  | True |

PR first run A/B: 19.0569 / 19.0569; last run A/B: 15.2324 / 15.2324.

PR first run A/B: 20.1514 / 20.1514; last run A/B: 19.7161 / 19.7161.

## Verdict

| metric | value |
|---|---|
| Records scored | 100 |
| Primary FPs (simple OR, PR < T) | 11 |
| Primary fired records | 14 |
| Fallback FPs (band_low hardening) | 7 |
| Fallback fired records | 7 |
| **Gate verdict** | **FAIL** |

**FAIL** — FPs persist even after the band_low hardening. STOP signal; no thresholds are negotiated or tuned.

## Primary mode summary (simple OR, PR < T)

| metric | value |
|---|---|
| Records | 100 |
| False positives | 11 |
| Records with ≥1 fire | 14 |
| Total fire steps | 42 |
| Byte-identical to dormant | 86 |
| EOS-terminated (active) | 10 |
| ΔDistinct-2 mean / median / p90 | 0.0193 / 0.0000 / 0.0435 |
| ΔDistinct-2 (firing only) mean / median / p90 | 0.1379 / 0.2174 / 0.3000 |
| Fire count per record mean / max | 0.4200 / 7.0000 |
| PR (per-record mean) mean / min / max | 17.9033 / 10.8521 / 21.0703 |

Primary FP record_ids: `[23, 34, 35, 47, 49, 76, 77, 83, 87, 98, 99]`

## Fallback mode summary (band_low hardening, PR < 8.22)

| metric | value |
|---|---|
| Records | 100 |
| False positives | 7 |
| Records with ≥1 fire | 7 |
| Total fire steps | 14 |
| Byte-identical to dormant | 93 |
| ΔDistinct-2 mean / median / p90 | 0.0202 / 0.0000 / 0.0000 |
| ΔDistinct-2 (firing only) mean / median / p90 | 0.2883 / 0.2642 / 0.4696 |
| Fire count per record mean / max | 0.1400 / 4.0000 |

Fallback FP record_ids: `[47, 49, 76, 83, 87, 98, 99]`

## Per-record table (final mode: fallback)

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
| 47 | 279 | 316 | 7 | 0.2899 | N | 1 | 2 | 0 | — | — | FP |
| 48 | 285 | 1840 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.2436 | 18.4762 | — |
| 49 | 298 | 1942 | 60 | 0.1739 | N | 2 | 2 | 2 | 15.2663 | 7.3571 | FP |
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
| 76 | 424 | 3229 | 128 | 0.7391 | N | 1 | 2 | 0 | 17.2184 | 11.0837 | FP |
| 77 | 427 | 533 | 128 | 0.0000 | Y | 0 | 0 | 0 | 10.0775 | 9.2733 | — |
| 78 | 429 | 612 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.7272 | 17.3216 | — |
| 79 | 430 | 468 | 128 | 0.0000 | Y | 0 | 0 | 0 | 14.1680 | 9.6566 | — |
| 80 | 436 | 1059 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.4686 | 18.1576 | — |
| 81 | 438 | 343 | 128 | 0.0000 | Y | 0 | 0 | 0 | 13.2501 | 12.3261 | — |
| 82 | 445 | 687 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.7875 | 10.6101 | — |
| 83 | 455 | 1078 | 14 | 0.2642 | N | 1 | 2 | 0 | — | — | FP |
| 84 | 456 | 698 | 128 | 0.0000 | Y | 0 | 0 | 0 | 19.5158 | 17.2997 | — |
| 85 | 464 | 1425 | 128 | 0.0000 | Y | 0 | 0 | 0 | 20.6264 | 19.3082 | — |
| 86 | 467 | 894 | 128 | 0.0000 | Y | 0 | 0 | 0 | 17.6165 | 16.0344 | — |
| 87 | 478 | 468 | 7 | 0.2899 | N | 1 | 2 | 0 | — | — | FP |
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
| 98 | 498 | 1794 | 128 | 0.1304 | N | 4 | 0 | 5 | 18.2139 | 6.8832 | FP |
| 99 | 499 | 1794 | 128 | 0.1304 | N | 4 | 0 | 5 | 18.2139 | 6.8832 | FP |

## Fire events (7 firing records, final mode: fallback)

### record 47 (corpus_id 279)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 6 | bigram | 2 | 0 | — | 0.2000 | 0.0000 | N |

### record 49 (corpus_id 298)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 16 | bigram | 2 | 0 | — | 0.2667 | 0.0000 | N |
| 59 | spectral | 0 | 2 | 7.3571 | 0.3043 | 0.0000 | N |

### record 76 (corpus_id 424)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 19 | bigram | 2 | 0 | — | 0.2778 | 0.0000 | N |

### record 83 (corpus_id 455)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 13 | bigram | 2 | 0 | — | 0.2500 | 0.0000 | N |

### record 87 (corpus_id 478)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 6 | bigram | 2 | 0 | — | 0.2000 | 0.0000 | N |

### record 98 (corpus_id 498)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 124 | spectral | 0 | 2 | 7.4957 | 0.4348 | 1.0000 | N |
| 125 | spectral | 0 | 3 | 6.8832 | 0.3913 | 1.0000 | N |
| 126 | spectral | 0 | 4 | 7.1216 | 0.4348 | 0.9167 | N |
| 127 | spectral | 0 | 5 | 7.5293 | 0.4783 | 1.0000 | N |

### record 99 (corpus_id 499)

| step | trigger | bigram_ctr | spectral_ctr | PR | token_diversity | trailing_ctr | is_code |
|---|---|---|---|---|---|---|---|
| 124 | spectral | 0 | 2 | 7.4957 | 0.4348 | 1.0000 | N |
| 125 | spectral | 0 | 3 | 6.8832 | 0.3913 | 1.0000 | N |
| 126 | spectral | 0 | 4 | 7.1216 | 0.4348 | 0.9167 | N |
| 127 | spectral | 0 | 5 | 7.5293 | 0.4783 | 1.0000 | N |

## Notes / caveats

- GATE: the pass/fail criterion is explicit and quoted from the task file. A FAIL is a STOP signal — the script exits non-zero; no thresholds are negotiated or tuned.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values), dropping compute from O(n^2) to O(n) vs the v1 full-sequence forward passes. The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer, identical to the DS-034b hook pattern.
- The dual-predicate OR rule is implemented exactly per RFC A3: bigram OR spectral, each with its own ≥2-consecutive persistence counter, code-context immunity forcing spectral_collapse=False when is_code_syntax_context is True.
- Actuation on is_collapsed uses the production logit-penalty path (token suppression cooldown=8 penalty=-5.0, kickstart counter=3 with -1e4/-5.0/-2.0, top_p=0.85). The actuation can fire multiple times per record; every fire is logged.
- Dormant baselines are generated LIVE (greedy, KV-cache, no hooks/penalties) for all 100 records. No JSONL reuse from the failed v1 run.
- FP = ΔDistinct-2 > 0 AND fire_count > 0. A record whose predicate fires but whose continuation is byte-identical to dormant is an actuation-gap, not an FP.
- Frozen thresholds (T_PR(2) = 10.954796, band_low = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified values; the script exits if they disagree.
- The gate verdict is scoped to prose-100 only. Hazard/schema corpora are deferred to DS-034c-hazard.
