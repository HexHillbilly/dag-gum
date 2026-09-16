# DS-006 — Corpus Profile Report

Date: 2026-08-08
Branch: `night/ds-006-corpus-profile`
Task: `batch/queue/ds-006-corpus-profile.md`
Deliverable: `docs/CORPUS_PROFILE.md` (exactly one new file)
Engine: automated engine — disposable clone, read-only analysis, **no model loading**

## 1. Scope & methodology

Profile the three certified corpora:

| label | path |
| --- | --- |
| valid prose | `data/t2s_bench/valid_subset_200.jsonl` |
| degenerate | `tests/fixtures/t2s_degenerate.jsonl` |
| schema | `tests/fixtures/schema_corpus.jsonl` |

The analysis is deterministic and read-only. No RNG is used by the profiler itself;
the only randomness in play is the fixed generation seeds already baked into the
corpora (see §6). No files other than this report are written, and no model is
loaded or invoked.

**Definitions.**

- **samples** — number of JSONL records (one per line, each with a `text` field).
- **chars** — `len(sample["text"])`.
- **words** — whitespace-delimited tokens, `len(sample["text"].split())`. This
  matches the token-counting convention used by the corpus builders
  (`scripts/build_t2s_degenerate.py` and `scripts/build_schema_corpus.py`).
- **lines** — non-empty lines, `sum(1 for ln in text.splitlines() if ln.strip())`,
  matching the `line_count` convention in `scripts/build_schema_corpus.py`.
- **Distinct-2** — unique-bigram / total-bigram ratio over a token sequence,
  `unique_bigrams / (n - 1)`.

**Distinct-2 metric.** The canonical implementation is
`compute_token_distinct_2_fast` in `AVG.governor.controller` (line ~70). Per the
DS-003 verified ground truth (`batch/queue/ds-003-metrics-tests.md`), `core/metrics.py`
has **no** Distinct-2 function; the canonical one lives in the controller. Imports
use the standard parents[2] root-as-package header; in this container it resolves
to `/` (the `/AVG` mount) and `import AVG.*` resolves correctly.

For each sample, the `text` is whitespace-tokenized, each token is mapped to an
integer via a deterministic global vocabulary (sorted unique tokens per corpus), and
the resulting `LongTensor` is passed to `compute_token_distinct_2_fast` with
`prompt_len=0, window_len=24` — the production fast-path ring-buffer window
(`fast_path_window=24` in `governor/controller.py`, and the function's own default).
This yields one Distinct-2 value per sample; the per-corpus distribution is then
summarized as mean / min / max. A full-sample variant (`window_len=n_tokens`) is
reported as a cross-check in §5.

## 2. Certified corpora at a glance

| corpus | samples | bytes | sha256 |
| --- | ---: | ---: | --- |
| valid prose | 200 | 1,620,945 | `0f7ccae3a302fb5f5c9ddd9b62206584904edb24764ee2880506e479b7da6d53` |
| degenerate | 100 | 94,472 | `e1c576997a549372dac01d76712964e9a36882d8065ac0bf6470d474e678dedf` |
| schema | 100 | 319,896 | `648e8330f999dc13edf874862387cc30d93aabcbbf28a0b6d356183fa1b490bb` |

## 3. Per-corpus profile

### 3.1 Valid prose — `data/t2s_bench/valid_subset_200.jsonl`

Certified prose subset of T2S-Bench-MR (arXiv:2603.03790); certification recorded in
`data/t2s_bench/valid_subset_200.cert.txt` (0/200 Track B trippers).

**Sample count:** 200

**Size distribution (min / median / max):**

| metric | min | median | max |
| --- | ---: | ---: | ---: |
| chars | 334 | 3,901 | 26,668 |
| words | 46 | 623 | 3,802 |
| lines (non-empty) | 2 | 9 | 85 |

**Distinct-2 distribution (24-token window):**

| stat | value |
| --- | ---: |
| mean | 0.9826 |
| min | 0.7391 |
| max | 1.0000 |

Supplementary: full-sample Distinct-2 mean 0.8681, min 0.6151, max 1.0000.
Vocabulary footprint: 23,602 unique word-tokens across the 200 samples.

### 3.2 Degenerate — `tests/fixtures/t2s_degenerate.jsonl`

Synthetic cyclic loops from a closed English-content vocabulary (fixed seed
`SEED = 2603`, see `scripts/build_t2s_degenerate.py`); certified with full-sample
Distinct-2 < 0.20 and 0/200 Track B trippers.

**Sample count:** 100

**Size distribution (min / median / max):**

| metric | min | median | max |
| --- | ---: | ---: | ---: |
| chars | 485 | 827.5 | 1,307 |
| words | 160 | 180 | 200 |
| lines (non-empty) | 1 | 1 | 1 |

**Distinct-2 distribution (24-token window):**

| stat | value |
| --- | ---: |
| mean | 0.2213 |
| min | 0.0435 |
| max | 0.3478 |

Supplementary: full-sample Distinct-2 mean 0.0285, min 0.0057, max 0.0494.
Vocabulary footprint: 132 unique word-tokens across the 100 samples (the closed-loop
design).

### 3.3 Schema — `tests/fixtures/schema_corpus.jsonl`

Synthetic bare, pretty-printed JSON (no markdown/fences/commentary) from a closed
English vocabulary (fixed seed `SEED = 4004`, see `scripts/build_schema_corpus.py`);
certified with >= 120 non-empty lines and 0/200 Track B trippers.

**Sample count:** 100

**Size distribution (min / median / max):**

| metric | min | median | max |
| --- | ---: | ---: | ---: |
| chars | 1,735 | 2,712 | 3,716 |
| words | 174 | 305.5 | 355 |
| lines (non-empty) | 120 | 124.5 | 130 |

**Distinct-2 distribution (24-token window):**

| stat | value |
| --- | ---: |
| mean | 0.9700 |
| min | 0.8261 |
| max | 1.0000 |

Supplementary: full-sample Distinct-2 mean 0.9883, min 0.9458, max 1.0000.
Vocabulary footprint: 3,857 unique word-tokens across the 100 samples (closed
English vocab plus JSON structural tokens).

## 4. Comparison table

| corpus | samples | chars med | words med | lines med | D-2 mean | D-2 min | D-2 max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| valid prose | 200 | 3,901 | 623 | 9 | 0.9826 | 0.7391 | 1.0000 |
| degenerate | 100 | 827.5 | 180 | 1 | 0.2213 | 0.0435 | 0.3478 |
| schema | 100 | 2,712 | 305.5 | 124.5 | 0.9700 | 0.8261 | 1.0000 |

Distinct-2 here is the canonical 24-token-window metric (mean/min/max across
samples). Medians are shown for the size metrics.

## 5. Fitness notes

**Degenerate vs. valid prose — clean separation on Distinct-2 alone.**
With the canonical 24-token window, the degenerate corpus never exceeds
D-2 = 0.3478 while valid prose never drops below 0.7391 — a discriminating band of
roughly (0.348, 0.739), so a single threshold in that band separates the two corpora
perfectly. On the full-sample metric the separation is even wider (degenerate max
0.0494 vs. valid prose min 0.6151), consistent with the builder's certification
assertion (full-sample D-2 < 0.20). A word-loop collapse detector using Distinct-2
alone would flag every degenerate sample and no valid-prose sample.

**Schema vs. valid prose — no low-side separation; schema sits *above* prose.**
The schema corpus is high-diversity: its 24-token D-2 never goes below 0.8261 (mean
0.9700), and on the full sample never below 0.9458 (mean 0.9883). It does not
separate from valid prose on the repetition side — both corpora reach 1.0000, and
valid prose's *minimum* is actually lower (0.7391 vs. 0.8261). That is by design:
the schema corpus models structured generation that a variety governor should *not*
flag as a collapse trap. Distinct-2 alone cannot tell schema from ordinary prose at
the top of the distribution; the two are separated only in that schema never dips
into prose's low-D-2 tail.

**Window sensitivity.** The 24-token fast-path window narrows the degenerate margin
relative to the full-sample metric (degenerate mean rises from 0.0285 full-sample to
0.2213 windowed) because a short window cannot average out longer motifs (up to 8
tokens). Even so, the degenerate maximum stays cleanly below the valid-prose minimum.
The discriminating band for any future threshold work is (0.348, 0.739) on the
24-token window.

**Vocabulary footprint.** Valid prose is open-domain (23,602 unique word-tokens over
200 samples); degenerate is an almost-closed loop (132 over 100); schema is a closed
English vocabulary plus JSON structural tokens (3,857 over 100). This is consistent
with the corpora's intended roles: degenerate models a repetition trap, schema models
diverse structured generation, prose models ordinary open-domain text.

## 6. Drift fingerprint

The date and the three sha256 values below pin this profile to the exact bytes
analyzed; any future regeneration or edit of a corpus will change one of these and
be detectable.

- date: **2026-08-08**
- valid prose: `0f7ccae3a302fb5f5c9ddd9b62206584904edb24764ee2880506e479b7da6d53`
  (matches `data/t2s_bench/valid_subset_200.cert.txt` / `.jsonl.sha256`)
- degenerate: `e1c576997a549372dac01d76712964e9a36882d8065ac0bf6470d474e678dedf`
  (matches DS-001 smoke baseline)
- schema: `648e8330f999dc13edf874862387cc30d93aabcbbf28a0b6d356183fa1b490bb`

Corpus generation seeds for provenance: valid prose subset certification refill
`random.Random(42)` (`valid_subset_200.cert.txt`); degenerate `SEED = 2603`
(`scripts/build_t2s_degenerate.py`); schema `SEED = 4004`
(`scripts/build_schema_corpus.py`).
