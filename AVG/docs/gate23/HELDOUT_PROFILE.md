# DS-025 — Held-Out Hard-Degenerate Slice (Part B)

> Fixture curation — no gate thresholds, no governor changes, no
> green/red verdict. The gap below is a REPORTED measurement; if
> induction had failed, that would be a valid negative result.

## Run metadata

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` |
| Revision | `8faed761d45a263340a0528343f099c05c9a4323` |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 4244 |
| grid seeds | SEED*1000 + sample_idx*10 + cell_idx |
| control selection seed | 42 |
| max_new_tokens / min_new_tokens | 128 / 128 |
| Prompt truncation | first 40 whitespace tokens |
| Runtime (wall clock) | 439.3 s |

## Decoding grid (degeneracy-induction)

All grid cells use `do_sample=True`, `top_k=40`, `top_p=1.0` 
(disabled), and `repetition_penalty=1.0` (no anti-repetition 
pressure).

| # | temperature | top_k | top_p |
|---|------------|-------|-------|
| 0 | 1.0 | 40 | 1.0 |
| 1 | 1.4 | 40 | 1.0 |

The grid yields 100 x 2 = 200 records; the 100 with the LOWEST last-24-token Distinct-2 are curated into the fixture.

## Determinism smoke (scaffolding)

Two grid records were regenerated with identical seeds and compared:

| sample_idx | cell_idx | seed | Run A sha256 | Run B sha256 | Identical |
|---|---|---|---|---|---|
| 0 | 0 | 4244000 | `bc5992d2b92a58ae7848142ac7d242642e1d8e2e509dc1cfbe6d27b6fa6864f9` | `bc5992d2b92a58ae7848142ac7d242642e1d8e2e509dc1cfbe6d27b6fa6864f9` | YES |
| 1 | 1 | 4244011 | `89a6b33984e5f1d9ea55deb9d76c408307782c798a57834e654f36ac7a6c1aaf` | `89a6b33984e5f1d9ea55deb9d76c408307782c798a57834e654f36ac7a6c1aaf` | YES |

All identical: **True**

## Distinct-2 (last 24 generated tokens)

Percentile convention: p10 uses the nearest-rank method (value actually present), matching the DS-023 builder; NumPy-linear percentile (used in Gate 2.2/2.3 reports) may differ slightly.

| Corpus | Mean | Median | p10 |
|--------|------|--------|-----|
| heldout_degenerate (curated) | 0.9609 | 1.0000 | 0.9130 |
| heldout-100 prose (teacher-forced text) | 0.9700 | 1.0000 | 0.8696 |


Curated corpus mean Distinct-2 is 0.0091 BELOW the heldout-100 prose mean (0.9609 vs 0.9700).

## Held-out prose slice selection (seeded complement)

100 of the 200 certified valid_subset_200 records were selected as the Gate 2.2 prose control with `random.Random(42).sample(records, 100)` sorted by corpus `id`. The heldout-100 slice is the COMPLEMENT of that control within `valid_subset_200` (ids below).

Heldout ids: `[3, 5, 7, 9, 12, 22, 33, 35, 40, 44, 47, 51, 61, 62, 63, 70, 73, 74, 91, 109, 110, 112, 116, 118, 125, 126, 136, 137, 138, 165, 166, 174, 183, 185, 193, 194, 197, 210, 211, 216, 237, 250, 252, 270, 273, 274, 278, 279, 285, 298, 299, 302, 306, 311, 313, 327, 330, 335, 338, 346, 347, 352, 357, 362, 366, 367, 373, 374, 379, 380, 386, 395, 405, 409, 411, 414, 424, 427, 429, 430, 436, 438, 445, 455, 456, 464, 467, 478, 480, 481, 482, 483, 484, 485, 486, 487, 492, 494, 498, 499]`

## Corpus curation provenance

Grid records were sorted by `(distinct_2, sample_idx, cell_idx)` 
and the first 100 taken; stable tie-break keeps selection 
byte-identical across runs.

## Fixture

- JSONL: `tests/fixtures/heldout_degenerate.jsonl`
- sha256: `6d6497649512a8c2d5af8d078208b2a283cdabad3d6269b95b3d90474214d99b`
