# DS-023 — Qwen Degeneracy-Induction Stimulus Corpus

> Corpus curation — no gate thresholds, no governor changes.
> The gap below is a REPORTED measurement, never a forced one;
> if induction had failed, that would be a valid negative result.

## Run metadata

| Field | Value |
|-------|-------|
| Model | `Qwen/Qwen2.5-1.5B` |
| Device | `cuda` |
| dtype | `torch.float16` |
| SEED | `4243` (grid seeds `SEED*1000 + sample_idx*10 + cell_idx`) |
| control selection seed | `42` (`random.Random(42).sample`, sorted by id) |
| max_new_tokens / min_new_tokens | `128` / `128` (every record is exactly 128 generated tokens) |
| Prompt truncation | first `40` whitespace tokens |
| Runtime (wall clock) | 1134.4 s (18.9 min) |

## Decoding grid (degeneracy-induction)

All grid cells use `do_sample=True` and
`repetition_penalty=1.0` (no anti-repetition pressure).

| # | temperature | top_p |
|---|------------|-------|
| 0 | 0.9 | 0.95 |
| 1 | 0.9 | 1.0 |
| 2 | 1.2 | 0.95 |
| 3 | 1.2 | 1.0 |

The grid yields 400 records (100 prompts x 4 cells); the 100 with the LOWEST last-24-token Distinct-2 are curated into the fixture.

## Determinism smoke (scaffolding)

Two grid records were regenerated with identical seeds and compared:

| sample_idx | cell_idx | seed | Run A sha256 | Run B sha256 | Identical |
|---|---|---|---|---|---|
| 0 | 0 | 4243000 | `b999247525b76ee98ff5e80389b08579499c47b9f2561eb4f67db92cfc7dbdc5` | `b999247525b76ee98ff5e80389b08579499c47b9f2561eb4f67db92cfc7dbdc5` | YES |
| 1 | 3 | 4243013 | `a44211189cb21db3b80c174335effdf95d2434a7500efc85a2f906a4bd07981b` | `a44211189cb21db3b80c174335effdf95d2434a7500efc85a2f906a4bd07981b` | YES |

## Distinct-2 (last 24 generated tokens)

| Corpus | Mean | Median | p10 |
|--------|------|--------|-----|
| qwen_degenerate (curated) | 0.1296 | 0.1304 | 0.0870 |
| valid_subset_200 control (greedy) | 0.7952 | 0.8261 | 0.5217 |


Curated corpus mean Distinct-2 is 0.6657 BELOW the greedy prose control mean (0.1296 vs 0.7952).

### valid_subset_200 control selection (seeded)

100 of the 200 certified valid_subset_200 records were selected deterministically with `random.Random(42).sample(records, 100)`, then sorted by corpus `id` for processing order.

Selected ids: `[0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80, 82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148, 150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209, 214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282, 283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349, 350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417, 433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496]`

### Corpus curation provenance

Grid records were sorted by `(distinct_2, sample_idx, cell_idx)` and
the first 100 taken; stable tie-break keeps selection
byte-identical across runs.  The 100 curated records draw from
38 distinct t2s_degenerate prompts (multiple grid cells of the same prompt are allowed).

## Fixture

- JSONL: `tests/fixtures/qwen_degenerate.jsonl`
- sha256: `2e0f21a02fc05308c00ac64a18f35e18c9f580839abe885716c81d6a39f991bf`

## Data

- Grid records (all 400, unsorted): not persisted as a fixture;
  the curated 100 are the deliverable.
