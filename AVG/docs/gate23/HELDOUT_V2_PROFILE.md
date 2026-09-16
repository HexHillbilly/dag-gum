# DS-027 — Rebuilt Held-Out Hard-Degenerate Slice (Part A, v2)

> Fixture curation — no gate thresholds, no governor changes, no
> green/red verdict. The gap below is a REPORTED measurement; if
> induction had failed, that would be a valid negative result.
> The slice-quality stop is applied per the gate-2.3b greenlight.

## Run metadata

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B` |
| Revision | `8faed761d45a263340a0528343f099c05c9a4323` |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 4245 |
| grid seeds | SEED*1000 + sample_idx*10 + cell_idx |
| control selection seed | 42 |
| max_new_tokens / min_new_tokens | 128 / 128 |
| Mutation | first 3 whitespace tokens x6 + original prompt (= first-3-token phrase x7) |
| Runtime (wall clock) | 893.3 s |

## Mutation spec (greenlight construction)

For each of the 100 held-out prose records, ONE mutated prompt is
built from the prose text:

```
first3 = first 3 whitespace tokens of the prose text
original_prompt = first3 (the unmutated 3-token prompt)
mutated_prompt = (first3 repeated 6 times) + " " + original_prompt
              = first3 repeated 7 times in total
```

The mutated prompt is `7 x 3` = `21` whitespace tokens of pure repetition with no prose continuation, so the model has no coherent context to escape to.

## Decoding grid (degeneracy-induction)

All grid cells use `do_sample=True` and
`repetition_penalty=1.0` (no anti-repetition 
pressure). This is the proven ds-023 grid.

| # | temperature | top_p |
|---|------------|-------|
| 0 | 0.9 | 0.95 |
| 1 | 0.9 | 1.0 |
| 2 | 1.2 | 0.95 |
| 3 | 1.2 | 1.0 |

The grid yields 100 x 4 = 400 records; the 100 with the LOWEST last-24-token Distinct-2 are curated into the fixture.

## Determinism smoke (scaffolding)

Two grid records were regenerated with identical seeds and compared:

| sample_idx | cell_idx | seed | Run A sha256 | Run B sha256 | Identical |
|---|---|---|---|---|---|
| 0 | 0 | 4245000 | `22ed914dca46d1ed9feaaa0897703fbf781095992ab3b1710a99c876ef837867` | `22ed914dca46d1ed9feaaa0897703fbf781095992ab3b1710a99c876ef837867` | YES |
| 1 | 3 | 4245013 | `a98ef4a760ada87f412756658fc9881fe2a023414f7c35376095077e982bb444` | `a98ef4a760ada87f412756658fc9881fe2a023414f7c35376095077e982bb444` | YES |

All identical: **True**

## Distinct-2 (last 24 generated tokens)

Percentile convention: p10 uses the nearest-rank method (value actually
present), matching the DS-023 builder; NumPy-linear percentile (used in
Gate 2.2/2.3 reports) may differ slightly.

| Corpus | Mean | Median | p10 |
|--------|------|--------|-----|
| heldout_degenerate_v2 (curated) | 0.1643 | 0.1739 | 0.1304 |
| heldout-100 prose (teacher-forced text) | 0.9700 | 1.0000 | 0.8696 |


Curated corpus mean Distinct-2 is 0.8057 BELOW the heldout-100 prose mean (0.1643 vs 0.9700).

## Slice-quality stop (gate-2.3b greenlight)

Stop condition: curated mean Distinct-2 must be <= 0.5 * heldout-100
prose mean (ds-025 measured 0.9700).

- heldout-100 prose mean (this run): 0.9700
- 0.5 * prose mean (this run): 0.4850
- curated mean Distinct-2: 0.1643
- PASS: 0.1643 <= 0.4850 -> YES

Slice-quality stop PASSED; scoring proceeds under the frozen
thresholds in `docs/gate23/FROZEN_THRESHOLDS.md`.

## Held-out prose slice selection (seeded complement)

100 of the 200 certified valid_subset_200 records were selected as
the Gate 2.2 prose control with `random.Random(42).sample(records, 100)` sorted by corpus `id`. The heldout-100 slice is the COMPLEMENT of that control within `valid_subset_200` (ids below).

Heldout ids: `[3, 5, 7, 9, 12, 22, 33, 35, 40, 44, 47, 51, 61, 62, 63, 70, 73, 74, 91, 109, 110, 112, 116, 118, 125, 126, 136, 137, 138, 165, 166, 174, 183, 185, 193, 194, 197, 210, 211, 216, 237, 250, 252, 270, 273, 274, 278, 279, 285, 298, 299, 302, 306, 311, 313, 327, 330, 335, 338, 346, 347, 352, 357, 362, 366, 367, 373, 374, 379, 380, 386, 395, 405, 409, 411, 414, 424, 427, 429, 430, 436, 438, 445, 455, 456, 464, 467, 478, 480, 481, 482, 483, 484, 485, 486, 487, 492, 494, 498, 499]`

## Corpus curation provenance

Grid records were sorted by `(distinct_2, sample_idx, cell_idx)` 
and the first 100 taken; stable tie-break keeps selection 
byte-identical across runs.

## Fixture

- JSONL: `tests/fixtures/heldout_degenerate_v2.jsonl`
- sha256: `50331d30338a9f92cd189cdaad3c57d82f91310f31073ebb6ed440bcadc0b181`
