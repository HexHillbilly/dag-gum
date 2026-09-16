# DS-011 Substrate-Leakage Measurement Probe

> MEASUREMENT probe — no thresholds, no gate logic, no asserts on
> outcomes. Data and report only; human review and humans interpret.

## Hypothesis under test (parked from the Jaynesian/adversarial critic exchange)

> "Blind orthogonal kicks at long rollout lengths (S>=128) cause
> measurable semantic-coherence degradation (substrate leakage)
> compared to dormant generation."

## Run metadata

| Field | Value |
|-------|-------|
| Model | `distilgpt2` |
| Device | `cuda` |
| SEED | `42` (torch.manual_seed + random.Random; cuda.manual_seed_all when cuda) |
| max_new_tokens | `128` |
| Prompt truncation | first `40` whitespace tokens |
| do_sample | `False` (greedy) |
| t2s_degenerate samples | 100/100 |
| valid_subset_200 samples | 100/200 (seeded select, see below) |
| Conditions per prompt | dormant (`intervene=False`), active (`intervene=True`) |
| Result records | 400 (200 dormant + 200 active) |
| Runtime (wall clock) | 344.0 s (5.7 min) |

### Calibration trusted texts (active condition)

1. `The quick brown fox jumps over the lazy dog.`
2. `In physics, spacetime is any mathematical model which fuses the three dimensions of space.`
3. `Photosynthesis is a process used by plants and other organisms to convert light energy into chemical energy.`

### valid_subset_200 selection (seeded)

100 of the 200 certified valid_subset_200 records were selected deterministically with `random.Random(42).sample(records, 100)`, then sorted by corpus `id` for processing order.

Selected ids: `[0, 24, 28, 32, 36, 46, 49, 52, 53, 54, 56, 57, 58, 65, 71, 72, 78, 80, 82, 83, 98, 101, 107, 114, 117, 119, 121, 122, 128, 135, 140, 142, 148, 150, 152, 157, 161, 172, 176, 181, 182, 186, 189, 196, 204, 205, 209, 214, 219, 222, 232, 235, 236, 255, 256, 258, 259, 260, 271, 276, 282, 283, 290, 308, 309, 316, 319, 321, 328, 329, 331, 332, 339, 343, 349, 350, 355, 359, 360, 376, 388, 390, 396, 404, 407, 412, 413, 415, 417, 433, 444, 454, 458, 466, 469, 470, 473, 474, 490, 496]`

## Determinism smoke (scaffolding)

- Subset: 2 prompt(s) — t2s_degenerate#0, t2s_degenerate#1
- Run A SHA-256: `0bd078ddecd2d237902f0fb455809196a13fdea0da88da57bd621b5dbe1bda19`
- Run B SHA-256: `0bd078ddecd2d237902f0fb455809196a13fdea0da88da57bd621b5dbe1bda19`
- Identical: **YES** (4 result records compared)

## Aggregate metric tables

### Distinct-2 (last 24 generated tokens) — t2s_degenerate

| Condition | Mean | Median | p90 |
|-----------|------|--------|-----|
| dormant | 0.5743 | 0.6087 | 0.9565 |
| active | 0.5743 | 0.6087 | 0.9565 |


### Coherent-token ratio (CTR) — t2s_degenerate

| Condition | Mean | Median | p90 |
|-----------|------|--------|-----|
| dormant | 0.9283 | 0.9196 | 1.0000 |
| active | 0.9283 | 0.9196 | 1.0000 |


### Code-syntax-context flag

| Condition | True count | Fraction |
|-----------|-----------|----------|
| dormant | 1 | 0.0100 |
| active | 1 | 0.0100 |

### Intervention counts

| Condition | Mean | Median | p90 | Total | Max |
|-----------|------|--------|-----|-------|-----|
| dormant | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| active | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

### Distinct-2 (last 24 generated tokens) — valid_subset_200

| Condition | Mean | Median | p90 |
|-----------|------|--------|-----|
| dormant | 0.9261 | 0.9565 | 1.0000 |
| active | 0.9261 | 0.9565 | 1.0000 |


### Coherent-token ratio (CTR) — valid_subset_200

| Condition | Mean | Median | p90 |
|-----------|------|--------|-----|
| dormant | 0.8644 | 0.8762 | 0.9196 |
| active | 0.8644 | 0.8762 | 0.9196 |


### Code-syntax-context flag

| Condition | True count | Fraction |
|-----------|-----------|----------|
| dormant | 1 | 0.0100 |
| active | 1 | 0.0100 |

### Intervention counts

| Condition | Mean | Median | p90 | Total | Max |
|-----------|------|--------|-----|-------|-----|
| dormant | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| active | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |

## Observations (strictly factual)

The following statements report measured values only. No verdict on
the hypothesis is offered; that is for human review and human
ratification.

- Across both corpora, 200/200 active continuations were byte-identical to their dormant counterpart on the same prompt (0 interventions were logged in the active condition).
- In `t2s_degenerate`, dormant mean Distinct-2 was 0.5743 (median 0.6087, p90 0.9565); active mean Distinct-2 was 0.5743 (median 0.6087, p90 0.9565).
- In `t2s_degenerate`, dormant mean CTR was 0.9283 (median 0.9196, p90 1.0000); active mean CTR was 0.9283 (median 0.9196, p90 1.0000).
- In `t2s_degenerate`, the active condition logged 0 total interventions (mean 0.0000 per continuation, median 0.0000, p90 0.0000, max 0). The dormant condition logged 0 interventions by construction (`intervene=False`).
- In `t2s_degenerate`, the code-syntax-context flag was True on 1/100 dormant continuations and 1/100 active continuations.
- In `valid_subset_200`, dormant mean Distinct-2 was 0.9261 (median 0.9565, p90 1.0000); active mean Distinct-2 was 0.9261 (median 0.9565, p90 1.0000).
- In `valid_subset_200`, dormant mean CTR was 0.8644 (median 0.8762, p90 0.9196); active mean CTR was 0.8644 (median 0.8762, p90 0.9196).
- In `valid_subset_200`, the active condition logged 0 total interventions (mean 0.0000 per continuation, median 0.0000, p90 0.0000, max 0). The dormant condition logged 0 interventions by construction (`intervene=False`).
- In `valid_subset_200`, the code-syntax-context flag was True on 1/100 dormant continuations and 1/100 active continuations.

## Data

- Results JSONL: `docs/substrate_leakage_results.jsonl`
