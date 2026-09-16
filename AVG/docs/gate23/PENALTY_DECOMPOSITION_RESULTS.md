# DS-030 — Logit-penalty decomposition: suppression vs kickstart

> MEASUREMENT REPORT. This document reports measured values only.
> No threshold is created or modified, no gate script is touched,
> no governor/controller.py edit, and no green/red verdict and no
> mechanism story are offered. Decomposition conclusions are the
> human's act.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Records | 50 seeded heldout-degenerate v2 (random.Random(42).sample, sorted) |
| Decoding | greedy (do_sample=False), max_new_tokens=128 |
| Analysis window | trailing 24 generated positions |
| Dormant | REUSED from DS-028 (docs/gate23/layer_effect_persistent_results.jsonl, join on record_id) |
| Conditions | B (combined, live), D (suppression-only, live), E (kickstart-only, live) |
| Suppression budget | cooldown=8, penalty=-5.0 (controller.py:137,745) |
| Kickstart budget | penalties={3: -10000.0, 2: -5.0, 1: -2.0}, trigger CTR<0.5 AND active_loop_ids (controller.py:691,753-758) |
| top_p | 0.85 (controller.py:138,663) |
| B CTR sub-sampling | every_k=2 or diversity<0.4 (production, controller.py:136,683); E computes CTR every step |
| bmm Triton override | deregistered |
| Wall clock (s) | 323.1 |

## Determinism smoke

One heldout-degenerate record, each live condition (B, D, E),
generated twice on the same model load; generated token ids must
match exactly.

| condition | run A tokens | run B tokens | identical |
|---|---|---|---|
| B | 1 | 1 | True |
| D | 83 | 83 | True |
| E | 1 | 1 | True |
|  |  | ALL | True |

## Method summary

- Per record, per live condition {B, D, E}: one greedy generation
  from the record's mutated_prompt; dormant is reused from DS-028.
- Condition B (combined): token suppression + kickstart, no
  residual hooks; production CTR sub-sampling (every_k=2 or
  diversity<0.40); EOS-terminating loop.
- Condition D (suppression-only): token suppression ONLY, applied
  at every decoding step; cooldown initial/reset 8, -5.0 penalty;
  top_p=0.85 when suppression active; NO kickstart, NO vocab cache.
- Condition E (kickstart-only): kickstart vocabulary steering ONLY;
  prose/non-prose sets via production _cache_vocabulary_subsets;
  trailing-CTR every step; kickstart_counter=3 when CTR<0.50 AND
  active_loop_ids; penalties -1e4/-5.0/-2.0; top_p=0.85 when active;
  NO token suppression.
- Distinct-2 on trailing 24 generated tokens (ds-025 convention);
  CTR is the coherent-token ratio of the decoded continuation.
- Δ vs dormant = condition − dormant (per record).

## Primary aggregation table

ΔDistinct-2 / ΔCTR are per-record deltas vs the DS-028 dormant
continuation; #byte-id counts records whose decoded continuation is
byte-identical to dormant; EOS rate is the fraction of records that
terminated early (n_generated < max_new_tokens).

| cond | n | ΔD2 mean | ΔD2 med | #ΔD2>0 | #byte-id | ΔCTR mean | EOS rate | mean n_gen | n_gen=1 | n_gen=128 |
|---|---|---|---|---|---|---|---|---|---|---|
| dormant | 50 | 0.0000 | 0.0000 | 0 | 50 | 0.0000 | 0.0000 | 128.00 | 0 | 50 |
| B | 50 | 0.8070 | 0.8261 | 50 | 0 | -0.0514 | 0.5400 | 64.30 | 19 | 23 |
| D | 50 | 0.7435 | 0.7826 | 48 | 2 | 0.1314 | 0.2800 | 111.86 | 3 | 36 |
| E | 50 | 0.2243 | 0.0000 | 15 | 27 | -0.0590 | 0.3000 | 89.94 | 13 | 35 |

## Continuation length distribution per condition

| cond | n_gen=1 | n_gen=2..23 | n_gen=24..127 | n_gen=128 | mean n_gen | median n_gen |
|---|---|---|---|---|---|---|
| dormant | 0 | 0 | 0 | 50 | 128.00 | 128.00 |
| B | 19 | 6 | 2 | 23 | 64.30 | 59.00 |
| D | 3 | 0 | 11 | 36 | 111.86 | 128.00 |
| E | 13 | 2 | 0 | 35 | 89.94 | 128.00 |

## Distinct-2 for records reaching n>=24 vs n<24

Distinct-2 is measured over the trailing 24 generated tokens;
records with fewer than 24 generated tokens use the cold-start
fallback (Distinct-2 = 1.0 when n<4). Splitting by n>=24 / n<24
separates prose-recovery continuations from early-EOS continuations.

| cond | subset | n | d2 mean | d2 med |
|---|---|---|---|---|
| dormant | n>=24 | 50 | 0.1696 | 0.1739 |
| dormant | n<24 | 0 | nan | nan |
| B | n>=24 | 25 | 0.9530 | 0.9565 |
| B | n<24 | 25 | 1.0000 | 1.0000 |
| D | n>=24 | 47 | 0.9075 | 0.9565 |
| D | n<24 | 3 | 1.0000 | 1.0000 |
| E | n>=24 | 35 | 0.1342 | 0.1304 |
| E | n<24 | 15 | 1.0000 | 1.0000 |

## Penalty activity stats (per record, per live condition)

Aggregated over the 50 records of each live condition. Suppression
counts are steps with at least one active suppression penalty;
n_suppression_applications is the total (token, step) penalty
applications; kickstart_events counts times kickstart_counter was
set to 3; kickstart_steps counts steps with a kickstart penalty.

| cond | supp steps | n supp appl | kick events | kick steps | top_p steps | max cooldown |
|---|---|---|---|---|---|---|
| B | 61.92 | 312.54 | 1.54 | 3.62 | 61.92 | 6.32 |
| D | 110.26 | 619.3 | 0.0 | 0.0 | 110.26 | 9.16 |
| E | 0.0 | 0.0 | 2.04 | 5.28 | 3.76 | 0.0 |

## Per-record interaction table

D/E/B rescues = per-record ΔDistinct-2 > 0 vs dormant. D ≡ E = D and
E generated token ids are identical. D ≠ dormant / E ≠ dormant = the
condition's decoded continuation differs from dormant. Genuine
interaction would be B rescues with D≠E and both ≠ dormant;
degenerate nulls are B also failing, or D≡E≡dormant.

| record_id | D rescues | E rescues | B rescues | D ≡ E? | D ≠ dormant? | E ≠ dormant? |
|---|---|---|---|---|---|---|
| 0 | Y | Y | Y | N | Y | Y |
| 2 | Y | N | Y | N | Y | N |
| 3 | Y | N | Y | N | Y | N |
| 4 | Y | N | Y | N | Y | N |
| 5 | Y | N | Y | N | Y | N |
| 6 | Y | N | Y | N | Y | N |
| 11 | Y | N | Y | N | Y | N |
| 13 | Y | N | Y | N | Y | N |
| 14 | Y | N | Y | N | Y | N |
| 16 | Y | N | Y | N | Y | N |
| 17 | Y | N | Y | N | Y | N |
| 19 | N | N | Y | Y | N | N |
| 20 | N | N | Y | Y | N | N |
| 22 | Y | N | Y | N | Y | Y |
| 24 | Y | Y | Y | N | Y | Y |
| 25 | Y | N | Y | N | Y | N |
| 27 | Y | N | Y | N | Y | N |
| 28 | Y | N | Y | N | Y | N |
| 29 | Y | N | Y | N | Y | N |
| 31 | Y | N | Y | N | Y | N |
| 35 | Y | Y | Y | N | Y | Y |
| 38 | Y | Y | Y | N | Y | Y |
| 43 | Y | Y | Y | N | Y | Y |
| 46 | Y | Y | Y | N | Y | Y |
| 51 | Y | N | Y | N | Y | N |
| 53 | Y | N | Y | N | Y | N |
| 54 | Y | N | Y | N | Y | N |
| 57 | Y | N | Y | N | Y | N |
| 58 | Y | N | Y | N | Y | N |
| 62 | Y | Y | Y | N | Y | Y |
| 64 | Y | N | Y | N | Y | Y |
| 67 | Y | N | Y | N | Y | Y |
| 68 | Y | N | Y | N | Y | Y |
| 69 | Y | N | Y | N | Y | Y |
| 71 | Y | N | Y | N | Y | N |
| 75 | Y | Y | Y | N | Y | Y |
| 77 | Y | N | Y | N | Y | N |
| 79 | Y | N | Y | N | Y | N |
| 81 | Y | N | Y | N | Y | N |
| 82 | Y | N | Y | N | Y | N |
| 84 | Y | Y | Y | N | Y | Y |
| 86 | Y | Y | Y | N | Y | Y |
| 88 | Y | Y | Y | N | Y | Y |
| 89 | Y | Y | Y | N | Y | Y |
| 90 | Y | N | Y | N | Y | Y |
| 93 | Y | N | Y | N | Y | Y |
| 94 | Y | Y | Y | N | Y | Y |
| 95 | Y | N | Y | N | Y | Y |
| 97 | Y | Y | Y | N | Y | Y |
| 99 | Y | Y | Y | N | Y | Y |

Interaction-table summary counts:

| metric | count |
|---|---|
| D rescues (ΔD2>0) | 48 |
| E rescues (ΔD2>0) | 15 |
| B rescues (ΔD2>0) | 50 |
| D ≡ E (token-identical) | 2 |
| D ≠ dormant | 48 |
| E ≠ dormant | 23 |
| both D and E null while B rescues | 2 |
| genuine-interaction shape (B rescues, D≠E, both ≠ dormant) | 23 |
| degenerate-null shape (B fails, or D≡E≡dormant) | 2 |

These are measurement categories, not a verdict.

## Per-record JSONL

Per-record results (dormant plus live B/D/E continuations, deltas,
penalty activity incl. per-step logs) are in
`penalty_decomposition_results.jsonl`.

## Notes / caveats

- MEASUREMENT ONLY. No verdict, no mechanism story.
- Condition B uses the production CTR sub-sampling (every_k=2 or
  diversity<0.40) and the production EOS-terminating loop;
  Condition E computes trailing-CTR at every step (DS-030 spec), so
  E's kickstart trigger is slightly more sensitive than B's.
- top_p follows the production post-decrement check (controller.py:762):
  on the third kickstart step (counter 1->0), the -2.0 penalty is
  applied but top_p is not, for both B and E.
- The controller's generate() does not break on EOS; the DS-029
  measurement loop did (early-EOS is a first-class outcome here).
- Dormant data is reused from DS-028 and is layer-independent;
  the first layer entry per record was joined on record_id.
- No layer sweep: logit mechanisms are layer-independent.
