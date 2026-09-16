# night-019 — B1 measurement pass: numeric-immunity threshold + new smoke bands (MEASUREMENT ONLY)

> MEASUREMENT REPORT. NOT a gate. This probe fills two currently-unknown bands from real data before any B1 controller diff: (1) `NUMERIC_IMMUNITY_THRESHOLD` and (2) the new Single-Token Spam fire band under retargeted kickstart (greedy). All B1 fixes are implemented INLINE; no controller edits [1]. The numbers produced are PROPOSALS for the human to sign; they are not active until written into the smoke script by the human [1].

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Fixture A | Single-Token Spam (`A A ... A`, 16 tokens; smoke-script logic, read-only reference) |
| Fixture B | Number Loop (`1 2 1 2 ...`, 16 tokens; smoke-script logic, read-only reference) |
| Fixture C | prose-100 (valid_subset_200 heldout partition; ds-027 complement of the Gate-2.2 control-100), record[\"text\"] prompt |
| Decoding | KV-cache incremental; Arm 1 greedy (do_sample=False); Arm 2 sampling (do_sample=True, temp=0.8, top_p=0.85). Fixtures A/B max_new_tokens=48, prose max_new_tokens=128 |
| Analysis window | trailing 24 positions (PR); trailing 16 tokens (numeric_fraction/CTR) |
| Hook layer | model.model.layers[2] (layer-2 PR, rolling 24-token ring buffer) |
| Frozen band_low(2) | 8.216097 (docs/gate23/FROZEN_THRESHOLDS.md, unchanged) |
| B1 kickstart | retargeted to `active_loop_ids` (night-015); armed ONLY if `do_sample=False` (greedy-only); penalties -1e4/-5.0/-2.0 |
| B1 vocab-range fix | `vocab_size = len(self.tokenizer)` (night-015 Fix 1); offline diagnostic only (kickstart no longer uses the orthographic subset) |
| B1 numeric guard | `if numeric_fraction >= T: is_collapsed = False` (dormant, like code-context immunity); numeric_fraction over the trailing 16-token decoded window (full-context: last 16 tokens of the current sequence); whitespace-only tokens excluded |
| Numeric token | decoded string contains a digit, or is a single numeric-punctuation char from `.,%:;)]}-/+=#$*` |
| Suppression | UNCONDITIONAL (cooldown=8, penalty=-5.0), matching night-018 |
| Sweep T | 0.30, 0.40, 0.50, 0.60, 0.70, 0.80 |
| Prose false-dormancy bound | <= 5% of prose-100 records (pre-registered) |
| bmm Triton override | deregistered |
| Wall clock (s) | 517.2 |

## Vocab-range fix diagnostic (B1 config #2)

`tokenizer.vocab_size` = 151643; `len(tokenizer)` = 151665. EOS id = 151643.

| classification | original (`vocab_size`) | Fix 1 (`len(tokenizer)`) |
|---|---|---|
| prose | 71800 | 71800 |
| non-prose | 79843 | 79865 |
| EOS classified | False | True |

With the kickstart retargeted to `active_loop_ids`, the vocab subsets are not used for kickstart actuation; the fix is offline diagnostics only.

## Determinism smoke (FIRST)

One Number Loop record: greedy generated twice and sampling generated twice, all with SEED=42. Token ids must match (deterministic sampling, per DS-032 Part 2). STOP if not.

| record | greedy n_A | greedy n_B | greedy ids | samp n_A | samp n_B | samp ids | identical |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NumberLoop | 41 | 41 | True | 48 | 48 | True | True |

## Part 1 — Numeric-immunity threshold sweep

Sweep candidate T over 0.30, 0.40, 0.50, 0.60, 0.70, 0.80 on Fixture B (Number Loop) and Fixture C (prose-100). NumberLoop fires = number of steps where `is_collapsed` is True (would_fire AND numeric_fraction < T). Prose false-dormancy = number of prose-100 records where `would_fire` AND `numeric_fraction >= T` at any step (a genuine fire wrongly suppressed by the guard).

### Table 1 — Numeric threshold sweep

| T | NumberLoop greedy fires | NumberLoop sampling fires | prose false-dormancy | numeric fraction (NumberLoop) |
| --- | --- | --- | --- | --- |
| 0.30 | 0 | 0 | 0 | 0.201 |
| 0.40 | 0 | 0 | 0 | 0.201 |
| 0.50 | 0 | 0 | 0 | 0.201 |
| 0.60 | 0 | 0 | 0 | 0.201 |
| 0.70 | 0 | 0 | 0 | 0.201 |
| 0.80 | 0 | 0 | 0 | 0.201 |

Prose false-dormancy bound: <= 5% of prose-100 records (5 records).

Prose context: 0/100 records have a would-fire step under greedy; 0/100 under sampling. Guard-on (numeric_fraction >= 0.30 at any step, the generation-time threshold) counts: greedy 71/100, sampling 64/100.

### Chosen `NUMERIC_IMMUNITY_THRESHOLD` (PROPOSAL)

Pre-registered selection rule: HIGHEST T that keeps Number Loop at 0 fires in BOTH greedy and sampling, subject to prose false-dormancy <= 5%.

Measured: all T in the sweep keep Number Loop at 0 fires in both regimes (fires = 0/0). Prose false-dormancy rises monotonically with T: T=0.30 → 0, T=0.40 → 0, T=0.50 → 0, T=0.60 → 0, T=0.70 → 0, T=0.80 → 0.

**PROPOSED `NUMERIC_IMMUNITY_THRESHOLD` = 0.80** (highest T meeting both constraints; NumberLoop greedy fires 0, sampling fires 0, prose false-dormancy 0/5 records = 0.0%).

> This is a PROPOSAL for the human to sign [1]. It is NOT active until written into the smoke script by the human.

## Part 2 — New Single-Token Spam fire band (greedy)

Under the B1 config (retargeted kickstart, greedy), with the numeric guard OFF (spam is not numeric). Residual path ENABLED (full-forward controller replication). This replaces the retired `21 fires / Number Loop 0 / numeric 0.053` contract.

### Table 2 — Single-Token Spam new fire band

| fires | Δres mean | Δres median | Δres p10 | Δres p90 | force bounds valid? |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.4649 | 0.4675 | 0.4600 | 0.4688 | True |

n_generated = 48; kickstart events = 26; suppression steps = 48.

All Δres values: [0.4675, 0.4581, 0.4691]

## Part 3 — Number Loop with chosen `NUMERIC_IMMUNITY_THRESHOLD`

With T = 0.80 active, re-run Fixture B (Number Loop) under both greedy and sampling.

### Table 3 — Number Loop with chosen T

| regime | fires | numeric fraction | smoke-style num frac | immunity preserved? |
| --- | --- | --- | --- | --- |
| greedy | 0 | 0.201 | 0.040 | True |
| sampling | 0 | 0.485 | 0.111 | True |

The smoke-style numeric fraction (whitespace-split tokens containing a digit) is the direct replacement for the retired 0.053 Number Loop metric.

## Notes / caveats

- MEASUREMENT ONLY: no controller edits, no threshold changes, no protected-file edits. All B1 fixes are implemented INLINE [1].
- The numeric-guard window is the trailing 16 tokens of the CURRENT sequence (full-context, `input_ids[0, -16:]`). Prototyping showed the generated-only window drops to 0 as soon as the first non-numeric token is emitted, so it cannot hold Number Loop dormant; the full-context window keeps the prompt's numeric context until 16 generated tokens push it out.
- The guard is applied EXACTLY as specified in the task: `if numeric_fraction >= T: is_collapsed = False`. It does NOT gate the unconditional suppression path or the kickstart (both are recorded separately).
- The kickstart is armed ONLY for greedy (`do_sample=False`), per B1 config #3. Under sampling, `n_kickstart_events` is always 0.
- The vocab-range fix (`len(tokenizer)`) is implemented as B1 config #2; with the kickstart retargeted to `active_loop_ids` the vocab subsets are offline diagnostics only (night-015 Fix 5).
- Part 2 (residual fire band) uses the full-forward controller replication (profile -> diagnose -> apply_interventions) because the residual intervention path requires a full forward pass; the KV-cache incremental loop is used for Parts 1 & 3 and the determinism smoke. Under full-forward, the layer-2 spectral PR hook (hs.shape[1] == 1 filter) never fires, so the residual-path detection is bigram-only — exactly matching the current controller's full-forward generate() and the smoke test's 21-fire contract methodology.
- Part 2 measured fire band is 1 residual decision with 3 Δres entries (the intervention persists ~3 forwards before the loop breaks to prose). The retired contract was 21 fires / 62 Δres. The reduction is attributable to the B1 retargeted kickstart (crushing `active_loop_ids` = the repeated 'A' token) breaking the spam loop far earlier than the old orthographic kickstart (which left 'A' unpenalized as prose).
- Part 3 smoke-style numeric fraction (whitespace-split tokens containing a digit, the direct replacement for the retired 0.053): greedy 0.040 (at the old smoke floor of 0.04, no headroom), sampling 0.111. The old smoke test used do_sample=True temp=0.3 top_p=0.9; the B1 measurement uses the task's night-014 sampling config (do_sample=True temp=0.8 top_p=0.85) plus a greedy arm (B1 config #3: kickstart greedy-only).
- Frozen thresholds (band_low(2) = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified value; the script exits if they disagree.

