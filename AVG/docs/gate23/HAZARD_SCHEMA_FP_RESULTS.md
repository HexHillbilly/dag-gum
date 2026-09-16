# DS-034c-hazard — Dual-predicate FP re-verification on hazard and schema corpora (GATE)

> GATE REPORT (RFC-004 Amendment A3). Dual-predicate OR rule — bigram OR spectral PR at layer 2 — re-verified on structured legitimate text (markdown, JSON, URLs, code-switched prose) under LIVE generation with the production cadence and hysteresis per RFC A3 (DS-034e config, band_low hardening). A FAIL is a STOP signal — no negotiation, no threshold tuning.

## Run metadata

| Field | Value |
|---|---|
| Model | Qwen/Qwen2.5-1.5B |
| Revision | 8faed761d45a263340a0528343f099c05c9a4323 |
| Device | cuda |
| dtype | torch.bfloat16 |
| torch | 2.13.0+cu130 |
| SEED | 42 |
| Corpora | schema_markdown_fenced (ds-007, N=50), schema_urls_strings (ds-007, N=50), prose_code_switch (ds-007, N=50), schema_corpus_v2 (ds-018, N=300); total N=450 |
| Prompt | record[\"text\"] |
| Decoding | greedy (do_sample=False), KV-cache incremental, max_new_tokens=128 |
| Analysis window | trailing 24 hidden states (layer-2 hook, rolling ring buffer) |
| Hook layer | model.model.layers[2] |
| Frozen T_PR(2) | 10.954796 (docs/gate23/FROZEN_THRESHOLDS.md, ds-025 Part A freeze) |
| Spectral band_low(2) | 8.216097 (DS-034e config: spectral_collapse threshold, frozen) |
| PR instrument | participation_ratio() from core/metrics.py (existing), float32 promotion |
| Cadence | every-step (confirmed by DS-034a at Δ = 0.7027 ms/token) |
| Persistence hysteresis | ≥2 consecutive is_collapsed steps (RFC A3 tightening; per-predicate counters) |
| Spectral-only corroboration | when bigram_ctr < 2, spectral_fire requires token_diversity < 0.40 (DS-034e) |
| Code-context immunity | spectral_fire forced False when is_code_syntax_context is True (RFC A3) |
| Actuation | production logit-penalty path: token suppression cooldown=8 penalty=-5.0, kickstart counter=3, top_p=0.85 |
| Dormant | LIVE (greedy, KV-cache, no hooks/penalties), generated for all 450 records |
| bmm Triton override | deregistered |
| Wall clock (s) | 142.6 |

## Gate criteria (quoted from the DS-034c-hazard task file)

> - Primary: 0 FPs on hazard corpora under simple OR → PASS.
> - If FPs > 0: the schema_markdown_fenced and schema_urls_strings corpora are expected to be covered by the code-context immunity (is_code_syntax_context fires on markdown fences and code patterns). The prose_code_switch and schema_corpus_v2 corpora are the primary risk — they contain unfenced structured text that may trigger spectral fires. Document per-corpus FP counts and which predicate (bigram vs spectral) triggered each FP.
> - A FAIL is a STOP signal — no negotiation, no threshold tuning [1].
> - The hazard/schema FP policy is scoped to these four corpora. The ratified prose-100 FP policy (7 documented FPs, Amendment A3) is unchanged.

FP criterion: a record is a false positive if ΔDistinct-2 > 0 vs dormant AND the predicate fired at least once (the dual-predicate triggered actuation that changed the continuation on legitimate structured text).

## Determinism smoke

Two-to-three hazard/schema records: dormant generated twice AND active (DS-034e dual-predicate) generated twice. Token ids AND the per-step PR log must match exactly. STOP if not. Includes one full-continuation record to exercise the layer-2 PR hook when one is available.

| fixture | corpus_id | prompt_len | n_gen | n PR | dormant id | active id | PR log id | identical |
|---|---|---|---|---|---|---|---|---|
| schema_markdown_fenced | 0 | 191 | 1 | 0 | True | True | True | True |
| prose_code_switch | 0 | 814 | 1 | 0 | True | True | True | True |
| schema_markdown_fenced | 10 | 127 | 128 | 105 | True | True | True | True |
|  | ALL |  |  |  |  |  |  | True |

PR first run A/B: — / —; last run A/B: — / —.

PR first run A/B: — / —; last run A/B: — / —.

PR first run A/B: 15.5287 / 15.5287; last run A/B: 17.1022 / 17.1022.

## Verdict

| metric | value |
|---|---|
| Records scored | 450 |
| Total FPs (all four corpora) | 0 |
| schema_markdown_fenced FPs | 0 |
| schema_urls_strings FPs | 0 |
| prose_code_switch FPs | 0 |
| schema_corpus_v2 FPs | 0 |
| **Gate verdict** | **PASS** |

**PASS** — 0 false positives across all four hazard/schema corpora under the frozen DS-034e dual-predicate.

## Measurement observations

Under LIVE greedy generation with record[\"text\"] as prompt, the model emits the EOS token as the FIRST generated token on 445/450 records (immediate termination). The hazard/schema fixtures are complete structured blocks (closing ``` fence, closing JSON brace) or closed word-salad prose; Qwen2.5-1.5B treats them as self-contained and stops at step 0. As a result:

- The generated continuation is empty on those records, so ΔDistinct-2 = 0 vs dormant (both terminate identically) and the dual-predicate is never evaluated past step 0.
- Only 5 records (all in schema_markdown_fenced: corpus_ids 10, 14, 16, 40, 44) produced a non-trivial continuation (n_gen=128). On those records the layer-2 PR stayed well above band_low (min PR 11.58–14.97 > 8.216097), bigram diversity stayed high, and no predicate fired — active was byte-identical to dormant.
- Therefore the PASS is real but largely **vacuous**: the dual-predicate did not fire on any of the four corpora, but on 445/450 records it had no generated tokens to evaluate. The risk hypothesis (unfenced structured text triggering spectral fires) is neither confirmed nor refuted by live generation on these fixtures.

## Per-corpus metrics

### schema_markdown_fenced (ds-007, N=50)

| metric | value |
|---|---|
| records | 50 |
| false positives | 0 |
| FP rate | 0.00% |
| fired records | 0 |
| total fire steps | 0 |
| bigram-only records | 0 |
| spectral-only records | 0 |
| both records | 0 |
| bigram fire steps | 0 |
| spectral fire steps | 0 |
| both fire steps | 0 |
| mean ΔD2 | 0.0000 |
| mean ΔD2 (firing only) | — |
| mean n_gen | 13.7 |
| mean suppression steps | 0.00 |
| mean kickstart events | 0.00 |
| byte-identical to dormant | 50 |

### schema_urls_strings (ds-007, N=50)

| metric | value |
|---|---|
| records | 50 |
| false positives | 0 |
| FP rate | 0.00% |
| fired records | 0 |
| total fire steps | 0 |
| bigram-only records | 0 |
| spectral-only records | 0 |
| both records | 0 |
| bigram fire steps | 0 |
| spectral fire steps | 0 |
| both fire steps | 0 |
| mean ΔD2 | 0.0000 |
| mean ΔD2 (firing only) | — |
| mean n_gen | 1.0 |
| mean suppression steps | 0.00 |
| mean kickstart events | 0.00 |
| byte-identical to dormant | 50 |

### prose_code_switch (ds-007, N=50)

| metric | value |
|---|---|
| records | 50 |
| false positives | 0 |
| FP rate | 0.00% |
| fired records | 0 |
| total fire steps | 0 |
| bigram-only records | 0 |
| spectral-only records | 0 |
| both records | 0 |
| bigram fire steps | 0 |
| spectral fire steps | 0 |
| both fire steps | 0 |
| mean ΔD2 | 0.0000 |
| mean ΔD2 (firing only) | — |
| mean n_gen | 1.0 |
| mean suppression steps | 0.00 |
| mean kickstart events | 0.00 |
| byte-identical to dormant | 50 |

### schema_corpus_v2 (ds-018, N=300)

| metric | value |
|---|---|
| records | 300 |
| false positives | 0 |
| FP rate | 0.00% |
| fired records | 0 |
| total fire steps | 0 |
| bigram-only records | 0 |
| spectral-only records | 0 |
| both records | 0 |
| bigram fire steps | 0 |
| spectral fire steps | 0 |
| both fire steps | 0 |
| mean ΔD2 | 0.0000 |
| mean ΔD2 (firing only) | — |
| mean n_gen | 1.0 |
| mean suppression steps | 0.00 |
| mean kickstart events | 0.00 |
| byte-identical to dormant | 300 |


## Overall summary (all four corpora)

| metric | value |
|---|---|
| Records | 450 |
| False positives | 0 (0.00%) |
| Fired records | 0 |
| Total fire steps | 0 |
| Bigram-only records | 0 |
| Spectral-only records | 0 |
| Both records | 0 |
| Bigram fire steps | 0 |
| Spectral fire steps | 0 |
| Both fire steps | 0 |
| Mean ΔD2 | 0.0000 |
| Mean ΔD2 (firing only) | — |
| Mean n_gen | 2.4 |
| Mean suppression steps | 0.00 |
| Mean kickstart events | 0.00 |
| Byte-identical to dormant | 450 |
| EOS-terminated (active) | 445 |

## Notes / caveats

- GATE: the pass/fail criterion is explicit and quoted from the task file. A FAIL is a STOP signal — the script exits non-zero; no thresholds are negotiated or tuned.
- Live generation uses KV-cache incremental decoding (pre-fill once, then single-token forwards with past_key_values), dropping compute from O(n^2) to O(n). The layer-2 spectral hook captures hidden states at each decoding step (seq_len==1) into a rolling 24-token ring buffer, identical to the DS-034b hook pattern.
- The DS-034e dual-predicate OR rule is implemented exactly per RFC A3: bigram_fire = bigram_ctr ≥ 2; spectral_collapse uses the frozen band_low 8.216097 with ≥2 consecutive spectral_ctr; when bigram_ctr < 2, spectral_fire requires token_diversity < 0.40 corroboration; code-context immunity forces spectral_fire=False when is_code_syntax_context is True.
- Actuation on is_collapsed uses the production logit-penalty path (token suppression cooldown=8 penalty=-5.0, kickstart counter=3 with -1e4/-5.0/-2.0, top_p=0.85). The actuation can fire multiple times per record; every fire is logged.
- Dormant baselines are generated LIVE (greedy, KV-cache, no hooks/penalties) for all 450 records. No pre-existing dormant data exists for these corpora.
- Prose-100 is REUSED for context only (DS-034c v2; ratified policy = 7 documented FPs, Amendment A3). Prose-100 is NOT re-run.
- FP = ΔDistinct-2 > 0 AND fire_count > 0. A record whose predicate fires but whose continuation is byte-identical to dormant is an actuation-gap, not an FP.
- Frozen thresholds (T_PR(2) = 10.954796, band_low = 8.216097) are read from docs/gate23/FROZEN_THRESHOLDS.md and verified against the task-specified values; the script exits if they disagree.
