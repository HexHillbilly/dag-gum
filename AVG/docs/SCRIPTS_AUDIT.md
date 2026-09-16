# DS-009 — scripts/ Directory Health Audit (READ-ONLY)

- **Branch**: `night/ds-009-scripts-audit`
- **Date**: 2026-08-08
- **Mode**: READ-ONLY inventory. No code changes, no model loading, no network downloads.
- **Deliverable**: this file (`docs/SCRIPTS_AUDIT.md`).

## Method & engagement evidence

- Every `.py` in `scripts/` was **read** in full (docstrings, comments, imports, bodies).
- **Static import resolution** was performed by parsing each file with the `ast` module
  (`ast.parse`) and cross-referencing every `AVG.*` import target against a symbol
  index built from all repo `.py` files (`core/`, `governor/`, `scripts/`, and the
  `/AVG` mount). No module was imported and no code was executed.
- **Import-path mechanism verified once** (per standing orders): for
  `scripts/evaluate_qwen.py`, `sys.path.insert(0, parents[2])` prints `path added: /`
  and `import AVG` resolves to `/AVG/__init__.py` — the container mount mechanism is
  working as designed.
- **Syntax validation**: every file was parsed with `ast.parse` only (see
  "Syntax-check results" below). All 23 files PASS.
- **Protected scripts** (`test_degeneracy_smoke.py`, `test_path2_code_filter.py`,
  `test_factual_safety.py`, `benchmark_latency.py`) were audited **by reading only**,
  never executed (they require GPUs/models).
- **Seed provenance**: this audit is deterministic — no RNG, no sampling, no seeded
  runs. All findings are reproducible by re-running the AST probes above. Seeds
  referenced below are the seeds *inside the audited scripts* (e.g. `SEED = 2603` in
  `build_t2s_degenerate.py`), recorded as engagement evidence for each corpus/gate.

---

## Per-file inventory

Classification key: **(a)** gate/CI script, **(b)** evaluation harness, **(c)** corpus
generator, **(d)** vendoring/data tool, **(e)** unclear/orphaned.
`STALE` = import target resolves to a name not defined anywhere in the repo.

### `scripts/__init__.py`
- **Imports**: none (empty, 0 bytes).
- **Class**: package marker (not a script; unclassified).
- **Purpose**: makes `scripts/` a package so `AVG.scripts.*` is importable via the `/AVG` mount.
- **Syntax**: PASS.

### `scripts/analyze_psychon_binding.py`
- **Imports**: `sys`, `pathlib.Path`; third-party `torch`, `collections.defaultdict`,
  `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`); AVG `governor.controller`
  (`ActiveVarietyGovernor`), `core.sae_loader` (`load_sae_map_for_model`). All resolve.
- **Class**: (b) evaluation harness.
- **Purpose**: "Track A Diagnostic: Psychon Binding Analysis" — phase-1 aggressive
  integration probes + phase-2 direct SAE machinery injection on Qwen2.5-1.5B to
  determine whether repetition attractors bind to monosemantic psychons.
- **Syntax**: PASS.

### `scripts/analyze_psychon_binding_gpt2.py`
- **Imports**: same import set as `analyze_psychon_binding.py` (sys, pathlib, torch,
  defaultdict, transformers, AVG `ActiveVarietyGovernor`, `load_sae_map_for_model`). All resolve.
- **Class**: (b) evaluation harness.
- **Purpose**: GPT-2 edition of the Psychon-binding diagnostic (layers 8–11).
- **Syntax**: PASS.

### `scripts/benchmark_latency.py` — PROTECTED (read-only)
- **Imports**: `sys`, `time`, `argparse`, `pathlib.Path`; third-party `torch`,
  `transformers` (`AutoTokenizer`, `AutoModelForCausalLM`); AVG `governor.controller`
  (`ActiveVarietyGovernor`). All resolve.
- **Class**: (a) gate/CI script.
- **Purpose**: AVG Path-1 dormant latency guardrail benchmark; measures minimum-of-3
  raw vs governed ms/token, exits 0/1 on an absolute governor-cost budget
  (default `--max-governor-cost-ms 7.0`). Wired into `make test`.
- **Syntax**: PASS (read-only, not executed).

### `scripts/build_schema_corpus.py`
- **Imports**: `hashlib`, `json`, `random`, `sys`, `pathlib.Path`, `typing`
  (`List`, `Union`), `from __future__ import annotations`; AVG `core.metrics`
  (`is_code_syntax_context`). All resolve.
- **Class**: (c) corpus generator.
- **Purpose**: deterministic DS-002 schema-context corpus (RFC-004 track) — 100 bare,
  pretty-printed JSON samples, depths 1–6, `SEED = 4004`; writes
  `tests/fixtures/schema_corpus.jsonl`.
- **Syntax**: PASS.

### `scripts/build_schema_hazards.py`
- **Imports**: `hashlib`, `json`, `random`, `sys`, `pathlib.Path`, `typing`
  (`List`, `Union`), `from __future__ import annotations`; AVG `core.metrics`
  (`is_code_syntax_context`). All resolve.
- **Class**: (c) corpus generator.
- **Purpose**: DS-007 schema-context hazard corpora (RFC-004) — three certified
  over-fire fixtures (`schema_markdown_fenced`, `schema_urls_strings`,
  `prose_code_switch`), `SEED = 4005`.
- **Syntax**: PASS.

### `scripts/build_t2s_degenerate.py`
- **Imports**: `json`, `math`, `random`, `sys`, `collections` (`Counter`),
  `pathlib.Path`, `typing` (`List`, `Sequence`, `Tuple`), `from __future__ import
  annotations`; AVG `core.metrics` (`is_code_syntax_context`). All resolve.
- **Class**: (c) corpus generator.
- **Purpose**: T2S-Bench-Degenerate corpus (RFC-003 §6) — 100 cyclic loops, ≥160
  whitespace tokens, low Distinct-2, `SEED = 2603`; writes
  `tests/fixtures/t2s_degenerate.jsonl`.
- **Syntax**: PASS.

### `scripts/compare_avg.py`
- **Imports**: `sys`, `pathlib.Path`; third-party `torch`, `transformers`
  (`AutoModelForCausalLM`, `AutoTokenizer`); AVG `core.sae_loader` (`load_sae`),
  `governor.controller` (`ActiveVarietyGovernor`),
  `AVG.scripts.profile_model` (`make_synthetic_trusted`),
  `core.metrics` (`compute_coherent_token_ratio`).
  - All four AVG symbols resolve **in the container mount** (`/AVG/scripts/profile_model.py`
    exists and defines `make_synthetic_trusted`). The `AVG.scripts.profile_model` import
    is fragile: it only works because the `/AVG` mount exposes `scripts/` (with its
    `__init__.py`) as a package, and `pyproject.toml` packages only `core` and `governor`.
    AGENTS.md (line 198) flags this import as a known issue.
- **Class**: (b) evaluation harness.
- **Purpose**: side-by-side generation comparison — ungoverned GPT-2 vs AVG-governed,
  with intervention summary telemetry.
- **Syntax**: PASS.

### `scripts/evaluate_avg.py`
- **Imports**: `argparse`, `sys`, `pathlib.Path`, `typing` (`Dict`, `List`, `Tuple`),
  `from __future__ import annotations`; third-party `torch`, `transformers`
  (`AutoModelForCausalLM`, `AutoTokenizer` — imported inside `main`); AVG
  `core.sae_loader` (`load_sae`), `governor.controller` (`ActiveVarietyGovernor`),
  `core.metrics` (`compute_coherent_token_ratio`). All resolve.
- **Class**: (b) evaluation harness.
- **Purpose**: comprehensive AVG evaluation across degeneracy failure modes + factual
  control group; tracks Dist-2/Dist-3 strictly on newly generated tokens.
- **Syntax**: PASS.

### `scripts/evaluate_long_horizon.py`
- **Imports**: `sys`, `pathlib.Path`; third-party `torch`, `torch.nn`, `torch.nn.functional`,
  `transformers` (inside `main`); AVG `core.sae_loader` (`BaseSAE`, `SAEOutput`),
  `governor.controller` (`ActiveVarietyGovernor`, `compute_token_distinct_2_fast`),
  `core.metrics` (`compute_coherent_token_ratio`). All resolve (uses the `_fast` variant).
- **Class**: (b) evaluation harness.
- **Purpose**: 256-token extended-horizon comparative evaluation for AVG v2.1 — long-context
  stability, gating precision, residual force tracking (Qwen2.5-1.5B).
- **Syntax**: PASS.

### `scripts/evaluate_multi_pass.py` — **STALE IMPORT**
- **Imports**: `sys`, `re`, `pathlib.Path`; third-party `torch`, `transformers`
  (inside `main`); AVG `core.sae_loader` (`load_sae`),
  `governor.controller` (**`ActiveVarietyGovernor`**, **`compute_token_distinct_2`** ← STALE),
  `core.metrics` (`compute_coherent_token_ratio`).
  - **STALE**: `compute_token_distinct_2` is not defined anywhere in the repo. The import
    statement at line 12 will raise `ImportError` at module load. Usage site: line 66.
- **Class**: (b) evaluation harness.
- **Purpose**: multi-pass (5×) Hard Zero-Context stress suite with Valid-Word-Ratio (VWR)
  metrics on GPT-2.
- **Syntax**: PASS.

### `scripts/evaluate_qwen.py` — **STALE IMPORT** + shadowed import
- **Imports**: `sys`, `re`, `dataclasses`, `inspect`, `pathlib.Path`; third-party `torch`,
  `torch.nn`, `torch.nn.functional`, `transformers` (inside `main`); AVG `core.sae_loader`
  (`BaseSAE`, `SAEOutput`), `governor.controller` (**`ActiveVarietyGovernor`**,
  **`compute_token_distinct_2`** ← STALE), `core.metrics` (`compute_coherent_token_ratio`
  ← **shadowed**).
  - **STALE**: `compute_token_distinct_2` at line 26 is not defined anywhere in the repo →
    `ImportError` at module load. Usage site: line 184.
  - **Shadowed**: the `core.metrics.compute_coherent_token_ratio` import at line 27 is
    fully overridden by a local `def compute_coherent_token_ratio` at line 107 → the
    import is dead (the local definition is used at line 185).
- **Class**: (b) evaluation harness.
- **Purpose**: cross-architecture evaluation harness for AVG on Qwen/Qwen2.5-1.5B with a
  language/code-aware CTR and a vocabulary-aligned unembed-dictionary SAE.
- **Syntax**: PASS.

### `scripts/generate_avg.py`
- **Imports**: `argparse`, `sys`, `pathlib.Path`, `from __future__ import annotations`;
  third-party `torch`, `transformers` (inside `main`); AVG `core.sae_loader` (`load_sae`),
  `governor.controller` (`ActiveVarietyGovernor`, `BaselineStats`), `core.metrics`
  (`compute_coherent_token_ratio`). All resolve.
- **Class**: (b) evaluation harness (generation demo).
- **Purpose**: demonstration of generation with AVG enabled vs disabled; can load a saved
  `BaselineStats` payload (`torch.load`) or calibrate on the fly.
- **Syntax**: PASS.

### `scripts/measure_force_landing_baseline.py` — **STALE IMPORT**
- **Imports**: `sys`, `re`, `dataclasses`, `inspect`, `pathlib.Path`; third-party `torch`,
  `torch.nn`, `torch.nn.functional`, `transformers` (inside `main`); AVG `core.sae_loader`
  (`BaseSAE`, `SAEOutput`), `governor.controller` (**`ActiveVarietyGovernor`**,
  **`compute_token_distinct_2`** ← STALE), `core.metrics` (`compute_coherent_token_ratio`).
  - **STALE**: `compute_token_distinct_2` at line 26 is not defined anywhere in the repo →
    `ImportError` at module load. The name is never used elsewhere in the file; the stale
    import alone breaks the script.
- **Class**: (b) evaluation harness (measurement probe).
- **Purpose**: baseline measurement probe — intervention force (Δres L2) vs post-exit
  landing domain (English prose / code syntax / CJK / patterned template).
- **Syntax**: PASS.

### `scripts/profile_model.py`
- **Imports**: `argparse`, `sys`, `pathlib.Path`, `typing` (`List`), `from __future__
  import annotations`; third-party `torch`, `transformers` (inside `main`); AVG
  `core.sae_loader` (`load_sae`), `governor.controller` (`ActiveVarietyGovernor`),
  `core.metrics` (`compute_coherent_token_ratio`). All resolve.
- **Class**: (b) evaluation harness (baseline-calibration utility).
- **Purpose**: baseline calibration script for AVG; builds `make_synthetic_trusted`
  traces and saves a calibration payload (`avg_baseline.pt`); also imported by
  `compare_avg.py`.
- **Syntax**: PASS.

### `scripts/test_active_modes_joint_gate.py`
- **Imports**: `sys`, `pathlib.Path`; third-party `torch`, `torch.nn`, `torch.nn.functional`,
  `transformers` (inside `main`); AVG `core.sae_loader` (`BaseSAE`, `SAEOutput`),
  `governor.controller` (`ActiveVarietyGovernor`), `core.metrics`
  (`compute_coherent_token_ratio`). All resolve.
- **Class**: (b) evaluation harness (verification probe — prints results, no exit-code gate).
- **Purpose**: "Degeneracy Trigger Verification Probe" — confirms genuine repetition loops
  satisfy the joint conjunction gate (diversity < 0.30 AND CTR < 0.30) and receive
  calibrated residual resets in [0.35, 0.75] L2.
- **Syntax**: PASS.

### `scripts/test_degeneracy_smoke.py` — PROTECTED (read-only)
- **Imports**: `sys`, `argparse`, `pathlib.Path`; third-party `torch`, `transformers`
  (`AutoTokenizer`, `AutoModelForCausalLM`, `set_seed`); AVG `governor.controller`
  (`ActiveVarietyGovernor`). All resolve.
- **Class**: (a) gate/CI script.
- **Purpose**: AVG Path-1 degeneracy recovery smoke gate (seeded, `seed=1`); asserts ≥1
  residual fire and force bounds [0.30, 0.85] L2; exits 0/1. Wired into `make test`.
- **Syntax**: PASS (read-only, not executed).

### `scripts/test_factual_safety.py` — PROTECTED (read-only)
- **Imports**: `sys`, `argparse`, `pathlib.Path`; third-party `torch`, `transformers`
  (`AutoTokenizer`, `AutoModelForCausalLM`); AVG `governor.controller`
  (`ActiveVarietyGovernor`). All resolve.
- **Class**: (a) gate/CI script.
- **Purpose**: factual-safety probe suite — 0-fire / 0.00 L2 immunity on factual prompts;
  exits 0/1. Wired into `make test`.
- **Syntax**: PASS (read-only, not executed).

### `scripts/test_path2_code_filter.py` — PROTECTED (read-only)
- **Imports**: `sys`, `argparse`, `pathlib.Path`; third-party `torch`, `transformers`
  (`AutoTokenizer`, `AutoModelForCausalLM`, `set_seed`); AVG `governor.controller`
  (`ActiveVarietyGovernor`). All resolve.
- **Class**: (a) gate/CI script.
- **Purpose**: Track B / Path 2 structural code-exclusion filter validation (seeded,
  `seed=19`); Probe A code-context immunity, Probe B word-loop rescue; exits 0/1.
  Wired into `make test`.
- **Syntax**: PASS (read-only, not executed).

### `scripts/test_rfc003_phase1_gates.py`
- **Imports**: `sys`, `time`, `pathlib.Path`; third-party `torch`, `transformers`
  (`AutoTokenizer`, `AutoModelForCausalLM`, `set_seed`); AVG `governor.controller`
  (`ActiveVarietyGovernor`). All resolve.
- **Class**: (a) gate/CI script (RFC-003 Phase 1 harness; not wired into `make test`).
- **Purpose**: RFC-003 Phase 1 quality-gate harness (Gate 1.1 factual safety, 1.2
  latency budget, 1.3 σ₁⁰ determinism); exits 0/1.
- **Syntax**: PASS.

### `scripts/test_rfc003_phase2_gates.py`
- **Imports**: `sys`, `json`, `pathlib.Path`; third-party `torch`, `transformers`
  (`AutoTokenizer`, `AutoModelForCausalLM`, `set_seed`); AVG `governor.controller`
  (`ActiveVarietyGovernor`), `core.metrics` (`is_code_syntax_context`). All resolve.
- **Class**: (a) gate/CI script (RFC-003 Phase 2b harness; not wired into `make test`).
- **Purpose**: Gate 2.1a prose immunity over certified Prose-200 (0 fires, shadow-log
  engagement); Gate 2.1b schema suppression is PARKED and prints measured evidence
  without asserting; exits 0/1.
- **Syntax**: PASS.

### `scripts/test_word_loop_fix.py`
- **Imports**: `sys`, `dataclasses`, `pathlib.Path`; third-party `torch`, `torch.nn`,
  `torch.nn.functional`, `transformers` (inside `main`); AVG `core.sae_loader`
  (`BaseSAE`, `SAEOutput`), `governor.controller` (`ActiveVarietyGovernor`),
  `core.metrics` (`compute_coherent_token_ratio`). All resolve.
- **Class**: (b) evaluation harness (verification probe — prints results, no exit-code gate).
- **Purpose**: targeted word-loop hard-cap & trajectory verification — Δres ≤ 0.75 L2 and
  coherent text exit across 3 seeds (100–102) over 128 tokens.
- **Syntax**: PASS.

### `scripts/vendor_t2s_bench.py`
- **Imports**: `hashlib`, `json`, `random`, `sys`, `pathlib.Path`, `typing`
  (`List`, `Set`); third-party `datasets.load_dataset` (imported inside function); AVG
  `core.metrics` (`is_code_syntax_context`). All resolve.
- **Class**: (d) vendoring/data tool.
- **Purpose**: vendors the T2S-Bench-MR validation subset — downloads, hashes, and
  prose-certifies a 200-sample subset (`SEED = 42`, refill on Track-B trippers);
  writes `data/t2s_bench/valid_subset_200.jsonl` + hashes/cert.
- **Syntax**: PASS.

---

## Summary table

| File | Class | AVG imports resolve? | Syntax |
|------|-------|----------------------|--------|
| `__init__.py` | (package marker) | n/a | PASS |
| `analyze_psychon_binding.py` | (b) | ✅ | PASS |
| `analyze_psychon_binding_gpt2.py` | (b) | ✅ | PASS |
| `benchmark_latency.py` (protected) | (a) | ✅ | PASS |
| `build_schema_corpus.py` | (c) | ✅ | PASS |
| `build_schema_hazards.py` | (c) | ✅ | PASS |
| `build_t2s_degenerate.py` | (c) | ✅ | PASS |
| `compare_avg.py` | (b) | ✅ (fragile `AVG.scripts.*` import) | PASS |
| `evaluate_avg.py` | (b) | ✅ | PASS |
| `evaluate_long_horizon.py` | (b) | ✅ | PASS |
| `evaluate_multi_pass.py` | (b) | ❌ **STALE** `compute_token_distinct_2` | PASS |
| `evaluate_qwen.py` | (b) | ❌ **STALE** `compute_token_distinct_2` (+ shadowed CTR import) | PASS |
| `generate_avg.py` | (b) | ✅ | PASS |
| `measure_force_landing_baseline.py` | (b) | ❌ **STALE** `compute_token_distinct_2` | PASS |
| `profile_model.py` | (b) | ✅ | PASS |
| `test_active_modes_joint_gate.py` | (b) | ✅ | PASS |
| `test_degeneracy_smoke.py` (protected) | (a) | ✅ | PASS |
| `test_factual_safety.py` (protected) | (a) | ✅ | PASS |
| `test_path2_code_filter.py` (protected) | (a) | ✅ | PASS |
| `test_rfc003_phase1_gates.py` | (a) | ✅ | PASS |
| `test_rfc003_phase2_gates.py` | (a) | ✅ | PASS |
| `test_word_loop_fix.py` | (b) | ✅ | PASS |
| `vendor_t2s_bench.py` | (d) | ✅ | PASS |

**Counts** (23 `.py` files; `__init__.py` is a package marker, unclassified):
- (a) gate/CI: **6** — `benchmark_latency`, `test_degeneracy_smoke`,
  `test_factual_safety`, `test_path2_code_filter`, `test_rfc003_phase1_gates`,
  `test_rfc003_phase2_gates`
- (b) evaluation harness: **12** — `analyze_psychon_binding`,
  `analyze_psychon_binding_gpt2`, `compare_avg`, `evaluate_avg`,
  `evaluate_long_horizon`, `evaluate_multi_pass`, `evaluate_qwen`, `generate_avg`,
  `measure_force_landing_baseline`, `profile_model`, `test_active_modes_joint_gate`,
  `test_word_loop_fix`
- (c) corpus generator: **3** — `build_schema_corpus`, `build_schema_hazards`,
  `build_t2s_degenerate`
- (d) vendoring/data tool: **1** — `vendor_t2s_bench`
- (e) unclear/orphaned: **0**

---

## Confirmed stale-import evidence (file:line)

Ground truth (human-verified) named the stale pair in `evaluate_multi_pass.py` and
`evaluate_qwen.py`. This audit confirms those two **and additionally finds a third
occurrence** in `measure_force_landing_baseline.py`. Only `compute_token_distinct_2_fast`
exists in the repo (`governor/controller.py:70`, invoked at `:672`); the non-`_fast`
name `compute_token_distinct_2` is not defined anywhere.

Verbatim import lines:

```
scripts/evaluate_multi_pass.py:12:
    from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2
    (usage site: line 66: dist2, _ = compute_token_distinct_2(gen_ids, window_len=24))

scripts/evaluate_qwen.py:26:
    from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2
    (usage site: line 184: dist2, _ = compute_token_distinct_2(gen_ids, window_len=32))

scripts/measure_force_landing_baseline.py:26:
    from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2
    (name not used elsewhere in the file; the stale import alone breaks module load)
```

All three files would raise `ImportError: cannot import name 'compute_token_distinct_2'`
at module load, before any `main()` runs. `evaluate_long_horizon.py` correctly imports
`compute_token_distinct_2_fast` (line 17) and is unaffected.

Related code-health finding (not a stale name — the symbol exists):
`scripts/evaluate_qwen.py` imports `compute_coherent_token_ratio` from
`AVG.core.metrics` at line 27 but a local `def compute_coherent_token_ratio` at line 107
shadows it completely; the import is dead code.

---

## Candidates for human decision

Inventory only — **no deletion recommendations**. Each item is flagged for human
review; deletion, repair, or retention are human decisions.

1. **`evaluate_multi_pass.py`** — broken at module load (stale
   `compute_token_distinct_2`, line 12). Evaluation harness is currently unrunnable.
2. **`evaluate_qwen.py`** — broken at module load (stale `compute_token_distinct_2`,
   line 26). Also contains a dead import (`core.metrics.compute_coherent_token_ratio`
   shadowed by a local definition at line 107).
3. **`measure_force_landing_baseline.py`** — broken at module load (stale
   `compute_token_distinct_2`, line 26; the name is imported but never used). This is a
   measurement-style probe; the stale import alone takes it offline.
4. **`compare_avg.py`** — imports `AVG.scripts.profile_model.make_synthetic_trusted`.
   Resolves **only** because the container mounts the repo root at `/AVG` and `scripts/`
   contains an `__init__.py`; `pyproject.toml` packages only `core` and `governor`.
   AGENTS.md (line 198) already flags this import as a layout mismatch. Fragile
   intra-repo coupling worth a human look.
5. **`test_active_modes_joint_gate.py`** and **`test_word_loop_fix.py`** — named
   `test_*` and gate-oriented by docstring, but they do **not** enforce pass/fail via
   exit codes (no `sys.exit(0/1)`), so they are not wired CI gates. They behave as
   verification probes. Naming-vs-behavior mismatch for human review.
6. **Doc references to absent scripts** — `AGENTS.md` (lines 56–57, 121–122, 168–170)
   references six scripts that are **not present** in this checkout:
   `generate_benchmark_assets_clean.py`, `plot_real_telemetry.py`,
   `evaluate_pareto_frontier.py`, `ablation_feature_resolution.py`,
   `positive_control_monosemantic.py`, `gen_plot.py`. These are orphaned *doc*
   references (no files to inventory); the checkout and docs are out of sync.
7. **Cross-reference note (tests/, out of audit scope)** — AGENTS.md (lines 114, 197)
   claims `tests/test_metrics.py` imports a non-existent `IdentitySAE`, but a grep of the
   current checkout finds no `IdentitySAE` reference in `tests/test_metrics.py` or
   `core/sae_loader.py`. This suggests the doc note is stale; recorded here as context
   only.

---

## Syntax-check results (ast.parse only, no execution)

All 23 `.py` files in `scripts/` were parsed with `ast.parse`. **Result: 23/23 PASS**,
0 FAIL. No `SyntaxError` in any file. (Full per-file pass list appears in the summary
table above.)
