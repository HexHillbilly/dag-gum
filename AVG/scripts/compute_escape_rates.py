#!/usr/bin/env python3
"""night-013: Retroactive escape-rate pass (DATA ANALYSIS ONLY).

Every prior night probe from DS-026 through night-012 reports rescue rate
(ΔDistinct-2 > 0 vs dormant) as the primary gate criterion. The rescue
metric conflates genuine prose recovery with early EOS bailouts — a 1-token
<|endoftext|> continuation has Distinct-2 of 1.0 vs the dormant loop's ~0.15
and counts as "rescued".

This probe computes two secondary diagnostic metrics — gross escape rate and
net escape rate — across all prior probes from existing per-record JSONL
files. The rescue-rate gate criterion stays unchanged. These are
contextualizing metrics, not a new gate.

Definitions
-----------
- Gross escape: the active continuation is NOT byte-identical to the dormant
  continuation. Computed as ``1.0 - byte_identical_rate``. Any divergence at
  any position counts as escape — including 1-token EOS bailouts.
- Net escape: gross escape AND n_generated >= 24. The n>=24 threshold is the
  project's standard analysis window for Distinct-2 computation. Filters out
  1-token EOS bailouts while keeping everything from short prose fragments to
  full completions.
- Rescue (unchanged): ΔDistinct-2 > 0 vs dormant. The primary gate criterion
  from every prior probe.

PURE DATA PASS. Zero GPU, zero model loading, zero re-running. The script
reads existing JSONL in docs/gate23/, computes two additional columns per
record, and produces a single trend table per fixture.

Ground truth (verified by human review)
----------------------------------
- Source: every probe with per-record JSONL in docs/gate23/ that contains the
  fields byte_identical (or byte-id), n_generated (or n_gen), and ΔDistinct-2
  (or ΔD2). The probes span DS-026 through night-012.
- Fixtures: t2s_degenerate, qwen_degenerate, heldout_degenerate_v2.
  Prose-100 FP probes are excluded — they measure false positives on healthy
  text, not escape from degenerate loops.
- No model. No GPU. No re-running. Pure data pass.

Method (auto-detection)
-----------------------
1. Scan docs/gate23/*.jsonl. Probe name is extracted from a ``probe`` field if
   present and consistent, otherwise from the sibling ``*.md`` report heading
   (``# <PROBE-ID> — ...``), otherwise from the JSONL filename stem.
2. For each record, auto-detect and normalize the three required fields:
     - byte-identity:  byte-id | byte_identical | byte-identical | token_match
       | byte_identical_to_dormant
     - n_generated:    n_gen | n_generated | n_gen_active
     - ΔDistinct-2:    ΔD2 | delta_d2 | delta_distinct_2
3. The measurement dict is the record's own ``active`` continuation when
   present (preferred over reused baselines). For multi-condition records the
   probe's canonical reference condition is used:
     - penalty_decomposition_results.jsonl (DS-030): arm ``D`` (production
       suppression-only, cooldown=8/penalty=-5.0 — the DS-030 primary-actuator
       finding, reused throughout the arc as the suppression-only reference).
     - cooldown_sweep_results.jsonl (DS-031): cell ``(5,-5)`` (the DS-031
       headline best-quality-at-equal-rescue finding).
   This is documented in ESCAPE_RATES.md.
4. Fixture is read from the record's ``fixture`` field; for records without
   one it is inferred by joining (record_id, source_prose_id) against an index
   built from records that DO carry a fixture field. Records whose fixture
   cannot be resolved are logged and skipped.
5. Records whose required fields cannot be resolved are logged and skipped.
6. Per (probe, fixture): rescue_rate, gross_escape_rate, net_escape_rate,
   median/IQR n_generated for escaped continuations, and the four-bucket
   histogram {n=1, 2-23, 24-127, n=128} over escaped continuations.
7. If fewer than MIN_PROBES_PER_FIXTURE probes have usable data for a fixture,
   report the partial result and STOP (exit 1).

Boundaries
----------
- DATA ANALYSIS ONLY. No GPU, no model loading, no re-running.
- Rescue-rate gate criterion unchanged — these are secondary metrics.
- READS existing JSONL in docs/gate23/ only.
- WRITES only docs/gate23/ESCAPE_RATES.md and docs/gate23/escape_rates.jsonl.
- Clean tree, never push.

Environment notes
-----------------
- Standard root-as-package header is present for consistency with the repo
  convention, but this script imports no AVG modules and no third-party
  packages: pure stdlib only. Percentiles use a self-contained linear-
  interpolation implementation that matches numpy.percentile's default.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SEED = 42
NET_ESCAPE_MIN_N_GEN = 24          # standard Distinct-2 analysis window
FULL_BUDGET_N_GEN = 128            # max_new_tokens used by every probe
MIN_PROBES_PER_FIXTURE = 5         # hard stop if a fixture drops below this
OUTPUT_JSONL_NAME = "escape_rates.jsonl"  # this script's own output (not input)
OUTPUT_MD_NAME = "ESCAPE_RATES.md"        # this script's own report (not input)

# Only the three degenerate-loop fixtures are in scope. prose-100 FP probes
# are excluded — they measure false positives on healthy text, not escape
# from degenerate loops.
TARGET_FIXTURES: Set[str] = {
    "t2s_degenerate",
    "qwen_degenerate",
    "heldout_degenerate_v2",
}

GATE23_DIR = Path(__file__).resolve().parent.parent / "docs" / "gate23"

# Normalized (lowercase, dash/space -> underscore) field aliases.
BYTE_IDENTITY_ALIASES = {
    "byte_id", "byte_identical", "byte_identical_to_dormant",
    "byte-id", "byte-identical", "token_match",
}
N_GENERATED_ALIASES = {"n_gen", "n_generated", "n_gen_active"}
DELTA_D2_ALIASES = {"Δd2", "delta_d2", "delta_distinct_2", "δd2", "dd2"}

# Canonical reference condition for multi-condition probes. Key = JSONL
# basename; value = (selector kind, condition label used as the dict key).
CANONICAL_CONDITIONS: Dict[str, Tuple[str, str]] = {
    # DS-030 logit-penalty decomposition: arm D = production suppression-only
    # (cooldown=8, penalty=-5.0), the DS-030 primary-actuator finding reused
    # throughout the arc as the suppression-only reference.
    "penalty_decomposition_results.jsonl": ("arm", "D"),
    # DS-031 cooldown sweep: cell (5,-5) = headline best-quality finding.
    "cooldown_sweep_results.jsonl": ("cell", "(5,-5)"),
}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _norm_key(key: Any) -> str:
    """Normalize a JSON key for alias matching."""
    return str(key).lower().replace("-", "_").replace(" ", "_")


def _percentile(sorted_values: List[int], q: float) -> float:
    """Linear-interpolation percentile (matches numpy.percentile default).

    ``sorted_values`` must be ascending. q in [0, 100].
    """
    n = len(sorted_values)
    if n == 0:
        raise ValueError("empty sequence")
    if n == 1:
        return float(sorted_values[0])
    pos = q / 100.0 * (n - 1)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


# --------------------------------------------------------------------------
# Field / measurement resolution
# --------------------------------------------------------------------------

def collect_candidates(record: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Return [(path, {byte_id, n_gen, dd2}), ...] for nested dicts that
    contain all three required fields (present, not None)."""
    candidates: List[Tuple[str, Dict[str, Any]]] = []

    def walk(obj: Any, path: str) -> None:
        if not isinstance(obj, dict):
            return
        found: Dict[str, Any] = {}
        for k, v in obj.items():
            nk = _norm_key(k)
            if nk in BYTE_IDENTITY_ALIASES:
                found["byte_id"] = v
            elif nk in N_GENERATED_ALIASES:
                found["n_gen"] = v
            elif nk in DELTA_D2_ALIASES:
                found["dd2"] = v
        if (set(found) == {"byte_id", "n_gen", "dd2"}
                and all(v is not None for v in found.values())):
            candidates.append((path, found))
        for k, v in obj.items():
            if isinstance(v, dict):
                walk(v, path + "/" + str(k))

    walk(record, "")
    return candidates


def resolve_measurement(
    record: Dict[str, Any],
    canonical_sel: Optional[Tuple[str, str]],
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    """Resolve the per-record measurement dict.

    Preference order:
      1. The record's own ``active`` continuation (the probe's live
         measurement, preferred over reused baselines such as DS-030 arms or
         cooldown_validation baseline_B/D_ds030).
      2. A single unambiguous nested dict carrying all three fields.
      3. The canonical reference condition for multi-condition probes
         (see CANONICAL_CONDITIONS).

    Returns (measurement_dict, source_path, reason_if_failed).
    """
    candidates = collect_candidates(record)
    if not candidates:
        return None, None, "missing required fields (byte_identity/n_generated/ΔD2)"

    if len(candidates) == 1:
        path, has = candidates[0]
        return has, path, None

    # Prefer the record's own active measurement over reused baselines.
    active = [c for c in candidates if c[0].split("/")[-1] == "active"]
    if len(active) == 1:
        path, has = active[0]
        return has, path, None

    # Multi-condition probe: apply the documented canonical reference.
    if canonical_sel is not None:
        kind, label = canonical_sel
        if kind in ("arm", "cell"):
            for path, has in candidates:
                if path.split("/")[-1] == label:
                    return has, path, None

    return None, None, (
        "ambiguous measurement (multiple dicts carry the required fields; "
        "no canonical condition applies)"
    )


def build_fixture_index() -> Dict[Tuple[Any, Any], Set[str]]:
    """Index (record_id, source_prose_id) -> {fixture} from all JSONL
    records that carry a fixture field."""
    index: Dict[Tuple[Any, Any], Set[str]] = {}
    for path in sorted(GATE23_DIR.glob("*.jsonl")):
        if path.name == OUTPUT_JSONL_NAME:
            # Never read this script's own output as input.
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(rec, dict) and "fixture" in rec and "record_id" in rec:
                    key = (rec["record_id"], rec.get("source_prose_id"))
                    index.setdefault(key, set()).add(str(rec["fixture"]))
    return index


def infer_fixture(record: Dict[str, Any], index: Dict[Tuple[Any, Any], Set[str]]) -> Optional[str]:
    """Fixture from the record's own field, else by (record_id, source_prose_id)
    join against the reference index."""
    if "fixture" in record and record["fixture"] is not None:
        return str(record["fixture"])
    key = (record.get("record_id"), record.get("source_prose_id"))
    fx = index.get(key)
    if fx and len(fx) == 1:
        return next(iter(fx))
    return None


# --------------------------------------------------------------------------
# Probe discovery / naming / ordering
# --------------------------------------------------------------------------

def probe_id_from_report(jsonl_path: Path) -> Optional[str]:
    """Extract the probe ID from the sibling .md report heading
    ``# <PROBE-ID> — ...``. Returns None if no report or no heading.

    The JSONL basenames are lowercase (e.g. ``controller_integration_results``)
    while the report basenames are uppercase (``CONTROLLER_INTEGRATION_RESULTS``),
    so the sibling lookup is case-insensitive.
    """
    stem_l = jsonl_path.stem.lower()
    md_paths = [p for p in GATE23_DIR.glob("*.md") if p.stem.lower() == stem_l]
    if not md_paths:
        return None
    try:
        text = md_paths[0].read_text(encoding="utf-8")
    except OSError:
        return None
    m = re.search(r"^#\s+(\S+)", text, flags=re.MULTILINE)
    if not m:
        return None
    candidate = m.group(1)
    # The heading token must look like a probe ID (DS-XXXX or night-XXX).
    if re.match(r"^(DS-|night-)", candidate, flags=re.IGNORECASE):
        return candidate
    return None


def probe_id_for_jsonl(jsonl_path: Path) -> str:
    """Probe ID: consistent ``probe`` field > sibling .md heading > filename stem."""
    # Try the sibling report heading first (deterministic, no data read).
    from_report = probe_id_from_report(jsonl_path)
    if from_report:
        return from_report
    # Try a consistent ``probe`` field across records.
    probe_values: Set[str] = set()
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and rec.get("probe") is not None:
                probe_values.add(str(rec["probe"]))
    if len(probe_values) == 1:
        return next(iter(probe_values))
    return jsonl_path.stem


def probe_sort_key(probe: str) -> Tuple[int, int, str]:
    """Chronological sort key: DS probes before night probes, then numeric,
    then letter suffix (DS-034d after DS-034)."""
    m = re.match(r"^(DS|night)-(\d+)([a-z]?)$", probe, flags=re.IGNORECASE)
    if not m:
        return (2, 0, probe)
    rank = 0 if m.group(1).upper() == "DS" else 1
    return (rank, int(m.group(2)), m.group(3))


# --------------------------------------------------------------------------
# Per-record flag extraction
# --------------------------------------------------------------------------

def record_escape_flags(
    probe: str,
    fixture: str,
    record_id: Any,
    source_path: str,
    m: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the per-record escape-flag record."""
    byte_identical = bool(m["byte_id"])
    n_gen = int(m["n_gen"])
    dd2 = float(m["dd2"])
    gross_escape = not byte_identical
    net_escape = gross_escape and n_gen >= NET_ESCAPE_MIN_N_GEN
    rescue = dd2 > 0.0

    bucket: Optional[str] = None
    if gross_escape:
        if n_gen == 1:
            bucket = "1"
        elif n_gen < NET_ESCAPE_MIN_N_GEN:
            bucket = "2-23"
        elif n_gen < FULL_BUDGET_N_GEN:
            bucket = "24-127"
        else:
            bucket = "128"

    return {
        "probe": probe,
        "fixture": fixture,
        "record_id": record_id,
        "source_path": source_path,
        "n_generated": n_gen,
        "byte_identical": byte_identical,
        "delta_distinct_2": dd2,
        "rescue": rescue,
        "gross_escape": gross_escape,
        "net_escape": net_escape,
        "bucket": bucket,
    }


# --------------------------------------------------------------------------
# Per (probe, fixture) metrics
# --------------------------------------------------------------------------

def bucket_of(n_gen: int) -> str:
    if n_gen == 1:
        return "1"
    if n_gen < NET_ESCAPE_MIN_N_GEN:
        return "2-23"
    if n_gen < FULL_BUDGET_N_GEN:
        return "24-127"
    return "128"


def aggregate_pairs(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Aggregate per-record rows into per-(probe, fixture) metric dicts."""
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault((r["probe"], r["fixture"]), []).append(r)

    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (probe, fixture), recs in grouped.items():
        n = len(recs)
        rescue = sum(1 for r in recs if r["rescue"])
        gross = sum(1 for r in recs if r["gross_escape"])
        net = sum(1 for r in recs if r["net_escape"])

        escaped_n_gen = sorted(r["n_generated"] for r in recs if r["gross_escape"])
        if escaped_n_gen:
            median = float(_percentile(escaped_n_gen, 50))
            p25 = _percentile(escaped_n_gen, 25)
            p75 = _percentile(escaped_n_gen, 75)
        else:
            median = p25 = p75 = None

        buckets = {"1": 0, "2-23": 0, "24-127": 0, "128": 0}
        for r in recs:
            if r["gross_escape"]:
                buckets[bucket_of(r["n_generated"])] += 1

        out[(probe, fixture)] = {
            "probe": probe,
            "fixture": fixture,
            "n": n,
            "rescue_rate": rescue / n,
            "gross_escape_rate": gross / n,
            "net_escape_rate": net / n,
            "median_n_gen_escaped": median,
            "p25_n_gen_escaped": p25,
            "p75_n_gen_escaped": p75,
            "bucket_1": buckets["1"],
            "bucket_2_23": buckets["2-23"],
            "bucket_24_127": buckets["24-127"],
            "bucket_128": buckets["128"],
        }
    return out


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------

def _fmt_rate(x: float) -> str:
    return f"{x:.4f}"


def _fmt_opt(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:g}"


# Rescue-rate ceilings cited in the night-013 task file (probe per fixture).
# heldout_degenerate_v2 has no cited ceiling; the max-rescue probe is used.
CEILING_PROBES = {
    "t2s_degenerate": "night-008",
    "qwen_degenerate": "night-007",
}


def _ceiling_probe_for(
    fixture: str,
    metrics: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the metric row for the task-cited ceiling probe, or the
    max-rescue probe when the fixture has no cited ceiling."""
    cited = CEILING_PROBES.get(fixture)
    if cited is not None:
        for m in metrics:
            if m["probe"] == cited:
                return m
    return max(metrics, key=lambda m: m["rescue_rate"])


def render_fixture_table(
    fixture: str,
    metrics: List[Dict[str, Any]],
) -> str:
    header = (
        "| probe | n | rescue_rate | gross_escape | net_escape | "
        "median_n_gen (esc) | IQR (esc) | n=1 | 2-23 | 24-127 | n=128 |"
    )
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    lines = [header, sep]
    for m in metrics:
        iqr = "—" if m["median_n_gen_escaped"] is None else (
            f"({m['p25_n_gen_escaped']:g}, {m['p75_n_gen_escaped']:g})"
        )
        lines.append(
            f"| {m['probe']} | {m['n']} | {_fmt_rate(m['rescue_rate'])} | "
            f"{_fmt_rate(m['gross_escape_rate'])} | {_fmt_rate(m['net_escape_rate'])} | "
            f"{_fmt_opt(m['median_n_gen_escaped'])} | {iqr} | "
            f"{m['bucket_1']} | {m['bucket_2_23']} | {m['bucket_24_127']} | {m['bucket_128']} |"
        )
    return "\n".join(lines)


def render_markdown(
    by_fixture: Dict[str, List[Dict[str, Any]]],
    skipped: List[str],
) -> str:
    fixture_order = ["t2s_degenerate", "qwen_degenerate", "heldout_degenerate_v2"]

    md: List[str] = []
    md.append("# night-013 — Retroactive Escape-Rate Pass (DATA ANALYSIS ONLY)")
    md.append("")
    md.append(
        "> Contextualizing secondary metrics. The rescue-rate gate criterion "
        "(ΔDistinct-2 > 0 vs dormant) is unchanged. Gross/net escape rates are "
        "computed from existing per-record JSONL; zero GPU, zero model loading, "
        "zero re-running."
    )
    md.append("")
    md.append("## Method")
    md.append("")
    md.append(f"- **SEED:** {SEED} (set for reproducibility; the pass is deterministic).")
    md.append(
        "- **Source:** every probe with per-record JSONL in `docs/gate23/` that "
        "contains the fields `byte_identical` (or `byte-id`), `n_generated` "
        "(or `n_gen`), and `ΔDistinct-2` (or `ΔD2`), spanning DS-026 through "
        "night-012."
    )
    md.append(
        "- **Gross escape:** `1.0 - byte_identical_rate`. Any divergence at any "
        "position counts as escape — including 1-token EOS bailouts."
    )
    md.append(
        f"- **Net escape:** gross escape AND `n_generated >= "
        f"{NET_ESCAPE_MIN_N_GEN}` (the standard Distinct-2 analysis window)."
    )
    md.append(
        "- **Rescue (unchanged):** `ΔDistinct-2 > 0` vs dormant. The primary "
        "gate criterion from every prior probe."
    )
    md.append(
        "- **Four-bucket histogram:** counts over *escaped* continuations: "
        "`n=1` (pure EOS bailout), `2-23` (escaped but too short for "
        "Distinct-2 analysis), `24-127` (net escape with analyzable text), "
        "`n=128` (escaped AND filled the full budget)."
    )
    md.append("")
    md.append("### Field-name normalization and multi-condition probes")
    md.append("")
    md.append(
        "Probes across the arc used slightly different field names. The script "
        "auto-detects and normalizes: byte-identity (`byte-id`, `byte_identical`, "
        "`byte-identical`, `token_match`, `byte_identical_to_dormant`), "
        "n_generated (`n_gen`, `n_generated`, `n_gen_active`), and ΔDistinct-2 "
        "(`ΔD2`, `delta_d2`, `delta_distinct_2`). The measurement is the "
        "record's own `active` continuation when present (preferred over reused "
        "baselines). For the two multi-condition probes the documented canonical "
        "reference condition is used:"
    )
    md.append("")
    md.append("- **DS-030** (`penalty_decomposition_results.jsonl`): arm **D** — "
              "production suppression-only (cooldown=8, penalty=-5.0), the DS-030 "
              "primary-actuator finding reused throughout the arc as the "
              "suppression-only reference.")
    md.append("- **DS-031** (`cooldown_sweep_results.jsonl`): cell **(5,-5)** — "
              "the DS-031 headline best-quality-at-equal-rescue finding.")
    md.append("")
    md.append("### Skipped / excluded sources")
    md.append("")
    md.append(
        "Prose-100 FP probes are excluded (they measure false positives on "
        "healthy text, not escape from degenerate loops). Sources that lack "
        "per-record escape fields, or whose required fields could not be "
        "resolved, are logged and skipped."
    )
    md.append("")
    for s in sorted(skipped):
        md.append(f"- `{s}`")
    md.append("")
    md.append("## Trend tables")
    md.append("")
    md.append(
        "Rows ordered chronologically (DS-026 earliest, night-012 latest). "
        "`IQR (esc)` is (p25, p75) of n_generated over escaped continuations. "
        "The `n=1 | 2-23 | 24-127 | n=128` buckets sum to the gross-escape "
        "count."
    )
    md.append("")
    for fixture in fixture_order:
        if fixture not in by_fixture:
            continue
        md.append(f"### Fixture: `{fixture}`")
        md.append("")
        md.append(render_fixture_table(fixture, by_fixture[fixture]))
        md.append("")

    md.append("## Key observations (computed, contextualizing)")
    md.append("")
    md.append(
        "The rescue-rate ceilings quoted across the arc — qwen 0.61 "
        "(night-007/008), t2s 0.23 (night-008) — are compared below with "
        "gross/net escape at the same probe. No gate verdict is offered; "
        "these are contextualizing metrics only."
    )
    md.append("")
    for fixture in fixture_order:
        ms = by_fixture.get(fixture, [])
        if not ms:
            continue
        m = _ceiling_probe_for(fixture, ms)
        if m is None:
            continue
        bailouts = m["bucket_1"] + m["bucket_2_23"]
        md.append(f"- **`{fixture}`** — at the rescue-ceiling probe "
                  f"`{m['probe']}` (max rescue {_fmt_rate(m['rescue_rate'])}): "
                  f"net escape {_fmt_rate(m['net_escape_rate'])} vs gross "
                  f"{_fmt_rate(m['gross_escape_rate'])}; {bailouts}/{m['n']} "
                  f"escaped records are too short for Distinct-2 analysis "
                  f"({m['bucket_1']} pure-EOS).")
    md.append("")
    md.append("## Per-record data")
    md.append("")
    md.append(
        "Per-record per-probe escape flags are written to "
        "`docs/gate23/escape_rates.jsonl` (one JSON object per line, "
        "chronological probe order)."
    )
    md.append("")
    md.append("## Caveats")
    md.append("")
    md.append(
        "- These are contextualizing metrics, not a gate. The rescue-rate gate "
        "criterion is unchanged."
    )
    md.append(
        "- DS-032 `heldout_degenerate_v2` spans two regimes (Part 1 greedy "
        "combined, Part 2 sampling suppression-only); the pair is aggregated "
        "into one row."
    )
    md.append(
        "- DS-030/DS-031 rows use the documented canonical reference condition "
        "so the trend table has one row per probe per fixture."
    )
    return "\n".join(md) + "\n"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    random.seed(SEED)

    print(f"[night-013] escape-rate pass  SEED={SEED}")
    print(f"[night-013] reading per-record JSONL from {GATE23_DIR}")

    fixture_index = build_fixture_index()
    print(f"[night-013] fixture index: {len(fixture_index)} (record_id, source) keys")

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for jsonl_path in sorted(GATE23_DIR.glob("*.jsonl")):
        fname = jsonl_path.name
        if fname == OUTPUT_JSONL_NAME:
            # Never read this script's own output as input.
            continue
        probe = probe_id_for_jsonl(jsonl_path)
        canonical_sel = CANONICAL_CONDITIONS.get(fname)

        n_target = 0
        n_skipped = 0
        n_missing_fields = 0
        n_ambiguous = 0
        n_no_fixture = 0

        with open(jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    n_skipped += 1
                    continue
                if not isinstance(rec, dict):
                    n_skipped += 1
                    continue

                fixture = infer_fixture(rec, fixture_index)
                if fixture is None:
                    n_no_fixture += 1
                    continue
                if fixture not in TARGET_FIXTURES:
                    # prose-100 and anything else outside the three degenerate
                    # fixtures are out of scope.
                    continue

                m, source_path, reason = resolve_measurement(rec, canonical_sel)
                if m is None:
                    if reason and "missing required fields" in reason:
                        n_missing_fields += 1
                    elif reason and "ambiguous" in reason:
                        n_ambiguous += 1
                    else:
                        n_skipped += 1
                    continue

                n_target += 1
                rows.append(record_escape_flags(
                    probe=probe,
                    fixture=fixture,
                    record_id=rec.get("record_id"),
                    source_path=source_path or "/active",
                    m=m,
                ))

        if n_target > 0:
            print(
                f"[night-013] {fname}: probe={probe} usable={n_target} "
                f"(missing_fields={n_missing_fields}, ambiguous={n_ambiguous}, "
                f"no_fixture={n_no_fixture}, skipped={n_skipped})"
            )
        else:
            skipped.append(fname)
            print(
                f"[night-013] {fname}: probe={probe} SKIPPED "
                f"(no usable degenerate-fixture records; "
                f"missing_fields={n_missing_fields}, ambiguous={n_ambiguous})"
            )

    # Re-sort rows chronologically by probe.
    rows.sort(key=lambda r: probe_sort_key(r["probe"]))

    # Per-(probe, fixture) aggregation.
    metrics = aggregate_pairs(rows)

    by_fixture: Dict[str, List[Dict[str, Any]]] = {}
    for (probe, fixture), m in metrics.items():
        by_fixture.setdefault(fixture, []).append(m)
    for fixture in by_fixture:
        by_fixture[fixture].sort(key=lambda m: probe_sort_key(m["probe"]))

    # Stop condition: fewer than MIN_PROBES_PER_FIXTURE usable probes.
    red_gates: List[str] = []
    for fixture in sorted(TARGET_FIXTURES):
        n_probes = len(by_fixture.get(fixture, []))
        print(f"[night-013] fixture {fixture}: {n_probes} usable probes")
        if n_probes < MIN_PROBES_PER_FIXTURE:
            red_gates.append(
                f"{fixture}: {n_probes} usable probes < "
                f"{MIN_PROBES_PER_FIXTURE} required"
            )

    if red_gates:
        print("[night-013] STOP: fixture(s) below usable-probe minimum", file=sys.stderr)
        for g in red_gates:
            print(f"  - {g}", file=sys.stderr)
        # Still write partial data so the report is available for diagnosis.
        _write_outputs(by_fixture, skipped, rows)
        return 1

    _write_outputs(by_fixture, skipped, rows)

    print(f"[night-013] wrote docs/gate23/{OUTPUT_MD_NAME} and "
          f"docs/gate23/{OUTPUT_JSONL_NAME} ({len(rows)} per-record rows)")
    return 0


def _write_outputs(
    by_fixture: Dict[str, List[Dict[str, Any]]],
    skipped: List[str],
    rows: List[Dict[str, Any]],
) -> None:
    md = render_markdown(by_fixture, skipped)
    md_path = GATE23_DIR / OUTPUT_MD_NAME
    md_path.write_text(md, encoding="utf-8")

    jsonl_path = GATE23_DIR / OUTPUT_JSONL_NAME
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
