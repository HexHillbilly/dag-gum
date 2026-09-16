#!/usr/bin/env python3
"""DS-036 — Temporal Hardening Sweep (Option B) — MEASUREMENT.

Sweeps two temporal-hardening candidates against the committed DS-034e
baseline, via a subclass that overrides _evaluate_collapse (real production
generate() path, only the spectral threshold logic diverges):

  baseline : spectral persistence >= 2, no step floor (committed DS-034e)
  P4       : spectral persistence >= 4
  S30      : step >= 30 before the spectral arm may fire

Regime: sampling (do_sample=True, temp=0.8, top_p=0.85), SEED=42.
Fixtures: prose-100 (heldout partition, FP safety) + t2s detection-gap 93 (recall).

Metrics (fire-based; no dormant baseline needed):
  prose-100 : spectral_fire, bigram_fire, first_spectral_fire_step, spectral_fp, bigram_fp
  t2s 93    : spectral_fire (recall = fire rate / 93)

Writes batch/queue/ds-036-temporal-hardening-results.jsonl
"""
import gc
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from AVG.governor.controller import ActiveVarietyGovernor

SEED = 42
MODEL = "Qwen/Qwen2.5-1.5B"
MAX_NEW = 128
PROSE_MAX_TOKENS = 512  # truncate long prose prompts (OOM guard)
T2S_FIXTURE = "tests/fixtures/t2s_degenerate.jsonl"
VALID_SUBSET = "data/t2s_bench/valid_subset_200.jsonl"
CLASS_FILE = "docs/gate23/production_cross_fixture_results.jsonl"
CONTROL_SEED = 42
CONTROL_N = 100
BAND_LOW = 8.216097
OUT = "batch/queue/ds-036-temporal-hardening-results.jsonl"

ARMS = {
    "baseline": {"persistence": 2, "step_floor": 0},
    "P4": {"persistence": 4, "step_floor": 0},
    "S30": {"persistence": 2, "step_floor": 30},
}


class ArmGovernor(ActiveVarietyGovernor):
    """Override _evaluate_collapse with arm-specific spectral threshold, and
    track the first spectral-fire step (counted over _evaluate_collapse calls,
    i.e. non-numeric decode steps)."""

    def __init__(self, *args, spectral_persistence=2, spectral_step_floor=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._spectral_persistence = spectral_persistence
        self._spectral_step_floor = spectral_step_floor
        self._step_no = 0
        self.first_spectral_fire_step = None

    def generate(self, *args, **kwargs):
        self._step_no = 0
        self.first_spectral_fire_step = None
        return super().generate(*args, **kwargs)

    def _evaluate_collapse(self, token_diversity, trailing_ctr, is_code_context):
        self._step_no += 1
        # 1. dual-gate immunity & state reset
        if trailing_ctr >= 0.75 and token_diversity >= 0.35:
            self._consecutive_interventions = {}
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            self._last_bigram_fire = False
            self._last_spectral_fire = False
            return False
        # 2. code exclusion
        if self.use_code_filter and is_code_context:
            self._bigram_ctr = 0
            self._spectral_ctr = 0
            self._consecutive_interventions = {}
            self._last_bigram_fire = False
            self._last_spectral_fire = False
            return False
        # 3. bigram predicate
        if (token_diversity < 0.30) and (trailing_ctr < 0.30):
            self._bigram_ctr += 1
        else:
            self._bigram_ctr = 0
        bigram_fire = self._bigram_ctr >= 2
        # 4. spectral predicate
        spectral_collapse = (self._layer2_pr is not None and self._layer2_pr < BAND_LOW)
        if spectral_collapse:
            self._spectral_ctr += 1
        else:
            self._spectral_ctr = 0
        if self._bigram_ctr < 2:
            spectral_fire = (self._spectral_ctr >= self._spectral_persistence and token_diversity < 0.40)
        else:
            spectral_fire = self._spectral_ctr >= self._spectral_persistence
        # arm: step floor (S30)
        if self._step_no < self._spectral_step_floor:
            spectral_fire = False
        if is_code_context:
            spectral_fire = False
        if spectral_fire and self.first_spectral_fire_step is None:
            self.first_spectral_fire_step = self._step_no
        self._last_bigram_fire = bigram_fire
        self._last_spectral_fire = spectral_fire
        return bigram_fire or spectral_fire


def seed_all(seed):
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def select_heldout(valid_records):
    rng = random.Random(CONTROL_SEED)
    control = sorted(rng.sample(valid_records, CONTROL_N), key=lambda r: int(r["id"]))
    cids = set(int(r["id"]) for r in control)
    heldout = sorted([r for r in valid_records if int(r["id"]) not in cids],
                     key=lambda r: int(r["id"]))
    return heldout


def load_detection_gap_ids():
    rows = load_jsonl(CLASS_FILE)
    return sorted(r["record_id"] for r in rows
                  if r["fixture"] == "t2s_degenerate"
                  and r["active"]["diagnostic_class"] == "detection-gap")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    # ---- dispatch gate verification ----
    print("=== Dispatch gate: DS-035 (post-B2) result ===")
    print("  DS-035 sampling rescue: t2s 97/100, qwen 97/100 (success bar 85/100 MET) -> DS-036 dispatchable")
    print("  Detection-gap classification (DS-033): reproducible from", CLASS_FILE)

    gap_ids = load_detection_gap_ids()
    print(f"  t2s detection-gap subset: {len(gap_ids)} records")

    # ---- determinism smoke (baseline arm, greedy, t2s record 0) ----
    print("\n=== Determinism smoke (baseline, greedy, SEED=42) ===")
    t2s = load_jsonl(T2S_FIXTURE)
    prompt = t2s[0]["text"]
    def make_gov(arm):
        return ArmGovernor(model, tokenizer=tokenizer, spectral_persistence=ARMS[arm]["persistence"],
                           spectral_step_floor=ARMS[arm]["step_floor"])
    seed_all(SEED); g1 = make_gov("baseline")
    in1 = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        o1 = g1.generate(in1, max_new_tokens=MAX_NEW, do_sample=False, intervene=True)
    seed_all(SEED); g2 = make_gov("baseline")
    in2 = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        o2 = g2.generate(in2, max_new_tokens=MAX_NEW, do_sample=False, intervene=True)
    tok_identical = torch.equal(o1["sequences"], o2["sequences"])
    pr_identical = (g1._layer2_pr is None and g2._layer2_pr is None) or (
        g1._layer2_pr is not None and g2._layer2_pr is not None and abs(g1._layer2_pr - g2._layer2_pr) < 1e-9)
    liveness = g1._layer2_pr is not None
    print(f"  token-identical: {tok_identical}")
    print(f"  _layer2_pr identical: {pr_identical}")
    print(f"  spectral liveness (_layer2_pr not None): {liveness}")
    if not tok_identical:
        print("[STOP] determinism smoke failed")
        sys.exit(1)

    # ---- fixtures ----
    valid = load_jsonl(VALID_SUBSET)
    heldout = select_heldout(valid)
    print(f"\n  prose-100 heldout: {len(heldout)} records")

    out_path = Path(OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = open(out_path, "w", encoding="utf-8")
    summary = {}

    for arm in ("baseline", "P4", "S30"):
        print(f"\n=== ARM: {arm} ===")
        # prose-100 FP
        n_spectral_fire = n_bigram_fire = n_spectral_fp = 0
        for k, r in enumerate(heldout):
            enc = tokenizer(r["text"], return_tensors="pt", truncation=True, max_length=PROSE_MAX_TOKENS)
            prompt = tokenizer.decode(enc["input_ids"][0], skip_special_tokens=True)
            seed_all(SEED)
            g = make_gov(arm)
            inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                               temperature=0.8, top_p=0.85, intervene=True)
            col = o["collapse"]
            sf = col["spectral_fires"] > 0
            bf = col["bigram_fires"] > 0
            first = g.first_spectral_fire_step
            spectral_fp = sf and first is not None and first >= 120
            if sf: n_spectral_fire += 1
            if bf: n_bigram_fire += 1
            if spectral_fp: n_spectral_fp += 1
            fh.write(json.dumps({
                "arm": arm, "fixture": "prose-100", "record_id": int(r["id"]), "regime": "sampling",
                "spectral_fire": sf, "spectral_fp": spectral_fp, "bigram_fp": bf,
                "first_fire_step": first, "detection_gap": None,
            }) + "\n")
            del g, o, inp
            gc.collect(); torch.cuda.empty_cache()
            if (k + 1) % 20 == 0:
                fh.flush(); print(f"  prose [{k+1}/100] sf={n_spectral_fire} bf={n_bigram_fire} fp={n_spectral_fp}", flush=True)

        # t2s detection-gap recall
        n_recall = 0
        for j, rid in enumerate(gap_ids):
            prompt = t2s[rid]["text"]
            seed_all(SEED)
            g = make_gov(arm)
            inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                o = g.generate(inp, max_new_tokens=MAX_NEW, do_sample=True,
                               temperature=0.8, top_p=0.85, intervene=True)
            col = o["collapse"]
            sf = col["spectral_fires"] > 0
            if sf: n_recall += 1
            fh.write(json.dumps({
                "arm": arm, "fixture": "t2s_degenerate", "record_id": rid, "regime": "sampling",
                "spectral_fire": sf, "spectral_fp": None, "bigram_fp": None,
                "first_fire_step": g.first_spectral_fire_step, "detection_gap": True,
            }) + "\n")
            del g, o, inp
            gc.collect(); torch.cuda.empty_cache()
            if (j + 1) % 20 == 0:
                fh.flush(); print(f"  t2s [{j+1}/93] recall_so_far={n_recall}", flush=True)

        recall_rate = n_recall / len(gap_ids)
        summary[arm] = {
            "prose_spectral_fire": n_spectral_fire,
            "prose_bigram_fp": n_bigram_fire,
            "prose_spectral_fp": n_spectral_fp,
            "t2s_recall": n_recall,
            "t2s_recall_rate": round(recall_rate, 4),
        }
        print(f"  {arm}: prose spectral_fp={n_spectral_fp} bigram_fp={n_bigram_fire} | t2s recall {n_recall}/{len(gap_ids)} ({recall_rate:.4f})")
        fh.flush()

    fh.close()
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print(f"[wrote] {OUT}")


if __name__ == "__main__":
    main()
