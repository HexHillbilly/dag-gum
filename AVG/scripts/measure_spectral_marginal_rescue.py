#!/usr/bin/env python3
"""Spectral detector marginal rescue (MEASUREMENT ONLY).

Question: in the CURRENT controller (unconditional suppression), does the
spectral-PR detector add rescues on macro-loop fixtures over suppression+bigram
alone? This is the adversarial-defense hypothesis for a served model: the
detector's value is catching macro-loop attacks that bigram-level detection
misses (DS-033: 93% bigram detection-gap on t2s).

Method: run ActiveVarietyGovernor on t2s_degenerate.jsonl (N=100 macro-loop
fixtures) under two configs:
  - spectral ON:  band_low = 8.216097 (production default)
  - spectral OFF: band_low = 1.0      (below the PR floor ~3, spectral never fires)
Suppression is UNCONDITIONAL in both (keys on active_loop_ids, not the collapse
predicate), so the ONLY difference is the spectral-triggered kickstart. Rescue =
dormant loops (tail d2 < 0.5) and active breaks it (tail d2 >= 0.5); EOS-death is
tracked separately. Greedy decoding (matches DS-035 macro-loop characterization).

MEASUREMENT ONLY. No controller edits.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # resolve `import AVG.*` from the repo root

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from AVG.governor.controller import ActiveVarietyGovernor, compute_token_distinct_2_fast

SEED = 42
MODEL = "Qwen/Qwen2.5-1.5B"
MAX_NEW = 128
LOOP_D2 = 0.5
T2S_DEG = "tests/fixtures/t2s_degenerate.jsonl"

SPECTRAL_ON = 8.216097   # production default band_low
SPECTRAL_OFF = 1.0       # below PR floor ~3 -> spectral predicate never fires


def seed_all(seed: int) -> None:
    set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tail_d2(ids: torch.Tensor) -> float:
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


@torch.no_grad()
def run_fixture(governor, tokenizer, prompt: str, device) -> Dict[str, Any]:
    seed_all(SEED)
    inp = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
    o_d = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=False, intervene=False)
    dorm = o_d["sequences"][:, o_d["prompt_len"]:]

    seed_all(SEED)
    o_a = governor.generate(inp, max_new_tokens=MAX_NEW, do_sample=False, intervene=True)
    act = o_a["sequences"][:, o_a["prompt_len"]:]
    col = o_a["collapse"]

    dd = tail_d2(dorm)
    ad = tail_d2(act)
    return {
        "dormant_d2": dd,
        "active_d2": ad,
        "dormant_len": int(dorm.shape[-1]),
        "active_len": int(act.shape[-1]),
        "bigram_fires": int(col["bigram_fires"]),
        "spectral_fires": int(col["spectral_fires"]),
        "looped": bool(dd < LOOP_D2),
        "rescued": bool(dd < LOOP_D2 and ad >= LOOP_D2),
        "eos_death": bool(act.shape[-1] < 24),
    }


def summarize(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    looped = [r for r in rows if r["looped"]]
    rescued = [r for r in looped if r["rescued"]]
    s = {
        "config": name,
        "n": n,
        "looped": len(looped),
        "rescued": len(rescued),
        "rescue_rate": len(rescued) / max(len(looped), 1),
        "eos_death": sum(1 for r in rows if r["eos_death"]),
        "bigram_fires": sum(1 for r in rows if r["bigram_fires"] > 0),
        "spectral_fires": sum(1 for r in rows if r["spectral_fires"] > 0),
        "mean_delta_d2": float(np.mean([r["active_d2"] - r["dormant_d2"] for r in rows])),
    }
    print(f"[{name}] looped {s['looped']}/{n} | rescued {s['rescued']}/{s['looped']} "
          f"({100*s['rescue_rate']:.0f}%) | EOS-death {s['eos_death']} | "
          f"bigram_fires {s['bigram_fires']} | spectral_fires {s['spectral_fires']} | "
          f"mean Δd2 {s['mean_delta_d2']:+.3f}", flush=True)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", default=T2S_DEG, help="fixture jsonl path")
    ap.add_argument("--field", default="text", help="prompt field name (text or prompt)")
    ap.add_argument("--n", type=int, default=100, help="number of fixtures (default 100)")
    ap.add_argument("--out", default="docs/gate23/spectral_marginal_rescue.jsonl")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device).eval()

    fixtures = [json.loads(l) for l in open(args.fixtures) if l.strip()][: args.n]
    print(f"fixtures: {len(fixtures)} from {args.fixtures} (field={args.field})", flush=True)

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for name, band_low in [("spectral_ON", SPECTRAL_ON), ("spectral_OFF", SPECTRAL_OFF)]:
        gov = ActiveVarietyGovernor(model, tokenizer=tokenizer, band_low=band_low)
        rows = []
        for i, rec in enumerate(fixtures):
            r = run_fixture(gov, tokenizer, rec[args.field], device)
            r["id"] = int(rec.get("id", i))
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"  {name} {i+1}/{len(fixtures)}", flush=True)
        all_results[name] = rows
        summarize(name, rows)

    # Marginal: records rescued by spectral_ON but NOT by spectral_OFF.
    on = {r["id"]: r for r in all_results["spectral_ON"]}
    off = {r["id"]: r for r in all_results["spectral_OFF"]}
    marginal = [
        i for i in on
        if on[i]["rescued"] and (not off[i]["rescued"]) and off[i]["looped"]
    ]
    print(f"\nSPECTRAL MARGINAL RESCUE: {len(marginal)} records rescued with spectral "
          f"ON but not with spectral OFF (of {summarize('spectral_OFF', all_results['spectral_OFF'])['looped']} "
          f"looped).", flush=True)
    if marginal:
        print(f"  record ids: {sorted(marginal)[:30]}", flush=True)

    # Write rows + summary.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for name in ("spectral_ON", "spectral_OFF"):
            for r in all_results[name]:
                r["config"] = name
                f.write(json.dumps(r) + "\n")
    print(f"[wrote] {out}", flush=True)


if __name__ == "__main__":
    main()
