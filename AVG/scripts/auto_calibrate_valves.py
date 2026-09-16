#!/usr/bin/env python3
"""Cross-model valve calibration: derive and lock per-model valves into one registry.

Runs the auto_tune pipeline (discovery -> band_low derivation -> rescue/EOS gate)
across a list of models and writes VALVE_REGISTRY.json, so AVG is plug-and-play
across architectures. Each entry freezes the model's band_low, the shared
suppression defaults, and the recommended decoding regime.

The 7B/8B models must be cached under HF_HOME pointing at a large disk
(e.g. <data-root>/lab/hf_cache) — they will not fit the home drive. Models
already in the registry (a pre-seeded band_low) are skipped unless --force.

VALVE_REGISTRY.json is GENERATED output and is gitignored — regenerate it on
demand with the command below rather than committing or hand-editing values.

Run (AVG venv):
  venv/bin/python scripts/auto_calibrate_valves.py \
      --models Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B \
      --out VALVE_REGISTRY.json
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import auto_tune as at  # reuse the proven pipeline (discovery -> band_low -> gate)

# Default model list. quant: "none" | "8bit" | "4bit" (7B/8B need 4bit on 12GB).
DEFAULT_MODELS = [
    {"id": "Qwen/Qwen2.5-0.5B", "quant": "none"},
    {"id": "Qwen/Qwen2.5-1.5B", "quant": "none"},
    {"id": "Qwen/Qwen2.5-3B", "quant": "none"},
    {"id": "HuggingFaceTB/SmolLM2-1.7B-Instruct", "quant": "none"},
    {"id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "quant": "none"},
    {"id": "unsloth/Llama-3.2-1B", "quant": "none"},
    # Large models (need the HF cache on a big disk, e.g. <data-root>/lab):
    {"id": "Qwen/Qwen2.5-7B", "quant": "4bit"},
    {"id": "NousResearch/Meta-Llama-3.1-8B", "quant": "4bit"},
    {"id": "mistralai/Mistral-7B-v0.3", "quant": "4bit"},
]

SHARED_DEFAULTS = {"suppression_strength": 5.0, "eos_guard": True, "top_p": 0.85}


def load_model(model_id, quant, device):
    tok = AutoTokenizer.from_pretrained(model_id)
    if quant in ("8bit", "4bit"):
        cfg = BitsAndBytesConfig(
            load_in_8bit=(quant == "8bit"),
            load_in_4bit=(quant == "4bit"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=cfg, device_map="auto",
            torch_dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    return model, tok


def calibrate_one(model_id, quant, prose_recs, device):
    """Return a registry entry (or None if the model cannot loop)."""
    print(f"\n===== {model_id} (quant={quant}) =====", flush=True)
    model, tok = load_model(model_id, quant, device)
    eos = tok.eos_token_id
    governor = at.ActiveVarietyGovernor(model, tokenizer=tok)

    # Phase 0: discovery (greedy + sampling, dormant).
    candidates = list(at.CANDIDATES) + at.make_word_loops()
    greedy_loopers, sampling_loopers = [], []
    for p in candidates:
        at.seed_all(at.SEED)
        inp = tok(p, return_tensors="pt")["input_ids"].to(device)
        o = governor.generate(inp, max_new_tokens=at.MAX_NEW, do_sample=False, intervene=False)
        if at.d2(o["sequences"][:, o["prompt_len"]:]) < at.LOOP_D2:
            greedy_loopers.append(p)
        at.seed_all(at.SEED)
        o = governor.generate(inp, max_new_tokens=at.MAX_NEW, do_sample=True, temperature=0.8, intervene=False)
        if at.d2(o["sequences"][:, o["prompt_len"]:]) < at.LOOP_D2:
            sampling_loopers.append(p)
    print(f"discovery: {len(greedy_loopers)} greedy / {len(sampling_loopers)} sampling loopers", flush=True)
    union = list(dict.fromkeys(greedy_loopers + sampling_loopers))
    if not union:
        print("!! no loopers discovered — cannot calibrate", flush=True)
        return None

    # Phase 1: band_low.
    band_low, p90, p10 = at.derive_band_low(model, tok, device, governor, union, prose_recs, eos)
    if band_low is None:
        return None
    governor.band_low = band_low

    # Phase 2: gate.
    entry = {
        "model": model_id,
        "quant": quant,
        "band_low": round(band_low, 4),
        "degenerate_p90": round(p90, 4),
        "prose_p10": round(p10, 4),
        "suppression_strength": SHARED_DEFAULTS["suppression_strength"],
        "eos_guard": SHARED_DEFAULTS["eos_guard"],
        "top_p": SHARED_DEFAULTS["top_p"],
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }
    if greedy_loopers:
        n, resc, eosd, mdd, mtb = at.run_gate(governor, tok, device, greedy_loopers, eos, do_sample=False)
        entry["greedy"] = {"n": n, "rescue": resc, "eos_death": eosd,
                           "mean_delta_d2": round(mdd, 4), "mean_tb_active": round(mtb, 2)}
        print(f"greedy rescue {resc}/{n} ({100*resc/max(n,1):.0f}%) | EOS-death {eosd}/{n}", flush=True)
    if sampling_loopers:
        n, resc, eosd, mdd, mtb = at.run_gate(governor, tok, device, sampling_loopers, eos, do_sample=True)
        entry["sampling"] = {"n": n, "rescue": resc, "eos_death": eosd,
                             "mean_delta_d2": round(mdd, 4), "mean_tb_active": round(mtb, 2)}
        print(f"sampling rescue {resc}/{n} ({100*resc/max(n,1):.0f}%) | EOS-death {eosd}/{n} ({100*eosd/max(n,1):.0f}%)", flush=True)

    # Free VRAM before the next model.
    del model, governor
    if device == "cuda":
        torch.cuda.empty_cache()
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated model ids (default: the built-in list)")
    ap.add_argument("--prose", default=str(ROOT / "data/t2s_bench/valid_subset_200.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "VALVE_REGISTRY.json"))
    ap.add_argument("--force", action="store_true", help="re-calibrate models already in the registry")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prose_recs = [json.loads(l) for l in open(args.prose) if l.strip()]

    models = DEFAULT_MODELS
    if args.models:
        models = [{"id": m.strip(), "quant": "none"} for m in args.models.split(",") if m.strip()]

    # Load any existing registry (skip already-locked models unless --force).
    registry = {}
    out_path = Path(args.out)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if isinstance(existing, dict) and "models" in existing:
            for m in existing["models"]:
                registry[m["model"]] = m

    def save(entries):
        out = {
            "schema": "avr.valve_registry.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "shared_defaults": SHARED_DEFAULTS,
            "models": entries,
        }
        out_path.write_text(json.dumps(out, indent=2) + "\n")

    entries = []
    for spec in models:
        mid = spec["id"]
        if mid in registry and not args.force:
            print(f"[skip] {mid} — already in registry (--force to redo)", flush=True)
            entries.append(registry[mid])
            continue
        entry = calibrate_one(mid, spec["quant"], prose_recs, device)
        if entry is not None:
            entries.append(entry)
            # Incremental save: persist after each model so a crash/interrupt
            # (GPU contention with a concurrent chat model, etc.) only loses the
            # in-flight model, not the whole run.
            save(entries)

    save(entries)
    print(f"\n[wrote] {out_path} ({len(entries)} models)", flush=True)
    for e in entries:
        print(f"  - {e['model']}: band_low={e['band_low']}", flush=True)


if __name__ == "__main__":
    main()
