#!/usr/bin/env python3
"""
Track A Diagnostic: Psychon Binding Analysis.

Phase 1: Aggressive integration probes (greedy decoding, long seeds)
Phase 2: Direct machinery injection (bypasses diversity gate, forces SAE analysis)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from AVG.governor.controller import ActiveVarietyGovernor
from AVG.core.sae_loader import load_sae_map_for_model

AGGRESSIVE_PROBES = [
    {
        "mode": "Long Phrase Echo",
        "prompt": "The rabbit ran away from home. The rabbit ran away from home. The rabbit ran away from home. The rabbit ran away from home. The rabbit ran away from home.",
    },
    {
        "mode": "Extreme Stutter",
        "prompt": "The the the the the the the the the the the the the the the the the the the the the the the the the the",
    },
    {
        "mode": "Number Lock",
        "prompt": "1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3",
    },
    {
        "mode": "Code Echo",
        "prompt": "def helper():\n    return 42\n\ndef helper():\n    return 42\n\ndef helper():\n    return 42\n\ndef helper():\n    return 42\n\ndef helper():\n    return 42\n\ndef helper():\n    return 42",
    },
    {
        "mode": "Punctuation Loop",
        "prompt": "................................................................",
    },
]


def test_integration_probes(governor, tokenizer, device):
    """Phase 1: Try to trigger natural collapse with greedy decoding."""
    print("\n--- Phase 1: Aggressive Integration Probes (Greedy) ---")
    path_counts = defaultdict(int)
    total_fires = 0

    for probe in AGGRESSIVE_PROBES:
        print(f"\n[Probe] {probe['mode']}")
        inputs = tokenizer(probe["prompt"], return_tensors="pt")["input_ids"].to(device)

        with torch.no_grad():
            out = governor.generate(
                inputs,
                max_new_tokens=32,
                do_sample=False,
                temperature=0.0,
                intervene=True,
            )

        fired = len(out["decisions"])
        total_fires += fired
        for entry in out.get("psychon_log", []):
            path_counts[entry["path"]] += 1

        print(f"  Fired: {fired} | Preview: {out['gen_only_text'][:70]!r}")

    return total_fires, path_counts


def test_direct_injection(governor, device):
    """
    Phase 2: Bypass diversity gate. Manually create a low-diversity state
    and force the governor to analyze + intervene.
    """
    print("\n--- Phase 2: Direct SAE Machinery Injection ---")

    dummy_prompt = "The the the the the the the the the the"
    inputs = governor.tokenizer(dummy_prompt, return_tensors="pt")["input_ids"].to(device)

    # Register hooks and get a profile
    governor.hook_manager.register_residual_hooks(
        layer_indices=governor.profiler.layer_indices,
        every_k=governor.every_k,
    )

    with torch.no_grad():
        profile = governor.profiler.profile(inputs)

    governor.hook_manager.remove_hooks()

    # DEBUG
    print(f"  Profile layers: {profile.monitored_layers}")
    print(f"  SAE map keys: {list(governor.sae_map.keys())}")
    print(f"  Baseline calibrated: {governor.baseline.is_calibrated()}")

    # Call 1: First collapse check — counter=1, return empty
    decisions1 = governor.diagnose(
        profile,
        input_ids=inputs,
        token_diversity=0.15,
        trailing_ctr=0.15,
    )
    print(f"  Call 1 (counter={governor._collapse_persistence_counter}): decisions={len(decisions1)}")

    # Call 2: Second collapse check — counter=2, hysteresis met, SHOULD fire
    decisions2 = governor.diagnose(
        profile,
        input_ids=inputs,
        token_diversity=0.15,
        trailing_ctr=0.15,
    )
    print(f"  Call 2 (counter={governor._collapse_persistence_counter}): decisions={len(decisions2)}")

    decisions = decisions2

    if not decisions:
        print("  ❌ Direct injection: Governor refused to fire.")
        return 0, {}

    print(f"  ✅ Direct injection: {len(decisions)} decision(s) fired.")
    print(f"  Target layer: {decisions[0].layer_idx}")

    # Apply intervention and inspect path
    governor.apply_interventions(decisions)

    target_layer = decisions[0].layer_idx
    sae = governor.sae_map.get(target_layer)

    if sae is None:
        print(f"  ⚠️  No SAE loaded for layer {target_layer} — cannot analyze Psychon binding.")
        print(f"      SAEs available for: {list(governor.sae_map.keys())}")
        # Still count the fire, but mark path as unknown
        return len(decisions), {"unknown_no_sae": 1}

    dummy_residual = torch.randn(1, 1, governor.model.config.hidden_size, device=device)

    fn = governor.hook_manager._interventions.get(target_layer)
    if fn is None:
        print("  ⚠️  Intervention fn not registered")
        return len(decisions), {}

    result = fn(dummy_residual)

    pattern = governor._analyze_feature_pattern(dummy_residual, sae)
    path = "monosemantic" if pattern["is_monosemantic"] else "polysemantic"

    delta = (result[:, -1, :] - dummy_residual[:, -1, :]).norm().item()
    print(f"  Path taken: {path}")
    print(f"  Feature PR: {pattern['feat_pr']:.2f} | n_significant: {pattern['n_significant']}")
    print(f"  Applied delta: {delta:.4f} L2")

    path_counts = defaultdict(int)
    path_counts[path] += 1

    governor.hook_manager.clear_interventions()
    governor.hook_manager.remove_hooks()
    # Reset state for clean exit
    governor._collapse_persistence_counter = 0
    governor._consecutive_interventions = {}

    return len(decisions), path_counts


def analyze():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "Qwen/Qwen2.5-1.5B"

    print("=" * 70)
    print("Track A: Psychon Binding Analysis")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    ).eval()

    # HuggingAnalist repo has layers 17, 18, 19, 21
    target_layers = [17, 18, 19, 21]
    print(f"\nLoading SAEs for layers {target_layers}...")
    sae_map = load_sae_map_for_model(model_name, model, target_layers, device=device)

    # CRITICAL: layer_indices must match SAE map keys, otherwise diagnose()
    # picks max(candidate_layers) from monitored layers that have no SAE
    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        layer_indices=target_layers,   # <-- THIS WAS MISSING
        sae_map=sae_map,
    )
    trusted = [tokenizer("The quick brown fox jumps over the lazy dog.", return_tensors="pt")["input_ids"]]
    governor.calibrate(trusted)

    int_fires, int_paths = test_integration_probes(governor, tokenizer, device)
    inj_fires, inj_paths = test_direct_injection(governor, device)

    all_paths = defaultdict(int)
    for k, v in int_paths.items():
        all_paths[k] += v
    for k, v in inj_paths.items():
        all_paths[k] += v

    total = sum(all_paths.values())

    print("\n" + "=" * 70)
    print("PSYCHON BINDING VERDICT")
    print("=" * 70)

    print(f"\nIntegration fires (natural collapse): {int_fires}")
    print(f"Injection fires (forced collapse):    {inj_fires}")

    if total == 0 and inj_fires == 0:
        print("\n❌ CRITICAL: No interventions fired in any mode.")
        return

    print(f"\nIntervention path distribution (n={total}):")
    for path, count in sorted(all_paths.items(), key=lambda x: -x[1]):
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {path}: {count} ({pct:.1f}%)")

    mono_pct = (all_paths.get("monosemantic", 0) / total) * 100 if total > 0 else 0
    if mono_pct >= 50:
        print(f"\n✅ VERDICT: Attractors bind to MONOSEMANTIC Psychons ({mono_pct:.0f}% of fires).")
        print(f"   Deploy single-latent clamps for ~10x cheaper steering.")
    else:
        print(f"\n⚠️  VERDICT: Attractors are POLYSEMANTIC manifolds ({100-mono_pct:.0f}% polysemantic).")
        print(f"   Continue Gram-Schmidt resets. Train larger SAE dictionaries for cleaner features.")

    if int_fires == 0 and inj_fires > 0:
        print(f"\n⚠️  NOTE: Qwen2.5-1.5B did not naturally degenerate, but injection validated machinery.")
        print(f"   SAEs are live and intervention paths are functional.")


if __name__ == "__main__":
    analyze()
