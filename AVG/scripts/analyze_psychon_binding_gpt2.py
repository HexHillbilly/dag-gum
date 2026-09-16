#!/usr/bin/env python3
"""
Track A Diagnostic: Psychon Binding Analysis — GPT-2 Edition.

Tests whether GPT-2's repetition attractors map to monosemantic Psychons
or polysemantic manifolds using live SAELens SAEs.
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
        "prompt": "The rabbit ran away from home. The rabbit ran away from home. The rabbit ran away from home.",
    },
    {
        "mode": "Extreme Stutter",
        "prompt": "The the the the the the the the the the the the the the the the",
    },
    {
        "mode": "Number Lock",
        "prompt": "1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3",
    },
    {
        "mode": "Token Spam",
        "prompt": "A A A A A A A A A A A A A A A A",
    },
]


def test_integration_probes(governor, tokenizer, device):
    """Phase 1: GPT-2 is much more prone to looping than Qwen."""
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
    """Phase 2: Force collapse and inspect SAE feature structure."""
    print("\n--- Phase 2: Direct SAE Machinery Injection ---")

    dummy_prompt = "The the the the the the the the the the"
    inputs = governor.tokenizer(dummy_prompt, return_tensors="pt")["input_ids"].to(device)

    governor.hook_manager.register_residual_hooks(
        layer_indices=governor.profiler.layer_indices,
        every_k=governor.every_k,
    )

    with torch.no_grad():
        profile = governor.profiler.profile(inputs)

    governor.hook_manager.remove_hooks()

    print(f"  Profile layers: {profile.monitored_layers}")
    print(f"  SAE map keys: {list(governor.sae_map.keys())}")

    # Call 1: counter = 1, silent
    decisions1 = governor.diagnose(
        profile,
        input_ids=inputs,
        token_diversity=0.15,
        trailing_ctr=0.15,
    )
    print(f"  Call 1 (counter={governor._collapse_persistence_counter}): decisions={len(decisions1)}")

    # Call 2: counter = 2, fire
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

    governor.apply_interventions(decisions)

    target_layer = decisions[0].layer_idx
    sae = governor.sae_map.get(target_layer)

    if sae is None:
        print(f"  ⚠️  No SAE loaded for layer {target_layer}")
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
    governor._collapse_persistence_counter = 0
    governor._consecutive_interventions = {}

    return len(decisions), path_counts


def analyze():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "gpt2"  # 124M params, 12 layers, d_model=768

    print("=" * 70)
    print("Track A: Psychon Binding Analysis — GPT-2 Edition")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # GPT-2-small: 12 layers. Mid-to-late commitment = layers 8, 9, 10, 11
    target_layers = [8, 9, 10, 11]
    print(f"\nLoading SAEs for layers {target_layers}...")
    sae_map = load_sae_map_for_model(model_name, model, target_layers, device=device)

    governor = ActiveVarietyGovernor(
        model,
        tokenizer=tokenizer,
        layer_indices=target_layers,
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

    if total == 0:
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
        print(f"   Continue Gram-Schmidt resets.")

    if int_fires > 0:
        print(f"\n✅ NOTE: GPT-2 naturally degenerated on {int_fires} probe(s).")
        print(f"   Live SAEs caught and steered real attractors.")
    elif inj_fires > 0:
        print(f"\n⚠️  NOTE: GPT-2 did not naturally degenerate, but injection validated machinery.")


if __name__ == "__main__":
    analyze()
