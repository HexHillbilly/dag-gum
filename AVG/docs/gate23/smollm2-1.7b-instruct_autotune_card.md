# Auto-tune valve card — HuggingFaceTB/SmolLM2-1.7B-Instruct

- **band_low** = `6.0009` (degenerate p90 = 2.8859, prose p10 = 13.1165)
- loopers: 19 greedy / 10 sampling
- **greedy**: rescue 18/19 (95%), EOS-death 0/19 (0%), mean ΔD2 +0.7254, mean tb-active 100.63
- **sampling**: rescue 9/10 (90%), EOS-death 0/10 (0%), mean ΔD2 +0.7696, mean tb-active 96.8
- shared defaults: suppression_strength 5.0, eos_guard True
