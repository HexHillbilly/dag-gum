# Auto-tune valve card — NousResearch/Meta-Llama-3-8B-Instruct

- **band_low** = `7.8703` (degenerate p90 = 2.9653, prose p10 = 18.0221)
- loopers: 21 greedy / 16 sampling
- **greedy**: rescue 20/21 (95%), EOS-death 0/21 (0%), mean ΔD2 +0.6398, mean tb-active 98.62
- **sampling**: rescue 16/16 (100%), EOS-death 0/16 (0%), mean ΔD2 +0.7717, mean tb-active 121.5
- shared defaults: suppression_strength 5.0, eos_guard True
