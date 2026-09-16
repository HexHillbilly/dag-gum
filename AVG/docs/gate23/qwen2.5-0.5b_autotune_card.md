# Auto-tune valve card — Qwen/Qwen2.5-0.5B

- **band_low** = `7.65` (degenerate p90 = 2.9752, prose p10 = 17.4249)
- loopers: 23 greedy / 14 sampling
- **greedy**: rescue 22/23 (96%), EOS-death 0/23 (0%), mean ΔD2 +0.7089, mean tb-active 87.61
- **sampling**: rescue 13/14 (93%), EOS-death 0/14 (0%), mean ΔD2 +0.7888, mean tb-active 98.29
- shared defaults: suppression_strength 5.0, eos_guard True
