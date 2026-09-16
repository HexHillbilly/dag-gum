# Auto-tune valve card — TinyLlama/TinyLlama-1.1B-Chat-v1.0

- **band_low** = `7.9526` (degenerate p90 = 2.9518, prose p10 = 18.255)
- loopers: 18 greedy / 13 sampling
- **greedy**: rescue 17/18 (94%), EOS-death 0/18 (0%), mean ΔD2 +0.7053, mean tb-active 110.67
- **sampling**: rescue 13/13 (100%), EOS-death 0/13 (0%), mean ΔD2 +0.786, mean tb-active 120.0
- shared defaults: suppression_strength 5.0, eos_guard True
