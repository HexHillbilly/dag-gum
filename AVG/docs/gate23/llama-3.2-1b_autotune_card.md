# Auto-tune valve card — unsloth/Llama-3.2-1B

- **band_low** = `8.0421` (degenerate p90 = 2.9733, prose p10 = 18.4723)
- loopers: 26 greedy / 15 sampling
- **greedy**: rescue 25/26 (96%), EOS-death 0/26 (0%), mean ΔD2 +0.6472, mean tb-active 101.62
- **sampling**: rescue 15/15 (100%), EOS-death 0/15 (0%), mean ΔD2 +0.7681, mean tb-active 111.33
- shared defaults: suppression_strength 5.0, eos_guard True
