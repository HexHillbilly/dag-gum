# Auto-tune valve card — microsoft/Phi-3-mini-4k-instruct

- **band_low** = `6.9467` (degenerate p90 = 2.9855, prose p10 = 15.5389)
- loopers: 11 greedy / 6 sampling
- **greedy**: rescue 10/11 (91%), EOS-death 0/11 (0%), mean ΔD2 +0.6561, mean tb-active 109.09
- **sampling**: rescue 6/6 (100%), EOS-death 0/6 (0%), mean ΔD2 +0.7899, mean tb-active 93.33
- shared defaults: suppression_strength 5.0, eos_guard True
