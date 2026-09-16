# High-Diversity Degeneration (emoji/symbol spam) — root-caused + detector landed

**Status:** root-caused + both fixes landed (2026-08-15) — reactive detector +
proactive root-cause suppression. Original finding below, then the correction.

## Observation

llama3-8b (4-bit, temp 0.8, top_p 0.85, 256-token horizon) generates authentic
in-character voice for ~150 tokens, then degenerates into distinct-token
emoji/symbol spam:

> "...i love making new sentences and learning more about the world. 🌸💫
> 𝕥𝕥𝕥 … 💚 Ⓔ𝕄 … ⚃💬 …"

## Why the governor misses it

All three LOW-diversity detectors target repeated-token attractors:

- **Repeated-token suppression** — fires on same-token re-occurrence; the spam
  tokens are each DISTINCT, so nothing repeats within the 24-token window.
- **Bigram distinct-2 < 0.30** — distinct-2 stays HIGH (every token is new).
- **Spectral PR < band_low** — no rank collapse (no ~4-token loop motif; the
  spam is a wide, flat distribution, not a low-rank one).

## Root cause (2026-08-15, detector measurement)

The emoji/symbol spam is **INDUCED by the governor's own suppression
actuation**, not an intrinsic model property. 3-way isolation on llama3-8b 4-bit
(3 seeds, same prompt):

| condition | symbol fires | full non-ASCII | max 24-window |
|---|---|---|---|
| dormant (`intervene=False`) | 0 | 0.000 | 0.000 |
| active, detector OFF | 0 | 0.25–0.55 | 1.000 |
| active, detector ON | 24–35 | 0.13–0.19 | 0.583 |

Plain HF `generate` (no governor) is also clean (0.0 non-ASCII, 5 seeds). The
spam only appears when the governor is actively intervening.

**Mechanism:** when suppression penalizes a loop token (−5.0) *and* `eos_guard`
masks `<eos>`, the displaced probability mass must go somewhere — and under
4-bit quantization noise at temp 0.8 it falls into the high-entropy emoji/symbol
region. A "logit vacuum" filled by non-linguistic tokens.

## Detector (landed, reactive)

Fourth collapse predicate: **non-ASCII token density** over the trailing 24
generated tokens (`compute_non_ascii_fraction` + a precomputed vocab mask).
Fires when density > `symbol_density_threshold` (default 0.50) for ≥2
consecutive steps; actuates by suppressing the whole non-ASCII *class* (−5.0,
existing cooldown), the class-level analog of repeated-token suppression.

It reduces the spam (full non-ASCII 0.25–0.55 → 0.13–0.19) but does **not**
eliminate it — it is reactive (waits for the density to build, then suppresses
for an 8-step cooldown), so brief emoji bursts persist (max 24-window 0.583).

## Proactive fix (landed)

Suppress the non-ASCII class *during* any suppression event (alongside the loop
token, in the same step), so the logit vacuum can never be filled by emoji in
the first place. Landed 2026-08-15: cuts full non-ASCII 0.25–0.55 → 0.0–0.098
(3 seeds), vs 0.13–0.19 for the reactive detector alone. Residual emoji bursts
still leak through cooldown gaps and kickstart vacuums (not yet covered).

## Caveats

- Compounded by 4-bit quantization tail noise (8-bit/fp16 may attenuate — unverified).
- `non-ASCII` also flags legitimate non-English scripts (CJK/Cyrillic/Arabic);
  out of scope for the English Daisy product, but a Unicode-block refinement
  (emoji/math-symbol blocks only) is the path if non-English output must be governed.
- Distinct from the "natural repetition is coherent" finding — this is INcoherent
  (semantic collapse into symbols).

## Owner note (VibeCheque, 2026-08-15)

"Genuinely interesting" — flagged for the AVG research roadmap. Production Daisy
stays on Ollama (ungoverned) for daily use; the governed `avg` backend is the
harness. Decision: land the reactive detector now, add the proactive fix next.
