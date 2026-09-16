# Instructed self-signal — RESULTS (does the marker fire reliably under the persona?)

Model: `microsoft/Phi-3-mini-4k-instruct`, persona = Emerson. 17 probes (7 tool /
3 reason / 7 chat) × 3 samples. Command: `scripts/measure_selfsignal.py`.

Tests the user's proposal: instead of a separate persona-free router, instruct the
model (still IN persona) to say "I need to think harder." and stop whenever it needs
external info or careful reasoning, then drop the persona and re-engage after EOS.

## Results

| label  | marker_fired | clean_stop |
|--------|--------------|------------|
| tool   | 0.67         | 0.57       |
| reason | 0.33         | 0.22       |
| chat   | **0.38**     | 0.14       |

## The marker is unreliable in BOTH directions

- **Under-fires on tool/reason** (recall miss): 33% of tool turns and 67% of
  reasoning turns produce no marker — the model fabricates ("current temperature is
  a brisk 16°C", "won by Joe Biden" [stale]) or philosophizes ("Each new step we
  take in thought reconciles twenty seemingly discordant facts…" for 17×43) instead
  of signaling.
- **Over-fires on chat** (precision miss): the marker fires on 38% of *trivial*
  turns — "What is 2 plus 2?" → "I need to think harder.", "Give me a recipe for
  pancakes" → "I need to think harder.". Precision 0.62 vs the router's 1.00.
- **Clean-stop is weak**: only 57% of fired markers actually stop (EOS). The rest
  continue with the answer in the same breath ("I need to think harder.\n\n
  Photosynthesis, in the essence of Emerson's spirit, is…"), which breaks EOS-based
  detection.

## Why it fails (the structural lesson)

The router and the self-signal are the SAME question asked in two places:

| | decision made | result |
|---|---|---|
| router | **outside** the persona (persona-free pass) | 1.00 / 1.00 |
| self-signal | **inside** the persona (model wears it while deciding) | 0.67 / 0.33 / 0.62 |

Putting the routing decision *inside* the persona lets the persona corrupt it — it
masks the "I need a tool / this is hard" signal (recall miss) *and* smears the marker
onto trivial turns (precision miss). The persona is a surface layer; it cannot also
reliably adjudicate whether it needs to be switched off. The decision has to be made
**persona-free**, which is exactly what the router does.

The "re-engage after EOS" half of the proposal is still right — it's the two-stage
resume, already measured at 1.00 grounding + ~0.29 voice.

## Verdict

**Router wins.** The self-signal is a clean, elegant idea, but the data says it is
strictly worse than the router on every axis (recall, precision, clean-stop). The
gate trigger is the persona-free few-shot router; the self-signal is recorded as a
tested-and-rejected negative.

(The "think hard" generalization is still worth pursuing later — but via extending
the *router's* few-shot set to include reasoning examples, not via the marker.)
