# Persona-drop gate — DESIGN (evidence-backed)

## Problem

The SCP persona makes a small model a bad tool-user. Three measured failures, all on
Phi-3-mini / Emerson:

| probe | metric | result |
|-------|--------|--------|
| tool-call fidelity | usable tool call under persona | **1.00 → 0.33** (persona suppresses the call) |
| tool-loop phase 1 | intent signaled under persona | **0/15** (persona masks "I need a tool", fabricates instead) |
| tool-loop phase 2 | resume uses the result | **1.00 → 0.53** (persona drifts off the result) |

The persona's voice value is real (`style_cos` 0.02 → 0.38), but it trades content for
voice — and at a tool boundary content is the point.

## The gate (3 phases, each measured)

```
 user turn ──► [1] ROUTER (persona-free) ──NO_TOOL──► persona ON, chat in voice
                   │ TOOL
                   ▼
              [2] TOOL CALL (persona OFF) ──► execute
                   ▼
              [3] RESUME (two-stage prompt) ──► grounded fact + voice elaboration
```

### [1] Router — persona-free few-shot trigger ✅ measured 1.00/1.00

Bare model, few-shot boundary prompt ("current/external → TOOL, memory/creative →
NO_TOOL", 4 labeled examples). **1.00 recall, 1.00 precision, 1.00 accuracy** on the
clean probe set. The persona never sees the tool decision — this is the fix for the
0/15 intent-masking failure.

### [2] Tool call — persona OFF ✅ measured 1.00 usable

Baseline (persona-free) tool-call emission is 1.00 usable. Dropping the persona for
the call recovers the ~67% the persona suppresses. This phase is already proven by the
tool-call-fidelity baseline.

### [3] Resume — two-stage "state-then-voice" ✅ measured 1.00 grounding + ~0.29 voice

Single persona'd turn with:
> "State the answer from the result in one plain sentence. Then, in your own voice,
> explain what it means."

Grounding fully recovered (`uses_result 1.00`, `topic_cos 0.702` ≈ baseline), voice
retained on the elaboration portion (`style_cos 0.291` ≈ 75% of pure persona). No
mid-turn persona drop needed — grounding is front-loaded, voice back-loaded, both in
one generation.

## The operating principle

The persona should never be active during tool **cognition** — deciding to use a tool
(router), emitting the call, or grounding the result. It belongs only on the
**surface**: the conversational voice and the post-hoc elaboration. Splitting the turn
into a neutral grounded core + a voiced shell is what breaks the voice-or-content
tradeoff.

## Caveats before production wiring

1. **Router is Phi-3-mini only** — production runs llama3-8b via Ollama; re-verify the
   few-shot router there (or run it on a dedicated small model).
2. **Clean probe set** — the 1.00 will degrade on edge cases ("is the capital *still*
   Canberra?"); needs a wider sweep.
3. **n is small** — 3 samples; strong but not exhaustive.

## Open design choices (decision points)

- Router on the production model vs a dedicated Phi-3-mini router (cheap, offloads the
  trigger).
- Where the gate lives: in `proxy.py` (per-turn routing) vs a pre-pass before the
  persona wrapper.
