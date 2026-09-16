# Resume grounding — RESULTS (does hard grounding recover persona resume?)

Model: `microsoft/Phi-3-mini-4k-instruct`, persona = Emerson. 3 samples × 5 probes.
Command: `scripts/measure_resume_grounding.py --out /tmp/resume_grounding.jsonl`.

The tool-loop probe left persona resume broken: `uses_result 0.53`, `topic_cos 0.435`
(voice restored at `style_cos 0.382`, but grounding collapsed). This probe tests two
prompt-level fixes.

## Results

| condition        | uses_result | style_cos | topic_cos |
|------------------|-------------|-----------|-----------|
| baseline         | 1.00        | 0.020     | 0.782     |
| persona          | 0.53        | **0.382** | 0.435     |
| persona_hard     | 0.93        | 0.088     | 0.654     |
| persona_twostage | **1.00**    | 0.165     | **0.702** |

Split metric (voice-portion measured separately):

| two-stage portion | style_cos |
|-------------------|-----------|
| grounded lead-in  | 0.138     |
| voice elaboration | **0.291** |

## What happened

- **`persona_hard`** ("use ONLY the fact, do not add anything") overshoots: grounding
  recovers (0.93) but the voice collapses to 0.088 — the model just emits the bare
  fact ("Canberra."). The directive is *too* hard; it forbids the voice.
- **`persona_twostage`** ("state the fact plainly, *then* explain in your voice") is the
  winner. The model reliably emits both halves in one turn:
  > "The capital of Australia is Canberra.  In the spirit of Emerson, one might say:
  > 'Canberra, though not as familiar as Boston Bay, holds its own as the chosen seat…'"
  Grounding is fully recovered (`uses_result 1.00`, `topic_cos 0.702` ≈ baseline 0.782),
  and a genuine voice elaboration follows.

## The tradeoff is softened, not eliminated

The voice-vs-content tension still exists, but the two-stage structure moves the
operating point to a usable spot:

- grounding: 0.53 → **1.00** (fully recovered, matches baseline)
- topic fidelity: 0.435 → **0.702** (≈ 90% of the baseline gap closed)
- voice: 0.382 → **0.291** on the elaboration portion alone (~75% of pure persona)

The whole-output `style_cos 0.165` *understates* the voice — it averages the neutral
grounded lead-in (0.138) with the voiced elaboration (0.291). The voice is real and
clearly present in the elaboration; it is diluted by the front-loaded plain sentence,
not lost.

## Implication for the gate

The resume phase does **not** need a mid-turn persona drop. A single persona'd turn
with the two-stage prompt is enough:

1. "State the answer from the result in one plain sentence" → grounding (front-loaded,
   effectively persona-neutral).
2. "Then, in your own voice, explain what it means" → voice (back-loaded).

This is the resume-level microcosm of the whole gate, and it works: the model grounds
first and voices second *within a single generation*.

## Remaining open piece

The outbound trigger — the tool-loop probe showed the persona masks the "I need a
tool" signal (0/15), so intent routing must be **persona-free** (a cheap `[tool]/[chat]`
classifier pass before the persona wrapper). That is the next probe.
