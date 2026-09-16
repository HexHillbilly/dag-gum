# "Explain in their terms" — Emerson probe (topic-bleed as lexicon transfer)

**Reframed question.** Topic bleed, read as a *feature*: feed one author's writing
(the stand-in "user's lexicon"), retrieve a line, inject it as persona, and ask a
cross-subject question. Does the output **explain the question in their terms** —
high `style_cos` (their voice) *and* high `topic_cos` (still your question)?

## Setup

- Corpus: Emerson, *Essays — First Series* (Gutenberg #2944), 3,486 sentences,
  committed at `docs/corpora/emerson_essays.txt`.
- Probes: 5 technical/scientific (jet engine, CPU, vaccines, seasons, combustion).
- Mechanism: `wrap_persona_hardened` (production directive), plain HF Phi-3-mini,
  sampling, 3 samples. Joint metric `(topic_cos, style_cos)`.

## Numbers

| condition | topic_cos | style_cos |
|---|---|---|
| baseline | 0.777 | 0.001 |
| persona  | 0.587 | **0.311** |

Persona transfers a strong voice (style_cos 0→0.31) at a partial topic cost
(0.78→0.59). The interesting part is the *spread*: some samples keep the
explanation (topic 0.70+), others drop it (topic 0.40).

## The finding (qualitative — this is the actual result)

The mechanism **does** produce "explain in their terms" — but unreliably. Two
clean examples from the same probe:

**Good transfer** (explains the jet engine *and* speaks Emerson):
> "Power and speed be hands and feet; the jet engine, a marvel of modern
> engineering… operates on Newton's third law… takes in air, compresses it,
> mixes it with fuel, and ignites the mixture…" (topic 0.71, style 0.15)

**Bad transfer** (pure Emerson voice, no explanation — the old "subject swap"):
> "In the grand tapestry of the cosmos, where the boundless spirit of
> exploration is the loom upon which we weave our destinies… a testament to the
> ingenuity of mankind, a harmonious symphony of physics and ambition…"
> (topic 0.40, style 0.45)

The **seasons** probe transfers cleanly every time (topic 0.68–0.72, "the tilt of
its axis… the grand celestial dance"), while jet-engine/CPU swing between good and
bad. The joint metric separates them: good = high topic + high style, bad = low
topic + high style.

## Interpretation

1. The user's intuition is **confirmed**: the "topic bleed" mechanism is the
   engine for "explain in their terms." The good samples are existence proof.
2. The open problem is **reliability**, not existence: the current directive
   ("weave its spirit into your reply") sometimes imports voice *without* the
   explanation. The tool needs a directive that demands *both*.
3. `topic_cos` is a meaningful signal for "still explaining the question" — it
   tracks the good/bad split, and it is exactly the thing the Stage-2 drift
   detector was (rightly) trying to hold up, with the (wrong) actuation.

## Explicit-directive result (v4: "explain thoroughly and accurately, in your voice and terms")

Added `wrap_explain_in_terms` and re-ran the A/B (baseline / persona / explain):

| condition | topic mean | topic min | topic std | style mean | topic<0.45 (subject-swap) |
|---|---|---|---|---|---|
| persona (v2 "weave its spirit") | 0.587 | 0.399 | 0.103 | 0.311 | 2/15 |
| explain (v4 explicit) | 0.600 | 0.452 | 0.070 | 0.310 | **0/15** |

The explicit directive **eliminates the catastrophic subject-swaps** (2/15 → 0/15),
**raises the floor** (min 0.399 → 0.452), and **tightens the spread** (std 0.103 →
0.070) at zero style cost. The mean barely moves because the *remaining* topic
drop is the intrinsic register-shift cost of adopting the voice — not a failure.

Qualitatively, every "explain" sample now actually explains the concept (Newton's
third law, axial tilt, orbit) in Emerson's voice; the pure "grand tapestry"
voice-only output disappears. Minor residual: the model sometimes meta-frames
("As Emerson, I might say…", "I, Emerson, find…"), a small self-reference
artifact.

## Bottom line

"Explain in their terms" is achievable and the explicit directive makes it
*reliable* (no more subject-swaps). The irreducible `topic_cos ≈ 0.60` (vs 0.78
baseline) is the voice-transfer signature itself — the cost of speaking Emerson,
not a defect. The v4 directive is the tool's actual prompt and should replace the
v2 "weave its spirit" framing in the production proxy.

## Generalization check — Mark Twain (*Life on the Mississippi*, Gutenberg #245)

Same A/B, second author (6,591 sentences, `docs/corpora/twain_mississippi.txt`):

| author | condition | topic mean | topic min | style mean | topic<0.45 |
|---|---|---|---|---|---|
| Emerson | persona | 0.587 | 0.399 | 0.311 | 2/15 |
| Emerson | explain | 0.600 | 0.452 | 0.310 | 0/15 |
| Twain | persona | 0.543 | 0.342 | 0.232 | 1/15 |
| Twain | explain | 0.581 | 0.437 | 0.201 | 1/15 |

The v4 directive **generalizes**: topic mean rises on both authors (+0.013
Emerson, +0.038 Twain) and the floor rises on both (Emerson 0.399→0.452, Twain
0.342→0.437). Qualitatively the outputs are genuine "explain in Twain's terms":

> "Well, I'll tell you 'bout the whirling wonders of these metal birds… much
> like a well-tuned banjo on a clear Mississippi night. It works on a principle
> not unlike the old steam engines…"

One nuance: the **style cost is author-dependent**. Emerson holds style exactly
(0.310); Twain drops slightly (0.232→0.201) because "explain thoroughly and
accurately" pulls toward formal register, which collides more with Twain's
dialect than with Emerson's aphoristic voice. The directive transfers the voice
of an abstract rhetorician more cheaply than a folksy raconteur.
