# Semantic Style Proxy — Research Program

> North-star thesis, hypotheses, and phases. The living research record;
> update as measurements land. Owner-ratified only.

## The one-line thesis

**Retrieval + prompt-injection can steer a local model toward any target
style's statistical fingerprint — and doing so moves the output away from the
fingerprint statistical AI-text detectors (GLTR/Panagram/watermark) test.**

Style transfer and detection-evasion are the same operation viewed from two
directions. The product is the style tool; the detection score is its reported
mirror metric.

## Relationship to the parent projects

- **semantic-context-proxy (SCP)** — the parent. Proved the primitive: prompt-
  injection persona transfers voice (style_cos +47%…+561% across 4 corpora),
  while logit-bias "skew" does not. SSP inherits the mechanism and the harness.
- **AVG** — the mid-generation sibling. SSP + AVG are the two intervention
  points of one weight-free steering framework; SSP conditions the *style*,
  AVG governs the *degeneracy*.

## Falsifiable core (SSP hypotheses)

- **S1 — Style transfer.** Injecting retrieved corpus context moves output
  toward the target style's fingerprint without destroying coherence or topic
  adherence. *(metric: style_cos, corpus_ppl, rep_rate — SCP Phase 3 already
  established the persona case; SSP generalizes it beyond "be X" to "write like X".)*
- **S2 — Generalization.** The mechanism transfers across corpora/voices, not
  just personas. A single `style` mode works for authors, personal writing, and
  brand voice alike.
- **S3 — Detection (mirror).** Style-matching reduces the detector signal:
  perplexity toward the human reference, and — the open problem — dampened
  **burstiness** (SCP Phase 4 overshot it: a variance tell).
- **S4 — Burstiness dampening.** The variance overshoot can be brought down
  without sacrificing style_cos.

## Phases

- **Phase A — Style mode.** Generalize the persona path into a `style` mode:
  corpus + voice descriptor → governed output. Measure S1/S2 with the existing
  benchmark on a held-out corpus. This is the product.
- **Phase B — Detection refinement.** Upgrade the proxy GLTR measure to a
  trained-classifier score, and dampen the burstiness overshoot (S3/S4). Report
  detection deltas as the side metric, never the goal.

## Red lines (mirroring AVG/SCP)

- Thresholds are **owner-signed after baseline measurement**, never pre-tuned.
- A red measurement is STOP-and-report, never a reason to tune.
- Detection claims require real detector scores quoted verbatim; no vibes.
- Personal writing / corpora stay gitignored (`*hexhillbilly*`, `daisy_2015*`, `*_corpus.txt`).

## Phase A — first run (Hemingway, *In Our Time*)

**Setup.** `docs/corpora/hemingway.txt` (278 sentences, Gutenberg #61085, public
domain). Added `core.wrap_style` (the "write like X" framing) and a `style`
condition to `scripts/benchmark.py`. Model: Phi-3-mini-4k-instruct; greedy +
sampling (3×); 10 probes. Raw results: `scripts/results_ssp_hemingway.jsonl`
(gitignored, regenerable).

**Result (mean over 10 probes).**

| condition | regime   | style_cos | corpus_ppl | rep_rate |
|-----------|----------|-----------|------------|----------|
| baseline  | greedy   | 0.009     | 387,686    | 0.049    |
| persona   | greedy   | 0.200     | 139,539    | 0.050    |
| style     | greedy   | 0.157     | 374,061    | 0.094    |
| baseline  | sampling | 0.022     | 382,371    | 0.047    |
| persona   | sampling | 0.144     | 274,452    | 0.026    |
| style     | sampling | 0.114     | 332,302    | 0.021    |

**Interpretation.**

1. **S1 confirmed (with a caveat).** Retrieval + prompt-injection moves output
   toward the target fingerprint — `style_cos` 0.009 → 0.157 (style) / 0.200
   (persona), greedy. "write like X" works. *Caveat:* corpus is not held-out —
   retrieval and the centroid share the same 278 lines, so style_cos is partly
   echo, not pure transfer. A train/test split is the next rigor step.
2. **Framing is secondary; the task dominates.** On prose tasks the `style`
   framing produced the cleanest Hemingway (see below); `persona` (the v1
   wrapper, still carrying the leaky "then answer naturally") reverts to
   instructional voice on some probes. On the poetry probe both framings
   collapse to poem form — register is forced by the task, not the style. The
   aggregate persona-vs-style gap (0.20 vs 0.16) sits inside this probe noise.
3. **Metrics are coarse.** `corpus_ppl` is the truer voice fingerprint (persona
   drops it 2.8×) but noisy — values are dominated by unseen-bigram penalties
   (baseline ~3.9e5, spread 9.8e4–6.5e5) and only mean anything relative and
   per-probe. `style_cos` overweights semantic/topic similarity.

**Representative output** (probe "How do I fix a broken network connection?"):
- baseline: *"To fix a broken network connection, you can follow these steps: 1. Check your hardware…"*
- style: *"The router blinked red. I unplugged it. Waited. Plugged it back in. The lights came on. The connection was lost again. I checked the cables. They were tight. I restarted the computer. The network came back. It was simple."*
- persona: *"The network cable was frayed. I replaced it. / To fix a broken network connection, you need to check the cable first…"*

**Open items for the next iteration.**
- The probe set is conversational (built for the Daisy persona); poetry and
  persona prompts dilute the style signal. Add prose-writing probes (narrate /
  describe / explain) for the style test.
- Fair framing A/B: benchmark `style` against `wrap_persona_hardened` (v2, no
  "answer naturally" leak) rather than the v1 wrapper it currently uses.
- Held-out split: retrieval on train lines, style_cos/ppl on unseen test lines.

## Phase A — tightened run (v2)

**Changes vs first run** (resolves the three open items above): (1) prose-writing
probes via `--probe-set style`; (2) fair framing — `persona` now uses
`wrap_persona_hardened` (no "answer naturally" leak) instead of the v1 wrapper;
(3) held-out split (`--train-ratio 0.8`) — retrieval + n-gram on 222 train lines,
`style_cos` centroid on 56 held-out test lines (kills the echo confound).

**Result (mean over 10 prose probes).**

| condition | regime   | style_cos | topic_cos | rep_rate | corpus_ppl |
|-----------|----------|-----------|-----------|----------|------------|
| baseline  | greedy   | 0.120     | 0.494     | 0.044    | 210,559    |
| persona   | greedy   | 0.352     | 0.538     | 0.017    | 116,406    |
| style     | greedy   | 0.385     | 0.536     | 0.039    | 217,768    |
| baseline  | sampling | 0.133     | 0.548     | 0.021    | 291,390    |
| persona   | sampling | 0.339     | 0.535     | 0.013    | 146,367    |
| style     | sampling | 0.370     | 0.527     | 0.018    | 235,807    |

**Interpretation.**

1. **S1 confirmed cleanly.** On a held-out centroid, `style_cos` 0.120 → 0.385
   (style, +221%) / 0.352 (persona, +193%). This is transfer, not echo — the
   centroid is 56 test lines the retrieval never sees.
2. **Style framing now ≥ persona on the north star** (0.385 vs 0.352). The first
   run's "persona beats style" was the leaky v1 wrapper, not a real effect.
3. **Topic adherence preserved** — `topic_cos` holds ~0.53 across conditions.
4. **No repetition cost** — `rep_rate` ≤ baseline for both (0.039 / 0.017 vs 0.044).
5. **`corpus_ppl` is confounded by topic mismatch, not a clean voice metric.**
   The prose probes sit outside *In Our Time*'s subject matter, so the corpus
   bigram model penalizes styled output for *topic*, not voice; length-normalized
   ppl/char (persona 705 < baseline 765 < style 936) tracks bigram genericity
   more than style. → de-emphasize `corpus_ppl` for style transfer unless the
   probes are topic-matched to the corpus.

**Representative output** ("Describe making a cup of coffee"):
- baseline: *"Making a cup of coffee is a simple and enjoyable process… 1. Gather your ingredients…"*
- style: *"I wake. Sunlight through the window. I walk to the kitchen. Coffee beans in the grinder. They crackle. I pour water into the kettle. It whistles…"*
- persona: *"I made coffee. Water boiled. Ground beans in pot. Pour hot water. Steam rose. Aroma filled room. I sipped. It was good."*

Both framings produce authentic Hemingway (short declaratives, concrete nouns,
no ornament); style sustains longer scenes, persona is terser. On "tell about a
fight" the style framing turned a baseline refusal into a bullfight narrative.

**Phase A verdict: S1 held.** Style transfer works on a held-out corpus; the
`style` framing is the product. Next: Phase B (detection mirror).

## Phase B — trained detector + detection delta

**Setup.** `scripts/detector.py`: logistic regression (numpy, gradient descent)
over [log-ppl, burstiness, rep_rate] under Phi-3-mini. Trained on baseline AI
outputs (n=40) vs a **non-target** human author (Austen, *Pride and Prejudice*,
`docs/corpora/austen.txt`, 6,488 lines) so the detector learns "general human vs
AI", not "target-style vs AI". `build_book_corpus.py` gained an illustration-
caption stripper for Austen's illustrated edition.

**Result (train_acc 0.99; held-out human n=120 scores P(AI)=0.018).**

| group            | P(AI) | log-ppl | burstiness | rep_rate |
|------------------|-------|---------|------------|----------|
| baseline (AI)    | 0.969 | 1.07    | 4.73       | 0.027    |
| style            | 0.304 | 1.93    | 295.27     | 0.023    |
| persona          | 0.288 | 2.35    | 105.60     | 0.014    |
| human (Austen)   | 0.018 | 3.52    | 37.96      | 0.008    |
| human (Hemingway)| 0.017 | 3.49    | 91.48      | 0.030    |

**Interpretation.**

1. **S3 confirmed.** Style transfer cuts P(AI) 0.969 → 0.304 (style) / 0.288
   (persona), a ~70% reduction toward human (0.018). The detection mirror works:
   the same operation that moves `style_cos` up moves P(AI) down.
2. **S4 quantified — burstiness is the residual tell.** Style-output burstiness
   (295) overshoots *both* human references (Austen 38, Hemingway 91) by ~3×.
   The naive linear detector *misses* this (its linear burstiness weight keeps
   extrapolating "higher = more human"), but a variance-aware detector would flag
   295 as an anomaly. This is exactly the Phase-4 overshoot, now pinned down.
3. **Persona is better-conditioned on rhythm than style.** persona burstiness
   (106) is close to Hemingway's own (91); style (295) is not. Mirrors the
   Phase-A `corpus_ppl` result: identity framing adopts the target's actual
   sentence rhythm, instruction framing over-applies "short sentences" erratically.
4. **log-ppl is the cleanest single signal** (baseline 1.07 → human ~3.5); the
   styled output reaches only 1.9–2.4 — closer to human, not there.

**Open item (S4 dampening).** Bring style-output burstiness from 295 toward
Hemingway's ~91 without sacrificing `style_cos`. Candidates: (a) framing — the
persona framing already does this (106); (b) descriptor refinement — drop the
over-applied "short sentences" cue for "measured, consistent rhythm"; (c) a
generation-time burstiness regularizer (logit penalty on per-sentence ppl
variance). (b) is the cheapest first probe.

## Phase B — S4 probe (candidate b): descriptor refinement — NEGATIVE

**Hypothesis.** The overshoot comes from the over-applied "short sentences" cue;
replacing it with a rhythm cue would dampen burstiness without hurting style_cos.

**Result.** It did not:

| style descriptor                | style_cos | P(AI) | burstiness |
|---------------------------------|-----------|-------|------------|
| "…short sentences…" (v2)        | 0.385     | 0.304 | 295        |
| "…measured, consistent rhythm…" (v3) | 0.367 | 0.477 | 345        |

`style_cos` held (~0.37, within noise) but burstiness *worsened* (295 → 345) and
P(AI) *worsened* (0.304 → 0.477). Prompt-level rhythm cues are unreliable — the
model applies style features piecemeal, and a descriptor tweak cannot regularize
the resulting sentence-length variance.

**Framing tradeoff (the useful finding).** The identity framing is the natural
dampener. persona burstiness (106) ≈ Hemingway's own (91); instruction framing
(style) overshoots (295–345). The two framings optimize *different* objectives:

| framing              | style_cos | burstiness | P(AI) |
|----------------------|-----------|------------|-------|
| style (instruction)  | 0.385     | 295        | 0.304 |
| persona (identity)   | 0.352     | 106        | 0.288 |

style wins the north star (semantic match); persona wins the detection mirror
(natural rhythm). No free lunch — the "best" framing depends on the objective.

**Remaining path (c).** A generation-time burstiness regularizer (logit penalty
on per-sentence perplexity variance / sentence-length regularization). The real
mechanism; substantial. Not attempted in this pass.

## Phase B — S4 probe (candidate c): generation-time rhythm regularizer — PARTIAL

**Mechanism.** `SentenceRhythmProcessor` (in `style_prior.py`): a logits processor
that penalizes sentence-ending tokens (./!/?/EOS) while the in-progress sentence
is shorter than `min_tokens` (suppresses fragments) and boosts them past
`max_tokens` (suppresses run-ons). Added as the `style_rhythm` benchmark condition
(= `style` framing + the processor). Two real tokenizer bugs fixed en route: the
standalone `.` (869) vs the word-attached `.` (29889) — only the attached form is
emitted, so penalizing 869 does nothing; and `\n` shares token 29871 with trailing
spaces (false resets), so newlines are excluded from the end set.

**Result** (Hemingway probes; burstiness target = 91):

| condition      | style_cos | burstiness | P(AI) | rep_rate |
|----------------|-----------|------------|-------|----------|
| style          | 0.385     | 295        | 0.304 | 0.039    |
| style_rhythm   | 0.391     | 198        | 0.399 | 0.021    |

1. **The regularizer works mechanically.** Burstiness dampened 295 → 198 (33%
   toward Hemingway's 91); fragments suppressed (coffee probe 4 → 1; "I wake." →
   "I wake, the world still dark."); `style_cos` held (0.391 vs 0.385), `rep_rate`
   improved. It is surgical — the already-rhythmic "morning" probe is byte-identical.
2. **But the gap to human is only partially closed** (198 vs 91) — the processor
   only fires on fragments; it does not yet regularize the mid-range length variance.
3. **The naive detector mis-measures this.** P(AI) went UP (0.304 → 0.399) because
   the linear detector's burstiness weight (-1.24) extrapolates "more bursty = more
   human", so it *rewards* the overshoot and penalizes its dampening. This exposes
   the detector's variance-blindness — and implies the S3 "−70%" was partly inflated
   by the overshoot, not pure evasion.

**Remaining.** (i) Tune the regularizer (tighter length band / a length-variance
penalty) to close 198 → 91; (ii) make the detector variance-aware — penalize
burstiness *distance from the human reference* (both too low AND too high), not
"more = human". The variance-aware detector is the honest measure of both S3 and S4.

## Phase B — variance-aware detector (correction to S3)

**Change.** `scripts/detector.py` now replaces the raw burstiness feature with
`|burstiness − ref|` (ref = mean human-train burstiness, 33.86), so being FAR from
human — both too-low (AI-uniform) and too-high (overshoot) — signals AI. The
linear "more bursty = more human" was rewarding the overshoot.

**Result** (same v5 outputs, honest scoring):

| group         | burstiness | P(AI) naive | P(AI) variance-aware |
|---------------|------------|-------------|----------------------|
| baseline (AI) | 4.7        | 0.969       | 0.965                |
| style         | 295        | 0.304       | **0.818**            |
| style_rhythm  | 198        | 0.399       | **0.744**            |
| human (held-out) | 38      | 0.018       | 0.024                |

**This corrects S3.** The naive "−69%" was ~4× inflated: the honest detectability
reduction is **−15%** for style (0.965 → 0.818) and **−23%** for style_rhythm
(0.965 → 0.744). Styled output remains clearly AI-detectable (P(AI) ~0.74–0.82);
the "two sides of the same coin" holds, but weakly — style transfer nudges the
detector, it does not defeat it.

**And it validates the regularizer.** Under the honest detector the dampened
output (style_rhythm 0.744) is *less* detectable than the overshooting output
(style 0.818) — the exact opposite of the naive detector's reading. Burstiness
dampening genuinely improves evasion; it just has to be measured variance-aware.

## Phase B — S4 regularizer tuning (sweep)

**Sweep** (variance-aware scoring; target burstiness 91):

| config                   | burstiness | log-ppl | P(AI) | style_cos |
|--------------------------|------------|---------|-------|-----------|
| style (no regularizer)   | 295        | 1.93    | 0.818 | 0.385     |
| rhythm min=4 max=40      | 198        | 1.95    | **0.744** | 0.391 |
| rhythm min=6 max=25      | 141        | 1.84    | 0.790 | 0.399     |

**Finding.** The regularizer's default (min=4) is the sweet spot. Tightening to
min=6 damps burstiness further (198 → 141, closer to the 91 target) but LOWERS
log-ppl (1.95 → 1.84) — the output becomes more predictable, which the detector
reads as *more* AI. Net P(AI) worsens (0.744 → 0.790). Real tradeoff: length
regularization damps the burstiness tell at the cost of predictability.

**S4 verdict.** The generation-time regularizer works, and its default is optimal:
burstiness 295 → 198, P(AI) 0.818 → 0.744 (style → style_rhythm), `style_cos` held.
Further tightening is a dead end (documented). The remaining 198 → 91 gap is
content-driven perplexity variance, not sentence length — beyond a length
regularizer.
