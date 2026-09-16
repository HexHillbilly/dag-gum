# Spontaneous loop formation — the loop period is always prompt-seeded

**Date:** 2026-08-14
**Script:** `scripts/probe_spontaneous_loops.py`
**Question:** on *non-looping* prompts, does a model **spontaneously** form a short loop
(≤8 tokens) — and what sets the period of whatever repetition forms? This is the reframed
loop-formation question left open after the PR-floor root-cause showed the "≈3 floor" was
fixture-primed.

## Method

`Qwen2.5-1.5B`, greedy, **512 tokens**, on 90 diverse non-looping prompts (30 natural prose +
30 code + 30 schema/JSON). Two detectors per prompt:

- **short-loop detector** — min trailing-24 distinct-token count (a `k`-token loop has ~`k`
  distinct; < 9 means a short loop formed);
- **periodicity detector** — the strongest self-agreement period `p` of the trailing 256 tokens
  (`agreement(p)` = fraction of positions where `seq[i]==seq[i−p]`).

## Results

| class | count (of 90) | note |
|---|---|---|
| periodic (any repetition, agreement ≥ 0.30) | 64 (71%) | — |
| **long-period phrase repetition (p ≥ 15)** | **34 (38%)** | dominant spontaneous degeneration |
| single-token fixed point (`0 0 0…`, `———…`) | 9 (10%) | prompt-seeded degenerate fixed point |
| short-period (p ≤ 8) flagged | 12 | mostly *structural*, see below |
| medium (9 ≤ p < 15) | 9 | mix of phrase + structural |
| no repetition detected | 26 | — |

The short-period (p ≤ 8) flags decompose into three kinds, only one of which is a genuine loop:

1. **Single-token collapse** — `0 0 0…` (7 cases) and `———…` (1 case). A real degenerate fixed
   point, but period-1, not a word-loop. It arises where the prompt makes a single token
   self-reinforce (code prompts end at a numeric slot; `0` predicts `0`).
2. **Structural continuation (NOT degeneration)** — code boilerplate/headers
   (`# -*- coding: utf-8 -*-`), comment stubs (`# TODO: add a cache…`), numbered lists
   (`# 1.62` / `# 1.63`), and JSON key/array patterns. The model is *correctly* continuing
   repetitive structure; the periodicity detector flags it, but it is not a loop.
3. **Genuine spontaneous short word-loop** — rare: `Array Object Array Object…` (schema/2),
   `range'][0']['range'][0']…` (schema/13), `ment desc=1 & job job require…` (schema/11),
   `abot/Alabot/Al…` (prose/9, a broken-token cycle). **≈4/90 (4%)**, and every one is
   *prompt-adjacent* — it emerges from the prompt's own short repetitive structure (JSON keys,
   array indices), never from a model-chosen period.

## Finding

**The loop period is always prompt-seeded — the model has no spontaneous preferred loop length.**

- **Short loops (2–8 tokens)** require a short *primed motif* (fixtures) or a short repetitive
  structure already in the prompt (schema/code). They do not self-organize on unstructured prose.
- **Spontaneous degeneration** on prose is **long-period phrase repetition** (15–61 tokens),
  matching the prompt's sentence/paragraph structure — and it never compresses the trailing-24
  window (PR stays 10–19, which is why the spectral detector fires 0/200 on prose).
- **The one genuinely spontaneous short attractor is the single-token fixed point** (`0 0 0`,
  `———`) — the "tight loop" that token suppression is built to break.

So the earlier "≈3 floor" story was a triple artifact: the PR *metric* (rank `k−1`), the *fixture*
motif length (2–8), and the absence of any spontaneous short-loop mechanism to contradict it. The
model's degenerate attractors, unprompted, are **1-token fixed points** and **phrase-length
cycles** — never free-standing 2–8-token word-loops.

## Pitfall (for future measurement)

A periodicity/self-agreement detector **conflates degeneration with legitimate structural
continuation**. Code comments, numbered lists, JSON keys, and schema arrays are all periodic by
construction; flagging them as "loops" inflates the apparent loop rate (here: 12 short-period
flags, of which only ~4 are genuine loops and ~8 are correct structural continuation). Any
loop-formation measurement must separate "repeats already-emitted content" from "continues a
legitimate repetitive template".
