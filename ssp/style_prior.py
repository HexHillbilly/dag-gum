"""Corpus style-prior: a decode-time logit bias that skews generation toward a
target corpus's n-gram distribution.

This is the "skew the distribution slightly" primitive from the research
program — the same bounded, per-step, weight-free intervention as AVG's logit
suppression, but with the opposite sign: AVG *suppresses* degenerate tokens,
this *boosts* tokens the target corpus finds likely.

Formally it is a product-of-experts prior over the LLM's next-token
distribution:

    logits' = logits + λ · log P_corpus(token | n-gram context)

P_corpus is a token-level n-gram (MLE) model built over the corpus re-tokenized
with the TARGET model's tokenizer, so the bias lives in the same token space as
the LLM. Unseen continuations get a uniform floor ε that cancels in softmax, so
the implementation only adds a positive boost to in-corpus continuations.
"""
import math
from collections import defaultdict

import torch
from transformers import LogitsProcessor


def build_ngram_model(tokenizer, lines, n=2):
    """Token-level n-gram MLE model over the corpus (target tokenizer space).

    Returns ctx -> {token_id: log_prob}, where ctx is a tuple of (n-1) token
    ids preceding the predicted token.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for line in lines:
        ids = tokenizer.encode(line, add_special_tokens=False)
        for i in range(n - 1, len(ids)):
            ctx = tuple(ids[i - (n - 1):i])
            counts[ctx][ids[i]] += 1
    model = {}
    for ctx, nxt_counts in counts.items():
        total = sum(nxt_counts.values())
        model[ctx] = {t: math.log(c / total) for t, c in nxt_counts.items()}
    return model


class CorpusNGramBias(LogitsProcessor):
    """Adds λ·log(P_corpus(t|ctx)/ε) to corpus-continuation tokens.

    Set `prompt_len` (token length of the prompt) before generation so the
    bias only applies to *generated* context, not the prompt's tail.
    """

    def __init__(self, ngram_model, n=2, lam=0.05, eps=1e-6, prompt_len=0):
        self.ngram_model = ngram_model
        self.n = n
        self.lam = lam
        self.log_eps = math.log(eps)
        self.prompt_len = prompt_len
        self._cache = {}  # (ctx, device) -> (token_ids tensor, bias values tensor)

    def __call__(self, input_ids, scores):
        # Skip until at least one token has been generated (context = that
        # token), so the prompt's tail never drives the bias.
        if input_ids.shape[1] <= self.prompt_len + (self.n - 2):
            return scores
        ctx = tuple(input_ids[0, -(self.n - 1):].tolist())
        table = self.ngram_model.get(ctx)
        if not table:
            return scores
        key = (ctx, scores.device)
        cached = self._cache.get(key)
        if cached is None:
            toks = torch.tensor(list(table.keys()), device=scores.device, dtype=torch.long)
            vals = torch.tensor(
                [self.lam * (lp - self.log_eps) for lp in table.values()],
                device=scores.device,
                dtype=scores.dtype,
            )
            cached = (toks, vals)
            self._cache[key] = cached
        toks, vals = cached
        # Single vectorized add — far faster than a per-token Python loop when
        # common contexts carry hundreds of continuations.
        scores[0, toks] += vals
        return scores


class SentenceRhythmProcessor(LogitsProcessor):
    """Decode-time sentence-length regularizer to dampen burstiness.

    Burstiness (std of per-sentence perplexity) overshoots under the instruction
    framing because the model emits erratic sentence lengths (fragments like
    "I wake." mixed with run-ons). This penalizes sentence-ending tokens
    (./!/?/newline/EOS) while the in-progress sentence is shorter than
    `min_tokens`, and boosts them past `max_tokens` — regularizing length, the
    dominant driver of the variance.

    Set `prompt_len` (token length of the prompt) before generation so the scan
    only counts generated tokens, never the prompt's tail.
    """

    def __init__(self, tokenizer, min_tokens=4, max_tokens=40, penalty=20.0, prompt_len=0):
        self.tokenizer = tokenizer
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.penalty = penalty
        self.prompt_len = prompt_len
        self.eos_id = tokenizer.eos_token_id
        self.end_ids = set()
        # Sentence terminators come in TWO token forms in Phi-3's tokenizer:
        # standalone (e.g. "." -> 869) and word-attached (e.g. "word." -> 29889).
        # The model emits the ATTACHED form, so capture both or the penalty hits
        # a token that is never generated. NB: do NOT add "\n" — it shares token
        # 29871 with trailing spaces (false resets on every word gap).
        for p in [".", "!", "?"]:
            self.end_ids.update(tokenizer.encode(p, add_special_tokens=False))
            attached = tokenizer.encode("word" + p, add_special_tokens=False)
            if attached:
                self.end_ids.add(attached[-1])

    def __call__(self, input_ids, scores):
        seq = input_ids[0].tolist()
        cur_len = 0
        for i in range(len(seq) - 1, self.prompt_len - 1, -1):
            if seq[i] in self.end_ids:
                break
            cur_len += 1
        targets = self.end_ids | {self.eos_id}
        vocab = scores.shape[1]
        if cur_len < self.min_tokens:
            # suppress ending a too-short sentence (fragments)
            for tid in targets:
                if tid is not None and tid < vocab:
                    scores[0, tid] -= self.penalty
        elif cur_len > self.max_tokens:
            # suppress continuing a too-long sentence (run-ons)
            for tid in targets:
                if tid is not None and tid < vocab:
                    scores[0, tid] += self.penalty
        return scores


def corpus_ngram_ppl(tokenizer, ngram_model, text, n=2, eps=1e-6):
    """Perplexity of `text` under the corpus n-gram model (fingerprint metric).

    Lower = the token sequence is more probable under the target corpus's
    statistical distribution, i.e. the output "looks more like" the corpus.
    Returns None when the text is too short to score.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < n:
        return None
    total_ll = 0.0
    count = 0
    for i in range(n - 1, len(ids)):
        ctx = tuple(ids[i - (n - 1):i])
        nxt = ids[i]
        logp = ngram_model.get(ctx, {}).get(nxt, math.log(eps))
        total_ll += logp
        count += 1
    return math.exp(-total_ll / count) if count else None
