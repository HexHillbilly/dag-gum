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
