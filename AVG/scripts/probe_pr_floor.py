#!/usr/bin/env python3
"""Root-cause the universal PR≈3 collapse floor (MEASUREMENT ONLY).

Question (RESEARCH.md §1.2, "candidate invariant, not a proven law"): why does the
participation ratio of the early-layer residual stream collapse to ≈3 across 9 models /
5 architectures? What IS the ≈3-dimensional subspace?

This probe runs the production model into a DORMANT loop and characterizes the collapse
subspace at the min-PR step:

  (1) full singular-value spectrum of the trailing-24 layer-2 hidden-state window
      (rank-3 cliff vs smooth decay — is ≈3 a hard rank or a soft effective-dim?);
  (2) logit readout of the top-3 right-singular vectors (what the subspace "says" to the
      unembedding);
  (3) nearest-token-embedding for each top-3 vector (is the subspace the span of ~3 tokens?);
  (4) token-embedding-only PR over the same 24-token window (is ≈3 already in the token
      sequence, or a property of the internal attractor?);
  (5) the decoded token sequence at min-PR (what is the loop?).

MEASUREMENT ONLY: no controller change, no threshold, no actuation. Writes a jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

import torch

try:
    from AVG.core.metrics import participation_ratio as _pr
except Exception:  # keep probe standalone if AVG import is unavailable
    _pr = None


def participation_ratio(activations: torch.Tensor, eps: float = 1e-8) -> float:
    if _pr is not None:
        return float(_pr(activations).item())
    x = activations.reshape(-1, activations.shape[-1]).to(dtype=torch.float32)
    x = x - x.mean(dim=0, keepdim=True)
    s = torch.linalg.svdvals(x)
    s = s.clamp(min=eps)
    return float((s.sum() ** 2) / (s.pow(2).sum() + eps))


def get_blocks(model):
    for attr in ("model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for part in attr.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError("could not locate transformer blocks")


def get_embed(model):
    for attr in ("model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"):
        obj = model
        ok = True
        for part in attr.split("."):
            if not hasattr(obj, part):
                ok = False
                break
            obj = getattr(obj, part)
        if ok:
            return obj
    raise RuntimeError("could not locate token embedding")


def load_fixtures(path, field):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs.append(rec[field] if field else (rec.get("text") or rec.get("prompt")))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--revision", default="8faed761d45a263340a0528343f099c05c9a4323")
    ap.add_argument("--fixtures", default="tests/fixtures/t2s_degenerate.jsonl")
    ap.add_argument("--field", default=None, help="jsonl key (default: auto-detect text/prompt)")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--layer-idx", type=int, default=2, help="block index for the PR hook")
    ap.add_argument("--out", default="docs/gate23/pr_floor_probe.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {args.model} @ {args.revision} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    blocks = get_blocks(model)
    embed = get_embed(model)
    lm_head = model.lm_head
    E = embed.weight.detach()  # (vocab, d)
    print(f"blocks={len(blocks)} embed={tuple(E.shape)} lm_head={tuple(lm_head.weight.shape)}", flush=True)

    texts = load_fixtures(args.fixtures, args.field)[: args.n]

    # top-3 per-record characterization helpers
    def nearest_tokens(vec, k=5):
        v = vec.float()
        v = v / (v.norm() + 1e-8)
        sims = (E.float() @ v)  # (vocab,)
        idx = sims.topk(k).indices.tolist()
        return [(tok.decode(i), round(float(sims[i]), 3)) for i in idx]

    def logit_readout(vec, k=5):
        logits = lm_head(vec.float().to(lm_head.weight.dtype))
        idx = logits.topk(k).indices.tolist()
        vals = logits.topk(k).values.tolist()
        return [(tok.decode(i), round(float(v), 2)) for i, v in zip(idx, vals)]

    out_recs = []
    for fi, text in enumerate(texts):
        inp = tok(text, return_tensors="pt").to(model.device)
        prompt_len = inp["input_ids"].shape[1]
        with torch.no_grad():
            out = model(**inp, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits

        # hook captures layer-2 (block idx) OUTPUT hidden state per generated token
        hs_buffer = []  # list of (d,) float32 tensors, one per generated step
        handle = blocks[args.layer_idx].register_forward_hook(
            lambda m, i, o: hs_buffer.append(
                (o[0] if isinstance(o, tuple) else o)[:, -1:, :].detach().float().squeeze(0).squeeze(0)
            )
        )

        gen_ids = []
        per_step_pr = []
        per_step_tok_emb_pr = []
        with torch.no_grad():
            for step in range(args.max_new):
                nxt = torch.argmax(next_logits[:, -1, :], dim=-1).item()
                gen_ids.append(nxt)
                nxt_inp = torch.tensor([[nxt]], device=model.device)
                out = model(input_ids=nxt_inp, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_logits = out.logits
                if len(hs_buffer) >= 24:
                    win = torch.stack(hs_buffer[-24:])  # (24, d)
                    per_step_pr.append(participation_ratio(win))
                    per_step_tok_emb_pr.append(participation_ratio(E[gen_ids[-24:]]))
                else:
                    per_step_pr.append(float("nan"))
                    per_step_tok_emb_pr.append(float("nan"))
        handle.remove()

        if len(hs_buffer) < 24:
            continue
        prs = [p for p in per_step_pr if not math.isnan(p)]
        if not prs:
            continue
        min_pr = min(prs)
        min_idx = per_step_pr.index(min_pr)  # step index (0-based, step 24+ corresponds to window end)

        # window of 24 hidden states ending at min_idx
        win = torch.stack(hs_buffer[min_idx - 23 : min_idx + 1])  # (24, d)
        x = (win.float() - win.float().mean(dim=0, keepdim=True))
        s = torch.linalg.svdvals(x)  # 24 singular values
        U, S, Vh = torch.linalg.svd(x, full_matrices=False)  # Vh: (24, d)
        spectrum = [round(float(v), 4) for v in (s / (s.sum() + 1e-8))]

        top3 = Vh[:3]  # (3, d) right singular vectors
        top3_logits = [logit_readout(v) for v in top3]
        top3_tokens = [nearest_tokens(v) for v in top3]

        # token-embedding-only PR over the same generated token ids
        window_ids = gen_ids[min_idx - 23 : min_idx + 1]
        emb_win = E[window_ids]  # (24, d)
        tok_emb_pr = participation_ratio(emb_win)

        # the attractor center (mean hidden state) logit readout
        center = win.float().mean(dim=0)
        center_logits = logit_readout(center, k=5)

        rec = {
            "fixture": fi,
            "min_pr": round(min_pr, 4),
            "min_step": min_idx,
            "prompt_len": prompt_len,
            "n_gen": len(gen_ids),
            "spectrum": spectrum,
            "top3_logit_readout": top3_logits,
            "top3_nearest_tokens": top3_tokens,
            "token_emb_only_pr": round(tok_emb_pr, 4),
            "window_distinct_tokens": len(set(window_ids)),
            "center_logit_readout": center_logits,
            "window_tokens": [tok.decode(i) for i in window_ids],
            "trajectory": [
                {"step": i + 24, "hidden_pr": round(per_step_pr[i], 3), "tok_emb_pr": round(per_step_tok_emb_pr[i], 3)}
                for i in range(len(per_step_pr))
                if not math.isnan(per_step_pr[i])
            ][-48:],
            "full_gen_tokens": [tok.decode(i) for i in gen_ids[:64]],
        }
        out_recs.append(rec)
        print(
            f"[{fi}] min_pr={min_pr:.3f} @step {min_idx} | tok_emb_pr={tok_emb_pr:.3f} "
            f"| distinct={len(set(window_ids))} | spectrum[:4]={spectrum[:4]} "
            f"| window={rec['window_tokens'][:12]}",
            flush=True,
        )

    with open(args.out, "w") as f:
        for r in out_recs:
            f.write(json.dumps(r) + "\n")

    # aggregate
    if out_recs:
        minprs = [r["min_pr"] for r in out_recs]
        embprs = [r["token_emb_only_pr"] for r in out_recs]
        gaps = [r["min_pr"] - r["token_emb_only_pr"] for r in out_recs]
        print(f"\n=== SUMMARY (n={len(out_recs)}) ===", flush=True)
        print(f"min-PR: mean {sum(minprs)/len(minprs):.3f} range [{min(minprs):.3f}, {max(minprs):.3f}]", flush=True)
        print(f"token-emb-only PR: mean {sum(embprs)/len(embprs):.3f} range [{min(embprs):.3f}, {max(embprs):.3f}]", flush=True)
        print(f"gap (hidden - tok_emb): mean {sum(gaps)/len(gaps):.3f} range [{min(gaps):.3f}, {max(gaps):.3f}]", flush=True)
        deep = [r for r in out_recs if r["min_pr"] < 4.0]
        print(f"deep-collapse records (min_pr<4): {len(deep)}", flush=True)
        for r in deep:
            print(f"  fi{r['fixture']}: min_pr={r['min_pr']:.3f} tok_emb={r['token_emb_only_pr']:.3f} "
                  f"distinct={r['window_distinct_tokens']} window={r['window_tokens'][:8]}", flush=True)
    print(f"[wrote] {args.out}", flush=True)


if __name__ == "__main__":
    main()
