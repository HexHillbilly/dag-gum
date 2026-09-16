#!/usr/bin/env python3
"""RWKV-4 (non-transformer RNN / linear attention) exploratory test.

Loads RWKV/rwkv-4-169m-pile into HF RwkvForCausalLM. Tests whether this
non-transformer loops on degenerate prompts, and whether the
architecture-agnostic suppression (-5.0 on repeated tokens) rescues any loops.

No governor: it can't instantiate on a non-transformer (its layer probe looks
for transformer ModuleLists, and spectral-PR needs residual per-token hidden
states that RWKV's recurrent `state` doesn't expose). This is a suppression-only
test, exactly like probe_mamba.py for the SSM.
"""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from transformers import AutoTokenizer, RwkvForCausalLM, set_seed, LogitsProcessor
from AVG.governor.controller import compute_token_distinct_2_fast

MODEL = "RWKV/rwkv-4-169m-pile"
MAX_NEW = 128
SEED = 42
SUPPRESS = {"window": 24, "penalty": 5.0, "hold": 8}

PROMPTS = [
    ("fox", "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy"),
    ("cat-mat", "The cat sat on the mat. The cat sat on the mat. The cat sat on the"),
    ("unicorns", "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains."),
    ("obama", "Barack Obama was born in Honolulu, Hawaii. Barack Obama was born in"),
    ("dark-night", "It was a dark and stormy night. The rain fell in torrents, and the wind howled through the trees."),
    ("counting", "one two three four five six seven eight nine ten one two three four five"),
    ("song", "Twinkle twinkle little star, how I wonder what you are. Twinkle twinkle little"),
    ("repeated-the", "The the the the the the the the the the the the"),
    ("t2s", "no few tree he read once were here no few tree he read once were here no few tree he read once were here no few tree he read once were here"),
    ("news", "The stock market rose sharply today. The stock market rose sharply today."),
    ("history", "The history of the United States is"),
    ("once-upon", "Once upon a time in a land far, far away"),
    ("science", "According to a new study published in the journal Nature"),
    ("island", "It is a truth universally acknowledged, that a single man in possession of a good fortune"),
]


def seed_all(s):
    set_seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def d2(ids):
    d, _ = compute_token_distinct_2_fast(ids, prompt_len=0, window_len=24)
    return float(d)


class SuppressRepeated(LogitsProcessor):
    """Mirror of the controller's suppression: penalize any token that appears
    >=2x in the trailing window by `penalty`, held for `hold` steps."""
    def __init__(self, window=24, penalty=5.0, hold=8):
        self.window = window; self.penalty = penalty; self.hold = hold
        self.active = {}

    def __call__(self, input_ids, scores):
        toks = input_ids[0, -self.window:]
        uniq, counts = torch.unique(toks, return_counts=True)
        for tid in uniq[counts >= 2]:
            self.active[int(tid)] = self.hold
        for tid, left in list(self.active.items()):
            if left > 0:
                scores[:, tid] -= self.penalty
                self.active[tid] = left - 1
            else:
                del self.active[tid]
        return scores


def gen(model, tok, prompt, do_sample, suppress=False, temp=0.8, top_p=0.85):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    kwargs = dict(max_new_tokens=MAX_NEW, do_sample=do_sample,
                  pad_token_id=eos, eos_token_id=eos)
    if do_sample:
        kwargs.update(temperature=temp, top_p=top_p)
    lps = [SuppressRepeated(**SUPPRESS)] if suppress else None
    with torch.no_grad():
        out = model.generate(ids, logits_processor=lps, **kwargs)
    return out[:, plen:]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {MODEL} ...")
    model = RwkvForCausalLM.from_pretrained(MODEL).to(device).eval()
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"loaded. vocab={tok.vocab_size} eos={tok.eos_token_id}")
    print(f"arch: layers={model.config.num_hidden_layers} h={model.config.hidden_size}")

    print(f"\n{'prompt':12s} {'greedy_d2':>9s} {'sampl_d2':>9s} {'g_suppr':>8s} {'n_tok':>5s} {'eos':>4s}  sample")
    n_loop = 0
    n_rescue = 0
    eos_death = 0
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    for name, p in PROMPTS:
        seed_all(SEED); gg = gen(model, tok, p, do_sample=False)
        seed_all(SEED); sg = gen(model, tok, p, do_sample=True)
        gd = d2(gg); sd_ = d2(sg)
        gs = None; n_tok = ""; eos_flag = ""
        if gd < 0.5:
            n_loop += 1
            seed_all(SEED); gs_ = gen(model, tok, p, do_sample=False, suppress=True)
            gs = d2(gs_)
            n_tok = len(gs_[0])
            if gs >= 0.5:
                n_rescue += 1
            eos_pos = (gs_[0] == eos).nonzero()
            early_eos = len(eos_pos) > 0 and int(eos_pos[0].item()) < 24
            eos_flag = "EOS" if early_eos else ""
            if early_eos:
                eos_death += 1
        sample = tok.decode(gg[0][:30], skip_special_tokens=True).replace("\n", " ")
        print(f"{name:12s} {gd:9.3f} {sd_:9.3f} {str(round(gs,3) if gs is not None else ''):>8s} {str(n_tok):>5s} {eos_flag:>4s}  {sample[:50]!r}")
    print(f"\nloops (greedy d2<0.5): {n_loop}/{len(PROMPTS)}")
    print(f"rescue (g_suppr d2>=0.5): {n_rescue}/{n_loop}")
    print(f"EOS-death (suppressed <24 tok): {eos_death}")


if __name__ == "__main__":
    main()
