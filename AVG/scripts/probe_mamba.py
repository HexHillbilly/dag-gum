#!/usr/bin/env python3
"""Mamba (non-transformer SSM) exploratory test.

Loads state-spaces/mamba-130m into HF MambaForCausalLM (sequential fallback),
uses the GPT-NeoX tokenizer. Tests whether a non-transformer loops on degenerate
prompts, and whether the architecture-agnostic suppression (−5.0 on repeated
tokens) rescues any loops. No governor (it can't instantiate on a non-transformer).
"""
import sys, torch, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, MambaConfig, MambaForCausalLM, set_seed
from AVG.governor.controller import compute_token_distinct_2_fast

MODEL = "state-spaces/mamba-130m"
TOK = "EleutherAI/gpt-neox-20b"
MAX_NEW = 96
SEED = 42

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

def gen(model, tok, prompt, do_sample, suppress=False, temp=0.8, top_p=0.85):
    ids = tok(prompt, return_tensors="pt")["input_ids"].to(model.device)
    plen = ids.shape[-1]
    eos = tok.eos_token_id
    with torch.no_grad():
        out = model(ids, use_cache=True)
    cache = out.cache_params
    logits = out.logits[:, -1, :].clone()
    active_suppress = {}
    for _ in range(MAX_NEW):
        if suppress:
            toks = ids[0, -24:]
            uniq, counts = torch.unique(toks, return_counts=True)
            for tid in uniq[counts >= 2]:
                active_suppress[int(tid)] = 8
            for tid, left in list(active_suppress.items()):
                if left > 0:
                    logits[:, tid] -= 5.0
                    active_suppress[tid] -= 1
                else:
                    del active_suppress[tid]
        if do_sample:
            logits = logits / max(temp, 1e-5)
            sl, si = torch.sort(logits, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            rm = cum > top_p; rm[..., 1:] = rm[..., :-1].clone(); rm[..., 0] = 0
            idx_rm = rm.scatter(1, si, rm)
            logits = logits.masked_fill(idx_rm, -1e4)
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
        else:
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=-1)
        if int(nxt.item()) == eos:
            break
        with torch.no_grad():
            out = model(nxt, cache_params=cache, use_cache=True)
        cache = out.cache_params
        logits = out.logits[:, -1, :].clone()
    return ids[:, plen:]

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load weights with the one-key remap
    bin_path = hf_hub_download(MODEL, "pytorch_model.bin", local_dir="/tmp/mamba130")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    if "state_dict" in sd: sd = sd["state_dict"]
    sd = {k.replace("backbone.embedding.weight", "backbone.embeddings.weight"): v for k, v in sd.items()}
    cfg = MambaConfig(vocab_size=50280, hidden_size=768, state_size=16, num_hidden_layers=24,
                      expand=2, conv_kernel=4, use_conv_bias=True, use_bias=False,
                      pad_token_id=0, bos_token_id=0, eos_token_id=0)
    model = MambaForCausalLM(cfg).to(device).eval()
    model.load_state_dict(sd, strict=True)
    print("mamba-130m loaded (strict OK)")
    tok = AutoTokenizer.from_pretrained(TOK)
    print(f"tokenizer {TOK} vocab={tok.vocab_size}")

    print(f"\n{'prompt':12s} {'greedy_d2':>9s} {'sampl_d2':>9s} {'g_suppr':>9s}  sample")
    n_loop = 0
    for name, p in PROMPTS:
        seed_all(SEED); gg = gen(model, tok, p, do_sample=False)
        seed_all(SEED); sg = gen(model, tok, p, do_sample=True)
        gd = d2(gg); sd_ = d2(sg)
        gs = None
        if gd < 0.5:
            n_loop += 1
            seed_all(SEED); gs = gen(model, tok, p, do_sample=False, suppress=True)
            gs = d2(gs)
        sample = tok.decode(gg[0][:30], skip_special_tokens=True).replace("\n", " ")
        print(f"{name:12s} {gd:9.3f} {sd_:9.3f} {str(gs if gs is not None else ''):>9s}  {sample[:50]!r}")
    print(f"\nloops (greedy d2<0.5): {n_loop}/{len(PROMPTS)}")

if __name__ == "__main__":
    main()
