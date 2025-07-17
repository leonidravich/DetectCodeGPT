#!/usr/bin/env python
import argparse, time, random, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from baselines import rank as rank_legacy
from baselines import rank_fast

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="codellama/CodeLlama-7b-hf")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    # Set padding token for CodeLlama tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(args.model).to(args.device)
    mdl.eval()

    class _Args: DEVICE = args.device
    argshim = _Args()
    model_cfg = {"base_model": mdl, "base_tokenizer": tok}

    seeds = [
        "def foo(x):\n    return x*2\n",
        "class A:\n    pass\n",
        "if x>0:\n    print('hi')\n",
        "for i in range(5):\n    x+=i\n",
    ]
    texts = [random.choice(seeds) for _ in range(args.n)]

    # correctness spot check
    lv = [rank_legacy.get_rank(t, argshim, model_cfg, log=True) for t in texts[:8]]
    fv = rank_fast.get_ranks_fast(texts[:8], argshim, model_cfg, log=True, batch_size=8)
    print("Correctness diff:", [abs(a-b) for a,b in zip(lv,fv)])

    # timing
    t0 = time.perf_counter()
    _ = [rank_legacy.get_rank(t, argshim, model_cfg, log=True) for t in texts]
    t1 = time.perf_counter()
    legacy_s = t1 - t0

    t2 = time.perf_counter()
    _ = rank_fast.get_ranks_fast(texts, argshim, model_cfg, log=True, batch_size=args.batch)
    t3 = time.perf_counter()
    fast_s = t3 - t2

    print(f"Legacy {legacy_s:.3f}s | Fast {fast_s:.3f}s | Speedup {legacy_s/fast_s:.1f}x")

if __name__ == "__main__":
    main()
