"""HF greedy reference for the 12B target on the e2e prompts (raw encoding, as text_demo_v2: no chat template,
literal <bos> -> one BOS at position 0). CPU, bf16. Prints the first N generated tokens per prompt.

    python hf_greedy_ref.py --out <json> [--new 48] [--prompts history,p1,p3,...]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from run_main_e2e import PROMPT  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
QUESTIONS = REPO / "models/ttt_compat/demo/sample_prompts/input_data_questions_prefill_128.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--new", type=int, default=48)
    p.add_argument("--prompts", default="history,p0,p1,p2,p3,p4,p5,p6,p7")
    args = p.parse_args()
    qs = json.loads(QUESTIONS.read_text())
    table = {"history": PROMPT, **{f"p{j}": "<bos>" + qs[j]["prompt"] for j in range(8)}}
    tok = AutoTokenizer.from_pretrained("google/gemma-4-12B-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-4-12B-it", dtype=torch.bfloat16)
    model.eval()
    out = {}
    for name in args.prompts.split(","):
        ids = tok.encode(table[name], add_special_tokens=False)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(torch.tensor([ids]), max_new_tokens=args.new, do_sample=False)
        new = gen[0, len(ids) :].tolist()
        out[name] = {"prompt_tokens": len(ids), "ids": new, "text": tok.decode(new)}
        print(f"{name} ({len(ids)} tokens, {time.time() - t0:.0f}s): {tok.decode(new)!r}", flush=True)
        args.out.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
