#!/usr/bin/env python3
"""12B 1x1 spec-decode e2e A/B on an upstream-main worktree (follow-up PR plan, not part of any PR).

    python run_main_e2e.py --arms "B+logitsl1,B+logitsl1+dec1d" --rounds 3 --output <dir> [--class-c]

Arms are named as in arms.py (or inline "label=K=V;K=V"). One fresh process per cell,
a discarded warm-up cell first, rounds counterbalanced (ABBA). Gates per cell: exit 0, 500 tokens. With --class-c
every cell must give the same text, acceptance and iteration count (stop at the first mismatch).
"""

import argparse
import fcntl
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

PROMPT = (
    "<bos>Write a comprehensive chapter on the history of computing from the abacus to modern computers. "
    "Include detailed sections on mechanical calculators, early electronic computers, programming "
    "languages, microprocessors, personal computers, the internet, and contemporary AI. Continue with "
    "examples and historical context until you have written at least 2,000 words."
)

sys.path.insert(0, str(Path(__file__).parent))
from arms import arm_env as named_arm_env  # noqa: E402


def arm_env(name):
    if "=" in name:  # inline "label=K=V;K=V"
        label, _, spec = name.partition("=")
        return label, dict(kv.split("=", 1) for kv in spec.split(";") if kv)
    return name, named_arm_env(name)


def base_env(repo, bos=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GEMMA4_", "TT_MM", "PLI_"))}
    env.pop("HF_MODEL", None)
    env.update(
        TT_METAL_HOME=str(repo),
        TT_METAL_RUNTIME_ROOT=str(repo),
        PYTHONPATH=f"{repo}:{repo}/ttnn:{repo}/tools",
        ARCH_NAME="blackhole",
        TT_VISIBLE_DEVICES="0",
        MESH_DEVICE="P150",
        HF_MODEL="google/gemma-4-12B-it",
        GEMMA4_ASSISTANT_MODEL="google/gemma-4-12B-it-assistant",
        GEMMA4_SPEC_TRACE="1",
        GEMMA4_SPEC_DRAFT_LEN="3",
        GEMMA4_MAX_NEW_TOKENS="500",
        GEMMA4_MAX_SEQ_LEN="4096",
        GEMMA4_SPEC_PROMPT=PROMPT if bos else PROMPT[len("<bos>") :],
    )
    return env


def parse(log):
    def get(pattern, cast=float):
        m = re.search(pattern, log)
        return cast(m.group(1)) if m else None

    text = re.search(r"== SPEC-DECODE GENERATION ==\n(.*?)\n\s*\d{4}-.*=== Speculative decoding metrics ===", log, re.S)
    rem = re.search(r"remaining: ([\d.]+)s, (\d+) tokens in (\d+) iterations", log)
    return {
        "generated_tokens": get(r"generated tokens: (\d+)", int),
        "accepted_per_3": get(r"mean accepted ([\d.]+)/3"),
        # steady state: everything after the first iteration (which carries the trace capture)
        "iterations": int(rem.group(3)) + 1 if rem else None,
        "ms_per_iteration": round(float(rem.group(1)) * 1e3 / int(rem.group(3)), 3) if rem else None,
        "tokens_per_s": get(r"Steady decode: ([\d.]+) tok/s/user"),
        "route": get(r"Effective route: ([\w-]+)", str),
        "text_sha256": hashlib.sha256(text.group(1).strip().encode()).hexdigest()[:16] if text else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--arms", required=True, help="comma-separated arm names (inline arms: 'label=K=V;K=V')")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--class-c", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    p.add_argument("--no-bos", action="store_true", help="drop the literal <bos> from the prompt")
    p.add_argument("--prompts-json", type=Path, help="run every arm on the first --prompt-count prompts of this file")
    p.add_argument("--prompt-count", type=int, default=8)
    p.add_argument("--test", default="models/demos/gemma4/demo/text_demo_v2.py")
    p.add_argument("-k", default="test_demo_spec_decode")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(f"/tmp/gemma4-gather-device{args.device}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    arms = [arm_env(a) for a in args.arms.split(",")]
    if args.prompts_json:
        data = json.loads(args.prompts_json.read_text())
        prompts = [(x["prompt"] if isinstance(x, dict) else x) for x in data][: args.prompt_count]
        bos = "" if args.no_bos else "<bos>"
        arms = [
            (f"{label}@p{j}", {**extra, "GEMMA4_SPEC_PROMPT": bos + pr})
            for j, pr in enumerate(prompts)
            for label, extra in arms
        ]
    order = [] if args.no_warmup else [("warmup", arms[0][1])]
    for r in range(args.rounds):
        order += arms if r % 2 == 0 else arms[::-1]
    py = str(args.repo / "python_env/bin/python")
    results = []
    for i, (label, extra) in enumerate(order):
        env = base_env(args.repo, bos=not args.no_bos)
        env.update(extra, TT_VISIBLE_DEVICES=str(args.device))
        name = f"{i:02d}_{label}"
        print("running", name, extra, flush=True)
        cmd = [py, "-m", "pytest", "-q", "-s", args.test, "-k", args.k]
        proc = subprocess.run(cmd, cwd=args.repo, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output / f"{name}.log").write_text(proc.stdout)
        row = {"index": i, "arm": label, "env": extra, "exit_code": proc.returncode, **parse(proc.stdout)}
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(
            {
                k: row[k]
                for k in (
                    "index",
                    "arm",
                    "ms_per_iteration",
                    "accepted_per_3",
                    "iterations",
                    "generated_tokens",
                    "tokens_per_s",
                    "text_sha256",
                )
            },
            flush=True,
        )
        if proc.returncode != 0 or (row["generated_tokens"] != 500 and not args.prompts_json):
            raise SystemExit(f"cell {name} failed (exit {proc.returncode}, tokens {row['generated_tokens']})")
        if args.class_c and label != "warmup":
            keys = {(r["text_sha256"], r["accepted_per_3"], r["iterations"]) for r in results if r["arm"] != "warmup"}
            if len(keys) != 1:
                raise SystemExit(f"cell {name}: class C identity broken: {keys}")
    for label, _ in arms:
        rows = [r for r in results if r["arm"] == label]
        ms = [r["ms_per_iteration"] for r in rows]
        print(
            f"{label}: ms/iter median {statistics.median(ms):.2f} (rounds {ms}); accepted/3 "
            f"{sorted({r['accepted_per_3'] for r in rows})}; iterations {sorted({r['iterations'] for r in rows})}; "
            f"tok/s/u {statistics.median(r['tokens_per_s'] for r in rows):.2f}; text {sorted({r['text_sha256'] for r in rows})}",
            flush=True,
        )


if __name__ == "__main__":
    sys.exit(main())
