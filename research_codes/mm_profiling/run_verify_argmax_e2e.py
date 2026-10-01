#!/usr/bin/env python3
"""End-to-end A/B of the target-verify argmax pad removal on the 12B demo.

Arms (same code; only GEMMA4_ARGMAX_PAD32 differs):
    pad_multirow  GEMMA4_ARGMAX_PAD32=multirow -- verify argmax (rows = K+1) padded to 32 (before)
    unpadded      unset                        -- no padding for any row count <= 32 (after)

Same protocol as run_down_proj_e2e.py, whose result parser and prompt this reuses:
text_demo_v2.py::test_demo_spec_decode, 12B + 12B assistant, 1x1, bf16, tuned matmuls,
canonical prompt, 500 new tokens, K=3, traced, everything in DRAM. One fresh process per cell;
round 0 is a discarded warm-up; the arm order alternates by round. Resumable.

    python research_codes/mm_profiling/run_verify_argmax_e2e.py --rounds 6
    python research_codes/mm_profiling/run_verify_argmax_e2e.py --csv
"""
import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_down_proj_e2e import PROMPT, TEST, parse  # noqa: E402

ARMS = {"pad_multirow": {"GEMMA4_ARGMAX_PAD32": "multirow"}, "unpadded": {}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--output", type=Path, default=Path("generated/verify_argmax_e2e"))
    p.add_argument("--csv", action="store_true")
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    out = args.output if args.output.is_absolute() else root / args.output
    out.mkdir(parents=True, exist_ok=True)
    if args.csv:
        rows = []
        for f in sorted(out.glob("r*_s*_*.log")):
            r, s, arm = f.stem.split("_", 2)
            if (out / f"{f.stem}.done.json").exists():
                rows.append(dict(round=r[1:], slot=s[1:], arm=arm, warmup=r == "r0", **parse(f)))
        path = Path(__file__).with_name("verify_argmax_e2e_cells.csv")
        with path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(sorted(rows, key=lambda d: (int(d["round"]), int(d["slot"]))))
        print(f"wrote {len(rows)} rows to {path}")
        return
    lock = open("/tmp/gemma4-gather-device0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {k: v for k, v in os.environ.items() if not k.startswith("GEMMA4_")}
    env.update(
        TT_METAL_HOME=str(root),
        PYTHONPATH=f"{root}:{root}/ttnn:{root}/tools",
        ARCH_NAME="blackhole",
        TT_VISIBLE_DEVICES="0",
        MESH_DEVICE="P150",
        HF_HUB_OFFLINE="1",
        HF_MODEL="google/gemma-4-12B-it",
        GEMMA4_ASSISTANT_MODEL="google/gemma-4-12B-it-assistant",
        GEMMA4_TUNE_MATMULS="1",
        GEMMA4_SHARD_ACTIVATIONS="0",
        GEMMA4_SPEC_TRACE="1",
        GEMMA4_MAX_SEQ_LEN="1024",
        GEMMA4_MAX_NEW_TOKENS="500",
        GEMMA4_PROMPT=PROMPT,
        GEMMA4_SPEC_PROMPT=PROMPT,
    )
    names = list(ARMS)
    for r in range(args.rounds + 1):  # r0 = warm-up, discarded
        for slot, arm in enumerate(names if r % 2 == 0 else names[::-1]):
            name = f"r{r}_s{slot}_{arm}"
            log, done = out / f"{name}.log", out / f"{name}.done.json"
            if done.exists():
                continue
            print(f"START {name}", flush=True)
            start = time.monotonic()
            with log.open("w") as fh:
                rc = subprocess.run(
                    [sys.executable, "-m", "pytest", "-s", "-q", "--timeout=3000", TEST],
                    cwd=root,
                    env=dict(env, **ARMS[arm]),
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    timeout=3100,
                ).returncode
            if rc:
                raise RuntimeError(f"Cell failed: {name}; campaign stopped, inspect {log}")
            res = parse(log)
            done.write_text(json.dumps(dict(res, seconds=time.monotonic() - start)))
            print(
                f"DONE {name}: {res['ms_per_iter']:.2f} ms/iter, accept {res['accept']:.2f}, text {res['text_sha']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
