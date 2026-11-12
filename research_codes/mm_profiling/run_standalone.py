#!/usr/bin/env python3
"""Standalone drafter campaign: one fresh process per cell, rounds counterbalanced, identity check per arm.

    python run_standalone.py --arms "B+logitsl1,B+logitsl1+dec1d" --rounds 3 --output <dir> [--class-c] [--census]

Each cell runs standalone_draft.py (context 512, max_seq_len 4096) under the arm's environment (arms.py) and
records ms per 3-step draft, the drafted token and the hidden SHA-256. --class-c: every cell must match the first
arm's token and hidden (stop at the first mismatch). --census: one untimed cell per arm that writes the L1 census.
"""

import argparse
import fcntl
import json
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arms import arm_env  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def base_env(device):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("GEMMA4_", "TT_MM", "PLI_"))}
    env.update(
        TT_METAL_HOME=str(REPO),
        TT_METAL_RUNTIME_ROOT=str(REPO),
        PYTHONPATH=f"{REPO}:{REPO}/ttnn:{REPO}/tools",
        ARCH_NAME="blackhole",
        TT_VISIBLE_DEVICES=str(device),
        MESH_DEVICE="P150",
    )
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arms", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--class-c", action="store_true")
    p.add_argument("--census", action="store_true")
    p.add_argument("--modes", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(f"/tmp/gemma4-gather-device{args.device}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    arms = args.arms.split(",")
    order = []
    for r in range(1 if args.census else args.rounds):
        order += arms if r % 2 == 0 else arms[::-1]
    py = str(REPO / "python_env/bin/python")
    results = []
    for i, arm in enumerate(order):
        env = base_env(args.device)
        env.update(arm_env(arm))
        name = f"{i:02d}_{arm}"
        cmd = [py, str(Path(__file__).parent / "standalone_draft.py"), str(args.ctx), str(args.max_seq_len)]
        cmd += ["--census", str(args.output / f"{name}.census.json")] if args.census else ["--rounds", "1"]
        cmd += ["--modes"] if args.modes and not args.census else []
        print("running", name, arm_env(arm), flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output / f"{name}.log").write_text(proc.stdout)
        m = re.search(r"mode=full ms_per_draft=([\d.]+).*token=\[(\d+)\] hidden_sha=(\w+)", proc.stdout)
        row = {"index": i, "arm": arm, "exit_code": proc.returncode}
        if m:
            row.update(ms_per_draft=float(m.group(1)), token=int(m.group(2)), hidden_sha=m.group(3))
        for mode, ms in re.findall(r"mode=(no_argmax|backbone) ms_per_draft=([\d.]+)", proc.stdout):
            row[f"ms_{mode}"] = float(ms)
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(row, flush=True)
        if args.census:
            print("\n".join(line for line in proc.stdout.splitlines() if line.startswith("CENSUS")), flush=True)
        if proc.returncode != 0 or (not m and not args.census):
            print("\n".join(proc.stdout.splitlines()[-25:]))
            raise SystemExit(f"cell {name} failed (exit {proc.returncode})")
        if args.class_c and not args.census:
            keys = {(r["token"], r["hidden_sha"]) for r in results}
            if len(keys) != 1:
                raise SystemExit(f"cell {name}: class C identity broken: {keys}")
    if not args.census:
        for arm in arms:
            rows = [r for r in results if r["arm"] == arm]
            ms = [r["ms_per_draft"] for r in rows]
            print(
                f"{arm}: ms per draft median {statistics.median(ms):.3f} (rounds {ms}); token "
                f"{sorted({r['token'] for r in rows})}; hidden {sorted({r['hidden_sha'] for r in rows})}",
                flush=True,
            )


if __name__ == "__main__":
    main()
