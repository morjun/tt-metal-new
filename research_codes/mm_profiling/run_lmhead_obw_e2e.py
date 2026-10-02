#!/usr/bin/env python3
"""End-to-end A/B of explicit lm_head out_block_w, with and without the drafter-head pin (12B).

Four arms, all with tuned matmuls (GEMMA4_TUNE_MATMULS=1), same protocol as run_lmhead_e2e.py
(fused traced route, 500 tokens, fresh process per cell, build_codex_current via a /tmp overlay):

  A  cols 0      auto lm_head configs                    (production, unpinned)
  B  cols 0      drafter head out_block_w 5, target 3    (best widths, unpinned)
  C  cols 38720  auto lm_head configs                    (production pin: tail -> 32, target -> 25)
  D  cols 38720  drafter tail out_block_w 8, target 3    (best widths, pinned)

Order: ABCD and DCBA alternating for --rounds rounds. Every arm is expected to be bit-exact, so
all cells should produce the same text, acceptance and iteration count; this is checked and
reported. Gates (abort): exit code, 500 tokens, fused-traced route, and the [lm-head-obw] log
line with the expected out_block_w in the explicit arms.

    python research_codes/mm_profiling/run_lmhead_obw_e2e.py --rounds 6
"""

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

from run_lmhead_e2e import OVERLAY, REPO, parse, setup_overlay

ARMS = {
    "A": {"cols": 0, "draft": None, "target": None},
    "B": {"cols": 0, "draft": 5, "target": 3},
    "C": {"cols": 38720, "draft": None, "target": None},
    "D": {"cols": 38720, "draft": 8, "target": 3},
}
PROMPT = (
    "Write a comprehensive chapter on the history of computing from the abacus to modern computers. "
    "Include detailed sections on mechanical calculators, early electronic computers, programming "
    "languages, microprocessors, personal computers, the internet, and contemporary AI. Continue with "
    "examples and historical context until you have written at least 2,000 words."
)


def env_for(arm):
    import os

    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("GEMMA4_") and not k.startswith("PLI_") and k not in ("HF_MODEL", "MESH_DEVICE")
    }
    env.update(
        {
            "TT_METAL_RUNTIME_ROOT": str(REPO),
            "TT_METAL_HOME": str(REPO),
            "PYTHONPATH": f"{OVERLAY}:{REPO}",
            "TT_VISIBLE_DEVICES": "0",
            "MESH_DEVICE": "P150",
            "HF_MODEL": "google/gemma-4-12B-it",
            "GEMMA4_ASSISTANT_MODEL": "google/gemma-4-12B-it-assistant",
            "GEMMA4_SPEC_ROUTE": "fused-batch-dim",
            "GEMMA4_SPEC_TRACE": "1",
            "GEMMA4_TUNE_MATMULS": "1",
            "GEMMA4_LMHEAD_L1_COLS": str(ARMS[arm]["cols"]),
            "GEMMA4_MAX_NEW_TOKENS": "500",
            "GEMMA4_MAX_SEQ_LEN": "4096",
            "GEMMA4_SPEC_PROMPT": PROMPT,
        }
    )
    if ARMS[arm]["draft"] is not None:
        env["GEMMA4_LMHEAD_DRAFT_OBW"] = str(ARMS[arm]["draft"])
        env["GEMMA4_LMHEAD_TARGET_OBW"] = str(ARMS[arm]["target"])
    return env


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--output", type=Path, default=REPO / "generated/lmhead_l1_profile/e2e_500_obw")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    setup_overlay()
    order = []
    for r in range(args.rounds):
        order += list("ABCD") if r % 2 == 0 else list("DCBA")
    results = []
    for index, arm in enumerate(order):
        name = f"{index:02d}_{arm}"
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-s",
            "models/demos/gemma4/demo/text_demo_v2.py",
            "-k",
            "test_demo_spec_decode",
            "--tt-arch",
            "blackhole",
        ]
        print("running", name, flush=True)
        proc = subprocess.run(
            cmd, cwd=REPO, env=env_for(arm), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        (args.output / f"{name}.log").write_text(proc.stdout)
        metrics = parse(proc.stdout)
        obw_lines = sorted(
            set(
                re.findall(
                    r"\[lm-head-obw\] (GEMMA4_LMHEAD_\w+): N=(\d+) per_core_N=(\d+) out_block_w=(\d+)", proc.stdout
                )
            )
        )
        row = {
            "index": index,
            "arm": arm,
            **ARMS[arm],
            "exit_code": proc.returncode,
            "log": f"{name}.log",
            "obw_lines": obw_lines,
            **metrics,
        }
        results.append(row)
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print({k: row[k] for k in ("index", "arm", "ms_per_iteration", "accepted_per_3", "iterations")}, flush=True)
        if proc.returncode != 0 or metrics["generated_tokens"] != 500 or metrics["route"] != "fused-traced":
            raise SystemExit(f"cell {name} failed its route/token gate")
        want = set()
        if ARMS[arm]["draft"] is not None:
            want = {
                ("GEMMA4_LMHEAD_DRAFT_OBW", str(ARMS[arm]["draft"])),
                ("GEMMA4_LMHEAD_TARGET_OBW", str(ARMS[arm]["target"])),
            }
        got = {(e, w) for e, _, _, w in obw_lines}
        if got != want:
            raise SystemExit(f"cell {name}: out_block_w knob lines {obw_lines} != expected {sorted(want)}")
    texts = {r["text_sha256"] for r in results}
    accepts = {r["accepted_per_3"] for r in results}
    iters = {r["iterations"] for r in results}
    print(
        "identical text / acceptance / iterations across all cells:",
        len(texts) == 1,
        len(accepts) == 1,
        len(iters) == 1,
        flush=True,
    )
    med = {a: statistics.median([r["ms_per_iteration"] for r in results if r["arm"] == a]) for a in ARMS}
    print("median ms/iteration:", med, flush=True)
    for a, b in (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D"), ("A", "D")):
        print(f"{b} - {a}: {med[b] - med[a]:+.3f} ms/iteration", flush=True)


if __name__ == "__main__":
    main()
