#!/usr/bin/env python3
"""E2b runner and analyzer (gemma4-12b record §9.2e): test_fatk_nt33.py per arm under tracy, NIU-delivered rates.

    python research_codes/mm_profiling/run_e2b_nt33.py --order 32,33,33,32 --output generated/dram_read_probe/e2b
    python research_codes/mm_profiling/run_e2b_nt33.py --analyze generated/dram_read_probe/e2b

Run: one process per arm (fresh device, timeout 600 s), the overlay build (run_lmhead_e2e.OVERLAY; not rebuilt here),
TT_MM1D_IN1_NIU_SAMPLE=1. Keeps the device log's IN1_NIU_RD_WORDS rows as <output>/<i>_nt<NT>/niu_samples.csv.
Analysis: per core and call, delivered rate = (last - first sample) x 64 B / (last - first sample time); calls are
split at gaps > 100 us in a core's samples; the first call (warm-up) is dropped. Per arm: the per-call median over
cores, then mean and SD over calls and processes; pooled SD over arms.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from run_lmhead_e2e import OVERLAY, REPO

LOGS = REPO / "generated/profiler/.logs"
MHZ, WORD = 1350.0, 64


def run(order, out):
    out.mkdir(parents=True, exist_ok=True)
    for i, nt in enumerate(order):
        cell = out / f"{i:02d}_nt{nt}"
        cell.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GEMMA4_")}
        env.update(
            TT_METAL_RUNTIME_ROOT=str(REPO),
            TT_METAL_HOME=str(REPO),
            PYTHONPATH=f"{OVERLAY}:{REPO}",
            TT_VISIBLE_DEVICES="0",
            TT_MM1D_IN1_NIU_SAMPLE="1",
            E2B_NT=str(nt),
        )
        cmd = [
            "timeout",
            "-k",
            "30",
            "600",
            sys.executable,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "-m",
            "pytest",
            "-s",
            "-q",
        ]
        cmd.append("research_codes/mm_profiling/test_fatk_nt33.py")
        print("running", cell.name, flush=True)
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (cell / "run.log").write_text(proc.stdout)
        if (LOGS / "profile_log_device.csv").exists():
            with open(LOGS / "profile_log_device.csv") as src, open(cell / "niu_samples.csv", "w") as dst:
                for j, line in enumerate(src):
                    if j < 2 or "IN1_NIU_RD_WORDS" in line:
                        dst.write(line)
        print(cell.name, "exit", proc.returncode, flush=True)
        if proc.returncode:
            sys.exit(proc.returncode)


def calls_of(path):
    """{core: [[(cycle, words), ...] per call]}"""
    per = defaultdict(list)
    with open(path) as f:
        next(f)
        for row in csv.DictReader(f, skipinitialspace=True):
            if (row.get("zone name") or "").strip() == "IN1_NIU_RD_WORDS":
                per[(int(row["core_x"]), int(row["core_y"]))].append(
                    (int(row["time[cycles since reset]"]), int(row["data"]))
                )
    out = {}
    for core, s in per.items():
        s.sort()
        calls, cur = [], [s[0]]
        for a, b in zip(s, s[1:]):
            if (b[0] - a[0]) / MHZ > 100:
                calls.append(cur)
                cur = []
            cur.append(b)
        calls.append(cur)
        out[core] = calls
    return out


def analyze(out):
    arms = defaultdict(list)
    detail = {}
    for cell in sorted(out.glob("*_nt*")):
        if not (cell / "niu_samples.csv").exists():
            continue
        nt = int(cell.name.split("_nt")[1])
        per = calls_of(cell / "niu_samples.csv")
        ncalls = {len(c) for c in per.values()}
        assert len(ncalls) == 1, f"{cell.name}: cores disagree on call count {ncalls}"
        n = ncalls.pop()
        cell_rows = []
        for k in range(1, n):  # drop the warm-up call
            rates = {}
            for core, calls in per.items():
                c = calls[k]
                dt = (c[-1][0] - c[0][0]) / MHZ
                rates[core] = (c[-1][1] - c[0][1]) * WORD / dt / 1e3 if dt > 0 else float("nan")
            med = statistics.median(rates.values())
            arms[nt].append(med)
            cell_rows.append(
                dict(call=k, cores=len(rates), median=med, min=min(rates.values()), max=max(rates.values()))
            )
        detail[cell.name] = cell_rows
    summary = {}
    for nt, v in sorted(arms.items()):
        summary[nt] = dict(n=len(v), mean=statistics.mean(v), sd=statistics.stdev(v) if len(v) > 1 else 0.0)
    pooled = (
        (sum((s["n"] - 1) * s["sd"] ** 2 for s in summary.values()) / sum(s["n"] - 1 for s in summary.values())) ** 0.5
        if summary
        else None
    )
    res = dict(summary=summary, pooled_sd=pooled, cells=detail)
    (out / "e2b.json").write_text(json.dumps(res, indent=1))
    for nt, s in summary.items():
        print(f"Nt {nt}: per-core delivered median {s['mean']:.2f} GB/s (SD {s['sd']:.2f}, {s['n']} calls)")
    if 32 in summary and 33 in summary:
        d = summary[33]["mean"] - summary[32]["mean"]
        print(f"Nt 33 - Nt 32: {d:+.2f} GB/s = {d / pooled:+.1f} pooled SD ({pooled:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order", default="32,33,33,32")
    p.add_argument("--output", type=Path, default=REPO / "generated/dram_read_probe/e2b")
    p.add_argument("--analyze", type=Path)
    a = p.parse_args()
    if a.analyze:
        analyze(a.analyze)
    else:
        run([int(v) for v in a.order.split(",")], a.output)
        analyze(a.output)


if __name__ == "__main__":
    main()
