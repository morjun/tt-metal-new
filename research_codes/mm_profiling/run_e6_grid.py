#!/usr/bin/env python3
"""E6 on the production kernel (gemma4-12b record §9.2e): test_e6_grid.py per arm under tracy, NIU-delivered rates.

    python research_codes/mm_profiling/run_e6_grid.py --output generated/dram_read_probe/e6_prod
    python research_codes/mm_profiling/run_e6_grid.py --analyze generated/dram_read_probe/e6_prod

Arms: gate_proj_8x8 (A64), gate_proj_86 (A86), gate_w2_8x8_pcn3 (W2), each without and with +ep0
(TT_MM1D_IN1_BANK0_XY=785); one process per arm, the six in order and then reversed. Run and sample rows as run_e2b_nt33.
Per call (warm-up dropped): per-core delivered rate (median over cores); read window = first core's first sample to last
core's last sample; tail = first core's last sample to last core's last sample; bytes-normalized rate = all cores'
delivered bytes / window. Per arm: mean and SD over calls and processes; contrasts in pooled SDs.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from run_e2b_nt33 import LOGS, MHZ, WORD, calls_of
from run_lmhead_e2e import OVERLAY, REPO

ARMS = ["gate_proj_8x8", "gate_proj_86", "gate_w2_8x8_pcn3"]
ARMS = ARMS + [a + "+ep0" for a in ARMS]


def run(order, out):
    out.mkdir(parents=True, exist_ok=True)
    for i, arm in enumerate(order):
        name, ep0 = arm.removesuffix("+ep0"), arm.endswith("+ep0")
        cell = out / f"{i:02d}_{arm}"
        cell.mkdir(exist_ok=True)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GEMMA4_") and k != "TT_MM1D_IN1_BANK0_XY"}
        env.update(
            TT_METAL_RUNTIME_ROOT=str(REPO),
            TT_METAL_HOME=str(REPO),
            PYTHONPATH=f"{OVERLAY}:{REPO}",
            TT_VISIBLE_DEVICES="0",
            TT_MM1D_IN1_NIU_SAMPLE="1",
            E6_ARM=name,
        )
        if ep0:
            env["TT_MM1D_IN1_BANK0_XY"] = "785"
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
        cmd.append("research_codes/mm_profiling/test_e6_grid.py")
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


def analyze(out):
    m = defaultdict(lambda: defaultdict(list))  # arm -> metric -> values per call
    for cell in sorted(out.glob("[0-9][0-9]_*")):
        if not (cell / "niu_samples.csv").exists():
            continue
        arm = cell.name[3:]
        per = calls_of(cell / "niu_samples.csv")
        n = {len(c) for c in per.values()}
        assert len(n) == 1, f"{cell.name}: cores disagree on call count {n}"
        for k in range(1, n.pop()):
            rates, words = [], 0
            for calls in per.values():
                c = calls[k]
                dt = (c[-1][0] - c[0][0]) / MHZ
                rates.append((c[-1][1] - c[0][1]) * WORD / dt / 1e3)
                words += c[-1][1] - c[0][1]
            first = min(c[k][0][0] for c in per.values())
            lasts = [c[k][-1][0] for c in per.values()]
            window = (max(lasts) - first) / MHZ
            m[arm]["median"].append(statistics.median(rates))
            m[arm]["window"].append(window)
            m[arm]["tail"].append((max(lasts) - min(lasts)) / MHZ)
            m[arm]["bytes_rate"].append(words * WORD / window / 1e3)
            m[arm]["cores"].append(len(rates))
    keys = ("median", "window", "tail", "bytes_rate")
    sd = {
        key: statistics.mean([statistics.variance(v[key]) for v in m.values() if len(v[key]) > 1]) ** 0.5
        for key in keys
    }
    summary = {a: {key: statistics.mean(v[key]) for key in keys} | {"n": len(v["window"])} for a, v in m.items()}
    lines = [f"pooled SD: " + ", ".join(f"{k} {sd[k]:.3f}" for k in keys), ""]
    lines += [
        "| arm | calls | per-core median GB/s | window µs | tail µs | bytes-normalized GB/s |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for a in ARMS:
        if a in summary:
            s = summary[a]
            lines.append(
                f"| {a} | {s['n']} | {s['median']:.2f} | {s['window']:.2f} | {s['tail']:.2f} | {s['bytes_rate']:.1f} |"
            )

    def d(a, b, key):
        if a in summary and b in summary:
            x = summary[a][key] - summary[b][key]
            return f"{a} − {b}, {key}: {x:+.2f} ({x / sd[key]:+.1f} SD)"
        return ""

    lines.append("")
    for suf in ("", "+ep0"):
        lines += [
            x
            for x in (
                d("gate_proj_86" + suf, "gate_proj_8x8" + suf, "window"),
                d("gate_w2_8x8_pcn3" + suf, "gate_proj_86" + suf, "tail"),
                d("gate_w2_8x8_pcn3" + suf, "gate_proj_8x8" + suf, "tail"),
                d("gate_w2_8x8_pcn3" + suf, "gate_proj_86" + suf, "bytes_rate"),
            )
            if x
        ]
    if all(a in summary for a in ARMS[:2] + ARMS[3:5]):
        g0 = summary["gate_proj_86"]["window"] - summary["gate_proj_8x8"]["window"]
        g1 = summary["gate_proj_86+ep0"]["window"] - summary["gate_proj_8x8+ep0"]["window"]
        lines.append(
            f"A86 − A64 window gap: base {g0:+.2f}, ep0 {g1:+.2f}; change {g1 - g0:+.2f} µs ({(g1 - g0) / (sd['window'] * 2**0.5):+.1f} SD of a difference)"
        )
    text = "\n".join(lines)
    (out / "e6_prod.md").write_text(text + "\n")
    (out / "e6_prod.json").write_text(json.dumps(dict(summary=summary, pooled_sd=sd, calls=m), indent=1))
    print(text)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=REPO / "generated/dram_read_probe/e6_prod")
    p.add_argument("--analyze", type=Path)
    a = p.parse_args()
    if a.analyze:
        analyze(a.analyze)
    else:
        run(ARMS + ARMS[::-1], a.output)
        analyze(a.output)


if __name__ == "__main__":
    main()
