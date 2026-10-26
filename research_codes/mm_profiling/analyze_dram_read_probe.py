#!/usr/bin/env python3
"""Analyzer for test_dram_read_probe.py groups (gemma4-12b record §9.2e).

    python research_codes/mm_profiling/analyze_dram_read_probe.py generated/dram_read_probe [--group gate|e1|e2]

gate: the pre-registered pass criteria (a) median per-core rate 9.27-11.33 GB/s, (b) bulk 324-430 GB/s and
      Spearman rho(logical row, finish) <= -0.8, (c) burst rate >= 45.6 GB/s; per repeat and over repeats.
e1:   per position (logical x, y; virtual coordinate) mean rate and mean block latency (issue -> barrier return) of
      the fat-K and thin-K strips over repeats; pooled SD over positions; position range in pooled SDs; Spearman of
      rate against logical row and column.
e2:   access x depth x banks table of mean rate (SD); pooled SD; the main-effect and two-way contrasts named in the
      prediction table.
Writes <dir>/<group>_summary.md next to the group's json.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

MHZ = 1350.0


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(order):
            r[i] = k
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def pooled_sd(groups):
    groups = [g for g in groups if len(g) > 1]
    dof = sum(len(g) - 1 for g in groups)
    return (sum((len(g) - 1) * statistics.variance(g) for g in groups) / dof) ** 0.5 if dof else float("nan")


def core_rate(c):
    return c["rate"]


def block_latency_us(c):
    return statistics.mean(((b[1] - b[0]) % (1 << 32)) / MHZ for b in c["blocks"])


def gate(rows):
    out = ["| check | repeat | value | criterion | pass |", "|---|---|---|---|---|"]
    verdict = {}
    for r in rows:
        rates = [core_rate(c) for c in r["cores"].values()]
        if r["name"].startswith("a_"):
            v = statistics.median(rates)
            ok = 9.27 <= v <= 11.33
            out.append(
                f"| (a) fat-K | {r['rep']} | median per-core {v:.2f} GB/s (min {min(rates):.2f}, max {max(rates):.2f}) | 9.27–11.33 | {ok} |"
            )
            verdict.setdefault("a", []).append(ok)
        elif r["name"].startswith("b_"):
            ok = r["bulk"] is not None and 324 <= r["bulk"] <= 430 and r["rho_row_end"] <= -0.8
            out.append(
                f"| (b) thin-K | {r['rep']} | bulk {r['bulk']:.1f} GB/s, ρ(row, end) {r['rho_row_end']:+.2f}, aggregate {r['agg']:.1f} |"
                f" 324–430 and ρ ≤ −0.8 | {ok} |"
            )
            verdict.setdefault("b", []).append(ok)
        elif r["name"].startswith("c_"):
            v = rates[0]
            ok = v >= 45.6
            out.append(f"| (c) burst | {r['rep']} | {v:.2f} GB/s | ≥ 45.6 | {ok} |")
            verdict.setdefault("c", []).append(ok)
    out.append("")
    out.append("Verdict (every repeat passes): " + ", ".join(f"({k}) {all(v)}" for k, v in sorted(verdict.items())))
    return "\n".join(out)


def e1(rows):
    per = defaultdict(lambda: defaultdict(list))  # pattern -> (x, y) -> [(rate, latency)]
    virt = {}
    for r in rows:
        pat, x, y = r["name"].split("_")
        ((key, c),) = r["cores"].items()
        xy = (int(x), int(y))
        per[pat][xy].append((core_rate(c), block_latency_us(c)))
        virt[xy] = tuple(c["virt"])
    out = []
    for pat, d in per.items():
        rates = {xy: [v[0] for v in vs] for xy, vs in d.items()}
        lats = {xy: [v[1] for v in vs] for xy, vs in d.items()}
        sd_r, sd_l = pooled_sd(list(rates.values())), pooled_sd(list(lats.values()))
        mr = {xy: statistics.mean(v) for xy, v in rates.items()}
        ml = {xy: statistics.mean(v) for xy, v in lats.items()}
        xs = sorted(mr)
        lo, hi = min(mr, key=mr.get), max(mr, key=mr.get)
        out.append(f"### {pat}-K strip, {len(xs)} positions × {len(next(iter(rates.values())))} repeats")
        out.append(
            f"- rate: median {statistics.median(mr.values()):.2f} GB/s, min {mr[lo]:.2f} at {lo} (virtual {virt[lo]}),"
            f" max {mr[hi]:.2f} at {hi} (virtual {virt[hi]}); pooled SD {sd_r:.3f}; range = {(mr[hi] - mr[lo]) / sd_r:.1f} pooled SD"
        )
        out.append(
            f"- block latency: median {statistics.median(ml.values()):.2f} µs, range {min(ml.values()):.2f}–{max(ml.values()):.2f};"
            f" pooled SD {sd_l:.3f}; range = {(max(ml.values()) - min(ml.values())) / sd_l:.1f} pooled SD"
        )
        out.append(
            f"- Spearman rate vs logical row {spearman([xy[1] for xy in xs], [mr[xy] for xy in xs]):+.2f},"
            f" vs logical column {spearman([xy[0] for xy in xs], [mr[xy] for xy in xs]):+.2f}"
        )
        gx, gy = max(x for x, _ in xs) + 1, max(y for _, y in xs) + 1
        out.append("")
        out.append("rate GB/s by logical position (rows y, columns x):")
        out.append("")
        out.append("| y \\ x | " + " | ".join(str(x) for x in range(gx)) + " |")
        out.append("|---" * (gx + 1) + "|")
        for y in range(gy):
            out.append(f"| {y} | " + " | ".join(f"{mr[(x, y)]:.2f}" if (x, y) in mr else "" for x in range(gx)) + " |")
        out.append("")
    return "\n".join(out)


def e2(rows):
    per = defaultdict(list)
    for r in rows:
        (c,) = r["cores"].values()
        per[r["name"]].append(core_rate(c))
    sd = pooled_sd(list(per.values()))
    m = {k: statistics.mean(v) for k, v in per.items()}
    out = [f"pooled SD {sd:.3f} GB/s; threshold for an effect 3 × = {3 * sd:.3f}", ""]
    accesses = ["strided2k", "contig2k", "burst16k"]
    out.append("| access | depth | 1 bank | 2 banks | 4 banks | 8 banks |")
    out.append("|---|---|---|---|---|---|")
    for a in accesses:
        for d in (1, 2, 4):
            cells = []
            for b in (1, 2, 4, 8):
                k = f"{a}_d{d}_b{b}"
                cells.append(f"{m[k]:.2f} ({statistics.stdev(per[k]):.2f})" if k in m else "")
            out.append(f"| {a} | {d} | " + " | ".join(cells) + " |")
    out.append("")

    def eff(k1, k2):
        return f"{m[k1]:.2f} → {m[k2]:.2f} ({(m[k2] - m[k1]) / sd:+.1f} SD)"

    out.append("Contrasts (pooled SD units):")
    out.append(
        f"- B1 banks at strided d1: 1 → 8 banks {eff('strided2k_d1_b1', 'strided2k_d1_b8')}; 1 → 2 {eff('strided2k_d1_b1', 'strided2k_d1_b2')}"
    )
    out.append(
        f"- B2 contiguous vs strided 2 KB, d1: 1 bank {eff('strided2k_d1_b1', 'contig2k_d1_b1')}; 8 banks {eff('strided2k_d1_b8', 'contig2k_d1_b8')}"
    )
    out.append(
        f"- B3 depth at 1 bank, strided: d1 → d2 {eff('strided2k_d1_b1', 'strided2k_d2_b1')}; d1 → d4 {eff('strided2k_d1_b1', 'strided2k_d4_b1')}"
    )
    out.append(
        f"- B4 burst vs contiguous 2 KB, d4: 1 bank {eff('contig2k_d4_b1', 'burst16k_d4_b1')}; 4 banks {eff('contig2k_d4_b4', 'burst16k_d4_b4')}"
    )
    out.append(f"- B4 at d1, 1 bank: {eff('contig2k_d1_b1', 'burst16k_d1_b1')}")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--group", default="gate")
    a = p.parse_args()
    rows = json.loads((a.dir / f"{a.group}.json").read_text())
    text = {"gate": gate, "e1": e1, "e2": e2}[a.group](rows)
    (a.dir / f"{a.group}_summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
