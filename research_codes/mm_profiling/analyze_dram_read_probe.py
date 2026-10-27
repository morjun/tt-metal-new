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


# ---- plan v3 (harness v2) groups -------------------------------------------------------------------------------------
ENDPOINT_Y = {0: 11, 1: 2, 2: 9, 3: 5, 4: 11, 5: 3, 6: 8, 7: 6}  # bank -> NoC0 endpoint row, physical (plan v3 G1)


def gate2(rows):
    """Gate v2 criteria (plan v3): (a) median per-core 9.27-11.33; (b) bulk 324-430 and rho(phys y, end) <= -0.8;
    (c) >= 45.6; every repeat. References: the generator-off runs."""
    out = [
        "| check | repeat | median per-core GB/s (min / max) | bulk GB/s | op µs | tail µs | ρ(phys y, end) | pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    verdict = defaultdict(list)
    for r in sorted(rows, key=lambda r: (r["name"], r["rep"])):
        rates = [c["rate"] for c in r["cores"].values()]
        med = statistics.median(rates)
        if r["name"] == "a_on":
            ok = 9.27 <= med <= 11.33
        elif r["name"] == "b_on":
            ok = r["bulk"] is not None and 324 <= r["bulk"] <= 430 and r["rho_y_end"] <= -0.8  # no bulk window: fail
        elif r["name"] == "c_burst":
            ok = med >= 45.6
        else:
            ok = None
        if ok is not None:
            verdict[r["name"]].append(ok)
        rho = "-" if r["rho_y_end"] is None else f"{r['rho_y_end']:+.2f}"
        out.append(
            f"| {r['name']} | {r['rep']} | {med:.2f} ({min(rates):.2f} / {max(rates):.2f}) |"
            f" {'-' if r['bulk'] is None else format(r['bulk'], '.1f')} |"
            f" {r['op_us']:.1f} | {r['tail_us']:.1f} | {rho} | {'' if ok is None else ok} |"
        )
    out += ["", "Verdict (every repeat): " + ", ".join(f"{k} {all(v)}" for k, v in sorted(verdict.items()))]
    return "\n".join(out)


def e1v3(rows):
    """E1 v3 predictions (plan v3): B1 band, B2-B4 range, B5 endpoint-row classes, B6, R1-R4 thin-K threshold."""
    per = defaultdict(list)
    for r in rows:
        (c,) = r["cores"].values()
        per[r["name"]].append(c["rate"])
    sd = pooled_sd(list(per.values()))
    m = {k: statistics.mean(v) for k, v in per.items()}
    fat = {k: v for k, v in m.items() if k.startswith("fat_")}
    thin = {k: v for k, v in m.items() if k.startswith("thin_")}

    def phys_of(name):
        _, x, y, _ = name.split("_")
        x, y = int(x), int(y)
        return (x + 1 if x < 6 else x + 5, y + 2)

    out = [
        f"pooled SD {sd:.3f} GB/s ({len(per)} configs × {len(next(iter(per.values())))} repeats); 3 SD = {3 * sd:.3f}",
        "",
    ]
    vals = list(fat.values())
    out.append(
        f"- fat-K (880 configs): min {min(vals):.2f}, median {statistics.median(vals):.2f}, max {max(vals):.2f} GB/s;"
        f" range {(max(vals) - min(vals)) / sd:.1f} SD; within 9.27–11.33: {sum(9.27 <= v <= 11.33 for v in vals)} / {len(vals)};"
        f" ≥ 10.3 + 3 SD: {sum(v >= 10.3 + 3 * sd for v in vals)} / {len(vals)}"
    )
    out += [
        "",
        "B5 classes per bank (endpoint row = cores in the bank's NoC0 endpoint row):",
        "",
        "| bank | endpoint row y | endpoint-row cores: n, mean (range) | others: n, mean (range) | Δ (SD) | within-class range (SD): row / others |",
        "|---|---|---|---|---|---|",
    ]
    for b in range(8):
        ks = [k for k in fat if k.endswith(f"_b{b}")]
        row = [fat[k] for k in ks if phys_of(k)[1] == ENDPOINT_Y[b]]
        oth = [fat[k] for k in ks if phys_of(k)[1] != ENDPOINT_Y[b]]
        d = statistics.mean(row) - statistics.mean(oth)
        out.append(
            f"| {b} | {ENDPOINT_Y[b]} | {len(row)}, {statistics.mean(row):.2f} ({min(row):.2f}–{max(row):.2f}) |"
            f" {len(oth)}, {statistics.mean(oth):.2f} ({min(oth):.2f}–{max(oth):.2f}) | {d:+.2f} ({d / sd:+.1f}) |"
            f" {(max(row) - min(row)) / sd:.1f} / {(max(oth) - min(oth)) / sd:.1f} |"
        )
    tv = list(thin.values())
    lo = [v for k, v in thin.items() if k.endswith("_lo")]
    hi = [v for k, v in thin.items() if k.endswith("_hi")]
    out += [
        "",
        f"- thin-K {{0–3}}: min {min(lo):.2f}, median {statistics.median(lo):.2f}, max {max(lo):.2f};"
        f" {{4–7}}: min {min(hi):.2f}, median {statistics.median(hi):.2f}, max {max(hi):.2f} GB/s;"
        f" ≥ 20.6 on both sets: {all(v >= 20.6 for v in tv)} ({sum(v >= 20.6 for v in tv)} / {len(tv)})",
    ]
    b0 = sorted((v, k) for k, v in fat.items() if k.endswith("_b0"))
    med = b0[len(b0) // 2]
    out.append(f"- E2 v3 position (median fat-K bank-0 rate): {med[1]} at {med[0]:.2f} GB/s (logical x, y in the name)")
    return "\n".join(out)


def e2v3(rows, dup_rows=None):
    """E2 v3 predictions (plan v3): access x depth x N_b table with request sizes, B1-B4 contrasts in pooled SD,
    between-process check on (strided, D 1, N_b 1)."""
    per = defaultdict(list)
    for r in rows:
        (c,) = r["cores"].values()
        per[r["name"]].append(c["rate"])
    sd = pooled_sd(list(per.values()))
    m = {k: statistics.mean(v) for k, v in per.items()}
    size = lambda a, nb: 2048 if a != "burst" else 16384 // nb  # noqa: E731
    out = [
        f"pooled SD {sd:.3f} GB/s; 3 SD = {3 * sd:.3f}",
        "",
        "| access | D | N_b 1 | N_b 2 | N_b 4 | N_b 8 |",
        "|---|---|---|---|---|---|",
    ]
    for a in ("strided2k", "contig2k", "burst"):
        for d in (1, 2, 4):
            cells = [
                f"{m[f'{a}_d{d}_b{nb}']:.2f} ({statistics.stdev(per[f'{a}_d{d}_b{nb}']):.2f}; {size(a, nb) // 1024} KB)"
                for nb in (1, 2, 4, 8)
            ]
            out.append(f"| {a} | {d} | " + " | ".join(cells) + " |")

    def z(k1, k2):
        return (m[k2] - m[k1]) / sd

    out += ["", "Contrasts (pooled SD):"]
    nb1 = [m[f"{a}_d{d}_b1"] for a in ("strided2k", "contig2k", "burst") for d in (1, 2, 4)]
    out.append(
        f"- B1: at N_b 1 the 9 cells span {min(nb1):.2f}–{max(nb1):.2f} ({(max(nb1) - min(nb1)) / sd:.1f} SD);"
        f" strided D1 by N_b: " + ", ".join(f"{m[f'strided2k_d1_b{nb}']:.2f}" for nb in (1, 2, 4, 8))
    )
    b2 = [(nb, d, z(f"strided2k_d{d}_b{nb}", f"contig2k_d{d}_b{nb}")) for nb in (1, 2, 4, 8) for d in (1, 2, 4)]
    out.append("- B2 (contig − strided): " + ", ".join(f"N_b {nb} D {d}: {v:+.1f}" for nb, d, v in b2))
    out.append(
        f"- B3 (N_b 1, strided): D1→D2 {z('strided2k_d1_b1', 'strided2k_d2_b1'):+.1f}, D1→D4"
        f" {z('strided2k_d1_b1', 'strided2k_d4_b1'):+.1f}, D2→D4 {z('strided2k_d2_b1', 'strided2k_d4_b1'):+.1f};"
        f" contig: D1→D2 {z('contig2k_d1_b1', 'contig2k_d2_b1'):+.1f}, D2→D4 {z('contig2k_d2_b1', 'contig2k_d4_b1'):+.1f}"
    )
    b4 = [(nb, d, z(f"contig2k_d{d}_b{nb}", f"burst_d{d}_b{nb}")) for nb in (1, 2, 4) for d in (1, 2, 4)]
    out.append(
        "- B4 (burst − contig 2 KB, N_b ≤ 4; burst size 16 / 8 / 4 KB): "
        + ", ".join(f"N_b {nb} D {d}: {v:+.1f}" for nb, d, v in b4)
    )
    if dup_rows:
        dv = [list(r["cores"].values())[0]["rate"] for r in dup_rows]
        mv = per["strided2k_d1_b1"]
        out.append(
            f"- between-process (strided, D 1, N_b 1): main {statistics.mean(mv):.3f} vs second process"
            f" {statistics.mean(dv):.3f} GB/s; difference {statistics.mean(dv) - statistics.mean(mv):+.3f}"
            f" = {(statistics.mean(dv) - statistics.mean(mv)) / sd:+.1f} pooled SD"
        )
    return "\n".join(out)


def p1_ep(rows, dup_rows=None):
    """P1 next limit: per config, each bank's summed per-core rate (fat-K 8 x 4: core "x,y" reads bank x) and the
    op-time proxy, mean (SD) over repeats; pooled SD of the bank-0 sum. With a second process, its bank-0 sums."""
    by = defaultdict(list)
    for r in rows:
        sums = defaultdict(float)
        for k, c in r["cores"].items():
            sums[int(k.split(",")[0])] += c["rate"]
        by[r["name"]].append((dict(sums), r["op_us"]))
    sd0 = pooled_sd([[s[0] for s, _ in v if 0 in s] for v in by.values()])
    out = [
        f"Pooled SD of the bank-0 sum: {sd0:.2f} GB/s.",
        "",
        "| config | bank sums GB/s, banks 0–7, mean (SD) | op µs |",
        "|---|---|---|",
    ]
    for name, v in by.items():
        banks = sorted({b for s, _ in v for b in s})
        cells = []
        for b in banks:
            x = [s[b] for s, _ in v]
            cells.append(f"{b}: {statistics.mean(x):.1f} ({statistics.stdev(x):.2f})")
        ops = [o for _, o in v]
        out.append(f"| {name} | {'; '.join(cells)} | {statistics.mean(ops):.1f} ({statistics.stdev(ops):.2f}) |")
    if dup_rows:
        d = defaultdict(list)
        for r in dup_rows:
            d[r["name"]].append(sum(c["rate"] for k, c in r["cores"].items() if k.split(",")[0] == "0"))
        out += [
            "",
            "Second process, bank-0 sum mean: " + ", ".join(f"{k} {statistics.mean(v):.1f}" for k, v in d.items()),
        ]
    return "\n".join(out)


def e4pc(rows):
    """E4 positive control: per config, per-core rate of A (logical 10,7) and B (9,7), their sum and A - B, mean (SD)
    over repeats; NOC_CTRL's priority field (bits 27-30) as read back after the stream."""
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r["cores"])
    out = ["| config | A GB/s | B GB/s | sum | A − B | priority field A / B |", "|---|---|---|---|---|---|"]
    ms = lambda v: f"{statistics.mean(v):.2f} ({statistics.stdev(v):.2f})"  # noqa: E731
    for name, reps in by.items():
        a = [c["10,7"]["rate"] for c in reps if "10,7" in c]
        b = [c["9,7"]["rate"] for c in reps if "9,7" in c]
        pr = lambda k: sorted({(c[k]["noc_ctrl"] >> 27) & 0xF for c in reps if k in c})  # noqa: E731
        row = [name, ms(a) if a else "—", ms(b) if b else "—"]
        row += [ms([x + y for x, y in zip(a, b)]), ms([x - y for x, y in zip(a, b)])] if a and b else ["—", "—"]
        row += [f"{pr('10,7')} / {pr('9,7')}"]
        out.append("| " + " | ".join(row) + " |")
    return "\n".join(out)


def e3(rows, e1_rows, dup_rows=None):
    """Plan v3 E3: per config, mean per-core rate, bank aggregate (bytes / op-time proxy), max / min, CV and
    rho(request hops, rate) per repeat (rho at k >= 8), the no-sharing control's CV (E1 v3 alone rates of the same
    cores and bank, strided only; another process), the fastest core's physical x, per-class means (endpoint row vs
    other rows) and the largest H4 response-link load from the mean rates. Pooled SDs of CV and rho over configs."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from noc_routes import ENDPOINTS_NOC0, hops, response_route

    phys = lambda x, y: (x + 1 if x < 6 else x + 5, y + 2)  # noqa: E731
    alone = defaultdict(list)
    for r in e1_rows:
        if r["name"].startswith("fat_"):
            _, x, y, b = r["name"].split("_")
            alone[(int(x), int(y), int(b[1:]))].append(next(iter(r["cores"].values()))["rate"])
    cv = lambda v: statistics.stdev(v) / statistics.mean(v)  # noqa: E731
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r)
    res = {}
    for name, reps in by.items():
        b = int(name.split("_")[0][1:])
        keys = list(reps[0]["cores"])
        xy = {k: phys(*map(int, k.split(","))) for k in keys}
        rates = {k: [r["cores"][k]["rate"] for r in reps] for k in keys}
        mean = {k: statistics.mean(v) for k, v in rates.items()}
        hop = {k: hops(xy[k], ENDPOINTS_NOC0[b])[0] for k in keys}
        per_cv = [cv([r["cores"][k]["rate"] for k in keys]) for r in reps] if len(keys) > 1 else []
        per_rho = (
            [spearman([hop[k] for k in keys], [r["cores"][k]["rate"] for k in keys]) for r in reps]
            if len(keys) >= 8 and "_own_" not in name
            else []
        )
        agg = [sum(c["issued"] * c["req_bytes"] for c in r["cores"].values()) / (r["op_us"] * 1e3) for r in reps]
        ctrl, ctrl_mean = None, None
        if "_str_" in name and "_own_" not in name:
            al = [alone.get((*map(int, k.split(",")), b)) for k in keys]
            if all(al):
                ctrl_mean = statistics.mean(statistics.mean(a) for a in al)
                if len(keys) > 1:
                    ctrl = [cv([a[j] for a in al]) for j in range(min(len(a) for a in al))]
        load = defaultdict(float)
        for k in keys if "_own_" not in name else []:
            for link in response_route(xy[k], ENDPOINTS_NOC0[b]):
                load[link] += mean[k]
        ey = ENDPOINTS_NOC0[b][1]
        cls = {c: [mean[k] for k in keys if (xy[k][1] == ey) == (c == "row")] for c in ("row", "other")}
        res[name] = dict(
            k=len(keys),
            mean=statistics.mean(mean.values()),
            agg=statistics.mean(agg),
            maxmin=max(mean.values()) / min(mean.values()),
            cv=per_cv,
            rho=per_rho,
            ctrl=ctrl,
            ctrl_mean=ctrl_mean,
            fastest=xy[max(mean, key=mean.get)],
            cls={c: statistics.mean(v) for c, v in cls.items() if v},
            maxlink=max(load.values()) if load else 0.0,
        )
    sd_cv = pooled_sd([r["cv"] for r in res.values()])
    sd_rho = pooled_sd([r["rho"] for r in res.values()])
    out = [
        f"Pooled SD of a repeat's CV {sd_cv:.4f}; of rho {sd_rho:.3f}. Control = E1 v3 alone rates (another process).",
        "",
        "| config | k | mean GB/s (÷ control mean) | aggregate GB/s | max/min | CV (control) Δ SD | ρ(req hops, rate) | fastest (phys)"
        " | class means row / other | max resp. link GB/s |",
        "|---|--:|--:|--:|--:|---|---|---|---|--:|",
    ]
    for name, r in res.items():
        c = statistics.mean(r["cv"]) if r["cv"] else None
        cvs = "—" if c is None else f"{c:.3f}"
        if c is not None and r["ctrl"]:
            cc = statistics.mean(r["ctrl"])
            cvs += f" ({cc:.3f}) {(c - cc) / sd_cv:+.1f}"
        rho = f"{statistics.mean(r['rho']):+.2f}" if r["rho"] else "—"
        ratio = f" ({r['mean'] / r['ctrl_mean']:.2f})" if r["ctrl_mean"] else ""
        cl = " / ".join(f"{r['cls'][c]:.2f}" if c in r["cls"] else "—" for c in ("row", "other"))
        out.append(
            f"| {name} | {r['k']} | {r['mean']:.2f}{ratio} | {r['agg']:.1f} | {r['maxmin']:.2f} | {cvs} | {rho} |"
            f" {r['fastest']} | {cl} | {r['maxlink']:.1f} |"
        )
    if dup_rows:
        d = defaultdict(list)
        for r in dup_rows:
            d[r["name"]].append(r)
        for name, reps in d.items():
            m = [statistics.mean(c["rate"] for c in r["cores"].values()) for r in reps]
            cvd = [cv([c["rate"] for c in r["cores"].values()]) for r in reps]
            out.append(
                f"\nSecond process, {name}: mean {statistics.mean(m):.2f} GB/s (first {res[name]['mean']:.2f}),"
                f" CV {statistics.mean(cvd):.3f} (first {statistics.mean(res[name]['cv']):.3f})"
            )
    return "\n".join(out)


def e3_r10(rows, e1_rows):
    """E3 re-run: the E3 table for the shared-bank configs, then each config's CV against the same-process alone
    control (the cores' alone rates in this run, per repeat)."""
    shared = [r for r in rows if not r["name"].startswith("alone_")]
    alone = defaultdict(list)
    for r in rows:
        if r["name"].startswith("alone_"):
            _, b, px, py = r["name"].split("_")
            alone[(int(b[1:]), int(px), int(py))].append(next(iter(r["cores"].values()))["rate"])
    cv = lambda v: statistics.stdev(v) / statistics.mean(v)  # noqa: E731
    phys = lambda x, y: (x + 1 if x < 6 else x + 5, y + 2)  # noqa: E731
    by = defaultdict(list)
    for r in shared:
        by[r["name"]].append(r)
    groups, lines = [], []
    for name, reps in by.items():
        b = int(name.split("_")[0][1:])
        keys = list(reps[0]["cores"])
        groups.append([cv([r["cores"][k]["rate"] for k in keys]) for r in reps])
        al = [alone[(b, *phys(*map(int, k.split(","))))] for k in keys]
        ctrl = [cv([a[j] for a in al]) for j in range(min(map(len, al)))]
        lines.append((name, statistics.mean(groups[-1]), statistics.mean(ctrl), ctrl))
    sd = pooled_sd(groups + [c for *_, c in lines])
    out = [e3(shared, e1_rows), "", f"Same-process alone control (pooled SD of a repeat's CV {sd:.4f}):", ""]
    out += ["| config | CV | same-process alone CV | Δ SD |", "|---|--:|--:|--:|"]
    out += [f"| {n} | {c:.4f} | {a:.4f} | {(c - a) / sd:+.1f} |" for n, c, a, _ in lines]
    return "\n".join(out)


def e7(rows):
    """Plan v3 E7: V's (logical 0,0) rate per config, mean (SD), and its change from V alone in pooled SDs of V's rate;
    the aggressors' summed rate and mean block latency (issue -> barrier return)."""
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r["cores"])
    vr = {n: [c["0,0"]["rate"] for c in reps] for n, reps in by.items() if "0,0" in reps[0]}
    sd = pooled_sd(list(vr.values()))
    base = statistics.mean(vr["V_alone"])
    out = [
        f"Pooled SD of V's rate {sd:.3f} GB/s; V alone {base:.2f}.",
        "",
        "| config | V GB/s (SD) | Δ vs V alone, SD | aggressors' sum GB/s | aggressors' block latency µs |",
        "|---|---|--:|--:|--:|",
    ]
    for name, reps in by.items():
        agg = [{k: c for k, c in r.items() if k != "0,0"} for r in reps]
        s = [sum(c["rate"] for c in a.values()) for a in agg] if agg[0] else None
        lat = [statistics.mean(block_latency_us(c) for c in a.values()) for a in agg] if agg[0] else None
        v = vr.get(name)
        out.append(
            f"| {name} | {f'{statistics.mean(v):.2f} ({statistics.stdev(v):.2f})' if v else '—'} |"
            f" {f'{(statistics.mean(v) - base) / sd:+.1f}' if v and name != 'V_alone' else '—'} |"
            f" {f'{statistics.mean(s):.1f}' if s else '—'} |"
            f" {f'{statistics.mean(lat):.3f} ({statistics.stdev(lat):.3f})' if lat else '—'} |"
        )
    return "\n".join(out)


def e8(rows):
    """E8 redesign: V's (logical 0,0) rate per config, mean (SD); per (generator, aggressors) the per-bank arm's rate
    minus the barrier arm's in pooled SDs of V's rate; aggressors' summed rate."""
    by = defaultdict(list)
    for r in rows:
        by[r["name"]].append(r["cores"])
    vr = {n: [c["0,0"]["rate"] for c in reps] for n, reps in by.items()}
    sd = pooled_sd(list(vr.values()))
    out = [
        f"Pooled SD of V's rate {sd:.3f} GB/s.",
        "",
        "| config | V GB/s (SD) | aggressors' sum GB/s |",
        "|---|---|--:|",
    ]
    for name, reps in by.items():
        agg = [sum(c["rate"] for k, c in r.items() if k != "0,0") for r in reps]
        v = vr[name]
        out.append(f"| {name} | {statistics.mean(v):.2f} ({statistics.stdev(v):.2f}) | {statistics.mean(agg):.1f} |")
    out.append("")
    for gen in ("on", "off"):
        for agg in ("none", "static", "rot"):
            b, p = vr.get(f"{gen}_{agg}_barrier"), vr.get(f"{gen}_{agg}_perbank")
            if b and p:
                d = statistics.mean(p) - statistics.mean(b)
                out.append(f"generator {gen}, aggressors {agg}: per-bank − barrier {d:+.3f} GB/s ({d / sd:+.1f} SD)")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dir", type=Path)
    p.add_argument("--group", default="gate")
    a = p.parse_args()
    rows = json.loads((a.dir / f"{a.group}.json").read_text())
    if a.group == "e2v3":
        dup = a.dir / "e2v3_dup.json"
        text = e2v3(rows, json.loads(dup.read_text()) if dup.exists() else None)
    elif a.group == "e3":
        dup = a.dir / "e3_dup.json"
        text = e3(
            rows, json.loads((a.dir / "e1v3.json").read_text()), json.loads(dup.read_text()) if dup.exists() else None
        )
    elif a.group == "e3_r10":
        text = e3_r10(rows, json.loads((a.dir / "e1v3.json").read_text()))
    elif a.group == "p1_ep":
        dup = a.dir / "p1_ep_dup" / "p1_ep.json"
        text = p1_ep(rows, json.loads(dup.read_text()) if dup.exists() else None)
    else:
        text = {
            "gate": gate,
            "gate_ls": gate,
            "e1": e1,
            "e2": e2,
            "gate2": gate2,
            "e1v3": e1v3,
            "p1_ref": p1_ep,
            "p1_tab": p1_ep,
            "e4pc": e4pc,
            "e7": e7,
            "e8": e8,
        }[a.group](rows)
    (a.dir / f"{a.group}_summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
