#!/usr/bin/env python3
"""P3 (gemma4-12b record §9.2e): per-block service order and spread of the whole lm_head from NIU samples.

For each call of `lm_head (whole, DRAM)` and each sampled block j (per-block samples, decimated to <= 100 per op), the
completion times of block j across the cores: their spread (max - min) and Spearman rho(physical y, completion). If
the order repeats every K-step while the spread stays bounded, the cores progress together (synchronized) and the
order costs no throughput. Also the whole-op per-core delivered rate (max / min, CV).

    python research_codes/mm_profiling/analyze_p3_blocks.py <niusample time cell> [...]
"""

import bisect
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resource_profile import label_session, load  # noqa: E402
from item6.analyze_niu_samples import LAYER_TYPES, MHZ, WORD_BYTES, read_samples  # noqa: E402


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r

    x, y = rank(a), rank(b)
    mx, my = statistics.mean(x), statistics.mean(y)
    num = sum((p - mx) * (q - my) for p, q in zip(x, y))
    den = (sum((p - mx) ** 2 for p in x) * sum((q - my) ** 2 for q in y)) ** 0.5
    return num / den


def analyze(cell):
    samples = read_samples(cell / "niu_samples.csv")
    ops = []
    for s in load(cell / "ops_perf.csv").values():
        for o, (_, name, _) in zip(s, label_session(s, LAYER_TYPES)):
            if o["op"] == "Matmul":
                ops.append((o["fw_start"], o["fw_end"], name))
    ops.sort()
    starts = [o[0] for o in ops]
    per_call = defaultdict(lambda: defaultdict(list))
    for t, core, w in samples:
        i = bisect.bisect_right(starts, t + 2 * MHZ) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] and ops[i][2].startswith("lm_head (whole"):
            per_call[i][core].append((t, w))
    spreads, rhos = defaultdict(list), defaultdict(list)
    rr, cvs, op_us = [], [], []
    nsamp = None
    for cores in per_call.values():
        n = min(len(s) for s in cores.values())
        nsamp = n
        t0 = min(s[0][0] for s in cores.values())
        ks = list(cores)
        for j in range(1, n):
            ts = [(cores[k][j][0] - t0) / MHZ for k in ks]
            spreads[j].append(max(ts) - min(ts))
            rhos[j].append(spearman([k[1] for k in ks], ts))
        rates = []
        for s in cores.values():
            words = sum((b - a) % (1 << 32) for (_, a), (_, b) in zip(s, s[1:]))
            rates.append(words * WORD_BYTES / ((s[-1][0] - s[0][0]) / MHZ) / 1e3)
        rr.append(max(rates) / min(rates))
        cvs.append(statistics.stdev(rates) / statistics.mean(rates))
        op_us.append(max((s[-1][0] - t0) / MHZ for s in cores.values()))
    js = sorted(spreads)
    sm = lambda sel: statistics.mean(statistics.mean(spreads[j]) for j in sel)  # noqa: E731
    rm = lambda sel: statistics.mean(statistics.mean(rhos[j]) for j in sel)  # noqa: E731
    op = statistics.mean(op_us)
    kstep = op / 240
    return (
        f"{cell.name}: {len(per_call)} calls, {nsamp} samples per core; op {op:.1f} µs (K-step {kstep:.2f} µs);"
        f" whole-op per-core rate max/min {statistics.mean(rr):.2f}, CV {statistics.mean(cvs):.3f};"
        f" per-block completion spread: first 10 {sm(js[:10]):.1f}, middle 10"
        f" {sm(js[len(js) // 2 - 5 : len(js) // 2 + 5]):.1f}, last 10 {sm(js[-10:]):.1f} µs"
        f" ({sm(js) / kstep:.1f} K-steps); rho(phys y, completion) mean {rm(js):+.2f}"
        f" (first 10 {rm(js[:10]):+.2f}, last 10 {rm(js[-10:]):+.2f})"
    )


if __name__ == "__main__":
    for c in sys.argv[1:]:
        print(analyze(Path(c)))
