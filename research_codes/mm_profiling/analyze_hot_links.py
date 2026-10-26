#!/usr/bin/env python3
"""E6b analysis (gemma4-12b record §9.2e): do gate_proj's DRAM reads that cross tt-npe's hot links read slower, and
does the wide grid's gain come from cores that avoid them?

Per cell pair (noc cell for routes, NIU time cell for delivered rates), gate_proj (sliding):
  - hot links: mean modelled demand > 100% over the matmul (plot_dram_read_grid.npe_link_means), and the hottest
  - per core: fraction of its DRAM read bytes whose tt-npe route crosses any hot link / the hottest link
  - per core: NIU-delivered GB/s per call (bytes between its first and last sample / that time; samples assigned to
    calls as item6/analyze_niu_samples.py does), mean and SD over calls
Reports Spearman(rate, crossing fraction) per cell and the crossing / non-crossing cores' rates on 64 vs 86 cores.

    python research_codes/mm_profiling/analyze_hot_links.py -> generated/lmhead_l1_profile/e6b_hot_links.{json,md}
"""

import bisect
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from analyze_resource_profile import label_session, load  # noqa: E402
from item6.analyze_niu_samples import LAYER_TYPES, MHZ, WORD_BYTES, read_samples  # noqa: E402
from plot_dram_read_grid import npe_link_means, npeviz  # noqa: E402

G = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile"
PAIRS = {
    "64 cores": (G / "phase2_vc/08_noc_B+logitsl1", G / "item6_niusample/01_time_B+logitsl1+niusample"),
    "86 cores": (
        G / "phase2_vc/10_noc_B+logitsl1+gridwide",
        G / "c1_iw_vc_niu/00_time_B+logitsl1+gridwide+headiw2+niusample",
    ),
}
LABEL = "gate_proj (sliding)"


def crossing(noc_cell):
    d = npeviz(noc_cell, LABEL, 0)
    mean, _, hot, *_ = npe_link_means(d)
    hot_set = {k for k, v in mean.items() if v > 100}
    reads = [t for t in d["noc_transfers"] if t["noc_event_type"] == "READ"]
    dram = {tuple(t["src"]) for t in reads} - {tuple(x) for t in reads for x in t["dst"]}
    tot, cross, cross_hot = defaultdict(int), defaultdict(int), defaultdict(int)
    for t in reads:
        if tuple(t["src"]) not in dram:
            continue
        core = tuple(t["dst"][0][1:])  # (row, col)
        links = {(r, c, k) for rt in t["route"] for _, r, c, k in rt["links"]}
        tot[core] += t["total_bytes"]
        if links & hot_set:
            cross[core] += t["total_bytes"]
        if hot in links:
            cross_hot[core] += t["total_bytes"]
    frac = {c: cross[c] / tot[c] for c in tot}
    frac_hot = {c: cross_hot[c] / tot[c] for c in tot}
    return frac, frac_hot, hot, mean[hot], len(hot_set)


def niu_rates(time_cell):
    """{(row, col): [GB/s per call]} for LABEL."""
    samples = read_samples(time_cell / "niu_samples.csv")
    ops = []
    for session in load(time_cell / "ops_perf.csv").values():
        for o, (_, name, _) in zip(session, label_session(session, LAYER_TYPES)):
            if o["op"] == "Matmul":
                ops.append((o["fw_start"], o["fw_end"], name))
    ops.sort()
    starts = [o[0] for o in ops]
    per_call = defaultdict(lambda: defaultdict(list))
    tol = 2.0 * MHZ
    for t, core, words in samples:
        i = bisect.bisect_right(starts, t + tol) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] and ops[i][2] == LABEL:
            per_call[i][core].append((t, words))
    rates = defaultdict(list)
    for cores in per_call.values():
        for (x, y), s in cores.items():
            if len(s) < 2:
                continue
            words = sum((b - a) % (1 << 32) for (_, a), (_, b) in zip(s, s[1:]))
            dt = (s[-1][0] - s[0][0]) / MHZ
            if dt > 0:
                rates[(y, x)].append(words * WORD_BYTES / dt / 1e3)
    return rates


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    res, lines = {}, ["# E6b: hot-link crossing vs NIU-delivered rate, gate_proj (sliding)", ""]
    for name, (noc_cell, time_cell) in PAIRS.items():
        frac, frac_hot, hot, hot_v, n_hot = crossing(noc_cell)
        rates = niu_rates(time_cell)
        cores = sorted(set(frac) & set(rates))
        mr = {c: statistics.mean(rates[c]) for c in cores}
        sd = statistics.mean(statistics.stdev(rates[c]) for c in cores if len(rates[c]) > 1)
        rho = spearman([frac[c] for c in cores], [mr[c] for c in cores])
        rho_hot = spearman([frac_hot[c] for c in cores], [mr[c] for c in cores])
        res[name] = dict(
            hot=list(hot),
            hot_pct=hot_v,
            n_hot=n_hot,
            cores=len(cores),
            calls=min(len(rates[c]) for c in cores),
            sd_over_calls=sd,
            rho_rate_frac=rho,
            rho_rate_frac_hottest=rho_hot,
            per_core={f"{r},{c}": dict(frac=frac[(r, c)], frac_hot=frac_hot[(r, c)], gbs=mr[(r, c)]) for r, c in cores},
            unmatched_npe=len(set(frac) - set(rates)),
            unmatched_niu=len(set(rates) - set(frac)),
        )
        lines += [
            f"## {name}",
            f"- hottest link {hot} at {hot_v:.0f}%; {n_hot} links > 100%; {len(cores)} cores matched"
            f" ({res[name]['unmatched_npe']} npe-only, {res[name]['unmatched_niu']} NIU-only); calls ≥ {res[name]['calls']}",
            f"- mean per-core SD over calls {sd:.3f} GB/s",
            f"- Spearman(rate, fraction crossing any hot link) {rho:+.2f}; (rate, fraction crossing the hottest) {rho_hot:+.2f}",
        ]
        for lo, hi, tag in ((0.5, 1.01, "crossing (≥ 0.5)"), (0.0, 0.5, "non-crossing (< 0.5)")):
            sel = [c for c in cores if lo <= frac[c] < hi]
            if sel:
                tot = sum(mr[c] for c in sel)
                lines.append(
                    f"- {tag}: {len(sel)} cores, mean {statistics.mean(mr[c] for c in sel):.2f} GB/s, sum {tot:.1f} GB/s"
                )
        lines.append("")
    a, b = res["64 cores"], res["86 cores"]
    common = sorted(set(a["per_core"]) & set(b["per_core"]))
    lines.append("## Cores present in both grids")
    for lo, hi, tag in ((0.5, 1.01, "crossing on 64 cores"), (0.0, 0.5, "non-crossing on 64 cores")):
        sel = [k for k in common if lo <= a["per_core"][k]["frac"] < hi]
        if sel:
            r64 = statistics.mean(a["per_core"][k]["gbs"] for k in sel)
            r86 = statistics.mean(b["per_core"][k]["gbs"] for k in sel)
            pooled = (a["sd_over_calls"] ** 2 / 2 + b["sd_over_calls"] ** 2 / 2) ** 0.5
            lines.append(
                f"- {tag}: {len(sel)} cores, {r64:.2f} → {r86:.2f} GB/s ({(r86 - r64) / pooled:+.1f} pooled SD of a core's"
                f" per-call rate, {pooled:.3f})"
            )
    new = sorted(set(b["per_core"]) - set(a["per_core"]))
    if new:
        lines.append(
            f"- cores only on 86: {len(new)}, mean {statistics.mean(b['per_core'][k]['gbs'] for k in new):.2f} GB/s,"
            f" sum {sum(b['per_core'][k]['gbs'] for k in new):.1f}; mean crossing fraction"
            f" {statistics.mean(b['per_core'][k]['frac'] for k in new):.2f}"
        )
    (G / "e6b_hot_links.json").write_text(json.dumps(res, indent=1))
    (G / "e6b_hot_links.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
