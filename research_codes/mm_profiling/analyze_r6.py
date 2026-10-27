#!/usr/bin/env python3
"""R6 analysis (gemma4-12b record §9.2e, plan v3 R6 analysis as frozen after Patch 7): does a gate_proj core's
NIU-delivered rate fall with the response-side load on its route?

Per grid (64 cores: item6_niusample/01_time_B+logitsl1+niusample; W, 86 cores: item6_niusample/00_time_W+niusample):
  - tile -> bank from the production column assignment (core i, row-major, columns i * pcN ...; Nt 256, Kt 32)
  - link loads: primary byte-based (each core's bytes per call on every link of its response routes, plan v3 H4),
    converted to GB/s with the op's median per-core NIU window; secondary rate-weighted (measured per-core rates)
  - per core: bottleneck load (max link load on its response routes), excess load (sum over its response links of
    max(0, load - capacity), capacity 77.6 GB/s = 57.7 B/cycle (tt-npe, 2 KB) x 1.3449 GHz), sum-form L_resp
    (secondary, with rho(sum form, response hops))
  - per core and bank sub-stream: bottleneck; R5 view = max over the core's banks vs mean
  - tests: Spearman rho(metric, rate); R2: rho(mean request hops, rate); excess informativeness (< 25% nonzero ->
    judged on bottleneck alone); R5: rho(max) - rho(mean) with (a) per-core bootstrap, (b) stratified 2 x 2-block
    bootstrap (strata = bank set {0-3} / {4-7} by majority of the core's columns; blocks = adjacent 2 x 2 physical core
    groups, edge blocks kept), 2,000 resamples each, 95% CI; Steiger's test for dependent correlations alongside.

    python research_codes/mm_profiling/analyze_r6.py -> generated/lmhead_l1_profile/r6_analysis.{json,md}
"""

import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_hot_links import niu_rates  # noqa: E402
from noc_routes import ENDPOINTS_NOC0, hops, request_route, response_route  # noqa: E402

G = Path(__file__).resolve().parents[2] / "generated/lmhead_l1_profile"
CELLS = {
    "64 cores": (G / "item6_niusample/01_time_B+logitsl1+niusample", 4),
    "W (86 cores)": (G / "item6_niusample/00_time_B+logitsl1+gridwide+vc1+niusample", 3),
}
CAP = 57.7 * 1.3449  # GB/s per link direction for 2 KB transfers (plan v3 G5)
KT, NT, TILE = 32, 256, 2048
LOG_X = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15]


def grid_cores(pcn):
    """Physical (x, y) of the production row-major cores: tuner 8 x 8 for pcN 4, the 11-wide prefix for pcN 3."""
    n = -(-NT // pcn)
    width = 8 if pcn == 4 else 11
    return [(LOG_X[i % width], 2 + i // width) for i in range(n)]


def spearman(a, b):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(o):  # average ranks for ties
            j = i
            while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = rank(a), rank(b)
    ma, mb = statistics.mean(ra), statistics.mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return num / den if den else float("nan")


def steiger(r12, r13, r23, n):
    """Steiger's Z for two dependent correlations sharing variable 1 (r12 vs r13), with r23 between the others."""
    rm = (r12**2 + r13**2) / 2
    f = min((1 - r23) / (2 * (1 - rm)), 1.0)
    h = (1 - f * rm) / (1 - rm)
    z = (math.atanh(r12) - math.atanh(r13)) * math.sqrt((n - 3) / (2 * (1 - r23) * h))
    p = math.erfc(abs(z) / math.sqrt(2))
    return z, p


def analyze(name, cell, pcn, rng):
    cores = grid_cores(pcn)
    rates_all = niu_rates(cell)  # {(y, x): [GB/s per call]}
    rate = {c: statistics.mean(rates_all[(c[1], c[0])]) for c in cores}
    bytes_bank = {}  # core -> {bank: bytes per call}
    for i, c in enumerate(cores):
        bb = defaultdict(int)
        for col in range(i * pcn, min(NT, (i + 1) * pcn)):
            bb[col % 8] += KT * TILE
        bytes_bank[c] = dict(bb)
    window_us = statistics.median(sum(bytes_bank[c].values()) / (rate[c] * 1e3) for c in cores)

    def link_loads(weight):
        L = defaultdict(float)
        for c in cores:
            for b, by in bytes_bank[c].items():
                for l in response_route(c, ENDPOINTS_NOC0[b]):
                    L[l] += weight(c, b, by)
        return L

    L_byte = link_loads(lambda c, b, by: by / (window_us * 1e3))  # GB/s
    L_rate = link_loads(lambda c, b, by: rate[c] * by / sum(bytes_bank[c].values()))

    def metrics(L):
        out = {}
        for c in cores:
            links = {l for b in bytes_bank[c] for l in response_route(c, ENDPOINTS_NOC0[b])}
            bott = max(L[l] for l in links)
            exc = sum(max(0.0, L[l] - CAP) for l in links)
            summ = sum(L[l] * bytes_bank[c][b] for b in bytes_bank[c] for l in response_route(c, ENDPOINTS_NOC0[b]))
            per_bank = {b: max(L[l] for l in response_route(c, ENDPOINTS_NOC0[b])) for b in bytes_bank[c]}
            out[c] = dict(
                bott=bott, exc=exc, sum=summ, bmax=max(per_bank.values()), bmean=statistics.mean(per_bank.values())
            )
        return out

    m = metrics(L_byte)
    m2 = metrics(L_rate)
    r = [rate[c] for c in cores]
    req = [statistics.mean(hops(c, ENDPOINTS_NOC0[b])[0] for b in bytes_bank[c]) for c in cores]
    resp = [statistics.mean(hops(c, ENDPOINTS_NOC0[b])[1] for b in bytes_bank[c]) for c in cores]
    res = dict(
        cores=len(cores),
        window_us=window_us,
        rho_bott=spearman([m[c]["bott"] for c in cores], r),
        rho_exc=spearman([m[c]["exc"] for c in cores], r),
        frac_exc_nonzero=sum(m[c]["exc"] > 0 for c in cores) / len(cores),
        rho_sum=spearman([m[c]["sum"] for c in cores], r),
        rho_sum_resp_hops=spearman([m[c]["sum"] for c in cores], resp),
        rho_req_hops=spearman(req, r),
        rho_bott_rateweighted=spearman([m2[c]["bott"] for c in cores], r),
        rho_exc_rateweighted=spearman([m2[c]["exc"] for c in cores], r),
        max_link=max(L_byte.values()),
        links_over_cap=sum(v > CAP for v in L_byte.values()),
    )
    # R5: rho(max over banks) - rho(mean over banks), bootstraps
    bmax = {c: m[c]["bmax"] for c in cores}
    bmean = {c: m[c]["bmean"] for c in cores}

    def dstat(sample):
        if len({bmax[c] for c in sample}) < 2 or len({bmean[c] for c in sample}) < 2:
            return None
        rr = [rate[c] for c in sample]
        return spearman([bmax[c] for c in sample], rr) - spearman([bmean[c] for c in sample], rr)

    res["r5_delta"] = dstat(cores)
    per = [dstat([rng.choice(cores) for _ in cores]) for _ in range(2000)]
    per = sorted(v for v in per if v is not None)
    res["r5_ci_per_core"] = (per[int(0.025 * len(per))], per[int(0.975 * len(per)) - 1])
    stratum = {
        c: 0 if sum(by for b, by in bytes_bank[c].items() if b < 4) * 2 >= sum(bytes_bank[c].values()) else 1
        for c in cores
    }
    xpair = {x: i // 2 for i, x in enumerate(LOG_X)}
    blocks = defaultdict(list)  # (stratum, block) -> cores
    for c in cores:
        blocks[(stratum[c], xpair[c[0]], (c[1] - 2) // 2)].append(c)
    by_stratum = defaultdict(list)
    for (s, *_), cs in blocks.items():
        by_stratum[s].append(cs)
    res["blocks_per_stratum"] = {s: len(v) for s, v in by_stratum.items()}
    strat = []
    for _ in range(2000):
        sample = []
        for s, bl in by_stratum.items():
            for _ in bl:
                sample += rng.choice(bl)
        v = dstat(sample)
        if v is not None:
            strat.append(v)
    strat.sort()
    res["r5_ci_stratified_2x2"] = (strat[int(0.025 * len(strat))], strat[int(0.975 * len(strat)) - 1])
    r_max = spearman([bmax[c] for c in cores], r)
    r_mean = spearman([bmean[c] for c in cores], r)
    r_mm = spearman([bmax[c] for c in cores], [bmean[c] for c in cores])
    res["rho_bmax"], res["rho_bmean"] = r_max, r_mean
    res["steiger_z_p"] = steiger(r_max, r_mean, r_mm, len(cores)) if abs(r_mm) < 1 else (float("nan"), float("nan"))
    return res


def main():
    rng = random.Random(0)
    out, lines = {}, ["# R6 analysis: response-side link load vs gate_proj (sliding) per-core NIU-delivered rate", ""]
    for name, (cell, pcn) in CELLS.items():
        r = analyze(name, cell, pcn, rng)
        out[name] = r
        lines += [
            f"## {name}",
            f"- cores {r['cores']}; median per-core window {r['window_us']:.1f} µs; byte-based max link load {r['max_link']:.1f} GB/s;"
            f" links over capacity ({CAP:.1f} GB/s): {r['links_over_cap']}",
            f"- R6: ρ(bottleneck load, rate) {r['rho_bott']:+.2f}; ρ(excess load, rate) {r['rho_exc']:+.2f};"
            f" cores with nonzero excess {100 * r['frac_exc_nonzero']:.0f}%",
            f"  - rate-weighted (secondary): ρ(bottleneck) {r['rho_bott_rateweighted']:+.2f}, ρ(excess) {r['rho_exc_rateweighted']:+.2f}",
            f"  - sum form (secondary): ρ(sum L_resp, rate) {r['rho_sum']:+.2f}; ρ(sum L_resp, response hops) {r['rho_sum_resp_hops']:+.2f}",
            f"- R2: ρ(mean request hops, rate) {r['rho_req_hops']:+.2f}",
            f"- R5: ρ(max-over-banks bottleneck, rate) {r['rho_bmax']:+.2f} vs ρ(mean over banks) {r['rho_bmean']:+.2f};"
            f" Δ {r['r5_delta']:+.3f}; per-core bootstrap 95% CI [{r['r5_ci_per_core'][0]:+.3f}, {r['r5_ci_per_core'][1]:+.3f}];"
            f" stratified 2 × 2-block bootstrap 95% CI [{r['r5_ci_stratified_2x2'][0]:+.3f}, {r['r5_ci_stratified_2x2'][1]:+.3f}]"
            f" (blocks per stratum {r['blocks_per_stratum']}); Steiger z {r['steiger_z_p'][0]:+.2f}, p {r['steiger_z_p'][1]:.3f}",
            "",
        ]
    (G / "r6_analysis.json").write_text(json.dumps(out, indent=1, default=str))
    (G / "r6_analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
