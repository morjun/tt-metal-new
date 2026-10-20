#!/usr/bin/env python3
"""Value density of one pinned drafter weight (gemma4-12b record §9.2).

Reads a run_pin_reload_standalone.py campaign (results.json) whose arms are the A0 baseline (B+logitsl1), the
same plus one "+pin<weight>[@<layer>][p<per_core_N>][r]" instance, and head-pin sizes R<cols>_…+logitsl1. Round
0 is a warm-up and is dropped. Per arm: the median over the remaining rounds of (arm − A0 in the same round),
in µs per 3-step draft, and from it

  Δ/call      ÷ 3 (one pinned instance runs once per step)
  Δ/MB        ÷ the instance's bytes
  ρ           ÷ its per-bank L1 charge in KiB (LOCKSTEP charges every bank: K·32·per_core_N·2)

Head arms use their pinned column bytes (cols · 2,048 / 110 per bank).

    python analyze_value_density.py <campaign_dir> [--baseline B+logitsl1] -> <campaign_dir>/value_density.{json,md}
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# pin name (and layer) -> (K, N)
SHAPES = {
    ("wqkv", "0"): (1024, 8192),
    ("wqkv", "3"): (1024, 9216),
    ("gate_proj", None): (1024, 8192),
    ("up_proj", None): (1024, 8192),
    ("down_proj", None): (8192, 1024),
    ("o_proj", "0"): (4096, 1024),
    ("o_proj", "1"): (4096, 1024),
    ("o_proj", "2"): (4096, 1024),
    ("o_proj", "3"): (8192, 1024),
    ("pre_projection", None): (7680, 1024),
    ("post_projection", None): (1024, 3840),
}
TUNER_CORES = {32: 32, 120: 40, 256: 64, 288: 48}  # Nt -> cores of the tuner's 8x8-capped grid


def describe(arm):
    m = re.search(r"\+pin([a-z_]+)(?:@([\d-]+))?(?:p(\d+))?(r)?$", arm)
    if m:
        name, layer, pcn, ring = m.groups()
        k, n = SHAPES.get((name, layer)) or SHAPES[(name, None)]
        nt = n // 32
        pcn = int(pcn) if pcn else nt // TUNER_CORES[nt]
        cores = -(-nt // pcn)
        return {
            "kind": "pin",
            "weight": name + (f" L{layer}" if layer else ""),
            "K": k,
            "N": n,
            "cores": cores,
            "per_core_N": pcn,
            "delivery": "shard_ring" if ring else "shard_mcast",
            "mb": k * n * 2 / 1e6,
            "bank_bytes": k * 32 * pcn * 2,
            "calls": 3,
        }
    m = re.match(r"R(\d+)_", arm)
    if m:
        cols = int(m.group(1))
        return {
            "kind": "head",
            "weight": f"lm_head {cols:,} cols",
            "cores": 110,
            "delivery": "shard_mcast + reload",
            "mb": cols * 1024 * 2 / 1e6,
            "bank_bytes": cols * 2048 // 110,
            "calls": 3,
        }
    return {"kind": "baseline", "weight": arm}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("--baseline", default="B+logitsl1")
    args = p.parse_args()
    results = json.loads((args.campaign / "results.json").read_text())
    arms = list(dict.fromkeys(r["arm"] for r in results))
    n = len(arms)
    rounds = defaultdict(dict)
    for r in results:
        rounds[r["index"] // n][r["arm"]] = r
    kept = [rd for k, rd in sorted(rounds.items()) if k > 0 and len(rd) == n]
    base_out = {(r["token"], r["hidden_sha"]) for rd in kept for a, r in rd.items() if a == args.baseline}
    assert len(base_out) == 1, f"baseline output differs across rounds: {base_out}"
    base_out = base_out.pop()

    rows = []
    for arm in arms:
        d = describe(arm)
        ms = [rd[arm]["fused_ms_per_iter"] for rd in kept]
        diff = [1e3 * (rd[arm]["fused_ms_per_iter"] - rd[args.baseline]["fused_ms_per_iter"]) for rd in kept]
        outs = {(rd[arm]["token"], rd[arm]["hidden_sha"]) for rd in kept}
        row = {
            "arm": arm,
            **d,
            "ms_median": statistics.median(ms),
            "delta_us": statistics.median(diff),
            "delta_us_rounds": diff,
            "same_sign": sum((x < 0) == (statistics.median(diff) < 0) for x in diff),
            "rounds": len(diff),
            "output": "same as A0" if outs == {base_out} else ("differs: " + ", ".join(f"{t}/{h}" for t, h in outs)),
        }
        if d["kind"] != "baseline":
            row["delta_us_per_call"] = row["delta_us"] / d["calls"]
            row["delta_us_per_mb"] = row["delta_us"] / d["mb"]
            row["rho_us_per_kib_bank"] = row["delta_us"] / (d["bank_bytes"] / 1024)
        rows.append(row)
    (args.campaign / "value_density.json").write_text(json.dumps(rows, indent=1) + "\n")

    md = [
        f"# Value density: {args.campaign.name}",
        "",
        f"{len(kept)} paired rounds (round 0 dropped); Δ = arm − {args.baseline} in the same round, median; "
        f"µs per 3-step draft.",
        "",
        "| arm | weight | delivery | cores / pcN | MB | B/bank | ms (median) | Δ µs/iter | rounds same sign | Δ µs/call "
        "| Δ µs/MB | ρ µs per KiB/bank | output |",
        "|---|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|",
    ]
    for r in rows:
        if r["kind"] == "baseline":
            md.append(f"| {r['arm']} | (A0) | | | | | {r['ms_median']:.3f} | | | | | | reference |")
            continue
        md.append(
            f"| {r['arm']} | {r['weight']} | {r['delivery']} | {r['cores']} / {r.get('per_core_N', '')} | "
            f"{r['mb']:.2f} | {r['bank_bytes']:,} | {r['ms_median']:.3f} | {r['delta_us']:+.1f} | "
            f"{r['same_sign']}/{r['rounds']} | {r['delta_us_per_call']:+.1f} | {r['delta_us_per_mb']:+.2f} | "
            f"{r['rho_us_per_kib_bank']:+.3f} | {r['output']} |"
        )
    (args.campaign / "value_density.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
