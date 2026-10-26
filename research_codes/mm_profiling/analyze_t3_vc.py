#!/usr/bin/env python3
"""T3 VC sweep analysis (gemma4-12b record §9.2e): per arm, µs per 3-step iteration of each op family and the span,
from every replay of every round (time cells of run_traced_opprof_arms.py), against the base arm.

Families (summed over layers and steps): lm_head (whole, DRAM), thin-K (qkv, gate, up, post), fat-K (pre, o_proj,
down), other (everything else, including gaps via the span). Noise: pooled SD of the per-replay values over all arms
(5 replays x rounds per arm). Effect threshold 3 pooled SD, "no change" within 1 (record §9.2e thresholds).

    python research_codes/mm_profiling/analyze_t3_vc.py generated/lmhead_l1_profile/t3_vc [--base B+logitsl1+gridwide]
        -> <root>/t3_vc.{json,md}
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_resource_profile import label_session, load  # noqa: E402

LAYER_TYPES = ["sliding", "sliding", "sliding", "global"]
THIN = ("qkv_proj", "gate_proj", "up_proj", "post_projection")
FAT = ("pre_projection", "o_proj", "down_proj")


def family(label):
    if label.startswith("lm_head"):
        return "lm_head"
    if label.startswith(THIN):
        return "thin-K"
    if label.startswith(FAT):
        return "fat-K"
    return "other ops"


def replays(cell):
    """[{family: µs, 'span': µs, label: µs}] per replay."""
    out = []
    for ops in load(cell / "ops_perf.csv").values():
        acc = defaultdict(float)
        for o, (_, name, _) in zip(ops, label_session(ops, LAYER_TYPES)):
            acc[family(name)] += o["dur"]
            acc[name] += o["dur"]
        acc["span"] = ops[0]["dur"] + sum(o["dur"] + o["gap"] for o in ops[1:])
        out.append(dict(acc))
    return out


def pooled(groups):
    groups = [g for g in groups if len(g) > 1]
    dof = sum(len(g) - 1 for g in groups)
    return (sum((len(g) - 1) * statistics.variance(g) for g in groups) / dof) ** 0.5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--base", default="B+logitsl1+gridwide")
    a = p.parse_args()
    arms = defaultdict(list)
    for d in sorted(a.root.iterdir()):
        if (d / "ops_perf.csv").exists():
            arms[d.name.split("_", 2)[2]].extend(replays(d))
    keys = ["span", "lm_head", "thin-K", "fat-K", "other ops"]
    labels = sorted({k for r in arms[a.base] for k in r if k not in keys})
    sd = {k: pooled([[r.get(k, 0.0) for r in rs] for rs in arms.values()]) for k in keys + labels}
    mean = {arm: {k: statistics.mean(r.get(k, 0.0) for r in rs) for k in keys + labels} for arm, rs in arms.items()}
    base = mean[a.base]
    lines = [
        f"# T3 VC sweep: µs per 3-step iteration, Δ vs `{a.base}` (Δ in pooled SD of a replay)",
        "",
        "pooled SD per replay: " + ", ".join(f"{k} {sd[k]:.1f}" for k in keys),
        "",
        "| arm | replays | span | " + " | ".join(keys[1:]) + " |",
        "|---|--:|--:|" + "--:|" * (len(keys) - 1),
    ]
    order = sorted(arms, key=lambda arm: mean[arm]["span"])
    for arm in order:
        cells = []
        for k in keys:
            d = mean[arm][k] - base[k]
            cells.append(
                f"{mean[arm][k]:.1f} ({d:+.1f}; {d / sd[k]:+.1f} SD)" if arm != a.base else f"{mean[arm][k]:.1f}"
            )
        lines.append(f"| `{arm.replace(a.base, '') or 'base'}` | {len(arms[arm])} | " + " | ".join(cells) + " |")
    lines += ["", "Per op label (µs per iteration; Δ vs base in pooled SD), arms with any |Δ| ≥ 3 SD:", ""]
    for arm in order:
        if arm == a.base:
            continue
        moved = [
            f"{k} {mean[arm][k] - base[k]:+.1f} ({(mean[arm][k] - base[k]) / sd[k]:+.1f})"
            for k in labels
            if sd[k] > 0 and abs(mean[arm][k] - base[k]) >= 3 * sd[k]
        ]
        if moved:
            lines.append(f"- `{arm.replace(a.base, '')}`: " + "; ".join(moved))
    w = [arm for arm in arms if arm.endswith("+vc1")]
    r4 = [arm for arm in arms if arm.endswith("+vcrow4")]
    if w and r4:
        lines += ["", "Check `+vcrow4` vs `+vc1` (same mapping, must match within 1%):"]
        for k in keys:
            x, y = mean[r4[0]][k], mean[w[0]][k]
            lines.append(f"- {k}: {x:.1f} vs {y:.1f} ({(x - y) / y * 100:+.2f}%)")
    out = {"sd": sd, "mean": mean, "replays": {arm: len(rs) for arm, rs in arms.items()}}
    (a.root / "t3_vc.json").write_text(json.dumps(out, indent=1))
    (a.root / "t3_vc.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
