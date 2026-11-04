#!/usr/bin/env python3
"""Knob values of a drafter arm, for the knob tables of gemma4-12b record §9.2a / §9.2b (plan v2, T1).

An arm name (run_pin_reload_standalone / run_traced_opprof_arms grammar) is turned into its env by
run_pin_reload_standalone.arm_spec and run_lmhead_obw_e2e.env_for; the env is then turned into the config every
matmul class gets, by the same rules the model applies. Those rules live in modules that import ttnn, so they are
mirrored here and checked against the runners' logs (`check` mode):
  - tuner grid / in0_block_w: matmul_tuning._pick_grid, _largest_divisor, _block_cap (default cap 8,
    GEMMA4_MM_BLOCK_CAP_SHAPES), _grid_override (GEMMA4_MM_GRID_SHAPES); out_block_w = per_core_N, out_subblock_w 1
  - DRAM-sharded fat-K: DecodeMatmulTuner._dram_sharded_linear (GEMMA4_DRAM_SHARD)
  - lm_head: lm_head_out_block_w_config (GEMMA4_LMHEAD_DRAFT_OBW / _PCN / _IW), the L1-slice config in
    tt/assistant/model.py (GEMMA4_LMHEAD_L1_COLS / _OBW / _OSW / _IW / _FIDELITY / _GATHER)
  - request VC (TT_MM1D_IN1_VC_MODE), NoC swap (TT_MM1D_MCAST_IN0_SWAP_NOCS), NIU sampler (TT_MM1D_IN1_NIU_SAMPLE)

    python arm_knobs.py table <arm> [<arm> ...] [--mode M] [--baseline B]   -> markdown knob table
    python arm_knobs.py check <cell_dir> [...]                              -> compare with the cell's run.log
"""

import argparse
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_lmhead_obw_e2e import env_for  # noqa: E402
from run_pin_reload_standalone import arm_spec  # noqa: E402

T, CORES, BANKS = 32, 110, 8
SHAPES = {
    "pre_projection": (7680, 1024),
    "qkv_proj (sliding)": (1024, 8192),
    "qkv_proj (global)": (1024, 9216),
    "o_proj (sliding)": (4096, 1024),
    "o_proj (global)": (8192, 1024),
    "gate_proj / up_proj": (1024, 8192),
    "down_proj": (8192, 1024),
    "post_projection": (1024, 3840),
}
FATK = ("pre_projection", "o_proj (sliding)", "o_proj (global)", "down_proj")
VC_MODES = {"0": "default (VC 1 for all)", "1": "row mod 4", "2": "column mod 4", "3": "(row + column) mod 4"}


def vc_mode_name(mode):
    """TT_MM1D_IN1_VC_MODE -> words; >= 16 is the T3 sweep encoding (kind * 16 + k, the in1 reader's MM_IN1_VC_MODE)."""
    if mode in VC_MODES:
        return VC_MODES[mode]
    kind, k = divmod(int(mode), 16)
    return {
        1: f"row mod {k}",
        2: f"column mod {k}",
        3: f"(row + column) mod {k}",
        4: f"first-column bank mod {k}",
        5: f"VC {k} for all",
    }[kind]


def pick_grid(nt, max_x=8, max_y=8):
    best = (1, 1)
    for gy in range(1, max_y + 1):
        for gx in range(1, max_x + 1):
            if nt % (gx * gy) == 0 and gx * gy > best[0] * best[1]:
                best = (gx, gy)
    return best


def block_cap(env, k, n):
    for entry in env.get("GEMMA4_MM_BLOCK_CAP_SHAPES", "").split(";"):
        if entry.strip() and entry.split("=")[0] == f"{k}x{n}":
            return int(entry.split("=")[1])
    return int(env.get("GEMMA4_MM_BLOCK_CAP", "8"))


def largest_divisor(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def grid_override(env, k, n):
    for entry in env.get("GEMMA4_MM_GRID_SHAPES", "").split(";"):
        if entry.strip() and entry.split("=")[0] == f"{k}x{n}":
            return int(entry.split("=")[1])
    return None


def pins(env):
    """{weight substring: (layers, per_core_N or None)} from GEMMA4_L1_PIN."""
    out = {}
    for entry in env.get("GEMMA4_L1_PIN", "").split(";"):
        if entry.strip():
            head, _, opts = entry.partition(":")
            pcn = opts.split(":")[0]  # per_core_N, then options such as "pc", "y<row>", "s"
            name, _, layers = head.partition("@")
            out[name] = (layers or "all", int(pcn) if pcn else None)
    return out


def backbone_cfg(env, name):
    """(delivery, cores, per_core_N, in0_block_w, grid source) of one backbone matmul."""
    k, n = SHAPES[name]
    kt, nt = k // T, n // T
    iw = largest_divisor(kt, block_cap(env, k, n))
    shard = [s for s in env.get("GEMMA4_DRAM_SHARD", "").split(",") if s]
    if name in FATK and any(s in name for s in shard):
        cores = max(
            c for c in range(1, 33) if (c <= 8 or c % 8 == 0) and kt % c == 0 and nt % c == 0 and (kt // c) % iw == 0
        )
        return ("DRAM-sharded", cores, nt // cores, iw, "in0 shard cores")
    pcn = grid_override(env, k, n)
    if pcn:
        cores, how = -(-nt // pcn), "wide"
    else:
        gx, gy = pick_grid(nt)
        cores, pcn, how = gx * gy, nt // (gx * gy), "tuner"
    ring = name in FATK and env.get("GEMMA4_GATHER_DRAM_WEIGHT") == "1" and name != "pre_projection"
    return ("gather_in0 ring" if ring else "1D multicast", cores, pcn, iw, how)


def backbone(env, names):
    """Compact text for a group of backbone matmuls: shared fields once, differing ones joined by ' / '."""
    cfgs = [backbone_cfg(env, n) for n in names]

    def field(i, fmt):
        vals = [fmt(c) for c in cfgs]
        return vals[0] if len(set(vals)) == 1 else " / ".join(vals)

    if all(c[0] == "DRAM-sharded" for c in cfgs):
        return f"DRAM-sharded (8 bank workers; in0 multicast from {field(1, lambda c: str(c[1]))} shard cores), iw{field(3, lambda c: str(c[3]))}"
    return (
        f"{field(1, lambda c: str(c[1]))} cores ({field(4, lambda c: c[4])}), pcN {field(2, lambda c: str(c[2]))},"
        f" iw{field(3, lambda c: str(c[3]))}, {field(0, lambda c: c[0])}"
    )


def lm_head(env):
    cols = int(env.get("GEMMA4_LMHEAD_L1_COLS", "0") or 0)
    iw = env.get("GEMMA4_LMHEAD_DRAFT_IW", "2")
    obw = env.get("GEMMA4_LMHEAD_DRAFT_OBW")
    rest_nt = (262144 - cols) // T
    pcn = env.get("GEMMA4_LMHEAD_DRAFT_PCN") or str(math.ceil(math.ceil(rest_nt / CORES) / 1))
    dram = f"pcN {pcn} / ow{obw} / iw{iw}" if obw else "ttnn automatic config (iw2)"
    if not cols:
        return f"whole head in DRAM: {dram}", "—"
    sl_pcn = cols // T // CORES
    sl = (
        f"{cols:,} cols L1 (pcN {sl_pcn}, ow{env.get('GEMMA4_LMHEAD_L1_OBW', sl_pcn)}, osw{env.get('GEMMA4_LMHEAD_L1_OSW', 1)},"
        f" iw{env.get('GEMMA4_LMHEAD_L1_IW', 2)}, {env.get('GEMMA4_LMHEAD_L1_FIDELITY', 'HiFi2')},"
        f" {'gather_in0 ring' if env.get('GEMMA4_LMHEAD_L1_GATHER') == '1' else '1D multicast'}) + DRAM tail {dram}"
    )
    if env.get("GEMMA4_LMHEAD_L1_RELOAD") != "1":
        return sl, "none"
    how = "DRISC" if env.get("GEMMA4_LMHEAD_L1_RELOAD_DRISC") == "1" else "Tensix"
    pace = env.get("GEMMA4_LMHEAD_L1_RELOAD_DRISC_GBPS")
    chunk = env.get("GEMMA4_LMHEAD_L1_RELOAD_DRISC_CHUNK")
    rope = " + rotary bracket" if env.get("GEMMA4_LMHEAD_L1_RELOAD_ROPE") == "1" else ""
    return sl, f"{how}{f' {pace} GB/s' if pace else ''}{f' {chunk}-page chunks' if chunk else ''}{rope}"


def knobs(arm):
    base, extra, _ = arm_spec(arm)
    env = {**{k: v for k, v in env_for(base).items() if k.startswith(("GEMMA4_", "TT_MM1D_"))}, **extra}
    head, reload = lm_head(env)
    p = pins(env)
    pin_txt = (
        "; ".join(
            f"{w}@{lay}"
            + (f" pcN {pcn}" if pcn else " (tuner grid)")
            + (" ring" if env.get("GEMMA4_GATHER_IN0") == "1" else "")
            for w, (lay, pcn) in p.items()
        )
        or "none"
    )
    return {
        "arm": arm,
        "thin-K: qkv sl, gate, up / qkv gl / post": backbone(
            env, ("qkv_proj (sliding)", "qkv_proj (global)", "post_projection")
        ),
        "fat-K: o_proj sl / down, o_proj gl / pre": backbone(env, ("o_proj (sliding)", "down_proj", "pre_projection")),
        "backbone L1 pins": pin_txt,
        "lm_head": head,
        "head reload": reload,
        "request VC": vc_mode_name(env.get("TT_MM1D_IN1_VC_MODE", "0")),
        "bank-0 endpoint": "(0, 0) (+ep0)" if env.get("TT_MM1D_IN1_BANK0_XY") == "785" else "default (0, 11)",
        "in1 NoC": "NoC1 (swapped)" if env.get("TT_MM1D_MCAST_IN0_SWAP_NOCS") else "NoC0",
        "logits": "L1" if env.get("GEMMA4_LMHEAD_LOGITS_L1") == "1" else "DRAM",
        "NIU sampler": env.get("TT_MM1D_IN1_NIU_SAMPLE", "off"),
    }


def table(arms, mode, baseline):
    rows = [knobs(a) for a in arms]
    cols = [c for c in rows[0] if c != "arm"]
    # drop columns identical across the section into a one-line "common" note
    common = {c: rows[0][c] for c in cols if all(r[c] == rows[0][c] for r in rows)}
    varying = [c for c in cols if c not in common]
    out = [
        f"**Knobs** (measurement: {mode}; baseline: {baseline}; every matmul HiFi2, fp32 dest off, packer L1 acc on;"
        " weights bf16; `arm_knobs.py`):",
        "",
    ]
    out += [f"- {c}: {v}" for c, v in common.items()]
    if varying:
        out += ["", "| arm | " + " | ".join(varying) + " |", "|---|" + "---|" * len(varying)]
        out += [f"| `{r['arm']}` | " + " | ".join(r[c] for c in varying) + " |" for r in rows]
    return "\n".join(out)


def check(cell):
    """Compare the derived thin-K / fat-K / lm_head configs with the [mm-tune] / [lm-head-obw] / DRAM_SHARDED lines."""
    if cell.is_file():  # a standalone run's log, e.g. 03_<arm>.log
        arm, log = re.sub(r"^\d+_", "", cell.stem), cell.read_text(errors="ignore")
    else:
        arm = re.sub(r"^\d+_(time|noc)_", "", cell.name)
        log = (cell / "run.log").read_text(errors="ignore") if (cell / "run.log").exists() else ""
    logged = set(
        re.findall(
            r"\[mm-tune:\w+\] M=\d+ K=(\d+) N=(\d+) -> TUNED grid=(\d+)x(\d+) in0_block_w=(\d+) per_core_N=(\d+)", log
        )
    )
    logged |= {
        (k, n, "ds", c, iw, pcn)
        for k, n, c, iw, pcn in re.findall(
            r"\[mm-tune:\w+\] M=\d+ K=(\d+) N=(\d+) -> DRAM_SHARDED in0 cores=(\d+) in0_block_w=(\d+) per_core_N=(\d+)",
            log,
        )
    }
    _, extra, _ = arm_spec(arm)
    env = {**{k: v for k, v in env_for(arm_spec(arm)[0]).items() if k.startswith(("GEMMA4_", "TT_MM1D_"))}, **extra}
    bad = []
    for k, n, gx, gy, iw, pcn in logged:
        k, n = int(k), int(n)
        name = next((s for s, kn in SHAPES.items() if kn == (k, n)), None)
        if name is None:
            continue
        delivery, _, d_pcn, d_iw, _ = backbone_cfg(env, name)
        if (gx == "ds") != (delivery == "DRAM-sharded") or int(iw) != d_iw or int(pcn) != d_pcn:
            bad.append(
                f"{name}: log {'DRAM-sharded' if gx == 'ds' else f'grid {gx}x{gy}'} iw{iw} pcN {pcn} vs derived {delivery} iw{d_iw} pcN {d_pcn}"
            )
    # lm_head: the DRAM-fed call's per_core_N / out_block_w, and the L1 slice's per_core_N / in0_block_w
    head, _ = lm_head(env)
    for n, pcn, obw in set(
        re.findall(r"\[lm-head-obw\] GEMMA4_LMHEAD_DRAFT_OBW: N=(\d+) per_core_N=(\d+) out_block_w=(\d+)", log)
    ):
        if f"pcN {pcn} / ow{obw}" not in head:
            bad.append(f"lm_head N={n}: log pcN {pcn} ow{obw} vs derived {head}")
        logged.add(("lm_head", n, pcn, obw))
    for pcn, iw in set(re.findall(r"\[lm_head-split\] pinning .* per_core_N=(\d+), in0_block_w=(\d+)", log)):
        if f"(pcN {pcn}," not in head or f"iw{iw}," not in head:
            bad.append(f"lm_head slice: log pcN {pcn} iw{iw} vs derived {head}")
        logged.add(("slice", pcn, iw))
    return arm, len(logged), bad


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("arms", nargs="+")
    t.add_argument("--mode", default="traced time mode")
    t.add_argument("--baseline", default="the first arm")
    c = sub.add_parser("check")
    c.add_argument("cells", nargs="+", type=Path)
    a = p.parse_args()
    if a.cmd == "table":
        print(table(a.arms, a.mode, a.baseline))
    else:
        for cell in a.cells:
            arm, n, bad = check(cell)
            print(f"{arm}: {n} logged configs, {'OK' if not bad else 'MISMATCH ' + '; '.join(bad)}")


if __name__ == "__main__":
    main()
