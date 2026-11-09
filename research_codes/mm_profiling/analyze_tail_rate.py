#!/usr/bin/env python3
"""G_read of gemma4-12b record §9.2b (M2): what the +16.8 µs per draft between F0's and H1's DRAM reads is made of.

1. Decomposition [derived], from analyze_m12.py's m12.json (sampled arms; per-draft windows R, NIU bytes per call B):
   G_read = R_tail,H1 - R_tail,F0 - R_up,F0 splits, at F0's tail rate r = B_tail,F0 / R_tail,F0, into
     up L0's bytes at the tail rate instead of up L0's own:  B_up / r - R_up
     H1's extra bytes beyond F0 tail + up L0:               (B_tail,H1 - B_tail,F0 - B_up) / r
     H1's tail slower than F0's per byte:                   R_tail,H1 - B_tail,H1 / r
2. Per-core NIU delivery of the tail [measured], every tail call of the +niusample cells: cores, MB per core, window
   (earliest first sample -> latest last sample), when 90% / 99% of the call's bytes had arrived, the rate up to 90%,
   the time after 90%, the spread of the cores' last samples, the core that ends the call.
3. Static DRAM-bank balance [model]: interleaved weight, page of tile (k, n) = k * W + n, bank = page mod 8; the last
   core's padded columns read pages past W (they run into the next K row). Per K-block of iw rows, read by all cores:
   busiest bank / mean. Each K-block covers iw * W consecutive pages, so this only tests the chip-wide aggregate; the
   order in which each core's requests reach the banks is not modelled.

    python research_codes/mm_profiling/analyze_tail_rate.py generated/lmhead_l1_profile/m12_1102
"""

import argparse
import bisect
import json
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "item6"))
from analyze_niu_samples import MHZ, WORD_BYTES, read_samples  # noqa: E402
from analyze_resource_profile import load, tensor_bytes  # noqa: E402

ARMS = (("F0", "*R59840_66_6*niusample*", 6322, 66), ("H1", "*R49280*niusample*", 6580, 75))
TAIL_MIN_BYTES = 300e6  # in1 of the DRAM tail; every other matmul reads < 20 MB


def decomposition(root):
    m = json.loads((root / "m12.json").read_text())["M2"]["arms"]
    F, H = m["F0 sampled"], m["H1 sampled"]
    RF, RU, RH = F["R_tail"]["mean"], F["R_up"]["mean"], H["R_tail"]["mean"]
    BF, BU, BH = F["B_tail_per_call"], F["B_up_per_call"], H["B_tail_per_call"]
    r = BF / RF  # bytes per µs of a per-draft window, per call
    parts = {
        "up L0's bytes at F0's tail rate instead of up L0's": BU / r - RU,
        "H1's bytes beyond F0 tail + up L0": (BH - BF - BU) / r,
        "H1's tail slower per byte than F0's": RH - BH / r,
    }
    print("1. G_read decomposition, µs per draft")
    print(f"   windows per draft: F0 tail {RF:.1f}, F0 up L0 {RU:.1f}, H1 tail {RH:.1f}; G_read {RH - RF - RU:+.1f}")
    print(f"   MB per call: F0 tail {BF / 1e6:.3f}, up L0 {BU / 1e6:.3f}, H1 tail {BH / 1e6:.3f}")
    gb = lambda b, R: b / (R / 3) / 1e3
    print(f"   GB/s: F0 tail {gb(BF, RF):.1f}, H1 tail {gb(BH, RH):.1f}, up L0 {gb(BU, RU):.1f}")
    for k, v in parts.items():
        print(f"   {k}: {v:+.1f}")
    print(f"   sum {sum(parts.values()):+.1f}")


def tail_calls(cell):
    samples = read_samples(cell / "niu_samples.csv")
    # every matmul, so that a sample is assigned to its own op (as in analyze_m12.niu_windows); keep the tail's
    ops = sorted(
        (o["fw_start"], o["fw_end"], (tensor_bytes(o["row"], "INPUT_1") or 0) > TAIL_MIN_BYTES)
        for session in load(cell / "ops_perf.csv").values()
        for o in session
        if o["op"] == "Matmul"
    )
    starts, tol = [o[0] for o in ops], 2 * MHZ
    per = defaultdict(lambda: defaultdict(list))
    for t, core, words in samples:
        i = bisect.bisect_right(starts, t + tol) - 1
        if i > 0 and t <= ops[i - 1][1]:
            i -= 1
        if i >= 0 and t <= ops[i][1] + tol and ops[i][2]:
            per[i][core].append((t, words))
    return [{c: s for c, s in cores.items() if len(s) >= 2} for _, cores in sorted(per.items())]


def niu_delivery(root):
    print("2. per-core NIU delivery of the DRAM tail, mean over calls")
    for arm, pat, _, _ in ARMS:
        R, last = defaultdict(list), Counter()
        for cell in sorted(root.glob(pat)):
            for cores in tail_calls(cell):
                t0 = min(s[0][0] for s in cores.values())
                ends = {c: (s[-1][0] - t0) / MHZ for c, s in cores.items()}
                ev = sorted(
                    ((tb - t0) / MHZ, (b - a) % (1 << 32) * WORD_BYTES)
                    for s in cores.values()
                    for (_, a), (tb, b) in zip(s, s[1:])
                )
                tot = sum(b for _, b in ev)
                acc, t90, t99 = 0, None, None
                for t, b in ev:
                    acc += b
                    t90 = t if t90 is None and acc >= 0.90 * tot else t90
                    t99 = t if t99 is None and acc >= 0.99 * tot else t99
                e = sorted(ends.values())
                R["cores"].append(len(cores))
                R["MB per call"].append(tot / 1e6)
                R["MB per core"].append(tot / 1e6 / len(cores))
                R["window"].append(e[-1])
                R["90% delivered"].append(t90)
                R["99% delivered"].append(t99)
                R["GB/s up to 90%"].append(0.9 * tot / t90 / 1e3)
                R["window after 90%"].append(e[-1] - t90)
                R["first to last core end"].append(e[-1] - e[0])
                last[max(ends, key=ends.get)] += 1
        n = len(R["window"])
        print(f"   {arm} ({n} calls): " + ", ".join(f"{k} {st.mean(v):.2f}" for k, v in R.items()))
        print(f"      core that ends the call: {last.most_common(3)}")


def bank_balance(iw=2, k_tiles=32):
    print(f"3. static DRAM-bank balance (iw{iw}, K {k_tiles} tiles; busiest bank / mean over the 8 banks)")
    for arm, _, W, pcn in ARMS:
        cores = -(-W // pcn)
        ratios = []
        for kb in range(k_tiles // iw):
            load_ = [0] * 8
            for k in range(kb * iw, kb * iw + iw):
                for n in range(cores * pcn):
                    load_[(k * W + n) % 8] += 1
            ratios.append(max(load_) / (sum(load_) / 8))
        print(
            f"   {arm}: W {W} (mod 8 = {W % 8}), {pcn} per core, {cores} cores, {cores * pcn - W} padded columns; "
            f"per K-block mean {st.mean(ratios):.4f}, max {max(ratios):.4f}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    a = p.parse_args()
    decomposition(a.root)
    niu_delivery(a.root)
    bank_balance()


if __name__ == "__main__":
    main()
