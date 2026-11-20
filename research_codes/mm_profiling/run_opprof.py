#!/usr/bin/env python3
"""Traced per-op device profiles of the standalone drafter on this branch, per arm and profiler mode (record §8 / §11).

Port of l1w run_traced_opprof_arms.py to this branch's standalone_draft.py (--profile-replays) and arms.py.
Modes:
  time  5 replays; plain device profile (kernel durations, PER CORE MAX, tensor shapes / memory)
  noc   1 replay;  --collect-noc-traces, analysed by tt-npe (NPE_ROOT, default ~/codes/tt-npe at 3ce4534)

    python run_opprof.py --cells time:M,noc:M --output generated/l1w_main/opprof_noc [--device 0]

Writes <output>/<index>_<mode>_<arm>/ with run.log, ops_perf.csv (legacy post-processor), cpp_device_perf_report.csv
(time mode) and npe_viz/ (noc mode). The raw device logs are not kept.
OPPROF_PROGRAMS sets the profiler program count (default 10000); OPPROF_NPE_WORKERS caps tt-npe's pool (default 16).
"""

import argparse
import fcntl
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from arms import arm_env  # noqa: E402
from run_standalone import REPO, base_env  # noqa: E402

MODES = {"time": (5, []), "noc": (1, ["--collect-noc-traces"])}
LOGS = REPO / "generated/profiler/.logs"
REPORTS = REPO / "generated/profiler/reports"
NPE_GLOBAL = REPO / "generated/profiler/npe_viz"
NPE_ROOT = Path(os.getenv("NPE_ROOT", Path.home() / "codes/tt-npe"))
# tt-npe analyses traces in a pool of 16 workers; at ctx >= 131072 each holds 10-25 GB, workers die for lack of memory
# and the pool then waits forever (2026-11-20). OPPROF_NPE_WORKERS=<n> runs both analyses (tracy's own report, in its
# parent process, and the legacy post-processor) with n workers instead. Usage: python -c WRAP <n> (-m <module> | <script>) ...
NPE_WORKERS_WRAP = """import functools, runpy, sys
import npe_analyze_noc_trace_dir as m
m.analyze_noc_traces_in_dir = functools.partial(m.analyze_noc_traces_in_dir, num_workers=int(sys.argv[1]))
if sys.argv[2] == "-m":
    sys.argv = sys.argv[3:]
    runpy.run_module(sys.argv[0], run_name="__main__", alter_sys=True)
else:
    sys.argv = sys.argv[2:]
    sys.path.insert(0, str(__import__("pathlib").Path(sys.argv[0]).resolve().parent))
    runpy.run_path(sys.argv[0], run_name="__main__")
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", required=True, help="comma-separated <mode>:<arm>[:<replays>]")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--start-index", type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = open(f"/tmp/gemma4-gather-device{args.device}.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    py = str(REPO / "python_env/bin/python")
    failed = 0
    for index, cell in enumerate(args.cells.split(","), args.start_index):
        mode, arm, *rep = cell.split(":")
        replays, flags = MODES[mode]
        replays = int(rep[0]) if rep else replays
        out = args.output / f"{index:02d}_{mode}_{arm}"
        out.mkdir(exist_ok=True)
        env = base_env(args.device)
        env.update(arm_env(arm))
        env["TT_METAL_DEVICE_PROFILER"] = "1"
        if mode == "noc":
            # NoC events overflow the default profiler DRAM buffer within one iteration (record §8 setup)
            env["TT_METAL_PROFILER_PROGRAM_SUPPORT_COUNT"] = os.getenv("OPPROF_PROGRAMS", "10000")
            inst = NPE_ROOT / "install"
            env["PYTHONPATH"] += f":{inst / 'lib'}:{inst / 'bin'}"
            env["PATH"] = f"{env['PATH']}:{inst / 'bin'}"
            env["TT_NPE_DEVICE_MODEL_CONFIG_DIR"] = str(inst / "share/tt_npe/device/models")
            # tt-npe never clears it: files of op IDs this run does not trace would survive from older runs
            shutil.rmtree(NPE_GLOBAL, ignore_errors=True)
        script = str(Path(__file__).parent / "standalone_draft.py")
        cmd = [
            py,
            "-m",
            "tracy",
            "-r",
            "-p",
            "-v",
            "--no-web-server",
            *flags,
            script,
            str(args.ctx),
            str(args.max_seq_len),
        ]
        cmd += ["--profile-replays", str(replays)]
        if mode == "noc" and os.getenv("OPPROF_NPE_WORKERS"):
            cmd = [py, "-c", NPE_WORKERS_WRAP, os.environ["OPPROF_NPE_WORKERS"], *cmd[1:]]
        print("running", out.name, arm_env(arm), flush=True)
        cell_start = time.time()
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log = proc.stdout
        if mode == "time" and (LOGS / "cpp_device_perf_report.csv").exists():
            shutil.copy(LOGS / "cpp_device_perf_report.csv", out / "cpp_device_perf_report.csv")
        rep_dir = REPORTS / f"{args.output.name}_{out.name}"
        if rep_dir.exists():
            shutil.rmtree(rep_dir)
        post = [py, "tools/tracy/process_ops_logs.py", "--force-legacy-device-logs", "-n", rep_dir.name]
        post += ["--analyze-noc-traces"] if mode == "noc" else []
        if mode == "noc" and os.getenv("OPPROF_NPE_WORKERS"):
            post = [py, "-c", NPE_WORKERS_WRAP, os.environ["OPPROF_NPE_WORKERS"], *post[1:]]
        pp = subprocess.run(post, cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        log += "\n==== legacy post-processing ====\n" + pp.stdout
        (out / "run.log").write_text(log)
        csvs = sorted(rep_dir.rglob("ops_perf_results*.csv"))
        if csvs:
            shutil.copy(csvs[-1], out / "ops_perf.csv")
        npe = sorted(rep_dir.rglob("npe_viz"))
        if npe and mode == "noc":
            shutil.rmtree(out / "npe_viz", ignore_errors=True)
            shutil.move(str(npe[-1]), str(out / "npe_viz"))
            shutil.rmtree(NPE_GLOBAL, ignore_errors=True)
        for d in REPORTS.glob("*/npe_viz"):  # every report of this cell carries an npe_viz copy (1-6 GB); keep one
            if d.stat().st_mtime >= cell_start - 1:
                shutil.rmtree(d)
        dropped = len(re.findall(r"markers were dropped", log))
        ok = proc.returncode == 0 and pp.returncode == 0 and bool(csvs) and "PROFILED" in proc.stdout
        m = re.search(r"PROFILED .*", proc.stdout)
        print(f"  {out.name}: exit {proc.returncode} / post {pp.returncode}, dropped {dropped}, {m and m.group(0)}")
        failed += not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
