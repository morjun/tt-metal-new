"""Arm names -> environment, for the standalone and end-to-end runners on the upstream-main research branch.

Grammar (l1w's run_pin_reload_standalone.arm_spec, mapped to this branch's knobs): <base>[+<suffix>...]

  base  M                                main as it is: every lm_head on ttnn's automatic config
        B                                drafter lm_head out_block_w 5, target lm_head out_block_w 3
        H<pcN>_<ow>                      whole drafter head in DRAM at per_core_N / out_block_w
        P<cols>_<pcN>_<ow>               drafter head: first <cols> columns pinned in L1, DRAM remainder at pcN / ow
        R<cols>_<pcN>_<ow>[_rope][_drisc<GB/s>c<pages>]
                                         the same pin with the reload around the global SDPA (Tensix reload;
                                         _rope: also around the global rotary embedding; _drisc: DRISC reload,
                                         paced at <GB/s> with <pages>-page DMA chunks)
  suffix +logitsl1                       drafter logits in L1
         +dec1d                          drafter backbone + pre_projection on main's explicit 1D decode config
                                         (in0_block_w up to 8, fp32 Dest accumulation). The grid is ttnn's automatic
                                         one, as without the suffix: l1w's +gridwide only removed the grid cap of
                                         l1w's own tuner, so it has no counterpart here
         +vc<mode>                       read-request VC mode of the 1D-multicast in1 reader (1 = by row)
         +vcrow<k> / +vccol<k> / +vcbank<k> / +vcfix<k>

The anchor of this branch is named in CLAUDE.md.
"""

import re

TARGET_OBW = {"GEMMA4_LM_HEAD_OUT_BLOCK_W": "3"}


def arm_env(arm: str) -> dict:
    base, *suffixes = arm.split("+")
    env = {}
    m = re.fullmatch(r"([PR])(\d+)_(\d+)_(\d+)(_rope)?(?:_drisc(\d+(?:\.\d+)?)c(\d+))?", base)
    if base == "M":
        pass
    elif base == "B":
        env.update(TARGET_OBW, GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W="5")
    elif h := re.fullmatch(r"H(\d+)_(\d+)", base):
        env.update(
            TARGET_OBW, GEMMA4_ASSISTANT_LM_HEAD_PER_CORE_N=h.group(1), GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W=h.group(2)
        )
    elif m:
        kind, cols, pcn, ow, rope, gbps, chunk = m.groups()
        env.update(
            TARGET_OBW,
            GEMMA4_ASSISTANT_LM_HEAD_L1_COLS=cols,
            GEMMA4_ASSISTANT_LM_HEAD_PER_CORE_N=pcn,
            GEMMA4_ASSISTANT_LM_HEAD_OUT_BLOCK_W=ow,
        )
        if kind == "P" and (rope or gbps):
            raise ValueError(f"{arm}: _rope / _drisc belong to an R arm")
        if kind == "R":
            env["GEMMA4_LMHEAD_L1_RELOAD"] = "1"
            if rope:
                env["GEMMA4_LMHEAD_L1_RELOAD_ROPE"] = "1"
            if gbps:
                env.update(
                    GEMMA4_LMHEAD_L1_RELOAD_DRISC="1",
                    GEMMA4_LMHEAD_L1_RELOAD_DRISC_GBPS=gbps,
                    GEMMA4_LMHEAD_L1_RELOAD_DRISC_CHUNK=chunk,
                )
    else:
        raise ValueError(f"unknown base arm {base!r} in {arm!r}")
    for s in suffixes:
        if s == "logitsl1":
            env["GEMMA4_ASSISTANT_LOGITS_L1"] = "1"
        elif s == "dec1d":
            env["GEMMA4_ASSISTANT_DECODE_1D"] = "1"
        elif v := re.fullmatch(r"vc(row|col|diag|bank|fix)?(\d+)", s):
            kind = {None: 0, "row": 1, "col": 2, "diag": 3, "bank": 4, "fix": 5}[v.group(1)]
            env["TT_MM1D_IN1_VC_MODE"] = str(kind * 16 + int(v.group(2)))
        else:
            raise ValueError(f"unknown suffix +{s} in {arm!r}")
    return env
