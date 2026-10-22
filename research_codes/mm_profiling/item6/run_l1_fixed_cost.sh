#!/bin/bash
# Item 6a (gemma4-12b record §9.2): L1-fed fat-K matmuls on the same 32 cores, multicast and ring, at three K
# (o_proj sliding Kt 128, pre_projection Kt 240, down_proj Kt 256), traced time mode, two cells per arm in
# A B C D E | E D C B A order. Fit with item6/analyze_l1_fixed_cost.py.
set -e
cd "$(dirname "$0")/.." && source ../../python_env/bin/activate
W="B+logitsl1+gridwide+vc1"
ARMS=("$W+pino_proj@0" "$W+pinpre_projection" "$W+pindown_proj@0" "$W+pino_proj@0r" "$W+pindown_proj@0r")
CELLS=""
for a in "${ARMS[@]}"; do CELLS="$CELLS,time:$a"; done
for ((i=${#ARMS[@]}-1; i>=0; i--)); do CELLS="$CELLS,time:${ARMS[$i]}"; done
python run_traced_opprof_arms.py --cells "${CELLS#,}" --output ../../generated/lmhead_l1_profile/item6_l1_fixed_cost
