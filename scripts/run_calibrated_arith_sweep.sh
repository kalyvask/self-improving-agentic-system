#!/usr/bin/env bash
# Calibrated arithmetic sweep -- PREPARED, NOT AUTO-RUN. Spends real OpenRouter
# credits (~$5-8 total). Run only with explicit go-ahead.
#
# Why these settings (from the offline analysis):
#   --benchmark arithmetic   free, exact grading -> we can afford enough tasks for
#                            statistical power, unlike tau-bench at ~$0.18/task.
#   --atomic 60 --multi 40 --underspecified 10   = 110 tasks (66 train / 44 eval),
#                            past the ~100-task power floor the reliability module
#                            computed for solve rate; cost CI resolves well below.
#   --budget 0.003           ~2x the median task cost ($0.00143), i.e. spent/budget
#                            ~= 0.5, where the exp(-cost_weight*spent/budget) credit
#                            term is sharpest (corr(credit,cost) = -0.99). At the old
#                            default 0.2 the cost-efficiency factor was inert.
#   --max-decisions 8        4-part multi tasks need room for DECOMPOSE to pay off.
#   three learners           bandit (round 0 of each) vs BC vs DPO vs KTO.
#
# Headline metric is PAIRED COST, not solve rate (binary solve rate is underpowered
# even at n=44). After the run:
#   python scripts/analyze_eval.py --ab traces/calib_dpo.jsonl --irt traces/calib_*.jsonl
#   python scripts/analyze_eval.py --verifier traces/calib_*.jsonl
#   python scripts/offline_ablations.py --arith traces/calib_dpo.jsonl
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="--benchmark arithmetic --atomic 60 --multi 40 --underspecified 10 \
        --budget 0.003 --max-decisions 8 --rounds 3 --seed 0 --overwrite"

for LEARNER in bc dpo kto; do
  echo "=== $LEARNER ==="
  python scripts/run_selfimprove.py --learner "$LEARNER" $COMMON \
      --out "traces/calib2_${LEARNER}.jsonl"
done
