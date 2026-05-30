"""Offline ablations on already-collected traces -- no API spend.

Two questions we can answer before paying for a rerun:

  A. Budget calibration (the cost lever). The credit cost term is
     exp(-cost_weight * spent/budget). If budget >> typical task spend the term
     is ~1 for every trace and carries no cost signal. We recompute credit over a
     budget x cost_weight grid and report the correlation between a solved trace's
     mean value-per-cost and its actual dollar cost. Inert budget -> corr ~ 0
     (credit ignores cost); calibrated budget -> corr strongly negative (cheaper
     solves earn more), which is the whole point of a cost-aware controller.

  B. Difficulty-feature signal. The feature is difficulty = 1 - first_process_score,
     derived from the cheap ProcessVerifier -- which just FAILED the alt-test as a
     binary predictor of success. So does the feature carry any task-difficulty
     signal at all? We reconstruct it from the first decision's process score and
     correlate it with the ground-truth outcome and with cost. Near-zero means the
     feature is noise and the policy should lean on the judge-independent
     structural features instead.

Usage:
    python scripts/offline_ablations.py --arith traces/curve_dpo_fix3.jsonl \
        --traces traces/curve_dpo_fix3.jsonl traces/taubench_haiku_dpo_fix.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.loop import TraceLog
from wdp.loop.trace import assign_credit
from wdp.allocator.policy import Action


def _corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def calibrate_budget(path: str) -> None:
    traces = TraceLog(path).read()
    solved = [t for t in traces if (t.solved or t.terminal_reward >= 0.99)]
    costs = np.array([(t.total_cost or {}).get(t.currency, 0.0) for t in solved])
    med = float(np.median(costs[costs > 0])) if (costs > 0).any() else 0.0
    print(f"=== A. budget calibration | {path} ===")
    print(f" {len(solved)} solved traces | median cost {med:.5f} (currency={solved[0].currency})\n")
    print(f" {'budget':>9}{'spent/bud':>11}{'corr(credit,cost)':>20}{'credit spread':>15}")
    # Grid anchored to the spend regime: way-too-big, then multiples of the median.
    for budget in [0.2, med * 10, med * 4, med * 2, med, med * 0.5]:
        means = []
        for t in solved:
            assign_credit(t, budget=budget, cost_weight=0.5)
            spend_d = [d.value_per_cost for d in t.decisions
                       if d.action != Action.STOP.value]
            means.append(float(np.mean(spend_d)) if spend_d else 0.0)
        means = np.array(means)
        spent_over = med / budget if budget else float("inf")
        print(f" {budget:>9.4f}{spent_over:>11.3f}{_corr(means, costs):>20.3f}"
              f"{means.max() - means.min():>15.3f}")
    print("\n (corr ~0 = credit ignores cost; strongly negative = cheaper solves"
          " earn more -> cost-aware. spread ~0 = inert.)\n")


def difficulty_signal(paths: list[str]) -> None:
    print("=== B. difficulty-feature signal (difficulty = 1 - first process score) ===")
    for p in paths:
        traces = TraceLog(p).read()
        diff, solved, cost = [], [], []
        for t in traces:
            ps = [d.process_score_after for d in t.decisions
                  if getattr(d, "process_score_after", None) is not None]
            if not ps:
                continue
            diff.append(1.0 - ps[0])
            solved.append(1.0 if (t.solved or t.terminal_reward >= 0.99) else 0.0)
            cost.append((t.total_cost or {}).get(t.currency, 0.0))
        n = len(diff)
        c_out = _corr(diff, solved)        # should be negative if it tracks hardness
        c_cost = _corr(diff, cost)         # harder -> pricier, if meaningful
        print(f" {p}")
        print(f"   n={n}  corr(difficulty, solved)={c_out:+.3f}  "
              f"corr(difficulty, cost)={c_cost:+.3f}")
    print("\n (a real difficulty signal is negatively correlated with solving and"
          " positively with cost; ~0 means the judge-derived feature is noise.)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arith", help="trace file for the budget-calibration sweep")
    ap.add_argument("--traces", nargs="*", default=[],
                    help="trace files for the difficulty-signal check")
    args = ap.parse_args()
    if args.arith:
        calibrate_budget(args.arith)
    if args.traces:
        difficulty_signal(args.traces)
    if not args.arith and not args.traces:
        ap.error("pass --arith and/or --traces")


if __name__ == "__main__":
    main()
