"""Offline measurement-science analysis of collected traces -- no API spend.

Answers the two questions a tiny eval cannot answer by eyeballing solve rates:

  1. Is a solve-rate difference real or just noise? -> Wilson CIs per arm, the
     minimum detectable effect at our n, the tasks needed to detect a target
     lift, McNemar on the paired binary outcomes, and -- the metric that actually
     has power at small n -- a paired bootstrap CI on per-task COST.
  2. How hard is each task, and which tasks discriminate? -> a Rasch (1PL) IRT
     fit over all responses, giving per-task difficulty and Fisher information at
     the agent's ability (the basis for choosing an informative small eval).

Usage:
    python scripts/analyze_eval.py --ab traces/eval_ab_haiku.jsonl
    python scripts/analyze_eval.py --ab traces/eval_ab_haiku.jsonl \
        --irt traces/taubench_haiku_dpo.jsonl traces/eval_ab_haiku.jsonl
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.loop import TraceLog
from wdp.metrics.reliability import (
    wilson_ci, min_detectable_effect, tasks_needed, mcnemar, paired_diff_ci,
)
from wdp.metrics.irt import fit_from_responses
from wdp.metrics.alt_test import alt_test, best_threshold


def _by_policy(traces):
    out = defaultdict(dict)            # policy -> {task_id: trace}
    for t in traces:
        out[t.policy][t.task_id] = t
    return out


def _pick_ab(names):
    """Choose the two policies to compare. Eval traces are now round-tagged
    (bandit@r0, bc@r1, bc@r2, bc@r3), so when there are more than two we compare the
    cold-start baseline (round 0) against the final learner round (highest @rN)."""
    if len(names) == 2:
        return names[0], names[1]
    def _round(p):
        return int(p.split("@r")[1]) if "@r" in p else -1
    baseline = next((p for p in names if _round(p) == 0), None)
    final = max(names, key=_round)
    if baseline is not None and final != baseline:
        return baseline, final
    return None


def analyze_ab(path: str) -> None:
    traces = TraceLog(path).read()
    pols = _by_policy(traces)
    names = list(pols)
    pick = _pick_ab(names)
    if pick is None:
        print(f"[ab] could not pick 2 policies in {path}, found {names}; skipping A/B.")
        return
    a_name, b_name = pick
    a, b = pols[a_name], pols[b_name]
    shared = sorted(set(a) & set(b))
    n = len(shared)
    print(f"=== paired A/B: {a_name} vs {b_name} | {n} shared tasks ===\n")

    def solved(t):  # robust to missing solved flag
        return bool(t.solved or t.terminal_reward >= 0.99)

    def cost(t):
        return (t.total_cost or {}).get(t.currency, 0.0)

    ka = sum(solved(a[t]) for t in shared)
    kb = sum(solved(b[t]) for t in shared)
    print(f" solve rate {a_name:>7}: {wilson_ci(ka, n)}")
    print(f" solve rate {b_name:>7}: {wilson_ci(kb, n)}")

    mc = mcnemar([(solved(a[t]), solved(b[t])) for t in shared])
    print(f"\n McNemar (paired solve): {b_name} wins {mc['b_only']}, {a_name} wins "
          f"{mc['c_only']}, p={mc['p_value']:.3f}  "
          f"({'no significant difference' if mc['p_value'] > 0.05 else 'significant'})")

    p_pool = (ka + kb) / (2 * n) if n else 0.0
    print(f"\n power on binary solve rate (pooled p={p_pool:.2f}):")
    print(f"   min detectable lift @ n={n}: +{min_detectable_effect(p_pool, n):.2f}")
    for d in (0.10, 0.15, 0.20):
        print(f"   to detect +{d:.2f}: need ~{tasks_needed(p_pool, d):.0f} tasks/arm")

    deltas = [cost(b[t]) - cost(a[t]) for t in shared]   # b - a (negative = cheaper)
    ci = paired_diff_ci(deltas)
    mean_a = sum(cost(a[t]) for t in shared) / n
    mean_b = sum(cost(b[t]) for t in shared) / n
    if ci.lo <= 0 <= ci.hi:
        sig = "  (straddles 0: not resolved)"
    elif ci.hi < 0:
        sig = "  <-- resolved CHEAPER (cost decrease)"
    else:
        sig = "  <-- resolved MORE EXPENSIVE (cost increase)"
    print(f"\n COST is the low-variance, paired metric with power at small n:")
    print(f"   mean cost {a_name}: {mean_a:.4f} | {b_name}: {mean_b:.4f}")
    print(f"   paired delta ({b_name}-{a_name}): {ci}{sig}")


def analyze_irt(paths: list[str]) -> None:
    responses = []
    for p in paths:
        for t in TraceLog(p).read():
            solved = 1.0 if (t.solved or t.terminal_reward >= 0.99) else 0.0
            responses.append((t.task_id, t.policy, solved))
    if not responses:
        print("[irt] no responses found.")
        return
    fit = fit_from_responses(responses)
    print(f"\n=== Rasch IRT difficulty | {fit.n_responses} responses, "
          f"{len(fit.items)} tasks, respondents={fit.respondents} ===")
    theta = float(fit.ability.mean())
    info = fit.information(theta)
    order = sorted(range(len(fit.items)), key=lambda i: fit.difficulty[i], reverse=True)
    print(f" ability theta (mean respondent) = {theta:+.2f}\n")
    print(f" {'task':<18}{'difficulty':>11}{'P(solve)':>10}{'info@theta':>12}")
    for i in order:
        print(f" {fit.items[i]:<18}{fit.difficulty[i]:>11.2f}"
              f"{fit.solve_prob(theta)[i]:>10.2f}{info[i]:>12.3f}")
    top = sorted(range(len(fit.items)), key=lambda i: info[i], reverse=True)[:5]
    print(f"\n most informative tasks at this ability (pick these for a small eval):")
    print("   " + ", ".join(fit.items[i] for i in top))


def analyze_verifier(paths: list[str]) -> None:
    """Alt-test verdict on whether the cheap ProcessVerifier is good enough to act
    on: does its best process score predict terminal success better than always
    guessing the majority outcome? Ground truth = the env-graded solved flag."""
    scores, truth = [], []
    for p in paths:
        for t in TraceLog(p).read():
            ps = [d.process_score_after for d in t.decisions
                  if getattr(d, "process_score_after", None) is not None]
            if not ps:
                continue
            scores.append(max(ps))
            truth.append(1.0 if (t.solved or t.terminal_reward >= 0.99) else 0.0)
    if not scores:
        print("[verifier] no process scores found.")
        return
    print(f"\n=== ProcessVerifier alt-test | {len(scores)} traces, "
          f"solve rate {sum(truth)/len(truth):.2f} ===")
    print(" " + str(alt_test(scores, truth, threshold=0.5, epsilon=0.05)).replace("\n", "\n "))
    print(" best operating point over thresholds:")
    print(" " + str(best_threshold(scores, truth, epsilon=0.05)).replace("\n", "\n "))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab", help="paired trace file with exactly two policies")
    ap.add_argument("--irt", nargs="*", default=[],
                    help="trace files to pool for the IRT difficulty fit")
    ap.add_argument("--verifier", nargs="*", default=[],
                    help="trace files to pool for the ProcessVerifier alt-test")
    args = ap.parse_args()
    if args.ab:
        analyze_ab(args.ab)
    if args.irt:
        analyze_irt(args.irt)
    if args.verifier:
        analyze_verifier(args.verifier)
    if not args.ab and not args.irt and not args.verifier:
        ap.error("pass --ab, --irt, and/or --verifier")


if __name__ == "__main__":
    main()
