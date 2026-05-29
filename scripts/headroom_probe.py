"""Headroom probe: does spending more compute actually move tau-bench reward?

This is lever #1 -- the diagnostic that gates the whole controller effort. The
Allocator can only help if compute *allocation* changes outcomes:

  - WIDER headroom: pass@k vs pass@1. If running k independent attempts solves
    many more tasks than one attempt, there is real room for the policy to spend
    extra attempts where they pay off. If pass@k ~= pass@1, WIDER buys nothing.
  - DEEPER headroom: take attempts that truncated (hit the step cap without the
    env finishing) and continue them; if reward improves, refining an in-flight
    conversation pays off. (Use --max-steps low enough that some attempts truncate,
    otherwise tau-bench conversations finish on their own and DEEPER has nothing to
    resume.)

Costs real OpenRouter credits: k full multi-turn conversations per task, each
talking to the live LLM user simulator. Start with a few tasks. Usage:

    python scripts/headroom_probe.py --env retail --n-tasks 5 --k 4
    python scripts/headroom_probe.py --env retail --n-tasks 5 --k 4 --max-steps 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wdp.config import load_config, require_openrouter_key
from wdp.cost import CostLedger
from wdp.llm.openrouter import OpenRouterClient
from wdp.metrics import pass_at_k, pass_hat_k


def run_probe(executor, tasks, *, k, threshold, deeper_steps, ledger):
    """Core probe loop. Returns (rows, successes) where rows is per-task
    (task_id, [reward per attempt], deeper_delta|None) and successes is the
    list-of-lists of bools that pass_at_k / pass_hat_k consume.

    `executor` only needs the wdp Executor surface (run / continue_from), so this
    is exercised offline with a fake env-backed executor in the smoke check."""
    rows = []
    successes: list[list[bool]] = []
    for task in tasks:
        attempts = [executor.run(task, ledger=ledger) for _ in range(k)]
        rewards = [float(a.reward or 0.0) for a in attempts]
        successes.append([r >= threshold for r in rewards])

        # DEEPER probe: resume the first unsolved attempt that truncated (env not
        # done) and see whether continuing the same conversation recovers reward.
        deeper_delta = None
        for a in attempts:
            if (a.reward or 0.0) < threshold and not a.done:
                before = float(a.reward or 0.0)
                cont = executor.continue_from(task, a, extra_steps=deeper_steps)
                deeper_delta = float(cont.reward or 0.0) - before
                break
        rows.append((task.id, rewards, deeper_delta))
    return rows, successes


def _format(rows, successes, *, k, ledger, currency) -> str:
    lines = [f"{'task':>22} {'attempt rewards (k=' + str(k) + ')':>28} {'deeper +/-':>10}"]
    for tid, rewards, dd in rows:
        rstr = " ".join(f"{r:.0f}" for r in rewards)
        ddstr = "-" if dd is None else f"{dd:+.2f}"
        lines.append(f"{tid:>22} {rstr:>28} {ddstr:>9}")

    p1 = sum(1 for s in successes if s and s[0]) / len(successes) if successes else 0.0
    pk = pass_at_k(successes)
    phk = pass_hat_k(successes)
    spent = ledger.amount(currency)
    lines += [
        "",
        f"pass@1 (one attempt):   {p1:.2f}",
        f"pass@k (any of {k}):      {pk:.2f}   <- WIDER headroom = pass@k - pass@1 = {pk - p1:+.2f}",
        f"pass^k (all {k}):         {phk:.2f}   (reliability)",
        f"total spend ({currency}):   {spent:.4f}",
    ]
    verdict = ("WIDER has real headroom; allocation can pay off."
               if pk - p1 >= 0.15 else
               "little WIDER headroom on these tasks; allocation gains will be small.")
    lines.append(f"\nverdict: {verdict}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="retail", help="tau-bench domain: retail | airline")
    ap.add_argument("--tb-split", default="test", help="task split: train | dev | test")
    ap.add_argument("--n-tasks", type=int, default=5)
    ap.add_argument("--k", type=int, default=4, help="independent attempts per task")
    ap.add_argument("--threshold", type=float, default=0.99, help="reward >= this counts as solved")
    ap.add_argument("--deeper-steps", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="agent step cap per attempt; set low to force truncation "
                         "so the DEEPER probe has something to resume")
    ap.add_argument("--agent-model", default=None,
                    help="model the agent uses (default: config models.executor)")
    ap.add_argument("--user-model", default="openai/gpt-4o-mini")
    ap.add_argument("--user-provider", default="openrouter")
    ap.add_argument("--currency", choices=["tokens", "latency", "dollars"], default="dollars")
    args = ap.parse_args()

    require_openrouter_key()
    cfg = load_config()
    from wdp.benchmarks import TauBenchBenchmark, TauReActExecutor

    bench = TauBenchBenchmark(env_name=args.env, split=args.tb_split,
                              task_indices=list(range(args.n_tasks)))
    tasks = bench.tasks()
    ledger = CostLedger()

    with OpenRouterClient() as client:
        executor = TauReActExecutor(
            client=client, model=args.agent_model or cfg["models"]["executor"],
            env_name=args.env, split=args.tb_split,
            user_model=args.user_model, user_provider=args.user_provider,
            max_steps=args.max_steps or cfg["executor"]["max_steps"],
            temperature=cfg["executor"]["temperature"],
        )
        print(f"probing {len(tasks)} {args.env}/{args.tb_split} tasks, "
              f"k={args.k} attempts each (this spends credits)...\n")
        rows, successes = run_probe(executor, tasks, k=args.k,
                                    threshold=args.threshold,
                                    deeper_steps=args.deeper_steps, ledger=ledger)

    print(_format(rows, successes, k=args.k, ledger=ledger, currency=args.currency))


if __name__ == "__main__":
    main()
