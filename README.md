# self-improving-agent-system (wdp-controller)

A self-improving controller that decides how to spend the next unit of compute on
a tool-using agent task. At each decision node it picks one of four actions:

- `WIDER`: spawn a fresh parallel Executor attempt from the current state
- `DEEPER`: continue and refine the current trajectory on tool feedback
- `DECOMPOSE`: hand the task to the Planner, producing a sub-task DAG
- `STOP`: stop spending and abstain (a safe non-attempt)

The controller (the Allocator) is a small, CPU-trainable policy over cheap numeric
features, not a fine-tuned LLM. Executors are frontier models called through
OpenRouter. The expensive part is collecting traces; the policy update is cheap.
The Allocator learns from its own logged traces, so the headline result is a
self-improvement curve: collect traces with the current policy, fit the next
policy from those traces, measure, repeat.

## How self-improvement works here

1. Round 0: the `BanditAllocator` cold-starts with no data (Thompson sampling over
   per-action value-per-cost) and collects traces.
2. Each later round: fit a fresh `BCAllocator` or `DPOAllocator` on all
   accumulated traces, run it to collect more traces, and evaluate it on a
   held-out task set.
3. The per-round scoreboard (solve rate, mean / p95 cost, generation-verification
   gap) is the self-improvement curve.

Three learners share one small linear-softmax policy core, so any difference
between them is attributable to the learning objective, not model capacity:

- `BanditAllocator`: Thompson sampling; works with zero training data.
- `BCAllocator`: behavior-cloning. Keeps the top fraction of traces by realized
  value-per-cost, then clones features to action, weighted by per-decision credit.
  Correct STOPs survive the filter, so it also learns when not to spend.
- `DPOAllocator`: preference learning. Fits a BC reference, mines preference pairs
  from realized value-per-cost, then runs the DPO objective against that
  reference.

GRPO is estimated, not run. The loop logs the per-call token and wall cost GRPO
would need, so the GRPO cost and expected ceiling are an extrapolation from
measured data rather than a guess.

## Cost currencies

Every LLM call is logged in three currencies at once, because the optimal
allocation policy depends on which one you are spending:

- tokens: prompt plus completion tokens
- latency: wall-clock seconds, where concurrent branches cost the max of their
  children, not the sum
- dollars: OpenRouter usage cost when available

## Layout

```
src/wdp/
  config.py            .env + YAML config loading
  cost/                per-call cost accounting in three currencies
  llm/                 OpenRouter chat client with usage-based cost
  allocator/           the policy core and four policies:
                         policy.py  Action, NodeFeatures, BanditAllocator (v0)
                         linear.py  shared CPU-trainable linear-softmax core
                         bc.py      BCAllocator (behavior cloning)
                         dpo.py     DPOAllocator (preference learning)
  verifier/            terminal (ground-truth) and process (cheap) scorers
  executor/            ReAct loop, tool protocol, Task/Trajectory types
  planner/             decomposability probe + sub-task DAG
  loop/                trace logging, credit assignment, round runner,
                       self-improvement driver
  metrics/             success@budget, pass^k, risk-coverage, CVaR, gen-verif gap
  benchmarks/          Benchmark protocol + local checkable arithmetic suite
  grpo/                GRPO cost estimator (measured per-rollout extrapolation)
tests/                 offline end-to-end tests (no key, no network)
scripts/               smoke_live, run_selfimprove, estimate_grpo
config/default.yaml    models, budgets, allocator and loop settings
```

## Setup

```bash
pip install -e ".[dev]"
```

Paste your OpenRouter key into `.env` (get one at https://openrouter.ai/keys):

```
OPENROUTER_API_KEY=sk-or-...
```

## Run

Offline tests (no key, no network):

```bash
python -m pytest -q
```

Live single-task check (costs a few cents):

```bash
python scripts/smoke_live.py
```

Self-improvement curve on the local arithmetic benchmark (costs credits, one
Executor run per task per round):

```bash
python scripts/run_selfimprove.py --learner bc --rounds 3
python scripts/run_selfimprove.py --learner dpo --rounds 3 --budget 0.15
```

GRPO cost estimate from collected traces (offline, no credits):

```bash
python scripts/estimate_grpo.py --traces traces/traces.jsonl
```

## Benchmarks

The repo ships a local `ArithmeticBenchmark` whose verifier is exact and free, so
a full self-improvement run is cheap enough to iterate on a laptop. It mixes
atomic tasks, multi-part decomposable tasks, and underspecified tasks where STOP
is the only good move. Real benchmarks (tau-bench, SWE-bench, ALFWorld) implement
the same `Benchmark` protocol: tasks, tools, and a terminal verifier.
