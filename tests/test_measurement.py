"""Offline tests for the measurement-science helpers: reliability + IRT.

These back the project's evaluation claims -- that a 10-task solve-rate gap is
underpowered, that cost is the metric with power, and that a Rasch fit recovers
task difficulty ordering -- so the numbers in the writeup are reproducible.
"""
from __future__ import annotations

import math

from wdp.metrics.reliability import (
    wilson_ci, min_detectable_effect, tasks_needed, mcnemar, paired_diff_ci,
)
from wdp.metrics.irt import fit_from_responses


def test_wilson_ci_is_wide_at_small_n():
    ci = wilson_ci(7, 10)
    assert ci.lo < 0.70 < ci.hi
    assert ci.hi - ci.lo > 0.4          # a 10-task solve rate is barely resolved
    # A large n tightens it a lot.
    tight = wilson_ci(700, 1000)
    assert (tight.hi - tight.lo) < 0.07


def test_mde_shrinks_with_n_and_tasks_needed_inverts_it():
    assert min_detectable_effect(0.7, 10) > min_detectable_effect(0.7, 100)
    # The 10-task eval cannot see anything smaller than a huge swing.
    assert min_detectable_effect(0.7, 10) > 0.4
    # Detecting a realistic +0.15 lift needs many dozens of tasks per arm.
    assert tasks_needed(0.7, 0.15) > 100
    assert tasks_needed(0.7, 0.30) < tasks_needed(0.7, 0.15)


def test_mcnemar_only_discordant_pairs_count():
    # 4 concordant (both solve), plus 3 B-wins and 0 A-wins -> B clearly better.
    pairs = [(True, True)] * 4 + [(False, True)] * 3
    r = mcnemar(pairs)
    assert r["b_only"] == 3 and r["c_only"] == 0 and r["discordant"] == 3
    assert r["p_value"] < 0.30          # exact two-sided binomial on 3 discordant
    # A perfect tie (no discordant pairs) is p=1.0, i.e. no evidence either way.
    assert mcnemar([(True, True), (False, False)])["p_value"] == 1.0


def test_paired_cost_ci_resolves_a_consistent_shift():
    # Every task got 0.05 cheaper -> the paired CI should exclude 0 even at n=10,
    # which is exactly why cost has power where binary solve rate does not.
    deltas = [-0.05] * 10
    ci = paired_diff_ci(deltas, seed=0)
    assert ci.hi < 0.0
    # A noisy zero-mean difference should straddle 0.
    noisy = paired_diff_ci([0.1, -0.1, 0.1, -0.1, 0.1, -0.1], seed=0)
    assert noisy.lo < 0 < noisy.hi


def test_rasch_recovers_difficulty_order():
    # Two respondents, three tasks of clearly increasing hardness. Pool repeated
    # responses; the fitted difficulty must rank easy < medium < hard.
    responses = []
    for user in ("p1", "p2"):
        for _ in range(5):
            responses += [("easy", user, 1.0), ("hard", user, 0.0)]
            responses += [("medium", user, 1.0), ("medium", user, 0.0)]
    fit = fit_from_responses(responses, epochs=3000)
    d = {t: fit.difficulty[i] for i, t in enumerate(fit.items)}
    assert d["easy"] < d["medium"] < d["hard"]
    # Difficulties are mean-centered for identifiability.
    assert abs(sum(fit.difficulty)) < 1e-6
