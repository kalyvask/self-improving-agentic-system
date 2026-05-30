"""Measurement-reliability helpers: error bars and power for small evals.

Lever from the AI-measurement literature (reliability_notes.pdf): a difference
between two policies is only meaningful relative to the standard error of that
difference. On a 10-task eval a 0.70-vs-0.70 solve rate is not a "tie" -- the
minimum detectable effect at that n is enormous, so the comparison simply has no
power. These helpers make that explicit and steer the comparison onto a
lower-variance, paired, continuous metric (cost) where a small eval can actually
resolve a difference.

All numpy/stdlib; no scipy. Proportion CIs use Wilson (well-behaved at small n);
paired differences use a percentile bootstrap (assumption-light).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# 80% power, two-sided alpha=0.05.
_Z_ALPHA = 1.959963985
_Z_POWER = 0.8416212336


@dataclass
class Interval:
    point: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.lo:.3f}, {self.hi:.3f}]"


def wilson_ci(k: int, n: int, z: float = _Z_ALPHA) -> Interval:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return Interval(0.0, 0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half))


def min_detectable_effect(p: float, n_per_arm: int,
                          z_alpha: float = _Z_ALPHA, z_power: float = _Z_POWER) -> float:
    """Smallest two-arm proportion difference detectable at the given power."""
    if n_per_arm <= 0:
        return 1.0
    return (z_alpha + z_power) * math.sqrt(2 * p * (1 - p) / n_per_arm)


def tasks_needed(p: float, delta: float,
                 z_alpha: float = _Z_ALPHA, z_power: float = _Z_POWER) -> float:
    """Per-arm task count needed to detect a proportion lift `delta`."""
    if delta <= 0:
        return math.inf
    return 2 * (z_alpha + z_power) ** 2 * p * (1 - p) / (delta * delta)


def mcnemar(pairs: list[tuple[bool, bool]]) -> dict:
    """Exact McNemar test on paired binary outcomes (a_i, b_i).

    Only discordant pairs (one succeeds, the other fails) carry signal. Returns
    the discordant counts and the exact two-sided binomial p-value -- the right
    test for "did B solve tasks A didn't, more than vice versa" on paired data.
    """
    b = sum(1 for a, bb in pairs if (not a) and bb)   # B wins
    c = sum(1 for a, bb in pairs if a and (not bb))   # A wins
    nd = b + c
    if nd == 0:
        return {"b_only": b, "c_only": c, "discordant": 0, "p_value": 1.0}
    k = min(b, c)
    # Two-sided exact binomial p at theta=0.5.
    tail = sum(math.comb(nd, i) for i in range(0, k + 1)) / (2 ** nd)
    p = min(1.0, 2 * tail)
    return {"b_only": b, "c_only": c, "discordant": nd, "p_value": p}


def paired_diff_ci(deltas: list[float], *, n_boot: int = 10000,
                   alpha: float = 0.05, seed: int = 0) -> Interval:
    """Percentile-bootstrap CI for the mean of paired differences (b_i - a_i).

    Used on per-task cost differences: pairing cancels task-to-task variance, so
    a continuous metric like cost resolves a difference at an n where the binary
    solve rate cannot."""
    d = np.asarray(deltas, dtype=float)
    if len(d) == 0:
        return Interval(0.0, 0.0, 0.0)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return Interval(float(d.mean()), float(lo), float(hi))
