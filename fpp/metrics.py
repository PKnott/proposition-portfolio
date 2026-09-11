"""Turning predictions into distributions, and scoring them.

Goals are near-Poisson (variance/mean ~1.1 in every league), so a Poisson pmf
built from the predicted rate is right.

Shots, shots on target and corners are materially overdispersed (variance/mean
2.1-3.9 for shots), which is why they use ``reg:tweedie``. But Tweedie predicts a
conditional *mean*, not a distribution -- and every over/under line and every
log-loss needs a pmf. So we map mean -> distribution through a **Negative
Binomial** whose dispersion is fitted per (target, league) on validation
residuals. Using Poisson there would badly understate the tails on exactly the
markets the workbook prices.

Match totals are the convolution of the two teams' pmfs, under the same
independence assumption the goals scoreline matrix already makes.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import nbinom, poisson

EPS = 1e-15


# --- Distributions --------------------------------------------------------


def poisson_pmf(lam: float, cap: int) -> np.ndarray:
    """PMF over 0..cap, with all mass above ``cap`` folded into the last cell."""
    lam = max(float(lam), 1e-12)
    ks = np.arange(cap + 1)
    p = poisson.pmf(ks, lam).astype(float)
    p[-1] += poisson.sf(cap, lam)
    return p


def nb_params(mu: float, dispersion: float) -> tuple[float, float]:
    """``(n, p)`` for scipy's negative binomial from mean and dispersion ``r``.

    Parameterised so ``var = mu + mu**2 / r``: large ``r`` -> Poisson limit.
    """
    mu = max(float(mu), 1e-12)
    r = max(float(dispersion), 1e-6)
    return r, r / (r + mu)


def nbinom_pmf(mu: float, dispersion: float, cap: int) -> np.ndarray:
    n, p = nb_params(mu, dispersion)
    ks = np.arange(cap + 1)
    out = nbinom.pmf(ks, n, p).astype(float)
    out[-1] += nbinom.sf(cap, n, p)
    return out


def pmf_for(dist: str, mu: float, cap: int, dispersion: float | None = None) -> np.ndarray:
    if dist == "poisson":
        return poisson_pmf(mu, cap)
    if dist == "nbinom":
        if dispersion is None:
            raise ValueError("negative binomial needs a fitted dispersion")
        return nbinom_pmf(mu, dispersion, cap)
    raise ValueError(f"unknown distribution {dist!r}")


def fit_nb_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    """Moment-match ``r`` in ``var = mu + mu**2 / r`` on held-out residuals.

    Returns a large value (effectively Poisson) when the residual variance does
    not exceed the mean -- i.e. when there is no overdispersion left to model.
    """
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    m = np.isfinite(y) & np.isfinite(mu) & (mu > 0)
    if m.sum() < 50:
        return 1e6
    y, mu = y[m], mu[m]
    excess = float(np.mean((y - mu) ** 2 - mu))
    if excess <= 0:
        return 1e6  # no overdispersion -> Poisson
    return float(np.mean(mu**2) / excess)


def convolve_pmf(a: np.ndarray, b: np.ndarray, cap: int) -> np.ndarray:
    """Distribution of the sum of two independent counts, capped and renormalised."""
    full = np.convolve(a, b)
    out = np.zeros(cap + 1)
    head = min(cap, len(full) - 1)
    out[:head] = full[:head]
    out[cap] = full[head:].sum()
    s = out.sum()
    return out / s if s > 0 else out


# --- Scoring --------------------------------------------------------------


def check_pmf(pmf: np.ndarray, cap: int) -> np.ndarray:
    assert len(pmf) == cap + 1, f"pmf length {len(pmf)} != {cap + 1}"
    assert np.isclose(pmf.sum(), 1.0, atol=1e-6), f"pmf sums to {pmf.sum():.6f}"
    return pmf


def bucket(actual: float, cap: int) -> int:
    return int(min(int(actual), cap))


def logloss(pmf: np.ndarray, actual: float, cap: int) -> float:
    return float(-np.log(max(pmf[bucket(actual, cap)], EPS)))


def over_prob(pmf: np.ndarray, line: float) -> float:
    """P(count > line) for a half-integer line."""
    k = int(np.floor(line))
    return float(1.0 - pmf[: k + 1].sum())


def binary_metrics(y_true, p_yes, name: str) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(p_yes, dtype=float), EPS, 1 - EPS)
    out = {
        "market": name,
        "logloss": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()),
        "brier": float(((p - y) ** 2).mean()),
        "accuracy": float(((p >= 0.5).astype(int) == y.astype(int)).mean()),
        "n": int(len(y)),
    }
    try:
        from sklearn.metrics import roc_auc_score

        out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    except Exception:
        out["auc"] = float("nan")
    return out


def reliability_bins(y_true, p, n_bins: int = 10):
    import pandas as pd

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append({
            "bin": b, "lo": edges[b], "hi": edges[b + 1], "n": int(m.sum()),
            "pred_mean": float(p[m].mean()) if m.any() else np.nan,
            "obs_freq": float(y[m].mean()) if m.any() else np.nan,
        })
    return pd.DataFrame(rows)


def ece(y_true, p, n_bins: int = 10) -> float:
    """Expected calibration error -- weighted mean gap between predicted and observed."""
    bins = reliability_bins(y_true, p, n_bins)
    bins = bins[bins["n"] > 0]
    if bins.empty:
        return float("nan")
    w = bins["n"] / bins["n"].sum()
    return float((w * (bins["pred_mean"] - bins["obs_freq"]).abs()).sum())


def bootstrap_ci(diff: np.ndarray, n_boot: int = 10000, seed: int = 2026) -> tuple[float, float]:
    """Percentile CI for the mean of a paired per-match difference."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


__all__ = [
    "poisson_pmf", "nbinom_pmf", "nb_params", "pmf_for", "fit_nb_dispersion",
    "convolve_pmf", "check_pmf", "bucket", "logloss", "over_prob",
    "binary_metrics", "reliability_bins", "ece", "bootstrap_ci",
]
