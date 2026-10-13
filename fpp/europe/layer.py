"""The European correction: league exchange rates on top of the league model.

    log E[y_i] = log mu_dom + c + theta_att[L_i] + theta_def[L_j] + gamma_comp + gamma_stage
                 (+ lambda * euroform_i)

``mu_dom`` (`fpp.europe.model`) is the league model's prediction with the
opponent converted at par -- as though the five leagues were equally strong.
``theta`` measures how far from par they are, per target: ``theta_att[L]`` is how
much more a club from L produces in Europe than its domestic priors say,
``theta_def[L]`` how much more its opponents produce against it. They are
estimated from every top-five club's European matches at once, which is what
gives them a usable sample: one club's European history is a handful of games a
season, a league's is hundreds.

Fitted as a ridge Poisson GLM with the offset folded in by the standard
equivalence: a Poisson fit of ``y`` with mean ``mu_dom * exp(Xb)`` has the same
likelihood as a Poisson fit of the rate ``y / mu_dom`` with mean ``exp(Xb)``,
weighted by ``mu_dom``. That keeps the fit to scikit-learn's `PoissonRegressor`,
already a dependency, rather than adding a stats package for one model.

About twenty coefficients on ~1,500 team-rows per target, and a ridge penalty
on all of them, so a league that rarely meets another shrinks toward par rather
than toward whatever its few matches happened to say.

``euroform`` is the club's own recent European over- or under-performance against
``mu_dom``, shrunk toward zero by how much evidence sits behind it. It is the
"recent European results" signal, and it enters only if walk-forward shows it
earns its place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from ..config import LEAGUES
from ..evaluate import significance_team
from ..metrics import fit_nb_dispersion, logloss, pmf_for
from ..spec import target_spec
from .config import EURO_COMPS

LEAGUE_KEYS = tuple(LEAGUES)
STAGES = ("league", "ko1", "ko2", "single")

# Euroform: Poisson-gamma shrinkage of a club's European actual/expected ratio,
# with the prior worth `FORM_PSEUDO` matches of its own expected volume. Only
# European matches inside `FORM_DAYS` count, so a run three seasons ago does not.
FORM_PSEUDO = 10.0
FORM_DAYS = 400


def stage_group(stage, leg) -> str:
    """Two-legged knockout legs apart, everything one-off together.

    The final, and the single-leg quarter- and semi-finals of the August 2020
    tournament, are one-off matches at a neutral venue. A first leg and a second
    leg are played differently -- the second is played to a scoreline -- so they
    are separated rather than averaged.
    """
    if stage == "final":
        return "single"
    if pd.notna(leg):
        return "ko1" if int(leg) == 1 else "ko2"
    if stage in ("group-stage", "league-phase") or stage is None or pd.isna(stage):
        return "league"
    return "single"  # a knockout round with no leg: a one-off tie


def euroform(rows: pd.DataFrame, mu_col: str = "mu_dom") -> np.ndarray:
    """Each club's shrunk log actual/expected over prior European matches, record-before-update."""
    out = np.zeros(len(rows))
    df = rows[["team", "date", "actual", mu_col]].copy()
    df["_i"] = np.arange(len(df))
    for _, g in df.sort_values("date").groupby("team", sort=False):
        d = g["date"].to_numpy()
        y, mu = g["actual"].to_numpy(dtype=float), g[mu_col].to_numpy(dtype=float)
        ok = np.isfinite(y) & np.isfinite(mu)
        for k in range(len(g)):
            prior = (d < d[k]) & (d >= d[k] - np.timedelta64(FORM_DAYS, "D")) & ok
            if not prior.any():
                continue
            mbar = mu[prior].mean()
            pseudo = FORM_PSEUDO * mbar
            out[g["_i"].iat[k]] = np.log((y[prior].sum() + pseudo) / (mu[prior].sum() + pseudo))
    return out


def design(rows: pd.DataFrame, form: bool = False) -> pd.DataFrame:
    """Fixed-column design matrix, so every fold and the live fit line up."""
    X = pd.DataFrame(index=rows.index)
    for lk in LEAGUE_KEYS:
        X[f"att_{lk}"] = (rows["team_league"] == lk).astype(float)
    for lk in LEAGUE_KEYS:
        X[f"def_{lk}"] = (rows["opp_league"] == lk).astype(float)
    for c in list(EURO_COMPS)[1:]:
        X[f"comp_{c}"] = (rows["comp"] == c).astype(float)
    sg = [stage_group(s, l) for s, l in zip(rows["stage"], rows["leg"])]
    for s in STAGES[1:]:
        X[f"stage_{s}"] = np.array([g == s for g in sg], dtype=float)
    if form:
        X["euroform"] = rows["euroform"].to_numpy(dtype=float)
    return X


@dataclass
class Layer:
    target: str
    model: PoissonRegressor | None
    columns: list[str]
    dispersion: float | None
    form: bool

    def mu(self, rows: pd.DataFrame) -> np.ndarray:
        base = rows["mu_dom"].to_numpy(dtype=float)
        if self.model is None:
            return base
        return base * self.model.predict(design(rows, self.form)[self.columns])

    def coefficients(self) -> pd.Series:
        if self.model is None:
            return pd.Series(dtype=float)
        return pd.Series(np.r_[self.model.intercept_, self.model.coef_],
                         index=["intercept", *self.columns])


def fit(rows: pd.DataFrame, target: str, *, alpha: float = 0.01, form: bool = False,
        offset_only: bool = False) -> Layer:
    """Fit the layer on European rows with a ``mu_dom`` and an ``actual``."""
    r = rows[rows["mu_dom"].notna() & rows["actual"].notna()]
    dist = target_spec(target).dist
    if offset_only:
        layer = Layer(target, None, [], None, False)
    else:
        X = design(r, form)
        mu = r["mu_dom"].to_numpy(dtype=float)
        model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=1000)
        model.fit(X, r["actual"].to_numpy(dtype=float) / mu, sample_weight=mu)
        layer = Layer(target, model, list(X.columns), None, form)
    if dist == "nbinom":
        layer.dispersion = fit_nb_dispersion(r["actual"].to_numpy(dtype=float), layer.mu(r))
    return layer


def team_logloss(rows: pd.DataFrame, mu: np.ndarray, target: str, dispersion) -> np.ndarray:
    s = target_spec(target)
    return np.array([logloss(pmf_for(s.dist, m, s.team_cap, dispersion), y, s.team_cap)
                     for m, y in zip(mu, rows["actual"].to_numpy(dtype=float))])


def paired(rows: pd.DataFrame, mu: np.ndarray) -> pd.DataFrame:
    """Team rows -> the fixture-level frame `fpp.evaluate` scores."""
    r = rows.assign(_mu=mu)
    h = r[r["is_home"] == 1].set_index("fixture_id")
    a = r[r["is_home"] == 0].set_index("fixture_id")
    common = h.index.intersection(a.index)
    return pd.DataFrame({
        "fixture_id": common,
        "league_key": h.loc[common, "comp"].to_numpy(),
        "mu_home": h.loc[common, "_mu"].to_numpy(), "mu_away": a.loc[common, "_mu"].to_numpy(),
        "actual_home": h.loc[common, "actual"].to_numpy(), "actual_away": a.loc[common, "actual"].to_numpy(),
    })


def baseline_mu(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """European average at venue, from training seasons only. Neutral one-offs use the pooled mean."""
    by_venue = train.groupby("is_home")["actual"].mean()
    pooled = train["actual"].mean()
    single = np.array([stage_group(s, l) == "single" for s, l in zip(test["stage"], test["leg"])])
    mu = test["is_home"].map(by_venue).to_numpy(dtype=float, copy=True)
    mu[single] = pooled
    return mu


def walk_forward(rows: pd.DataFrame, target: str, seasons: list[int], *,
                 alpha: float = 0.01, form: bool = False, min_train_seasons: int = 2) -> dict:
    """Train on European seasons ``< y``, score ``y``, for each ``y`` in ``seasons``.

    Returns per-row predictions for every variant, so the comparison is paired on
    exactly the same rows: the venue baseline, the offset alone, and the layer.
    """
    r = rows[rows["mu_dom"].notna() & rows["actual"].notna()].copy()
    if form:
        r["euroform"] = euroform(r)
    preds = []
    for y in seasons:
        tr, te = r[r["year"] < y], r[r["year"] == y]
        if tr["year"].nunique() < min_train_seasons or te.empty:
            continue
        off = fit(tr, target, offset_only=True)
        lay = fit(tr, target, alpha=alpha, form=form)
        base_disp = (fit_nb_dispersion(tr["actual"].to_numpy(dtype=float), baseline_mu(tr, tr))
                     if target_spec(target).dist == "nbinom" else None)
        preds.append(te.assign(
            mu_base=baseline_mu(tr, te), disp_base=base_disp,
            mu_off=off.mu(te), disp_off=off.dispersion,
            mu_layer=lay.mu(te), disp_layer=lay.dispersion,
        ))
    return {"rows": pd.concat(preds, ignore_index=True) if preds else pd.DataFrame()}


def score(pred: pd.DataFrame, target: str) -> pd.DataFrame:
    """Mean per-team log loss of each variant, and significance of each step up."""
    out = []
    for v in ("base", "off", "layer"):
        ll = np.concatenate([team_logloss(g, g[f"mu_{v}"].to_numpy(), target, g[f"disp_{v}"].iat[0])
                             for _, g in pred.groupby("year")])
        out.append({"variant": v, "logloss": ll.mean(), "n": len(ll)})
    return pd.DataFrame(out)


def significance(pred: pd.DataFrame, target: str, model: str, base: str) -> dict:
    """`evaluate.significance_team` of ``model`` over ``base``, season dispersions respected.

    Dispersion varies by fold, and `significance_team` takes one per league key --
    so the fixtures are relabelled by season and passed a per-season dict.
    """
    p_m = paired(pred, pred[f"mu_{model}"].to_numpy()).assign(
        league_key=lambda d: d["fixture_id"].map(pred.drop_duplicates("fixture_id").set_index("fixture_id")["year"]))
    p_b = paired(pred, pred[f"mu_{base}"].to_numpy()).assign(league_key=p_m["league_key"].to_numpy())
    disp = lambda v: pred.groupby("year")[f"disp_{v}"].first().to_dict()  # noqa: E731
    return significance_team(p_m, p_b, target, disp(model), disp(base))


__all__ = ["LEAGUE_KEYS", "STAGES", "stage_group", "euroform", "design", "Layer", "fit",
           "team_logloss", "paired", "baseline_mu", "walk_forward", "score", "significance"]
