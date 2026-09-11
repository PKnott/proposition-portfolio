"""The declarative feature spec -- the single source of truth for what exists.

Why this module is the centre of the rebuild
--------------------------------------------
In the old pipeline the five prior columns were hardcoded in *six* places:

1. ``compute_priors``' output column list
2. the five per-team history lists (``gf_hist``, ``xgf_hist``, ...)
3. ``KEY_TO_COLS``
4. ``build_model_table``'s select + rename maps
5. ``build_team_priors_lookup``
6. the two hand-written dicts in ``predict_lambdas_from_priors``

Widening from one target family to four would have meant editing all six, four
times over. Instead everything downstream is *generated* from the registries
here, so adding a stat is a one-line change to ``STATS``.

The candidate feature space is a cross product::

    scope {team, opp} x stat x side {for, against} x venue {all, venue}

plus per-league venue-aware league priors and a handful of context columns.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from .config import SCORING_MODE

# --- Stats ----------------------------------------------------------------


@dataclass(frozen=True)
class StatSpec:
    """One measurable quantity, recorded for and against each team per match."""

    key: str
    display: str
    for_col: str  # column in the long team-perspective table
    against_col: str
    is_target: bool = False  # do we model it as a target family?
    objective: str | None = None  # xgboost objective, when is_target
    dist: str | None = None  # "poisson" | "nbinom", when is_target
    # Bucket cap for the pmf/log-loss of the *match total*; last bucket is "cap+".
    total_cap: int | None = None
    # Bucket cap for the pmf/log-loss of ONE team's count. Reasoned per stat, not
    # derived from `total_cap`: each sits near 60% of its match-total counterpart
    # rather than half, because in a lopsided match one side takes well over half
    # the combined total, and a strict 50/50 split would recreate the same
    # tail-clipping one level down.
    team_cap: int | None = None
    # Default over/under ladders for the workbook, anchored on measured means.
    team_lines: tuple[float, ...] = ()
    match_lines: tuple[float, ...] = ()


STATS: tuple[StatSpec, ...] = (
    StatSpec(
        "goals", "Goals", "goals_for", "goals_against",
        is_target=True, objective="count:poisson", dist="poisson",
        total_cap=10, team_cap=6,
        team_lines=(0.5, 1.5, 2.5, 3.5), match_lines=(0.5, 1.5, 2.5, 3.5),
    ),
    StatSpec(
        "shots", "Shots", "shots_for", "shots_against",
        is_target=True, objective="reg:tweedie", dist="nbinom",
        total_cap=40, team_cap=25,
        team_lines=(8.5, 10.5, 12.5, 14.5), match_lines=(20.5, 22.5, 24.5, 26.5),
    ),
    StatSpec(
        "sot", "Shots on Target", "sot_for", "sot_against",
        is_target=True, objective="reg:tweedie", dist="nbinom",
        total_cap=20, team_cap=12,
        team_lines=(2.5, 3.5, 4.5, 5.5), match_lines=(6.5, 7.5, 8.5, 9.5),
    ),
    StatSpec(
        "corners", "Corners", "corners_for", "corners_against",
        is_target=True, objective="reg:tweedie", dist="nbinom",
        total_cap=25, team_cap=15,
        team_lines=(3.5, 4.5, 5.5, 6.5), match_lines=(8.5, 9.5, 10.5, 11.5),
    ),
    # Predictors only -- never modelled as targets, but legitimate candidate
    # features for any target family (a shot prior may help predict goals).
    StatSpec("xg", "xG", "xg_for", "xg_against"),
    StatSpec("npxg", "npxG", "npxg_for", "npxg_against"),
    StatSpec("points", "Points", "points_for", "points_against"),
)

STAT_BY_KEY: dict[str, StatSpec] = {s.key: s for s in STATS}
TARGETS: tuple[str, ...] = tuple(s.key for s in STATS if s.is_target)

# Stats that measure the same underlying thing compete inside one group, so the
# search can find that (say) shots-for is redundant once goals-for and xG-for are
# already in. Grouping one stat with only its own venue variant -- the previous
# structure -- could never ask that question.
#
# `points` sits on its own: it is an outcome summary (win/draw/loss), not a
# shot-quality measure, and folding it into attack/defence would let it mask the
# process stats it correlates with.
STAT_FAMILIES: dict[str, tuple[str, ...]] = {
    "end_product": ("goals", "xg", "npxg"),
    "process": ("shots", "sot", "corners"),
    "form": ("points",),
}

# Group names, by (family, side). A whole family shares one group so its stats
# fight each other directly.
#
# Splitting a family across groups to save evaluations was a false economy: each
# stat then fought alone against a backdrop that still contained near-duplicates
# of itself sitting in *other* groups, so nothing ever looked necessary and stage
# 1 discarded too much before stage 2 could test anything against the full-model
# anchor. Goals vs xG has to be settled inside one group, the way the reference
# framework did it.
#
# The cost is real: six members means 64 subsets per group rather than 16.
FAMILY_GROUP_NAMES: dict[tuple[str, str], str] = {
    ("end_product", "for"): "attack",
    ("end_product", "against"): "defence",
    ("process", "for"): "for",
    ("process", "against"): "against",
}


def target_spec(target: str) -> StatSpec:
    s = STAT_BY_KEY[target]
    if not s.is_target:
        raise ValueError(f"{target!r} is a predictor stat, not a target family")
    return s


def scoring_hash() -> str:
    """Fingerprint of the scoring *definition*: mode, distribution and caps.

    A cached score is only valid for the definition that produced it, and
    ``point_id`` encodes none of this -- it carries the group id and its subset,
    nothing about what the resulting number means. Changing the mode or a cap
    changes the number itself, so without this the same ``point_id`` would serve
    a score measured under the old definition as if it were fresh.

    Same discipline as ``artifacts.spec_hash()``, applied to the metric rather
    than to the feature spec.
    """
    payload = {
        "mode": SCORING_MODE,
        "targets": {
            s.key: {"dist": s.dist, "total_cap": s.total_cap, "team_cap": s.team_cap}
            for s in STATS if s.is_target
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


# --- Buffers & features ---------------------------------------------------

SIDES = ("for", "against")
VENUES = ("all", "venue")
PERSPECTIVES = ("team", "opp")


@dataclass(frozen=True)
class BufferSpec:
    """One rolling buffer: a stat, a side, and whether it is venue-restricted.

    ``all`` uses window ``L`` over every prior match; ``venue`` uses window
    ``L_V = ceil(0.5 * L)`` over prior matches at the same venue only.
    """

    stat: str
    side: str
    venue: str

    @property
    def name(self) -> str:
        suffix = "_v" if self.venue == "venue" else ""
        return f"{self.stat}_{self.side}{suffix}"


@dataclass(frozen=True)
class FeatureSpec:
    """A buffer read from one perspective -- the team's own, or its opponent's."""

    buffer: BufferSpec
    perspective: str

    @property
    def name(self) -> str:
        return f"{self.perspective}_avg_{self.buffer.name}"


def enumerate_buffers(stats: tuple[StatSpec, ...] = STATS) -> tuple[BufferSpec, ...]:
    return tuple(
        BufferSpec(s.key, side, venue) for s in stats for side in SIDES for venue in VENUES
    )


def enumerate_features(buffers: tuple[BufferSpec, ...] | None = None) -> tuple[FeatureSpec, ...]:
    buffers = buffers if buffers is not None else enumerate_buffers()
    return tuple(FeatureSpec(b, p) for b in buffers for p in PERSPECTIVES)


BUFFERS: tuple[BufferSpec, ...] = enumerate_buffers()
BUFFER_NAMES: tuple[str, ...] = tuple(b.name for b in BUFFERS)
BUFFER_INDEX: dict[str, int] = {n: i for i, n in enumerate(BUFFER_NAMES)}

FEATURES: tuple[FeatureSpec, ...] = enumerate_features()

# Stats whose buffers are still computed but which are NOT offered to feature
# selection. `points` is excluded by decision, not by evidence -- recorded here
# so the exclusion is visible rather than being an absence nobody notices. Its
# buffers still exist, so putting it back is deleting one entry.
NON_CANDIDATE_STATS: frozenset[str] = frozenset({"points"})

CANDIDATE_FEATURES: tuple[FeatureSpec, ...] = tuple(
    f for f in FEATURES if f.buffer.stat not in NON_CANDIDATE_STATS
)
PRIOR_FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in CANDIDATE_FEATURES)

# Venue-aware league priors: the league's own recent mean for this stat, routed
# by the team's venue.
#
# DELIBERATELY NOT CANDIDATE FEATURES -- they are excluded from
# ``ALL_FEATURE_NAMES`` below. The pooled model gets a per-league offset from the
# `league` categorical, and each team carries its own decayed form. What is given
# up is the *time-varying* league level (rates drifting across seasons); if a
# per-league segment breakdown ever shows drift, adding these back is a one-line
# change here plus restoring the loop in ``build.build_feature_table``.
# ``priors.BufferWindows.league_priors()`` remains implemented and working.
#
# Note for anyone tempted to collapse the for/against pair: they do NOT converge.
# ``priors`` groups the league blocks by (league, is_home), so on a home row
# `_for` is the league's home-scoring rate and `_against` its away-scoring rate.
# They differ by home advantage.
LEAGUE_FEATURE_NAMES: tuple[str, ...] = tuple(
    f"league_avg_{s.key}_{side}" for s in STATS for side in SIDES
)

CONTEXT_FEATURES: tuple[str, ...] = (
    "is_home",
    "league",
    "game_week_normalised",
    # Schedule. `rest_days` measures club quality far more than fatigue -- see
    # the docstring on `clean.attach_schedule_context` -- and `matches_14d` is
    # the properly-posed congestion question. Both are candidates; which of them
    # earns a place is the search's decision, not this file's.
    "team_rest_days",
    "opp_rest_days",
    "team_matches_14d",
    "opp_matches_14d",
    # Movement. `is_new_to_league` is the live signal today; `rank_delta` is the
    # generalised mechanism that activates when lower tiers are added. See the
    # LEAGUE_RANK note in config.py.
    "team_is_new_to_league",
    "opp_is_new_to_league",
    "team_rank_delta",
    "opp_rank_delta",
)

ALL_FEATURE_NAMES: tuple[str, ...] = PRIOR_FEATURE_NAMES + CONTEXT_FEATURES

# A small, fixed set used for the (L, alpha) window search. Measuring a
# smoothing window through the full feature space adds variance and cost without
# changing which alpha wins.
CORE_FEATURES: tuple[str, ...] = (
    "team_avg_goals_for",
    "team_avg_goals_against",
    "team_avg_xg_for",
    "team_avg_xg_against",
    "opp_avg_goals_for",
    "opp_avg_goals_against",
    "opp_avg_xg_for",
    "opp_avg_xg_against",
    "is_home",
    "league",
)


def _venue_forms(perspective: str, stat: str, side: str) -> list[str]:
    """Both venue representations of one stat, all-venue first."""
    return [
        FeatureSpec(BufferSpec(stat, side, venue), perspective).name for venue in VENUES
    ]


def feature_groups() -> dict[str, list[str]]:
    """Stage-1 groups: 10, tested **in isolation** with no other features present.

    A whole stat family shares one group, so goals/xG/npxG argue against each
    other directly and likewise shots/SOT/corners. Members are both venue forms.

    These are deliberately small, because stage 1 asks a narrow question: does
    this group carry *standalone* signal? Testing against a backdrop of everything
    else -- as earlier versions did -- meant a group with real signal could lose
    simply because 50-odd other features already implied the same thing. With no
    backdrop, the only way to lose is to carry nothing on its own.

    ``context`` and ``movement`` have no venue axis. ``points`` is excluded from
    the candidate space entirely -- see ``NON_CANDIDATE_STATS``.

    Cost: 8 x 64 + 128 + 16 = 656 subsets, and each fit is cheap (4-6 columns
    rather than the whole space) precisely because there is no backdrop. The
    ``context`` term is the one that moves -- it was 32 while the group had five
    members, and 128 since `matches_14d` joined it.
    """
    groups: dict[str, list[str]] = {}

    for family, side in (
        ("end_product", "for"),
        ("end_product", "against"),
        ("process", "for"),
        ("process", "against"),
    ):
        stem = FAMILY_GROUP_NAMES[(family, side)]
        for p in PERSPECTIVES:
            groups[f"{p}_{stem}"] = [
                name
                for s in STAT_FAMILIES[family]
                if s not in NON_CANDIDATE_STATS
                for name in _venue_forms(p, s, side)
            ]

    groups["context"] = [
        c for c in CONTEXT_FEATURES if "new_to_league" not in c and "rank_delta" not in c
    ]
    groups["movement"] = [
        c for c in CONTEXT_FEATURES if "new_to_league" in c or "rank_delta" in c
    ]
    return groups


# Stage 2 merges each side's end-product and process groups into one, so goals
# and shots finally compete directly -- but only over what survived stage 1.
# Merging the raw groups would be 12 members and 4,096 subsets each; restricting
# to survivors is what keeps it affordable.
MERGED_GROUP_SOURCES: dict[str, tuple[str, str]] = {
    "team_attack_merged": ("team_attack", "team_for"),
    "team_defence_merged": ("team_defence", "team_against"),
    "opp_attack_merged": ("opp_attack", "opp_for"),
    "opp_defence_merged": ("opp_defence", "opp_against"),
}

# No merge partner, so they continue as themselves. Still searched and still
# eligible for elimination -- nothing leaves the pipeline without losing on merit.
CARRY_THROUGH_GROUPS: tuple[str, ...] = ("context", "movement")


def merged_groups(stage1_winners: dict[str, list[str]]) -> dict[str, list[str]]:
    """Stage-2+ groups, built from stage-1 survivors.

    Deliberately distinct names from the stage-1 groups: ``team_attack`` (stage 1,
    end-product only) and ``team_attack_merged`` (stage 2, end-product *and*
    process) are different objects, and reading them as the same thing at two
    points in the pipeline is exactly the confusion to avoid.

    A stage-1 group that lost everything contributes nothing here.
    """
    out: dict[str, list[str]] = {}
    for gid, sources in MERGED_GROUP_SOURCES.items():
        out[gid] = [f for src in sources for f in stage1_winners.get(src, [])]
    for gid in CARRY_THROUGH_GROUPS:
        out[gid] = list(stage1_winners.get(gid, []))
    return out


def validate() -> None:
    """Structural invariants of the candidate space and its grouping.

    With venue folded into the group members, the groups partition the *whole*
    candidate space again -- a stronger invariant than the split one it replaces,
    because a feature that belongs to no group can never be selected.
    """
    assert len(set(BUFFER_NAMES)) == len(BUFFER_NAMES), "duplicate buffer names"
    assert len(set(ALL_FEATURE_NAMES)) == len(ALL_FEATURE_NAMES), "duplicate feature names"

    # Both caps must exist for every target family: `score_team_logloss` reads
    # `team_cap` and `score_total_logloss` reads `total_cap`, and `pmf_for`
    # given None builds a zero-length pmf that fails far from the cause.
    for s in STATS:
        if not s.is_target:
            continue
        assert s.total_cap and s.team_cap, (
            f"target {s.key!r} needs both total_cap and team_cap "
            f"(got total_cap={s.total_cap}, team_cap={s.team_cap})"
        )
        assert s.team_cap < s.total_cap, (
            f"target {s.key!r}: team_cap {s.team_cap} must be below total_cap "
            f"{s.total_cap} -- one team cannot out-range the match"
        )
        # A cap at or below the top priced line would fold that line's own
        # decision boundary into the tail bucket.
        assert not s.team_lines or s.team_cap > max(s.team_lines), (
            f"target {s.key!r}: team_cap {s.team_cap} does not clear its top "
            f"team line {max(s.team_lines)}"
        )
        assert not s.match_lines or s.total_cap > max(s.match_lines), (
            f"target {s.key!r}: total_cap {s.total_cap} does not clear its top "
            f"match line {max(s.match_lines)}"
        )

    groups = feature_groups()
    grouped = [f for members in groups.values() for f in members]
    dupes = {f for f in grouped if grouped.count(f) > 1}
    assert not dupes, f"features in multiple groups: {dupes}"

    expected = set(ALL_FEATURE_NAMES)
    assert set(grouped) == expected, (
        f"group coverage mismatch: missing={sorted(expected - set(grouped))} "
        f"extra={sorted(set(grouped) - expected)}"
    )

    # Every stat group carries both venue forms, so the subset search decides the
    # representation rather than inheriting it.
    for gid, members in groups.items():
        if gid in ("context", "movement"):
            continue
        venue = [m for m in members if m.endswith("_v")]
        assert len(venue) * 2 == len(members), (
            f"group {gid!r} is not an even split of all-venue and venue-specific "
            f"forms: {members}"
        )

    for f in CORE_FEATURES:
        assert f in ALL_FEATURE_NAMES, f"CORE_FEATURES has unknown feature {f!r}"

    # Every family's stats must be real, and the families must cover every stat --
    # otherwise a stat silently drops out of the candidate space.
    familied = {s for stats in STAT_FAMILIES.values() for s in stats}
    known = {s.key for s in STATS}
    assert familied <= known, f"STAT_FAMILIES names unknown stats: {sorted(familied - known)}"
    assert familied == known, f"stats missing from STAT_FAMILIES: {sorted(known - familied)}"

    # A family must live in ONE group per (perspective, side), so its stats
    # compete directly. Splitting a family apart is the failure this guards.
    for family, stats in STAT_FAMILIES.items():
        candidates = [s for s in stats if s not in NON_CANDIDATE_STATS]
        if not candidates or (family, "for") not in FAMILY_GROUP_NAMES:
            continue
        for p in PERSPECTIVES:
            for side in SIDES:
                gid = f"{p}_{FAMILY_GROUP_NAMES[(family, side)]}"
                members = set(groups[gid])
                for s in candidates:
                    assert set(_venue_forms(p, s, side)) <= members, (
                        f"{s!r} of family {family!r} is not in {gid!r} -- family split "
                        f"across groups means its stats never compete head-to-head"
                    )

    # Non-candidate stats must be absent from the candidate space entirely, not
    # merely unused -- otherwise they reappear the next time something iterates
    # FEATURES instead of CANDIDATE_FEATURES.
    for stat in NON_CANDIDATE_STATS:
        leaked = [f for f in ALL_FEATURE_NAMES if f"_{stat}_" in f]
        assert not leaked, f"non-candidate stat {stat!r} leaked into the space: {leaked}"

    # Every merged group must draw from real stage-1 groups, and between them the
    # merges plus carry-throughs must cover every stage-1 group -- otherwise a
    # group's survivors are silently discarded on the way into stage 2.
    sourced = {s for srcs in MERGED_GROUP_SOURCES.values() for s in srcs}
    unknown_sources = sourced - set(groups)
    assert not unknown_sources, f"merged groups name unknown stage-1 groups: {sorted(unknown_sources)}"
    uncovered = set(groups) - sourced - set(CARRY_THROUGH_GROUPS)
    assert not uncovered, (
        f"stage-1 group(s) {sorted(uncovered)} feed no merged group and are not "
        f"carried through -- their survivors would vanish after stage 1"
    )

    # League priors are intentionally out of the candidate space (see above).
    assert not (set(LEAGUE_FEATURE_NAMES) & set(ALL_FEATURE_NAMES)), (
        "league priors are documented as non-candidates but appear in ALL_FEATURE_NAMES"
    )


__all__ = [
    "StatSpec", "BufferSpec", "FeatureSpec",
    "STATS", "STAT_BY_KEY", "STAT_FAMILIES", "FAMILY_GROUP_NAMES", "TARGETS", "target_spec",
    "SCORING_MODE", "scoring_hash",
    "BUFFERS", "BUFFER_NAMES", "BUFFER_INDEX",
    "FEATURES", "CANDIDATE_FEATURES", "NON_CANDIDATE_STATS",
    "PRIOR_FEATURE_NAMES", "LEAGUE_FEATURE_NAMES", "CONTEXT_FEATURES",
    "ALL_FEATURE_NAMES", "CORE_FEATURES",
    "MERGED_GROUP_SOURCES", "CARRY_THROUGH_GROUPS", "merged_groups",
    "enumerate_buffers", "enumerate_features", "feature_groups", "validate",
]
