"""Filesystem layout.

Two roots, deliberately separated:

* ``PROJECT_ROOT`` -- the repo. Resolved from *this file's* location, never from
  ``os.getcwd()``. The old notebooks used
  ``os.path.abspath(os.path.join(os.getcwd(), "..", ".."))``, which silently
  breaks the moment a notebook moves or is run from a different directory.

* ``CACHE_ROOT`` -- large, churny, regenerable files. This repo lives inside
  iCloud Drive, where writing hundreds of MB of grid checkpoints causes constant
  upload churn, and "Optimise Mac Storage" can evict files to dataless stubs --
  turning a checkpoint read into a multi-second network stall in the middle of a
  grid search. So search checkpoints and the cleaned parquet go *outside* iCloud.
  Override with the ``FPP_CACHE_DIR`` environment variable.

Only the kilobyte-scale frozen artifacts (``artifacts/``) live in the repo.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from .config import SCORING_MODE

# fpp/paths.py  ->  fpp/  ->  project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUTS = PROJECT_ROOT / "Inputs"
OUTPUTS = PROJECT_ROOT / "Outputs"
ARTIFACTS = PROJECT_ROOT / "artifacts"

# --- Inputs ---------------------------------------------------------------
FIXTURES_DIR = INPUTS / "Fixtures"
MAPPING_DIR = INPUTS / "Mapping"
STATS_DIR = INPUTS / "Stats"
SOCCERDATA_CACHE = INPUTS / "soccerdata_cache"

CLUB_MAPPING_ALL = MAPPING_DIR / "Club_Mapping_All.csv"
ESPN_TEAM_IDS = MAPPING_DIR / "ESPN_Team_IDs.csv"
ESPN_MATCH_STATS = STATS_DIR / "espn_match_stats.csv"

HISTORIC_ALL_COMP = FIXTURES_DIR / "historic_competitions_fixtures.csv"


def all_comp_fixtures(season_label: str) -> Path:
    """``Inputs/Fixtures/all_competitions_fixtures_2025-26.csv``."""
    return FIXTURES_DIR / f"all_competitions_fixtures_{season_label}.csv"


# --- Cache (outside iCloud) ----------------------------------------------
def _default_cache_root() -> Path:
    env = os.environ.get("FPP_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.home() / ".cache" / "football_prediction"


CACHE_ROOT = _default_cache_root()
CLEAN_CACHE = CACHE_ROOT / "clean"
SEARCH_CACHE = CACHE_ROOT / "search"


def search_checkpoint(target: str, stage: str, mode: str = SCORING_MODE,
                     fingerprint: str | None = None) -> Path:
    """``<cache>/search/goals__window_grid__team__a1b2c3d4.csv``.

    The mode segment is what makes per-team and match-total scores physically
    incapable of sharing a file. Stronger than tagging rows: a whole file becomes
    the wrong vintage rather than individual rows silently matching a `point_id`.
    It defaults to the configured mode so callers cannot forget it -- the guard
    should not depend on anyone remembering to pass an argument.

    ``fingerprint`` is the same discipline applied to everything else a cached
    score depends on and `point_id` does not encode. The window grid keys its
    points on ``(L, alpha)`` alone and the hyperparameter descent on the axes it
    is sweeping -- neither says anything about *which features were in the model*
    or *which seasons were the folds*. So changing the selected feature set, or
    withholding a season from tuning, left every cached row matching by
    ``point_id`` and silently reusable. That is exactly what made a rerun after
    the internals changed return the old answer.

    Build one with `search.fingerprint_for`, which decides what belongs in it.
    """
    stem = f"{target}__{stage}__{mode}"
    if fingerprint:
        stem = f"{stem}__{fingerprint}"
    return SEARCH_CACHE / f"{stem}.csv"


# --- Outputs --------------------------------------------------------------
#
# Three files you open, and everything else in a folder.
#
# `Outputs/` accumulated a fortnight of dated workbooks, their `.bak` copies,
# two kinds of scratch JSON and the leftovers of five v1 league notebooks -- at
# which point "which predictions workbook is the current one" became a question
# you answered by reading dates off a listing. It is now answered by *position*:
# the current run sits in the root and everything it supersedes is moved into
# `Archive/<kind>/` by the writer that superseded it.
#
#   Outputs/
#     predictions_<date>.xlsx     the read-only view
#     odds_input_<date>.xlsx      the form to fill
#     edge_book_<date>.html       the page
#     Data/                       what the pipeline reads and writes, not you
#     Archive/<kind>/             every superseded copy
#     Evaluation/                 unchanged; written by 04_Evaluation
#
# `Data/` holds no history on purpose. The two JSON payloads rebuild from the
# workbook and the form, the odds-gathering files are consumed within one run,
# and yesterday's page is already archived whole -- so a dated copy of any of
# them is a file nothing would ever read.

OUTPUTS_DATA = OUTPUTS / "Data"
OUTPUTS_ARCHIVE = OUTPUTS / "Archive"
EVALUATION = OUTPUTS / "Evaluation"

# The ledger -- the one store here that history is the whole point of, and the
# only thing in the project that cannot be rebuilt from source. A model, a clean
# table and both payloads all regenerate; a result nobody captured is gone.
#
# It sits outside `Data/` and outside the root deliberately. `keep_newest`,
# `stow_working` and `prune_data` all sweep one of those two, and `_write_json`
# deletes every `Data/portfolios_*.json` that is not today's -- so anything the
# ledger needs to keep would be housekeeping's to delete. Nothing below reaches
# in here.
ANALYSIS = OUTPUTS / "Analysis"
ANALYSIS_PENDING = ANALYSIS / "Pending"     # 06 fills it, 07 drains it
ANALYSIS_REPORTS = ANALYSIS / "Reports"     # the CSVs and figures 07 writes

# kind -> the globs belonging to it, relative to the `Outputs` root. One
# definition: the writers archive against it, `tidy_outputs` sweeps it, and
# `tests/test_outputs_layout.py` walks it rather than restating the names.
OUTPUT_KINDS: dict[str, tuple[str, ...]] = {
    "predictions": ("predictions_*.xlsx",),
    "odds": ("odds_input_*.xlsx",),
    "portfolios": ("portfolios_*.xlsx", "split_*.xlsx"),
    "edge_book": ("edge_book_*.html",),
}

# Files that are working state rather than output: they belong in `Data/`
# wherever they turn up, and are never archived because nothing reads an old one.
WORKING_FILES: tuple[str, ...] = (
    "raw_odds.json", "odds_gathered.json", "_layout_check.xlsx",
    "*.xlsx.bak", "*_unmatched.txt", "predictions_*.json", "portfolios_*.json",
)

_DATE_IN_NAME = re.compile(r"(\d{4}-\d{2}-\d{2})")


def evaluation_dir(target: str) -> Path:
    return OUTPUTS / "Evaluation" / target


def predictions_workbook(date_label: str) -> Path:
    return OUTPUTS / f"predictions_{date_label}.xlsx"


def archive_dir(kind: str) -> Path:
    """`Outputs/Archive/<kind>/`, created on demand."""
    if kind not in OUTPUT_KINDS:
        raise ValueError(f"unknown output kind {kind!r}; expected one of {list(OUTPUT_KINDS)}")
    d = OUTPUTS_ARCHIVE / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def _recency(p: Path) -> tuple[str, str]:
    """Sort key: the date *in the name*, then the name.

    Not the mtime. Rebuilding an old page to pick up a template change touches
    the file without making it current, and this is what decides which one gets
    to stay in the root.
    """
    m = _DATE_IN_NAME.search(p.name)
    return (m.group(1) if m else "", p.name)


def _relocate(src: Path, dest_dir: Path) -> Path:
    """Move `src` into `dest_dir`, replacing any same-named file already there.

    Replacing is right: a same-named file in the archive is an earlier build of
    the same day, and the one arriving is the later one. `os.replace` rather
    than `shutil.move` because move onto an existing path raises.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    os.replace(src, dest)
    return dest


def keep_newest(kind: str, keep: Path | None = None, root: Path | None = None) -> list[Path]:
    """Move every file of `kind` out of the root except `keep`. Returns what moved.

    `keep` is passed by the writer that just wrote it, so "current" is a fact
    rather than an inference. Called without one -- a manual tidy -- it falls
    back to the newest date in a name, and on a tie to the kind's *primary*
    glob: `portfolios` also sweeps the retired `split_*.xlsx` name, and on a
    same-date tie a plain sort by filename let the retired one outrank the
    current one and stay in the root.
    """
    root = Path(root) if root else OUTPUTS
    rank: dict[Path, int] = {}
    for i, g in enumerate(OUTPUT_KINDS[kind]):
        for f in root.glob(g):
            rank[f] = min(rank.get(f, i), i)
    files = sorted(rank, key=lambda f: (_recency(f)[0], -rank[f], f.name))
    if keep is None:
        if not files:
            return []
        keep = files[-1]
    keep = Path(keep).resolve()
    dest = OUTPUTS_ARCHIVE / kind if root == OUTPUTS else root / "Archive" / kind
    return [_relocate(f, dest) for f in files if f.resolve() != keep]


def stow_working(root: Path | None = None) -> list[Path]:
    """Move working files into `Data/`. Returns what moved."""
    root = Path(root) if root else OUTPUTS
    data = (OUTPUTS_DATA if root == OUTPUTS else root / "Data")
    moved = []
    for pattern in WORKING_FILES:
        for f in root.glob(pattern):
            if f.is_file():
                moved.append(_relocate(f, data))
    return moved


def prune_data(root: Path | None = None) -> list[Path]:
    """Delete every dated file in `Data/` except the newest of its kind.

    Deleting, not archiving. These are the odds-gathering scratch, the form's
    `.bak`, the unmatched list and the two JSON payloads -- all either consumed
    within a single run or rebuildable in seconds from the workbook and the form
    that are archived alongside. Keeping dated copies of them was the single
    biggest contributor to the mess this layout exists to fix.
    """
    data = OUTPUTS_DATA if (root is None or Path(root) == OUTPUTS) else Path(root) / "Data"
    if not data.is_dir():
        return []
    dated = [(m.group(1), f) for f in data.iterdir()
             if f.is_file() and (m := _DATE_IN_NAME.search(f.name))]
    if not dated:
        return []
    # One date, not one of each family. Yesterday's unmatched list surviving
    # because today's fill happened to produce none reads exactly like today's,
    # which is worse than not having it.
    current = max(d for d, _ in dated)
    dropped = [f for d, f in dated if d != current]
    for f in dropped:
        f.unlink()
    return dropped


def tidy_outputs(root: Path | None = None) -> dict[str, list[Path]]:
    """Put the whole `Outputs` root back into shape. Safe to call repeatedly."""
    root = Path(root) if root else OUTPUTS
    moved = {kind: keep_newest(kind, root=root) for kind in OUTPUT_KINDS}
    moved["working"] = stow_working(root)
    moved["pruned"] = prune_data(root)
    return moved


def publish(path: Path, kind: str) -> Path:
    """Archive whatever `path` supersedes, if it landed in the `Outputs` root.

    A caller who passed an explicit `out_path` elsewhere -- a test, a scratch
    build -- gets no filesystem housekeeping done behind its back.
    """
    path = Path(path)
    if path.parent.resolve() == OUTPUTS.resolve():
        keep_newest(kind, keep=path)
    return path


def ensure_dirs() -> None:
    """Create every directory the pipeline writes to. Safe to call repeatedly."""
    for d in (
        INPUTS,
        OUTPUTS,
        OUTPUTS_DATA,
        OUTPUTS_ARCHIVE,
        ANALYSIS,
        ANALYSIS_PENDING,
        ANALYSIS_REPORTS,
        ARTIFACTS,
        FIXTURES_DIR,
        MAPPING_DIR,
        STATS_DIR,
        SOCCERDATA_CACHE,
        CACHE_ROOT,
        CLEAN_CACHE,
        SEARCH_CACHE,
    ):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT",
    "INPUTS",
    "OUTPUTS",
    "ANALYSIS",
    "ANALYSIS_PENDING",
    "ANALYSIS_REPORTS",
    "ARTIFACTS",
    "FIXTURES_DIR",
    "MAPPING_DIR",
    "STATS_DIR",
    "SOCCERDATA_CACHE",
    "CLUB_MAPPING_ALL",
    "ESPN_TEAM_IDS",
    "ESPN_MATCH_STATS",
    "HISTORIC_ALL_COMP",
    "CACHE_ROOT",
    "CLEAN_CACHE",
    "SEARCH_CACHE",
    "all_comp_fixtures",
    "search_checkpoint",
    "OUTPUTS_DATA",
    "OUTPUTS_ARCHIVE",
    "EVALUATION",
    "OUTPUT_KINDS",
    "WORKING_FILES",
    "evaluation_dir",
    "predictions_workbook",
    "archive_dir",
    "keep_newest",
    "stow_working",
    "prune_data",
    "tidy_outputs",
    "publish",
    "ensure_dirs",
]
