"""Frozen artifacts: the handoff from R&D to production.

The Run notebook is cheap and dumb by design -- it never re-derives anything the
research notebooks already solved, it just loads the latest saved answer. These
are the files it loads: kilobytes of JSON, versioned, kept in the repo.

No symlinks: a ``latest ->`` symlink does not survive iCloud sync. A
``manifest.json`` pointer does.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .paths import ARTIFACTS
from .config import SCORING_MODE
from .spec import ALL_FEATURE_NAMES, TARGETS, scoring_hash

STAGES = ("window", "features", "params", "cv", "provenance")


def spec_hash() -> str:
    """Fingerprint of the feature spec *and* the scoring definition, so stale
    artifacts cannot be scored with.

    The scoring half matters as much as the feature half: an artifact stores the
    `logloss` and `baseline_logloss` it was frozen against, and those numbers are
    only meaningful under the metric that produced them. A renamed feature
    silently becomes a NaN column; a changed metric silently becomes a
    comparison between two different quantities, which is harder to spot.
    """
    return hashlib.sha256(
        ("|".join(ALL_FEATURE_NAMES) + "#" + scoring_hash()).encode()
    ).hexdigest()[:12]


def _git_sha() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() or None
    except Exception:
        return None


def default_version() -> str:
    return "v" + dt.date.today().isoformat()


def working_version() -> str:
    """The version currently being *built* -- distinct from the frozen one.

    Every research notebook must agree on this. Resolving it from the manifest's
    ``current`` instead would mean a Features run writes today's version while a
    subsequent Tuning run reads and overwrites whatever was last *frozen* --
    silently ignoring the features that were just selected.

    Override with ``FPP_VERSION`` to rebuild or amend a specific version.
    """
    return os.environ.get("FPP_VERSION") or default_version()


def orphan_versions() -> list[str]:
    """Version directories on disk that the manifest does not list."""
    known = set(read_manifest().get("versions", []))
    found: set[str] = set()
    for target_dir in ARTIFACTS.iterdir() if ARTIFACTS.exists() else []:
        if target_dir.is_dir() and target_dir.name in TARGETS:
            found |= {d.name for d in target_dir.iterdir() if d.is_dir()}
    return sorted(found - known)


def manifest_path() -> Path:
    return ARTIFACTS / "manifest.json"


def read_manifest() -> dict:
    p = manifest_path()
    return json.loads(p.read_text()) if p.exists() else {"current": None, "versions": []}


def set_current(version: str) -> None:
    m = read_manifest()
    m["current"] = version
    m.setdefault("versions", [])
    if version not in m["versions"]:
        m["versions"].append(version)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    manifest_path().write_text(json.dumps(m, indent=2))


def artifact_dir(target: str, version: str) -> Path:
    return ARTIFACTS / target / version


def save_artifact(target: str, stage: str, payload: dict, version: str | None = None) -> Path:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; expected one of {STAGES}")
    version = version or default_version()
    d = artifact_dir(target, version)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{stage}.json"
    p.write_text(json.dumps(payload, indent=2, default=str))
    return p


def load_artifact(target: str, stage: str, version: str = "current") -> dict:
    if version == "current":
        version = read_manifest().get("current") or default_version()
    p = artifact_dir(target, version) / f"{stage}.json"
    if not p.exists():
        raise FileNotFoundError(f"No {stage} artifact for {target} at {version} ({p})")
    return json.loads(p.read_text())


def write_provenance(target: str, version: str, *, n_rows: int, data_hash: str, extra: dict | None = None) -> Path:
    return save_artifact(target, "provenance", {
        "spec_hash": spec_hash(),
        # Recorded in the clear as well as inside `spec_hash`, so reading the
        # provenance file answers "which metric is this number?" without having
        # to recompute a hash to find out.
        "scoring_mode": SCORING_MODE,
        "scoring_hash": scoring_hash(),
        "git_sha": _git_sha(),
        "n_rows": int(n_rows),
        "data_hash": data_hash,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        **(extra or {}),
    }, version)


def freeze(version: str | None = None, targets: tuple[str, ...] = TARGETS) -> str:
    """Mark a version current, after checking every target family is complete."""
    version = version or default_version()
    missing = []
    for t in targets:
        for stage in ("window", "features", "params"):
            if not (artifact_dir(t, version) / f"{stage}.json").exists():
                missing.append(f"{t}/{stage}")
    if missing:
        raise FileNotFoundError(f"Cannot freeze {version}: missing {missing}")
    set_current(version)
    print(f"Froze {version} as current ({len(targets)} target families)")

    orphans = [v for v in orphan_versions() if v != version]
    if orphans:
        print(f"  note: {len(orphans)} version dir(s) on disk are not in the manifest: {orphans}")
        print("        left in place -- delete them if they are stale.")
    return version


@dataclass
class RunContext:
    """Everything production needs, loaded from disk. No computation."""

    version: str
    windows: dict[str, dict] = field(default_factory=dict)
    features: dict[str, list[str]] = field(default_factory=dict)
    params: dict[str, dict] = field(default_factory=dict)
    dispersion: dict[str, dict] = field(default_factory=dict)
    n_estimators: dict[str, int] = field(default_factory=dict)

    @classmethod
    def load(cls, version: str = "current", targets: tuple[str, ...] = TARGETS) -> "RunContext":
        if version == "current":
            version = read_manifest().get("current")
            if not version:
                raise FileNotFoundError("No current artifact version -- run freeze() first.")

        ctx = cls(version=version)
        for t in targets:
            ctx.windows[t] = load_artifact(t, "window", version)
            ctx.features[t] = load_artifact(t, "features", version)["selected"]
            p = load_artifact(t, "params", version)
            ctx.params[t] = p["params"]
            ctx.dispersion[t] = p.get("dispersion", {})
            ctx.n_estimators[t] = int(p.get("n_estimators_production", p["params"].get("n_estimators", 400)))

            # Refuse to score with an artifact frozen against a different feature
            # spec or a different scoring definition -- otherwise a renamed
            # feature silently becomes a NaN column, and a changed metric
            # silently turns every stored comparison into one between two
            # different quantities.
            try:
                prov = load_artifact(t, "provenance", version)
            except FileNotFoundError:
                continue
            if prov.get("spec_hash") and prov["spec_hash"] != spec_hash():
                raise RuntimeError(
                    f"{t}: artifact {version} was frozen against feature+scoring spec "
                    f"{prov['spec_hash']}, current spec is {spec_hash()} "
                    f"(scoring mode {SCORING_MODE!r}, scoring hash {scoring_hash()}). "
                    f"Re-run Features/Tuning."
                )
        return ctx


__all__ = [
    "STAGES", "spec_hash", "default_version", "working_version", "orphan_versions",
    "read_manifest", "set_current", "artifact_dir", "save_artifact", "load_artifact",
    "write_provenance", "freeze", "RunContext",
]
