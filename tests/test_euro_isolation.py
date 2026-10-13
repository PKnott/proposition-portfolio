"""League -> UEFA is a one-way street.

European code may read the league model; nothing in the league model may read,
import or be changed by European code. If the European experiment is abandoned,
deleting `fpp/europe/` must leave the league model exactly as it is on `main`.
These two tests are what make that a checked fact rather than an intention.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "fpp"
EUROPE = PKG / "europe"


def _imports(path: Path) -> set[str]:
    """Absolute module names a file imports, with relative imports resolved."""
    tree = ast.parse(path.read_text())
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    pkg = parts if path.name == "__init__.py" else parts[:-1]

    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def test_no_league_module_imports_europe():
    offenders = []
    for path in PKG.rglob("*.py"):
        if EUROPE in path.parents:
            continue
        hits = [m for m in _imports(path) if m == "fpp.europe" or m.startswith("fpp.europe.")]
        if hits:
            offenders.append(f"{path.relative_to(ROOT)}: {sorted(hits)}")
    assert not offenders, "league code must not depend on European code:\n" + "\n".join(offenders)


def test_league_feature_tables_unchanged(clean_table):
    from tests.domestic_fingerprint import SNAPSHOT, fingerprint

    if not SNAPSHOT.exists():
        pytest.skip("no domestic fingerprint captured -- run tests.domestic_fingerprint on main")
    want = json.loads(SNAPSHOT.read_text())
    got = fingerprint()
    if got["artifact_version"] != want["artifact_version"] or got["clean_rows"] != want["clean_rows"]:
        pytest.skip(f"fingerprint was captured on artifact {want['artifact_version']} / "
                    f"{want['clean_rows']} rows, this machine has {got['artifact_version']} / "
                    f"{got['clean_rows']} -- re-capture on main")
    assert got == want, "the league model's priors or features changed"
