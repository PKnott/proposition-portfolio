"""Guards on the shape of `Outputs/`.

The layout is a claim about *position*: whatever sits in the root is the current
run, and anything it superseded has been moved out from under it. That claim is
only worth making if the writers actually keep it, and the failure mode when they
do not is quiet -- two predictions workbooks in the root, `latest_form` picking
whichever name sorts last, and a day's odds entered against the wrong fixtures.

Everything below runs against a temporary root, so no test can move a real
output. `keep_newest` and friends take `root` for exactly that reason.
"""

from __future__ import annotations

import json

import pytest

from fpp import paths


@pytest.fixture
def out(tmp_path):
    """A stand-in `Outputs/` holding two runs' worth of everything."""
    root = tmp_path / "Outputs"
    root.mkdir()
    for name in (
        "predictions_2026-08-19.xlsx", "predictions_2026-08-21.xlsx",
        "odds_input_2026-08-19.xlsx", "odds_input_2026-08-21.xlsx",
        "edge_book_2026-08-19.html", "edge_book_2026-08-21.html",
        "portfolios_2026-08-19.xlsx", "split_2026-08-19.xlsx",
        "raw_odds.json", "odds_gathered.json", "_layout_check.xlsx",
        "odds_input_2026-08-19.xlsx.bak", "odds_input_2026-08-21.xlsx.bak",
        "odds_input_2026-08-19_unmatched.txt",
        "predictions_2026-08-19.json", "predictions_2026-08-21.json",
        "portfolios_2026-08-19.json", "portfolios_2026-08-21.json",
    ):
        (root / name).write_text(name)
    return root


def root_files(root):
    return sorted(p.name for p in root.iterdir() if p.is_file())


# --- archiving --------------------------------------------------------------


def test_only_the_newest_of_each_kind_stays_in_the_root(out):
    paths.tidy_outputs(out)
    assert root_files(out) == [
        "edge_book_2026-08-21.html",
        "odds_input_2026-08-21.xlsx",
        "portfolios_2026-08-19.xlsx",     # legacy kind, and the primary name of it
        "predictions_2026-08-21.xlsx",
    ]


def test_a_retired_name_does_not_outrank_the_current_one_on_a_date_tie(out):
    """`portfolios` sweeps `split_*.xlsx` too, and they share a date here.

    Sorted by filename alone, `split_` comes last and would be the one left in
    the root -- a name nothing has written since v1 presented as today's output.
    """
    paths.keep_newest("portfolios", root=out)
    assert (out / "portfolios_2026-08-19.xlsx").exists()
    assert not (out / "split_2026-08-19.xlsx").exists()


def test_the_current_run_is_the_one_named_current_not_the_one_touched_last(out):
    """Recency is the date in the name, never the mtime.

    Rebuilding an old page to pick up a template change touches the file. If that
    decided which one stayed in the root, a layout tweak would silently promote
    last week's fixtures back to being "today's".
    """
    old = out / "predictions_2026-08-19.xlsx"
    old.touch()                       # newest mtime, oldest run
    paths.keep_newest("predictions", root=out)
    assert not old.exists()
    assert (out / "predictions_2026-08-21.xlsx").exists()


def test_the_writer_names_what_it_kept(out):
    """`publish` passes the file it just wrote, so "current" is a fact."""
    moved = paths.keep_newest("predictions", keep=out / "predictions_2026-08-19.xlsx", root=out)
    assert (out / "predictions_2026-08-19.xlsx").exists()
    assert [p.name for p in moved] == ["predictions_2026-08-21.xlsx"]


def test_rebuilding_the_same_day_replaces_its_archived_copy(out):
    """Two builds of one date must not collide -- the later one wins."""
    paths.keep_newest("edge_book", root=out)
    archived = out / "Archive" / "edge_book" / "edge_book_2026-08-19.html"
    assert archived.exists()

    (out / "edge_book_2026-08-19.html").write_text("rebuilt")
    paths.keep_newest("edge_book", root=out)
    assert archived.read_text() == "rebuilt"


def test_archiving_is_idempotent(out):
    paths.tidy_outputs(out)
    before = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    paths.tidy_outputs(out)
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == before


def test_every_kind_has_somewhere_to_go(out):
    """Each glob in the table must resolve to its own archive folder."""
    paths.tidy_outputs(out)
    for kind in paths.OUTPUT_KINDS:
        assert (out / "Archive" / kind).is_dir(), f"{kind} has no archive"


def test_an_unknown_kind_is_refused():
    with pytest.raises(ValueError, match="unknown output kind"):
        paths.archive_dir("nonsense")


# --- Data/ ------------------------------------------------------------------


def test_working_files_leave_the_root(out):
    paths.tidy_outputs(out)
    names = sorted(p.name for p in (out / "Data").iterdir())
    assert "raw_odds.json" in names and "odds_gathered.json" in names
    assert "_layout_check.xlsx" in names
    assert not any(n.endswith(".bak") for n in root_files(out))
    assert not any(n.endswith("_unmatched.txt") for n in root_files(out))


def test_data_keeps_one_run_and_no_history(out):
    """The answer to "do we need the old odds JSON": no, and nothing keeps it."""
    paths.tidy_outputs(out)
    names = sorted(p.name for p in (out / "Data").iterdir())
    assert "predictions_2026-08-21.json" in names
    assert "portfolios_2026-08-21.json" in names
    assert not any("2026-08-19" in n for n in names), names


def test_nothing_from_data_is_archived(out):
    """Working files are deleted, never archived -- an archived one reads nothing."""
    paths.tidy_outputs(out)
    archived = [p.name for p in (out / "Archive").rglob("*") if p.is_file()]
    assert not any(n.endswith(".json") for n in archived)
    assert not any(n.endswith(".bak") for n in archived)


# --- the writers hold to it -------------------------------------------------


def test_publish_leaves_a_path_outside_outputs_alone(tmp_path, monkeypatch):
    """A test or a scratch build gets no housekeeping done behind its back."""
    monkeypatch.setattr(paths, "OUTPUTS", tmp_path / "Outputs")
    (tmp_path / "Outputs").mkdir()
    elsewhere = tmp_path / "scratch"
    elsewhere.mkdir()
    (elsewhere / "predictions_2026-08-19.xlsx").write_text("x")
    target = elsewhere / "predictions_2026-08-21.xlsx"
    target.write_text("y")

    paths.publish(target, "predictions")
    assert (elsewhere / "predictions_2026-08-19.xlsx").exists()
    assert not (elsewhere / "Archive").exists()


def test_json_writers_drop_the_previous_dated_copy(tmp_path):
    from fpp.report import edge_book as eb

    (tmp_path / "portfolios_2026-08-19.json").write_text("{}")
    out = tmp_path / "portfolios_2026-08-21.json"
    eb._write_json({"counts": {"exported": 0}}, out, "portfolios")
    assert out.exists()
    assert not (tmp_path / "portfolios_2026-08-19.json").exists()
    assert json.loads(out.read_text())["counts"]["exported"] == 0


def test_the_page_is_built_from_the_data_folder(tmp_path, monkeypatch):
    """`latest_json` looks in `Data/`, which is where the writers put it."""
    from fpp.report import edge_book as eb

    data = tmp_path / "Data"
    data.mkdir()
    (data / "portfolios_2026-08-21.json").write_text("{}")
    monkeypatch.setattr(paths, "OUTPUTS_DATA", data)
    assert eb.latest_json("portfolios").name == "portfolios_2026-08-21.json"
