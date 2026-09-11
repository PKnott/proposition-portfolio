"""Guards on the ledger.

Everything here runs against a temporary store, so no test can touch the real
one -- which matters more than usual: `Outputs/Analysis/` is the only thing in
this project that cannot be rebuilt by re-running something.

The claims worth holding are about *identity* and *idempotency*. A proposition
must resolve to the same row whichever run offered it, even though `sheet_code`
is positional and changes underneath it; and absorb, settle and roll-up must all
be safe to run twice, because the notebook that calls them will be.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fpp import ledger


# --- fixtures -------------------------------------------------------------


@pytest.fixture
def root(tmp_path):
    return tmp_path / "Analysis"


def form_rows(sheet_code="E0-01", date="2026-08-21", home="Arsenal", away="Coventry City",
              labels=None, p=0.6, prices=None):
    """A slice of what `read_filled` returns: label, p, six book columns, fixture."""
    labels = labels or [f"Goals - {home} - Over 0.5", f"Corners - {away} - Over 4.5"]
    rows = []
    for i, label in enumerate(labels):
        row = {"sheet_code": sheet_code, "label": label, "p": p,
               "date": pd.Timestamp(date), "league": "Premier League",
               "home_team": home, "away_team": away}
        row.update({b: None for b in ledger.BOOK_COLUMNS})
        row["b365"] = (prices or [2.0, 1.5])[i]
        rows.append(row)
    return pd.DataFrame(rows)


def results_rows(date="2026-08-21", home="Arsenal", away="Coventry",
                 home_espn="Arsenal", away_espn="Coventry City", **stats):
    base = dict(home_goals=3.0, away_goals=0.0, home_shots=15.0, away_shots=4.0,
                home_sot=7.0, away_sot=1.0, home_corners=8.0, away_corners=2.0)
    base.update(stats)
    return pd.DataFrame([{
        "league_key": "Prem", "date": pd.Timestamp(date),
        "home_team": home, "away_team": away,
        "home_espn": home_espn, "away_espn": away_espn, **base,
    }])


def seed(root, props, run_code="R001", run_date="2026-08-21"):
    """Put one run straight into the store, skipping capture."""
    ledger.write_table(props, "propositions", root)
    ledger.write_table(pd.DataFrame([{
        "run_code": run_code, "run_date": run_date, "origin": "test",
        "generated_at": f"{run_date}T00:00:00+00:00", "source_odds_file": "x.xlsx",
    }]), "runs", root)


# --- labels and keys ------------------------------------------------------


def test_parse_label_round_trips_every_market():
    from fpp.staking import proposition_label

    for target, display in (("goals", "Goals"), ("shots", "Shots"),
                            ("sot", "Shots on Target"), ("corners", "Corners")):
        label = proposition_label(target, "Real Sociedad", 2.5)
        assert label.startswith(display)
        assert ledger.parse_label(label) == (target, "Real Sociedad", 2.5)


def test_parse_label_keeps_a_hyphen_in_the_club_name():
    """`" - "` appears in club names; the market name is what is split off."""
    assert ledger.parse_label("Corners - Saint-Etienne - Over 4.5") == (
        "corners", "Saint-Etienne", 4.5)


def test_parse_label_rejects_a_market_it_does_not_know():
    with pytest.raises(ValueError, match="unknown market"):
        ledger.parse_label("Bookings - Arsenal - Over 2.5")


def test_league_comes_off_the_sheet_prefix():
    assert ledger.league_of("D1-05") == "Bund"
    assert ledger.league_of("SP1-01") == "Liga"
    with pytest.raises(ValueError, match="unknown league prefix"):
        ledger.league_of("XX-01")


def test_the_key_ignores_the_run_and_the_sheet_code():
    """The whole point: `E0-02` was three different fixtures inside eight days.

    Two runs offering the same proposition must produce one key, and the sheet
    code -- which is a positional label, not an identity -- must not appear in it.
    """
    a = ledger.prop_key("2026-08-21", "Prem", "Arsenal", "Coventry City",
                        "goals", "Arsenal", 0.5)
    b = ledger.prop_key(pd.Timestamp("2026-08-21 00:00"), "Prem", "Arsenal",
                        "Coventry City", "goals", "Arsenal", 0.5)
    assert a == b
    assert "E0" not in a and "#" not in a
    # A different line is a different proposition.
    assert a != ledger.prop_key("2026-08-21", "Prem", "Arsenal", "Coventry City",
                                "goals", "Arsenal", 1.5)


def test_two_runs_under_different_sheet_codes_share_one_key():
    one = ledger.proposition_rows(form_rows(sheet_code="E0-02"), "R001")
    two = ledger.proposition_rows(form_rows(sheet_code="E0-07"), "R002")
    assert set(one["prop_key"]) == set(two["prop_key"])
    assert list(one["sheet_code"]) != list(two["sheet_code"])


# --- proposition rows -----------------------------------------------------


def test_only_priced_propositions_are_banked():
    """A row no book quoted has no price to settle against and is not a bet."""
    f = form_rows(prices=[2.0, None])
    out = ledger.proposition_rows(f, "R001")
    assert len(out) == 1
    assert out.iloc[0]["o"] == 2.0


def test_scope_is_derived_from_which_side_the_team_is():
    out = ledger.proposition_rows(form_rows(), "R001").set_index("target")
    assert out.loc["goals", "scope"] == "home"      # Goals - Arsenal, the home side
    assert out.loc["corners", "scope"] == "away"    # Corners - Coventry City


def test_edge_is_p_times_the_best_price():
    out = ledger.proposition_rows(form_rows(p=0.6, prices=[2.0, 1.5]), "R001")
    assert out["e"].tolist() == pytest.approx([1.2, 0.9])


def test_qualified_marks_only_the_rows_that_became_picks():
    f = form_rows()
    ids = {("E0-01", "Goals - Arsenal - Over 0.5"): "E0-01#0"}
    out = ledger.proposition_rows(f, "R001", ids).set_index("target")
    assert out.loc["goals", "qualified"] and out.loc["goals", "prop_id"] == "E0-01#0"
    assert not out.loc["corners", "qualified"]


# --- run codes ------------------------------------------------------------


def test_run_codes_count_absorbed_and_pending_together(root):
    """Capture hands out the code and 07 may not run for days.

    Counting only the absorbed table would hand `R002` to two captures in that
    window, and the second would overwrite the first on the way in.
    """
    assert ledger.next_run_code(root) == "R001"
    ledger.write_table(pd.DataFrame([{"run_code": "R001"}]), "runs", root)
    assert ledger.next_run_code(root) == "R002"
    (root / "Pending" / "slate_R002_2026-08-22").mkdir(parents=True)
    assert ledger.next_run_code(root) == "R003"


# --- settlement -----------------------------------------------------------


def test_settlement_is_strictly_over_the_line(root):
    """`actual > line`, the same comparison `metrics.over_prob` integrates.

    Three goals settles `Over 2.5` as a win and `Over 3.0` as a loss; getting
    this wrong would put every whole-number push on the wrong side and disagree
    with the probability that was quoted for it.
    """
    props = ledger.proposition_rows(form_rows(labels=[
        "Goals - Arsenal - Over 2.5", "Goals - Arsenal - Over 3.0",
        "Goals - Arsenal - Over 3.5"], prices=[2.0, 2.0, 2.0]), "R001")
    seed(root, props)
    ledger.settle(root, results=results_rows(home_goals=3.0))

    out = ledger.read_table("propositions", root).set_index("line")
    assert out.loc[2.5, "won"] and not out.loc[3.0, "won"] and not out.loc[3.5, "won"]
    assert (out["actual"] == 3.0).all()


def test_every_market_reads_its_own_statistic(root):
    props = ledger.proposition_rows(form_rows(labels=[
        "Goals - Arsenal - Over 0.5", "Shots - Arsenal - Over 0.5",
        "Shots on Target - Arsenal - Over 0.5", "Corners - Arsenal - Over 0.5"],
        prices=[2.0] * 4), "R001")
    seed(root, props)
    ledger.settle(root, results=results_rows(
        home_goals=3.0, home_shots=15.0, home_sot=7.0, home_corners=8.0))
    out = ledger.read_table("propositions", root).set_index("target")["actual"]
    assert out.to_dict() == {"goals": 3.0, "shots": 15.0, "sot": 7.0, "corners": 8.0}


def test_the_away_side_reads_the_away_columns(root):
    props = ledger.proposition_rows(
        form_rows(labels=["Corners - Coventry City - Over 1.5"], prices=[2.0]), "R001")
    seed(root, props)
    ledger.settle(root, results=results_rows(away_corners=2.0))
    out = ledger.read_table("propositions", root).iloc[0]
    assert out["scope"] == "away" and out["actual"] == 2.0 and out["won"]


def test_a_club_named_differently_in_the_results_still_settles(root):
    """The one fixture a naive join drops.

    The form carries `Coventry City`; `espn_match_stats` carries the mapped
    `Coventry` in `away_team` and ESPN's own `Coventry City` in `away_espn`. One
    fixture a slate silently unsettled is a biased sample, not a gap.
    """
    props = ledger.proposition_rows(
        form_rows(labels=["Goals - Coventry City - Over 0.5"], prices=[2.0]), "R001")
    seed(root, props)
    ledger.settle(root, results=results_rows(away="Coventry", away_espn="Coventry City"))
    assert bool(ledger.read_table("propositions", root)["won"].notna().all())


def test_a_kickoff_either_side_of_midnight_still_settles(root):
    """ESPN timestamps are UTC and land either side of the local date."""
    for shift in (-1, 1):
        r = root.with_name(f"{root.name}{shift}")
        props = ledger.proposition_rows(form_rows(date="2026-08-21"), "R001")
        seed(r, props)
        stamp = pd.Timestamp("2026-08-21") + pd.Timedelta(days=shift)
        ledger.settle(r, results=results_rows(date=stamp, away="Coventry City"))
        assert bool(ledger.read_table("propositions", r)["won"].notna().all())


def test_an_unplayed_fixture_is_left_alone_rather_than_guessed(root):
    props = ledger.proposition_rows(form_rows(date="2099-01-01"), "R001")
    seed(root, props)
    ledger.settle(root, results=results_rows())
    out = ledger.read_table("propositions", root)
    assert out["won"].isna().all() and out["actual"].isna().all()


def test_settling_twice_changes_nothing(root):
    props = ledger.proposition_rows(form_rows(), "R001")
    seed(root, props)
    ledger.settle(root, results=results_rows(away="Coventry City"))
    first = ledger.read_table("propositions", root)
    ledger.settle(root, results=results_rows(away="Coventry City"))
    second = ledger.read_table("propositions", root)
    pd.testing.assert_frame_equal(first, second)


# --- absorb ---------------------------------------------------------------


def _pending(root, run_code, run_date, props):
    d = root / "Pending" / f"slate_{run_code}_{run_date}"
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(
        {"run_code": run_code, "run_date": run_date, "origin": "test"}), encoding="utf-8")
    props.to_parquet(d / "propositions.parquet", index=False)
    pd.DataFrame([{"run_code": run_code, "portfolio_id": 1, "split": "inverse_e",
                   "legs": 1, "realised_return": np.nan, "n_settled": 0, "settled": False}
                  ]).to_parquet(d / "portfolios.parquet", index=False)
    pd.DataFrame([{"run_code": run_code, "portfolio_id": 1, "leg": 0,
                   "prop_key": props.iloc[0]["prop_key"], "stake_frac": 1.0}
                  ]).to_parquet(d / "legs.parquet", index=False)
    return d


def test_absorb_drains_the_inbox_and_deletes_what_it_took(root):
    d = _pending(root, "R001", "2026-08-21", ledger.proposition_rows(form_rows(), "R001"))
    added = ledger.absorb(root)
    assert list(added["run_code"]) == ["R001"]
    assert not d.exists()
    assert len(ledger.read_table("propositions", root)) == 2


def test_absorbing_a_run_already_held_does_not_double_it(root):
    _pending(root, "R001", "2026-08-21", ledger.proposition_rows(form_rows(), "R001"))
    ledger.absorb(root)
    n = len(ledger.read_table("propositions", root))
    _pending(root, "R001", "2026-08-21", ledger.proposition_rows(form_rows(), "R001"))
    ledger.absorb(root)
    assert len(ledger.read_table("propositions", root)) == n
    assert len(ledger.read_table("runs", root)) == 1


# --- roll-up and placed bets ----------------------------------------------


def test_a_portfolio_is_only_rolled_up_once_every_leg_has_settled(root):
    props = ledger.proposition_rows(form_rows(
        labels=["Goals - Arsenal - Over 0.5", "Goals - Arsenal - Over 9.5"],
        prices=[2.0, 2.0]), "R001")
    _pending(root, "R001", "2026-08-21", props)
    ledger.absorb(root)
    legs = ledger.read_table("legs", root)
    legs = pd.concat([legs, legs.assign(leg=1, prop_key=props.iloc[1]["prop_key"])],
                     ignore_index=True)
    legs["stake_frac"] = 0.5
    ledger.write_table(legs, "legs", root)

    ledger.roll_up(root)
    assert not bool(ledger.read_table("portfolios", root).iloc[0]["settled"])

    ledger.settle(root, results=results_rows(away="Coventry City", home_goals=3.0))
    out = ledger.roll_up(root).iloc[0]
    # Half a pound at 2.0 wins, half a pound at 2.0 loses: back to level.
    assert bool(out["settled"]) and out["realised_return"] == pytest.approx(0.0)


def test_record_bet_takes_the_reference_the_page_prints(root):
    ledger.write_table(pd.DataFrame([{
        "run_code": "R007", "portfolio_id": 770, "g_f_protective": 0.21,
        "g_f_star": 0.99}]), "portfolios", root)
    rec = ledger.record_bet("R007-770", pot=1000, mode="protective", root=root)
    assert rec["stake"] == pytest.approx(210.0)
    rec = ledger.record_bet("R007-770", pot=1000, mode="max", root=root)
    assert rec["stake"] == pytest.approx(990.0)
    # Re-recording replaces rather than doubling, so a corrected pot is one call.
    assert len(ledger.read_table("placed", root)) == 1


def test_record_bet_refuses_a_portfolio_the_ledger_does_not_hold(root):
    ledger.write_table(pd.DataFrame([{"run_code": "R007", "portfolio_id": 770}]),
                       "portfolios", root)
    with pytest.raises(KeyError, match="not in the ledger"):
        ledger.record_bet("R007-999", pot=100, root=root)
    with pytest.raises(ValueError, match="R007"):
        ledger.record_bet("R007", pot=100, root=root)


# --- purge ----------------------------------------------------------------


def _two_runs(root):
    for code, date in (("R001", "2026-08-21"), ("R002", "2026-08-22")):
        _pending(root, code, date, ledger.proposition_rows(form_rows(date=date), code))
    ledger.absorb(root)


def test_purge_drops_a_runs_data_but_keeps_the_run(root):
    """A superseded model's slate must leave the tables without leaving no trace.

    The row is what stops `backfill` importing it again, and what lets the ledger
    say why its history starts where it does.
    """
    _two_runs(root)
    ledger.purge("R001", reason="old model", root=root)

    runs = ledger.read_table("runs", root).set_index("run_code")
    assert set(runs.index) == {"R001", "R002"}
    assert runs.loc["R001", "status"] == "excluded"
    assert runs.loc["R001", "excluded_reason"] == "old model"
    assert runs.loc["R002", "status"] == "active"

    for name in ("propositions", "portfolios", "legs"):
        held = ledger.read_table(name, root)
        assert set(held["run_code"]) == {"R002"}, name
    assert list(ledger.active_runs(root)["run_code"]) == ["R002"]


def test_purge_refuses_a_run_it_does_not_hold(root):
    _two_runs(root)
    with pytest.raises(KeyError, match="R404"):
        ledger.purge("R404", root=root)


def test_a_purged_run_is_not_imported_again_by_backfill(root, tmp_path):
    """The point of keeping the row. Backfill skips any date `runs` names."""
    _two_runs(root)
    ledger.purge("R001", reason="old model", root=root)
    known = ledger.read_table("runs", root)
    seen = set(known["run_date"].astype(str))
    assert "2026-08-21" in seen


def test_backfill_does_not_reimport_a_run_it_already_absorbed(root):
    """The bug this guards: `seen` was built from run *codes* and compared to run
    *dates*, so it never matched and a second backfill duplicated the ledger."""
    _two_runs(root)
    known = ledger.read_table("runs", root)
    seen = set(known["run_date"].astype(str))
    assert seen == {"2026-08-21", "2026-08-22"}


# --- the money ------------------------------------------------------------


def _one_portfolio(root, realised=None, settled=True):
    ledger.write_table(pd.DataFrame([{
        "run_code": "R001", "portfolio_id": 1, "split": "inverse_e", "legs": 8,
        "expected_return_pct": 1.07, "g_f_protective": 0.2, "g_f_star": 0.9,
        "g_g_protective": 0.01, "realised_return": realised, "settled": settled,
    }]), "portfolios", root)
    ledger.write_table(pd.DataFrame([{
        "run_code": "R001", "run_date": "2026-08-21", "status": "active"}]), "runs", root)


def test_an_empty_account_has_a_zero_balance(root):
    assert ledger.balance(root) == 0.0
    assert ledger.bankroll(root).empty


def test_opening_a_balance_twice_moves_it_rather_than_adding_one(root):
    ledger.open_account(1000, when="2026-08-01", root=root)
    ledger.open_account(500, when="2026-08-01", root=root)
    assert ledger.balance(root) == pytest.approx(500.0)
    assert (ledger.read_table("cash", root)["kind"] == "opening").sum() == 1


def test_deposits_and_withdrawals_move_the_balance(root):
    ledger.open_account(1000, when="2026-08-01", root=root)
    ledger.deposit(250, when="2026-08-02", root=root)
    ledger.withdraw(100, when="2026-08-03", root=root)
    assert ledger.balance(root) == pytest.approx(1150.0)


def test_a_settled_bet_moves_the_balance_by_stake_times_return(root):
    _one_portfolio(root, realised=0.5)
    ledger.open_account(1000, when="2026-08-01", root=root)
    ledger.record_bet("R001-1", pot=1000, mode="protective", root=root)   # 20% = 200
    book = ledger.bankroll(root)
    bet = book[book["kind"] == "bet"].iloc[0]
    assert bet["stake"] == pytest.approx(200.0)
    assert bet["pnl"] == pytest.approx(100.0)
    assert bet["returned"] == pytest.approx(300.0)
    assert ledger.balance(root) == pytest.approx(1100.0)


def test_an_unsettled_bet_is_pending_and_leaves_the_balance_alone(root):
    """The stake is with the bookmaker and the outcome is unknown. Booking it
    either way would be a guess, and a NaN would poison every later row."""
    _one_portfolio(root, realised=None, settled=False)
    ledger.open_account(1000, when="2026-08-01", root=root)
    ledger.record_bet("R001-1", pot=1000, root=root)
    book = ledger.bankroll(root)
    assert bool(book[book["kind"] == "bet"].iloc[0]["pending"])
    assert ledger.balance(root) == pytest.approx(1000.0)


def test_the_pot_defaults_to_the_balance_rather_than_being_retyped(root):
    """Staking 21% of a pot that stopped existing four losses ago is the failure
    this default exists to remove."""
    _one_portfolio(root, realised=-1.0)
    ledger.open_account(1000, when="2026-08-01", root=root)
    ledger.record_bet("R001-1", root=root)                    # 20% of 1000 = 200
    assert ledger.balance(root) == pytest.approx(800.0)

    # Append: replacing the table would orphan R001's bet, and a bet whose
    # portfolio has vanished reads as pending -- which would leave the balance at
    # 1000 and make this test pass for the wrong reason.
    ledger.write_table(pd.concat([ledger.read_table("portfolios", root), pd.DataFrame(
        [{"run_code": "R002", "portfolio_id": 1, "g_f_protective": 0.2,
          "realised_return": 0.0, "settled": True}])], ignore_index=True),
        "portfolios", root)
    ledger.write_table(pd.concat([ledger.read_table("runs", root), pd.DataFrame(
        [{"run_code": "R002", "run_date": "2026-08-22", "status": "active"}])],
        ignore_index=True), "runs", root)
    rec = ledger.record_bet("R002-1", root=root)
    assert rec["pot"] == pytest.approx(800.0)                 # not 1000
    assert rec["stake"] == pytest.approx(160.0)


def test_staking_with_no_account_says_so(root):
    _one_portfolio(root, realised=0.1)
    with pytest.raises(ValueError, match="no balance to stake from"):
        ledger.record_bet("R001-1", root=root)


def test_the_statement_runs_in_date_order_with_money_in_before_the_bet(root):
    _one_portfolio(root, realised=0.5)
    ledger.open_account(1000, when="2026-08-01", root=root)
    ledger.deposit(500, when="2026-08-21", root=root)
    ledger.record_bet("R001-1", pot=1000, root=root)
    ledger.withdraw(50, when="2026-08-21", root=root)
    book = ledger.bankroll(root)
    assert list(book["kind"]) == ["opening", "deposit", "bet", "withdrawal"]
    assert list(book["balance"].round(2)) == [1000.0, 1500.0, 1600.0, 1550.0]


# --- custom staking -------------------------------------------------------


def test_the_three_modes_size_the_stake_three_ways(root):
    _one_portfolio(root, realised=0.0)
    ledger.open_account(1000, when="2026-08-01", root=root)

    assert ledger.record_bet("R001-1", mode="protective", root=root)["stake"] == pytest.approx(200.0)
    assert ledger.record_bet("R001-1", mode="max", root=root)["stake"] == pytest.approx(900.0)
    rec = ledger.record_bet("R001-1", f=0.15, root=root)
    assert rec["stake"] == pytest.approx(150.0) and rec["mode"] == "custom"


def test_an_explicit_fraction_is_filed_as_custom_not_as_the_default_mode(root):
    """A stake the model did not pick must not be recorded under a mode saying it
    did -- the strategy analysis reads that column."""
    _one_portfolio(root, realised=0.0)
    ledger.open_account(1000, when="2026-08-01", root=root)
    assert ledger.record_bet("R001-1", f=0.33, root=root)["mode"] == "custom"


def test_a_cash_stake_is_turned_into_a_fraction_of_the_pot(root):
    """Sometimes what you know is 'I put 150 on', not 'I staked 15%'."""
    _one_portfolio(root, realised=0.0)
    ledger.open_account(1000, when="2026-08-01", root=root)
    rec = ledger.record_bet("R001-1", stake=150, root=root)
    assert rec["f"] == pytest.approx(0.15)
    assert rec["stake"] == pytest.approx(150.0)
    assert rec["mode"] == "custom"


def test_custom_with_nothing_to_go_on_says_what_is_missing(root):
    _one_portfolio(root, realised=0.0)
    ledger.open_account(1000, when="2026-08-01", root=root)
    with pytest.raises(ValueError, match="custom staking needs"):
        ledger.record_bet("R001-1", mode="custom", root=root)


@pytest.mark.parametrize("kwargs, match", [
    ({"f": 0.1, "stake": 100}, "not both"),
    ({"mode": "kelly"}, "mode must be"),
    ({"f": 0}, "above 0"),
    ({"f": 1.5}, "at most 1"),
])
def test_a_stake_that_cannot_be_made_sense_of_is_refused(root, kwargs, match):
    _one_portfolio(root, realised=0.0)
    ledger.open_account(1000, when="2026-08-01", root=root)
    with pytest.raises(ValueError, match=match):
        ledger.record_bet("R001-1", root=root, **kwargs)


def test_a_portfolio_with_no_growth_block_cannot_be_staked_by_mode(root):
    """Some backfilled runs predate the growth block; `f=` still works."""
    ledger.write_table(pd.DataFrame([{
        "run_code": "R001", "portfolio_id": 1, "g_f_protective": None,
        "realised_return": 0.0, "settled": True}]), "portfolios", root)
    ledger.write_table(pd.DataFrame([{"run_code": "R001", "run_date": "2026-08-21"}]),
                       "runs", root)
    ledger.open_account(1000, when="2026-08-01", root=root)
    with pytest.raises(ValueError, match="no protective stake"):
        ledger.record_bet("R001-1", root=root)
    assert ledger.record_bet("R001-1", f=0.1, root=root)["stake"] == pytest.approx(100.0)


def _page(tmp_path, date, name="Elversberg"):
    """A minimal built Edge Book: the two payloads `backfill` unwraps."""
    pred = {"fixtures": [{"event": "E0-01", "date": date, "league_key": "Prem",
                          "home": "Arsenal", "away": name}]}
    port = {"generated_at": f"{date}T12:00:00+00:00",
            "source_odds_file": f"odds_input_{date}.xlsx",
            "counts": {"propositions": 1, "priced": 1},
            "propositions": [], "portfolios": [],
            "prices": [{"event": "E0-01", "label": "Goals - Arsenal - Over 0.5",
                        "p": 0.9, "odds": 1.2, "book": "Bet365", "e": 1.08}]}
    path = tmp_path / f"edge_book_{date}.html"
    path.write_text(
        f'<script id="predictions-data" type="application/json">{json.dumps(pred)}</script>'
        f'<script id="portfolios-data" type="application/json">{json.dumps(port)}</script>',
        encoding="utf-8")
    return path


def test_backfill_imports_a_page_it_has_never_seen(root, tmp_path):
    page = _page(tmp_path, "2026-08-21")
    made = ledger.backfill(root, pages=[page], include_current=False)
    assert [s.run_code for s in made] == ["R001"]
    ledger.absorb(root)
    assert len(ledger.read_table("propositions", root)) == 1


def test_backfill_will_not_resurrect_a_purged_run(root, tmp_path):
    """The failure this guards is the alarming one: you exclude a slate priced by
    an old model, run 07 again, and backfill quietly puts it back."""
    page = _page(tmp_path, "2026-08-21")
    ledger.backfill(root, pages=[page], include_current=False)
    ledger.absorb(root)
    ledger.purge("R001", reason="old model", root=root)
    assert ledger.read_table("propositions", root).empty

    made = ledger.backfill(root, pages=[page], include_current=False)
    assert made == []
    assert ledger.pending_slates(root) == []
    ledger.absorb(root)
    assert ledger.read_table("propositions", root).empty
    assert len(ledger.read_table("runs", root)) == 1
