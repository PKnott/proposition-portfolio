# Edge capacity: what portfolio size actually does

A study of how the number of legs in a portfolio drives variance, the stake it can
carry, and long-run growth. It rebuilt the staking model and retired two of the
three numbers the project used to publish.

Everything below was measured against the project's own ledger — two settled
slates, 286 qualified propositions, 4,327 exported portfolios — using an engine
written independently of `fpp` and then pinned against it: expected return and
standard deviation agree to 4e-6, and the grid convolution matches brute-force
2ⁿ enumeration exactly on threshold probabilities.

## The question

More legs is supposed to be safer. With 100 legs at 60% you expect 60 winners and
get something close to it; with 10 you expect 6 and could get 2. Does that hold,
how fast, and what should it do to the stake?

## The mechanism is not the one you'd assume

With identical legs, three laws fall out exactly:

- **Spread falls as 1/√n.** At 20 legs the standard deviation is ×0.707 of the
  ten-leg book, at 40 legs ×0.500, at 160 legs ×0.250.
- **Risk-adjusted edge rises as √n.**
- **The stake you can safely carry rises *linearly* in n.** Holding risk tolerance
  fixed at "5% chance of a 30% drawdown within 100 rounds", the admissible fraction
  goes 0.156 → 0.293 → 0.548 → 0.998 as legs go 10 → 20 → 40 → 100.

The third is the one that matters, and it is not obvious. Leg count is better
understood not as a way to *reduce* risk but as a way to *convert* risk into
permitted stake — and since growth is roughly `f·μ`, doubling the permitted stake
doubles the compounding rate off the same edge. A hundred-leg book compounds about
**6.3× faster** than a ten-leg book on an identical edge, because it is allowed six
times as much money on the table.

It stops at the edge of the bankroll. You cannot stake more than you own, so growth
is capped at `ln(1 + edge)` — 9.53% per round at a 10% edge, however many legs. A
hundred legs already reaches 96% of it. Past that, extra legs buy certainty, not
growth.

## Leg count enters through exactly one number

Give each proposition its standalone Sharpe ratio:

```
S_i = (o_i p_i − 1) / (o_i √(p_i (1 − p_i)))
```

Weight the portfolio by `mu_i / v_i` — inverse variance times edge. Then:

```
Sharpe_P² = Σ S_i²  ≡  C
```

**Exactly.** Checked against the discrete distribution on both settled slates at
n = 4, 8, 12 and full size: agreement to 4e-16. It is an algebraic identity, not a
fit, and it is pinned by a test.

Three consequences:

1. **Adding a leg can never hurt.** It adds `S_i² ≥ 0` to `C`. Any measurement
   showing more legs making a portfolio worse is measuring the weighting.
2. **Leg count and leg quality are the same currency.** One proposition at
   `S = 0.30` is worth exactly nine at `S = 0.10`.
3. **Growth is linear in `C`**, so linear in leg count while leg quality holds up.

### The pool exhausts faster than you'd like

Ranking each slate's events by their best proposition's Sharpe, eight events carry
**89%** of everything the slate has to offer. The remaining eleven on one real card
were worth 13% between them. So the route to a materially bigger book is not more
events on the same slate.

## The split was the bug, not the leg count

Read the exported portfolios and leg count looks like it has an optimum — Sharpe
climbing to 0.541 at thirteen legs, then *falling* to 0.510 by nineteen.

It is an artefact. Both previous splits ignored edge:

```
1/E       s_i ∝ 1 / e_i
Min-var   s_i ∝ 1 / (o_i² p_i q_i)
Growth    s_i ∝ (o_i p_i − 1) / (o_i² p_i q_i)      ← min-var weight × edge
```

On one real placed book, minimum-variance put **25.5% of stake on the worst
proposition in it** (edge 1.003, Sharpe 0.006) and 4.3% on the best (Sharpe 0.367).
Not slightly wrong — inverted, because at odds 1.21 the worst leg barely moves the
total and min-variance rewards exactly that.

Under edge-aware weights the Sharpe curve never turns down. Worth +39% growth over
min-variance and +92% over 1/E at full leg count, and it nearly doubles how wrong
the model could be before the edge disappears (−8.0pp against −4.2pp).

The cost: growth weights **concentrate**. Effective legs — the inverse Herfindahl
of each leg's share of variance — fall from 12.6 to 5.6 on a nineteen-leg book,
with 24% of stake on one proposition. Hence `MAX_LEG_STAKE`.

## The two published stakes were one number and a ceiling

`f_star`, the growth optimum, sat at its own cap in **66.4%** of exported
portfolios and **99%** of those with sixteen or more legs. The unconstrained
optimum wants leverage it cannot have, so the column was reading back a constant.

`f_protective` was doing real work — and turns out to be a fixed fraction of Kelly:

```
f_protective = 0.111 × μ / σ²        r = 0.998, worst residual 6.2%
```

The 4,000-path drawdown simulation is computing **one-ninth Kelly**. That is not a
criticism of the method; it is a discovery about what the method means. The risk
appetite that sets that constant was three constants in a config file that nothing
surfaced.

## What leg count cannot fix

Resampling real settled legs into portfolios of each size shows the spread
narrowing exactly as expected — on one slate the 5th-percentile return climbs from
0.00 at two legs to 0.65 at sixteen. But look where each fan points: as `n` reaches
the slate size, every portfolio converges on **the slate's own outcome, not the
model's expectation**. One slate settled at 0.901 and every full-size portfolio
built from it lost money with certainty; the other settled at 1.884 and every one
won.

Leg count removes the variance *within* a slate. It removes none of the variance
*between* slates — and that is the term with two observations behind it.

The two terms that price it, `PESSIMISM_B` and `SLATE_TAU`, are the mean and the
spread of one quantity: the error shared by every leg on the card. Independent
per-leg error costs nothing at all — the extra uncertainty in `p` is exactly offset
by less coin-flip variance in the outcome, and simulated books from 6 to 96 legs
are indistinguishable from a perfect model. Only the shared part survives
diversification.

## A correction

An earlier version of this study reported that allowing a second proposition per
match was worth **1.72–1.76×** edge capacity, and called it the largest lever
available.

That was wrong. The factor model behind it had a sign flip: legs on the same team
should load on the match factor with the *same* sign, and they were opposed, which
made same-team pairs look correlated at 0.075 instead of the measured 0.455. The
corrected figure is **1.27–1.30×**, cross-checked two independent ways — full
matrix inversion and plain pairwise arithmetic — agreeing to four decimals.

It changes a recommendation: two-per-match is no longer bigger than everything
else, it is about the same size as the split change, and the two compound to
roughly 1.8× together.

## What this cannot tell you

- **Nothing about whether the edge is real.** Every figure is conditional on the
  probabilities. If they are optimistic by more than about 8 percentage points
  across the board, this describes an efficient way to lose money.
- **`PESSIMISM_B` and `SLATE_TAU` are placeholders.** Both zero, neither measured.
  The two settled slates ran *hot*, not optimistic, so a positive haircut today
  would be caution dressed as measurement.
- **Between-slate variance is unmeasured** and is what leg count provably cannot
  reduce. Two observations: 0.901 and 1.884.
- **The correlation model is fitted on two slates.** 1,438 same-match pairs is a
  decent sample and the placebo is clean at −0.013, but the factor structure is
  imposed and should not be trusted past three propositions per match.
- **Independence is assumed between matches.** The placebo supports it, but
  same-day fixtures could share a referee-strictness or weather effect that 133
  matches cannot resolve.
