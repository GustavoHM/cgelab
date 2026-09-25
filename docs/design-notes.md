# Design notes

Why the package does what it does. Every entry below is a decision with a
plausible alternative, and every failure described is one the alternative
produces **silently** — which is the defining hazard of CGE code, since the
output of a wrong model looks exactly like the output of a right one.

---

## Purchaser prices versus basic prices

A SAM records intermediate and final purchases **inclusive** of product taxes.
Quantities in the model are in base-year basic-price units, so those flows are
divided by `1 + t` at calibration:

```python
io = inter / X0 / (1 + ptax)[:, None]
```

Skipping it makes the tax appear twice. The symptom is not an error but a base
year that does not reproduce, which is why base-year reproduction is a gate and
not a nicety.

The tax base matters equally. Absorption's base quantity is
`dom0 + imp0 + margin_in` and its base purchaser price is `1 + t`, so the rate is
`tax / A0` — **not** `tax` over the commodity row total, which already contains
the tax.

## The margin block

National accounts offset trade and transport margins inside the commodity block:
margin commodities sell services into every other commodity's supply column
against a large **negative** entry on their own diagonal. Two things follow.

Ignoring the block loses that entire value, and it surfaces as a large residual
on one commodity — a diagnosis that is easy once you know to look and mystifying
otherwise.

The diagonal entries are pure accounting offsets appearing on both sides of the
same balance, so dropping them preserves the balance exactly. What remains
off-diagonal is economic: delivering a commodity requires trade and transport
services, Leontief, on top of the Armington composite. That is what `mq` is, and
it is why the price block is a **fixed point** rather than a formula — margin
prices are themselves delivered prices.

## Household income is the full row total

`hh_income0` is the account's entire row: factor income, transfers from
government or abroad, and anything else the SAM credits to it. Summing a chosen
list of columns instead silently drops whatever was not listed, and the saving
rate computed from it is then wrong in a way no check downstream will find.

---

## The quantity system is written on commodity output

With more commodities than activities, the commodity balances cannot determine
activity levels directly — that system is overdetermined, and solving it by least
squares would be arithmetic dressed as economics. So the balances determine
commodity output and activity levels follow from fixed market shares:

```
(I - B G - MQ) A = final,   G = diag(dsh * arm_share / (1 - export share)),
B = IO @ MSH,               Q = G A,   X = MSH @ Q
```

## Own-account activities

An activity with no market share in any commodity does not appear in the
commodity balance at all, so `X = MSH @ Q` returns zero for it and the base year
silently loses the whole activity. `calibrate` detects these
(`Economy.own_account`) and the solver carries them at their base level, adding
their intermediate demand to final demand so the balance stays exact.

They are also the **only** activities `Scenario.activity_scale` may touch.
Scaling an activity whose level the commodity balance already determines is a
contradiction, not an approximation, and the solver refuses it by name rather
than producing a number.

---

## Walras' law is a check, not an equation

The numeraire factor's price is fixed and its market is **not** in the solved
system. That is the whole design: with every other market cleared and every
budget constraint imposed, the last market clears by Walras' law, so its residual
is independent evidence about the accounting rather than an identity the solver
was instructed to satisfy.

A nonzero residual means income is being lost or manufactured somewhere. Two real
instances, both caught this way and both invisible otherwise:

- **A leak on an own-account activity.** Holding its revenue fixed in nominal
  terms while its cost moved with the factor price violates zero profit.
- **A productivity path applied to prices only.** Hicks-neutral productivity
  scales every input requirement, so it belongs in the coefficient matrices and
  not only in the unit cost. Applied to the price block alone, the price system
  says one thing and the quantity system another; the difference is a residual
  that **grows** with the productivity path — negligible in a single period,
  unmistakable over twenty-five years. `test_walras_holds_at_every_date` exists
  for exactly this.

## Every closure needs its own adjusting variable

A closure that pins a magnitude without freeing one somewhere else does not close
the model. Pinning the exchange rate while foreign saving stays at its base value
leaves the current-account shortfall nowhere; fixing real investment while
nothing absorbs the saving gap leaves the difference nowhere. Neither raises. Both
show up as a Walras residual, and both were found that way here.

So `foreign="fixed_rate"` makes foreign saving an unknown with the identity
`trade = foreign_saving` as its residual, and `investment="fixed_real"` adds a
forced-saving unknown with the saving–investment identity as its residual. Five
closure combinations are in the test matrix and all hold Walras to `~2e-15` under
a 20% tax shock.

---

## Wedge revenue is an infinite series, summed in closed form

Crediting recycled revenue to income raises spending, which raises the wedge,
which raises income again. Demand is affine in spending under both demand systems,
so `revenue_h = A_h + B_h · spend_h` exactly and the total solves

```
R = [sum_h A_h + sum_h B_h ((1-s_h) Y0_h - direct_h)] / [1 - sum_h B_h (1-s_h) alpha_h]
```

Iterating instead converges, but leaves small discontinuities that wreck the
outer Newton's finite-difference Jacobian — in a long path, several periods that
fail to converge for no visible reason.

Note the denominator uses the **total** wedge `1 + W + Wg`, since it is the total
that determines the quantity bought, while the numerator splits by recipient.
Using `1 + W` where a government wedge is also present would credit the network a
revenue that the refund has already reduced.

## Who receives the revenue is a modelling choice with a sign

Capital ownership is concentrated at the top. Under `"capital"` recycling the
poor pay the wedge on their purchases and the rich receive it as income, so part
of any measured regressivity is manufactured by that assumption. The convex form
`wedge_revenue_to=λ` exists so the finding can be given a magnitude rather than a
binary: there is generally a weight at which the distributional gradient crosses
zero, and that fraction is what says whether the result has room.

## Population shares are an input, not a derived quantity

The weight of a per capita transfer is the number of households, which is neither
the income share nor the expenditure share. Deriving it from money would reverse
the distributional sign of any lump-sum recycling. `calibrate(population=...)`
takes it explicitly and defaults to equal weights, which is a stated assumption
rather than a silent inference.

---

## Convergence is judged on the residual

Never on the solver's success flag. `hybr` reports failure whenever its
finite-difference Jacobian stops improving, which includes the case where the
start point already **is** the solution — residuals of `5e-14` flagged as
failures. It also terminates early from a warm start sitting exactly on the
previous period's answer. So four start points are tried — warm, cold, perturbed,
Levenberg–Marquardt — and the best residual wins, judged against
`Closure.tolerance`.

## Savings and tax revenue determine each other, affinely

Investment drives output, output drives tax revenue, revenue is government
saving, and saving funds investment. The relation is affine in saving, so it is
solved in closed form. Iterating it worked here too, and left the same class of
small discontinuity in the outer Jacobian.

## Real magnitudes use a scenario-independent yardstick

`real["hh_real"]`, `gov_real` and `inv_real` are absorption at **base-year**
prices, `1 + ptax`. Comparing a real EV sum against a nominal factor-income
difference across scenarios with different price levels is not a reconciliation,
it is a category error. For the same reason `p_base` does not follow a policy
rate: a yardstick that moved with the policy would compare each scenario against
its own ruler.

Investment is financed by three saving pools and the identity needs them apart.
Equivalent variation measures **household** income, which funds consumption and
household saving; government and foreign saving enter absorption with no
household counterpart at all. A reconciliation comparing the EV sum against total
absorption is incomplete by construction, not approximately right.

---

## Negative value added is data

An activity can report a negative operating surplus — a sector that collapsed in
the base year. No CES nest accepts a negative exponent, so `ModelSpec` chooses:
`"leontief"` puts the activity on fixed proportions and keeps the SAM's factor
quantities intact; `"net"` rescales its factor demands to the net total, which
changes them, and then aggregate factor supply must fall by the same amount or
the base year will not clear — `calibrate` does that adjustment, and it is the
kind of consequence that is easy to miss when the switch is flipped by hand.

## Sign-preserving RAS, used reluctantly

Standard RAS is multiplicative and breaks on negative entries, which real SAMs
have. Scaling is applied to magnitudes with signs fixed, targeting the midpoint
of each account's receipts and outlays.

The docstring says plainly when not to use it. A residual large enough to see is
evidence about how the table was built, and closing it before diagnosing it
destroys that evidence. The routine **fails** rather than writing a SAM that
looks balanced.

## The CET elasticity is the one parameter with no source

Input–output sources supply Armington substitution and factor substitution;
they supply no output CET. Fixing export shares keeps that parameter out of the
baseline entirely, so no headline number rests on an invented value, and
`export_rule="cet"` remains available as a robustness variant. This is a
methodological position, and the default encodes it.

---

## Composition is not welfare

An account is not the same population in two scenarios whenever households move
between accounts — adopters and non-adopters, formal and informal. Per-account
real income carries that size difference and reports it as welfare.

The instance that motivated the design: in an applied model, the per-account
"welfare" change of the adopters was almost exactly the difference in their
numbers between scenarios, and a gradient by decile turned out to be the adoption
rate by decile, rescaled. Nothing in the numbers looked wrong.

So `equivalent_variation` names its per-account column `ev_pct_account`, and
`ev_by_group` sums real incomes within a group **before** dividing, which makes
the set of households summed over identical across scenarios. Every account must
be assigned to a group; a missing one raises rather than being dropped, because a
dropped account understates the group and the error looks like a result.
