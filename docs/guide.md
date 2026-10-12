# cgelab user guide

1. [The SAM](#1-the-sam)
2. [Specifying the model](#2-specifying-the-model)
3. [Calibration](#3-calibration)
4. [Diagnostics: the three gates](#4-diagnostics-the-three-gates)
5. [Solving one period](#5-solving-one-period)
6. [Closures](#6-closures)
7. [Scenarios](#7-scenarios)
8. [Wedges and revenue recycling](#8-wedges-and-revenue-recycling)
9. [Recursive dynamics](#9-recursive-dynamics)
10. [Welfare](#10-welfare)

---

## 1. The SAM

A SAM is square and every account's receipts must equal its outlays. `cgelab`
holds it in long form: `row`, `col`, `value`, where the entry is the payment
**from** the column account **to** the row account — a column is an account's
outlays, a row its receipts.

```python
sam = ck.SAM.read("sam_2021.parquet")            # .parquet or .csv, long form
sam = ck.SAM.from_long(df, row="row", col="col", value="value")
sam = ck.SAM.from_square(square_df)              # indexed by account on both axes
```

Inspect and check:

```python
sam.accounts, sam.total
sam.receipts(), sam.outlays(), sam.imbalance()
sam.matrix()                                     # dense square frame
sam.block(rows, cols)                            # a sub-matrix as an array

sam.check_balance()                              # raises, naming the offenders
```

`check_balance` is a **hard gate**, and `calibrate` calls it by default. A CGE
calibrated on an unbalanced SAM produces confident nonsense, and nothing in the
output makes the nonsense visible. The test is relative to the SAM total with an
absolute floor, so a small account is not held to an impossible standard and a
large one cannot hide a large gap behind a small ratio.

### Balancing, and why to be reluctant about it

```python
sam = sam.balance()          # sign-preserving RAS
```

Standard RAS is multiplicative and breaks on negative entries, and real SAMs have
them: government dissaving, destocking, a subsidy exceeding the production tax, a
negative operating surplus in a year a sector collapsed. So scaling is applied to
**magnitudes** with signs held fixed, and each account's target is the midpoint of
its receipts and outlays.

Use it only after the residual has been diagnosed. Any residual large enough to
see is evidence about how the table was built, and closing it before diagnosing
it destroys the evidence. If the routine cannot reach tolerance it **fails**
rather than returning a SAM that merely looks balanced.

---

## 2. Specifying the model

```python
spec = ck.ModelSpec(
    activity_prefix="a_", commodity_prefix="c_", household_prefix="hh_",
    factors=("f_lab", "f_cap"),
    government="gov", investment="s_i", rest_of_world="row",
    numeraire_factor="f_lab",
    value_added="ces", sigma_va=0.5,
    sigma_armington=2.0,
    export_rule="fixed_shares",
    household_demand="cobb_douglas",
    negative_value_added="leontief",
    margins=True,
)
```

Accounts are found by prefix unless explicit lists are given
(`activities=[...]`, `commodities=[...]`, `households=[...]`). Explicit lists win
and are the safer choice for a SAM whose naming is not systematic: a prefix rule
that silently misses an account produces a model that balances and is wrong.

`sigma_va` and `sigma_armington` accept a scalar or a mapping keyed by account
name (with `"__default__"` as a fallback).

### Choices worth stating in a paper

**`export_rule`.** `"fixed_shares"` is the default because the CET elasticity is
the one parameter of a standard CGE with no input–output source. If no headline
number may rest on an invented parameter, the baseline keeps it out entirely and
`"cet"` enters as a robustness variant. Under `"cet"` output splits between the
home market and exports along a transformation frontier with elasticity
`sigma_cet`, exports are sold at `exchange_rate × Scenario.export_price`, and the
domestic price is the one at which the frontier's revenue equals unit cost.
`export_price` is refused under fixed shares, where it would do nothing.

**`export_demand` and `import_supply`.** Under CET the economy is still a price
taker abroad: `export_price` is what its exports fetch, whatever the volume. A
region that sells a large share of a product faces a demand curve instead.
`export_demand=η` gives every exported commodity one,
`E / E0 = (pWe / export_price)^(-η)`, and `import_supply=ψ` gives every imported
commodity a supply curve, `M / M0 = (pWm / world_price)^ψ`. These are the
equations of Hosoe's `lrgcge`. The scenario fields become the shifts of the
curves, and the solved world prices are `PeriodResult.world_export_price` and
`world_import_price`. Both take a scalar or a mapping by commodity; a commodity
missing from the mapping, or with no base-year volume, stays a price taker. Each
commodity on a curve adds one unknown to the outer solve, so on an applied SAM
put on curves only the commodities the study is about. `export_demand` requires
`export_rule="cet"`: with fixed shares, the export volume is set by output and
no world price could clear a demand curve for it.

**`tariff_account` and `indirect_tax_account`.** A SAM can record the import
tariff and an indirect tax on output on accounts of their own, Hosoe's TRF and
IDT. Named here, or passed as rates with `calibrate(..., tariffs=,
output_taxes=)`, each tax moves out of the product tax and into the price it
falls on, as in Hosoe's `stdcge`. The Armington nest is calibrated on the
tariff-inclusive value of imports, `(1 + tm0) × M0`. The CET is calibrated on
the tax-inclusive value of output, `(1 + tauz0) × Z0`, so home sales and exports
both carry the indirect tax. Government receives each as a level, the rate
times its base. `Scenario.tariff` and `Scenario.output_tax` change the rates.
Abolishing a tariff leaves the domestic variety untaxed. When the tariff was
instead left inside a product tax spread over the whole composite, the domestic
variety went on paying it after the abolition. At the calibrated rates the base
year is the SAM, in quantities and in what buyers pay.

**`negative_value_added`.** An activity can have a negative operating surplus,
and that is real data, not an error to clean. No CES nest accepts a negative
exponent, so either the activity goes on fixed proportions (`"leontief"`, the
default, which keeps the SAM's factor quantities intact) or its factor demands
are rescaled to the net total (`"net"`, which changes them — and then aggregate
factor supply must fall by the same amount or the base year will not reproduce;
`calibrate` does this for you).

**`household_demand="les"`.** The linear expenditure system with subsistence
quantities calibrated from `frisch`. Supernumerary income is `-Y / frisch` and
`p·γ = p·q - β·Y*`. A `frisch` too close to zero implies a negative subsistence
quantity, which is not a calibration but a contradiction, and `calibrate` refuses
it with that explanation rather than clipping.

---

## 3. Calibration

```python
econ = ck.calibrate(sam, spec,
                    income_elasticities={"c_food": 0.6, "c_services": 1.3},
                    population={"hh_d1": 8.3e6, ...})
```

`income_elasticities` are used only under LES. `population` is the weight of a
**per capita** transfer — neither the income share nor the expenditure share.
Getting it wrong reverses the distributional sign of any lump-sum recycling, so
it is a separate input rather than a default derived from money.

### What comes out

`Economy` carries every coefficient the solver uses. The ones you are most likely
to read or override:

| Field | Meaning |
|---|---|
| `X0`, `Q0` | base activity and commodity output |
| `io` | intermediate per unit of activity output, in **basic-price** units |
| `msh` | market shares, `(n_act, n_com)`, columns summing to one |
| `theta`, `vunit`, `aunit`, `leontief` | the value-added nest |
| `own_account` | activities selling only to households, absent from the commodity balance |
| `ptax`, `atax` | product tax on absorption; specific tax per unit of activity output |
| `mq`, `arm_share`, `A0` | the margin block and the Armington composite per unit of absorption |
| `hh_budget`, `hh_savrate`, `hh_facshare`, `hh_direct`, `hh_committed` | the household block |
| `fac_supply`, `gov_dem`, `inv_dem`, `foreign_sav` | the exogenous blocks |
| `diagnostics` | negative-value-added activities and their treatment, own-account list |

Helpers: `econ.commodity("c_x")`, `econ.activity`, `econ.household`,
`econ.factor` return indices; `econ.coms`, `econ.acts`, `econ.hhs`,
`econ.factors` are the ordered name lists.

---

## 4. Diagnostics: the three gates

```python
print(ck.report(econ))
```

```
cgelab diagnostics -- Economy(2 activities, 2 commodities, 2 households, 2 factors)
  base-year reproduction : PASS (worst 1.42e-16)
  Walras residual        : 0.00e+00 [f_lab]
  price homogeneity      : PASS (worst 1.14e-15)
```

**Base-year reproduction.** With no shock the model must return the SAM: every
price one, every quantity its base-year value. This is what catches a units error
in the calibration — a purchaser price treated as a basic price, a tax levied on
the wrong base — and it catches it before the error has been absorbed into a
result. A model that fails this is not calibrated.

**Walras' law.** The numeraire factor's market is not in the solved system, so
its residual is independent evidence. A nonzero residual means income is being
lost or manufactured somewhere, and the size of the leak bounds nothing about the
size of the error it causes.

**Homogeneity.** Scaling the numeraire must leave every real magnitude untouched.
A failure means a nominal magnitude denominated in the numeraire was held fixed
while the price level moved — foreign saving, the trade-balance target, or a
specific tax that enters additively rather than ad valorem. Each shifts a real
allocation and none is visible in the output.

Run all three on any new SAM, any new specification, and after any change to the
solver. `base_year_reproduction`, `homogeneity_test` and `walras_check` return
dicts with the per-check detail when you need to see which one moved.

---

## 5. Solving one period

```python
res = ck.solve_period(econ, closure, scenario, warm_start=previous)
print(res.summary())
```

| Attribute | Meaning |
|---|---|
| `factor_prices`, `exchange_rate` | the solved unknowns |
| `pa`, `pd`, `pq`, `pqf`, `pbase` | activity, domestic, consumer, firm and pre-tax prices |
| `Q`, `X`, `absorption`, `imports`, `exports` | the quantity system |
| `hh_quantity`, `hh_income`, `hh_commodity_spend`, `hh_direct_outlay`, `hh_wedge_paid` | the household block |
| `gov_quantity`, `investment_quantity`, `savings`, `factor_demand` | the rest of final demand |
| `wedge_revenue`, `government_cost` | recycled revenue; cost falling on the budget |
| `real` | real absorption at base-year prices, real saving by source, the household price index and real income |
| `walras`, `gaps`, `residual`, `converged`, `attempt` | the evidence |

Convergence is judged on the **residual**, never on the solver's flag: `hybr`
reports failure whenever its finite-difference Jacobian stops improving, which
includes the case where the start point already is the solution. Four start
points are tried — warm, cold, perturbed, Levenberg–Marquardt — and the best
residual wins.

`warm_start` accepts a previous `PeriodResult` (or a raw `z`) and is what makes a
twenty-five year path cheap.

---

## 6. Closures

```python
closure = ck.Closure(numeraire=1.0,
                     investment="savings_driven",   # or "fixed_real"
                     government="real_fixed",       # or "nominal_fixed", "saving_rate"
                     foreign="fixed_saving",        # or "fixed_rate"
                     foreign_saving_unit="numeraire",   # or "foreign_currency"
                     exchange_rate=1.0,
                     tolerance=1e-9)
```

Every closure carries its own adjusting variable. A closure that pins a magnitude
without freeing one somewhere else does not close the model, and the symptom is a
Walras residual rather than a solver failure.

| Setting | Pinned | Adjusts |
|---|---|---|
| `investment="savings_driven"` | nothing | investment absorbs household, government and foreign saving |
| `investment="fixed_real"` | investment quantities | forced household saving (a per-capita levy) |
| `foreign="fixed_saving"` | the current account, equal to `econ.foreign_sav` | the exchange rate |
| `foreign="fixed_rate"` | the exchange rate | foreign saving |
| `government="real_fixed"` | real public purchases | public spending moves with prices |
| `government="nominal_fixed"` | the public budget | real purchases move instead |
| `government="saving_rate"` | the share of its income government saves | public spending, in base-year value shares |
| `foreign_saving_unit="foreign_currency"` | foreign saving in dollars, under `fixed_saving` | what it finances at home moves with the exchange rate |

`government="saving_rate"` and `foreign_saving_unit="foreign_currency"` together are
the closure of Hosoe's `stdcge` (`cgelab.examples.hosoe.CLOSURE`). Under them a
tariff abolition cuts public consumption by the lost revenue, and the depreciation
it causes raises what foreign saving finances. The base-year saving rate is
`econ.gov_savrate`: government saving over income net of transfers to households.

`numeraire` exists for the homogeneity test only; no real magnitude may depend on
it. All five combinations in the test matrix reproduce the base year and hold
Walras to `~2e-15` under a 20% product-tax shock.

---

## 7. Scenarios

Everything a counterfactual can change lives in one object, so a run can be
logged, compared and swept:

```python
sc = ck.Scenario(
    name="reform 2027",
    product_tax=new_consumer_rates,
    product_tax_firm=non_creditable_part,
    factor_supply=supply_vector,
    factor_ownership=new_shares,          # (n_hh, n_fac), columns sum to one
    productivity=tfp_by_activity,
    activity_scale={"a_own": 2.4},
    consumer_wedge=W, government_wedge=Wg,
    wedge_revenue_to="capital",
    transfers=lump_sum_by_household,
    government_absorbs=cost_charged_to_the_budget,
    world_price=import_price_multipliers,
    export_price=export_price_multipliers,  # export_rule="cet" only
)
sc2 = sc.with_(wedge_revenue_to="lumpsum")     # a copy with one field replaced
```

**`product_tax` vs `product_tax_firm`.** The first is what final consumers pay;
the second is the **non-creditable** part, that is, what actually burdens a buyer
purchasing for production. Under a cascading turnover tax the two coincide and
the two-price block collapses to the single-price one exactly — a property the
base-year reproduction test verifies. Under a fully non-cumulative VAT only the
residual old system cascades.

**`activity_scale`.** Only for own-account activities — those selling directly to
households with no market share in any commodity. Scaling an activity whose level
the commodity balance already determines is a contradiction, not an
approximation, and the solver refuses it by name.

**`factor_ownership`.** The hook for an extensive-margin story in which
households **move** between accounts rather than each doing more of something.
See §10 for what that does to a welfare measure.

---

## 8. Wedges and revenue recycling

```python
W = ck.consumer_wedge(econ, {"c_electricity": 0.031})              # all households
W = ck.consumer_wedge(econ, {"c_electricity": 0.031},
                      households=["hh_d1_npr", "hh_d2_npr"])       # a subset
```

A `consumer_wedge` is an ad valorem charge on household purchases whose revenue
is **recycled to households**. A `government_wedge` is a second wedge whose
revenue accrues to the budget instead — a refund granted at the till, a levy
earmarked to the Treasury. Keeping them apart matters: routing a refund through
household recycling would credit the money to the wrong agent.

### The revenue is an infinite series

Crediting the revenue to income raises spending, which raises the wedge, which
raises income again. Demand is affine in spending under both demand systems, so
`revenue_h = A_h + B_h · spend_h` exactly, and the series is summed in closed
form. Iterating it converges but leaves small discontinuities that wreck the
outer Newton's finite-difference Jacobian — a real failure mode, not a
hypothetical one.

The identity `sum(res.hh_wedge_paid) == res.wedge_revenue` holds exactly and is
tested.

### Who receives it

```python
wedge_revenue_to="capital"    # in proportion to ownership of wedge_revenue_factor
wedge_revenue_to="lumpsum"    # per capita, using econ.pop_share
wedge_revenue_to=0.5          # convex combination of the two
wedge_revenue_to=lambda econ: my_shares
```

This is **not** a detail of implementation. Capital ownership is concentrated at
the top, so under `"capital"` the poor pay the wedge on their purchases while the
rich receive it as income, and part of any measured regressivity is manufactured
by that assumption. The convex form exists so the finding can be given a
magnitude: there is generally a weight at which the distributional gradient
crosses zero, and that fraction — not the mere existence of the boundary — says
whether the result has room.

A robustness exercise that does not vary this is not a robustness exercise.

---

## 9. Recursive dynamics

```python
growth = ck.Growth(labour=0.005, tfp=0.010,
                   depreciation=0.05, discount=0.05,
                   labour_factor="f_lab", capital_factor="f_cap")

path = ck.run_path(econ, closure, years=range(2021, 2046),
                   scenario=lambda year, t: ck.Scenario(...),
                   growth=growth)
```

Each year is a static equilibrium; what carries across is the capital stock, the
labour endowment and total factor productivity. No agent optimises over the
horizon — the defining limitation of a recursive-dynamic CGE, and worth stating
in any paper that uses one.

**Units.** A SAM's capital row is the annual capital **service** payment, not the
stock. Adding investment to it directly compounds a flow onto a flow and the
stock runs away — over twenty-five years by orders of magnitude. The stock is
recovered from the user cost, `r = discount + depreciation`, and services scale
with the stock. `path.capital_stock` records it.

Factor supplies set inside a per-year scenario **override** the accumulated ones:
an exercise that fixes an endowment by hand should not have the growth block
quietly undo it. Anything the scenario leaves as `None` is filled from the path.

### Changes that are not scenario fields

A technology adopted over time changes input coefficients; a financing plan
changes foreign saving. Neither is a price, a tax or an endowment, so neither is
a `Scenario` field. Pass the whole economy for the year instead:

```python
path = ck.run_path(econ, years=years, growth=growth,
                   economy=lambda year, t: dataclasses.replace(econ, io=io_path[t]))
```

### Several capital stocks, and who owns them

```python
growth = ck.Growth(capital_factor=("f_cap", "f_cap_h2"),
                   depreciation={"f_cap": 0.05, "f_cap_h2": 0.065},
                   allocation=2.0)          # investment follows relative rents
path = ck.run_path(econ, years=years, growth=growth,
                   scenario=lambda year, t: ck.Scenario(fdi={"f_cap_h2": capex[t]}))
```

Each capital factor is accumulated from its own service flow. Domestic
investment is shared by `allocation`. FDI raises that year's foreign saving —
so it finances investment — and the capital it buys is added to the
**foreign-owned** part of the stock it targets. `path.foreign_capital` records
that part, each year's `foreign_factor_share` is set from it, and its returns
leave through the current account. Counting FDI only as foreign saving would hand
residents a capital stock they did not pay for, and every gain it produced would
be credited to them.

### Reading a path

```python
path.aggregates()        # one row per year: prices, real absorption, savings, Walras
path.households()        # one row per year and account: income, spend, price index
path.commodity("c_x")    # price and output of one commodity
path[2035]               # the PeriodResult for that year
path.converged, path.worst_walras
```

`worst_walras` is the number to look at first. A residual that **grows** along
the path is the signature of a coefficient the price block saw and the quantity
block did not — a productivity path applied to unit costs but not to input
requirements, for instance. It is invisible in a single period and unmistakable
over a horizon.

---

## 10. Welfare

```python
ev = ck.equivalent_variation(reference_path, alternative_path)
```

Columns: `ev` (valued at the reference scenario's prices), `ev_base` (at
base-year prices), `ev_pct_account`, plus the components
`wedge_paid_change` and `direct_outlay_change`.

**Two yardsticks.** `ev` and `ev_base` differ by the price drift accumulated over
the horizon, which is a level effect that does **not** shrink as the policy shock
shrinks. Reconciling a sum of EVs against real absorption — itself measured at
base-year prices — needs `ev_base`. Comparing like with like is not a detail.

**Composition.** An account is not the same population in two scenarios whenever
households move between accounts. `ev_pct_account` then carries that size
difference and reports it as welfare. The column is named to say so; the
distributional object is:

```python
ck.ev_by_group(ev, {"hh_d1_pro": "d1", "hh_d1_npr": "d1", "hh_d2_pro": "d2", ...})
```

which sums real incomes **within** a group before dividing, so the set of
households summed over is the same in both scenarios and the ratio does not
depend on how many migrated. Every account must appear in the mapping: silently
dropping one would understate the group, and the error would look like a result.

The test `test_group_aggregation_is_immune_to_migration_between_accounts` moves
income between two identical twin accounts, which changes no aggregate and no
price. The per-account measure reports more than ±5%; the group measure returns
the zero that is actually there.

```python
ck.compensating_variation(reference_path, alternative_path)
```

The same move valued at the alternative's prices. Report it alongside EV when the
price change is large enough for the two to differ; quoting only one while calling
it "the welfare change" hides a choice of reference prices.

---

## 11. Multipliers and input–output analysis

A multiplier model reads the same SAM with every price fixed. Quantities answer
to an injection into the exogenous accounts, in base-year proportions. It is what
a reader will set a CGE result against. When the two disagree, the difference is
the work done by prices and by supply constraints.

```python
m = ck.accounting_multipliers(sam, spec=spec)     # gov, investment, rest of world exogenous
m.M                                                # (I - A)^-1, account by account
m.impact({"c_agr": 100.0})                         # an injection into agriculture's demand
m.leakage                                          # what each account pays the exogenous ones
```

`M[i, j]` is the effect on account `i` of one unit injected into account `j`.
`exogenous=` or `endogenous=` states a split other than the usual one.

**Fixed-price multipliers.** `fixed_price_multipliers(sam, spec=spec,
households=..., income_elasticities={"c_agr": 0.6, ...})` replaces each
household's average propensity to consume a commodity by the marginal one,
elasticity times average. The marginal propensities of a household must add up to
its average propensity to consume, or an extra unit of income would create or
destroy spending. Elasticities that break this are refused unless
`engel="rescale"`. All-ones elasticities give back the accounting multipliers.

**Leontief.** `leontief_multipliers(sam, spec)` makes only activities and
commodities endogenous. Commodity columns leak imports and taxes, activity columns
leak value added, and the activity block of `M` is the domestic Leontief inverse.
On Miller and Blair's two-sector example it is their `L` to every printed digit.

**Linkages.** `linkages(m, activities)` returns, for each account, the backward
linkage (the column sum of `M`), the forward linkage (the row sum of the Ghosh
inverse) and both divided by their mean. Above one is above average. The forward
measure is the Ghosh row sum and not the Leontief one. The Leontief row sum
answers a uniform rise in every final demand, which is a different question.

**Structural paths.** `structural_paths(m, "c_agr", "hh_1")` lists the
elementary paths from an injection to a destination (Defourny and Thorbecke,
1984). The direct influence is the product of the propensities along a path. The
total influence multiplies it by the feedback from circuits off the path,
`det(I - A_p) / det(I - A)`. The global influence is `M[destination, origin]`.
Over every elementary path the total influences add up to the global one
(Lantner, 1974); the tests check that identity to 1e-10. On a real SAM the search
stops at `max_length` arcs and at a direct influence below `threshold`, so the
`share` column then sums to less than one, and the shortfall is what was cut.

A real SAM can show a negative `leakage`. A margin commodity with a negative
diagonal entry, or an account with negative outlays, spends more inside the
endogenous block than its total. That is the data, and the inverse exists as long
as the block as a whole leaks.
