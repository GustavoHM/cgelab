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
11. [Multipliers and input–output analysis](#11-multipliers-and-inputoutput-analysis)
12. [Systematic sensitivity](#12-systematic-sensitivity)
13. [Subtotals: which shock did what](#13-subtotals-which-shock-did-what)
14. [Welfare decomposition](#14-welfare-decomposition)
15. [GTAP data and HAR files](#15-gtap-data-and-har-files)
16. [Emissions and a carbon price](#16-emissions-and-a-carbon-price)
17. [Forward-looking investment, through dsgelab](#17-forward-looking-investment-through-dsgelab)
18. [Physical units](#18-physical-units)
19. [Microsimulation: poverty and inequality](#19-microsimulation-poverty-and-inequality)
20. [Building a SAM from supply and use tables](#20-building-a-sam-from-supply-and-use-tables)

The examples assume `import cgelab as cge` (and `numpy as np` where it appears).
They are fragments: `econ` is a calibrated economy and `...` marks where your own
data go. To run one without data, build `econ` from the bundled toy economy
(`from cgelab.examples import toy`, then `sam, econ = toy.build()`); the
[site](https://gustavohm.github.io/cgelab/) has complete examples that run as they are.

---

## 1. The SAM

A SAM is square and every account's receipts must equal its outlays. `cgelab`
holds it in long form: `row`, `col`, `value`, where the entry is the payment
**from** the column account **to** the row account — a column is an account's
outlays, a row its receipts.

```python
sam = cge.SAM.read("sam_2021.parquet")           # .parquet or .csv, long form
sam = cge.SAM.from_long(df, row="row", col="col", value="value")
sam = cge.SAM.from_square(square_df)             # indexed by account on both axes
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
spec = cge.ModelSpec(
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
econ = cge.calibrate(sam, spec,
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
print(cge.report(econ))
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
res = cge.solve_period(econ, closure, scenario, warm_start=previous)
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

### Accounts

```python
cge.national_accounts(res)         # GDP by expenditure and by income, current prices
cge.price_indices(res)             # Laspeyres, Paasche, Fisher, against the base year
cge.price_indices(res, reference=baseline)  # or against another solved period
cge.government_account(res)        # revenue by instrument, outlays, saving
```

`national_accounts` builds GDP at market prices twice. The expenditure side adds
household consumption (at what households pay, consumer wedges and own-account
purchases included), government consumption, investment and exports, less
imports at the border. The income side adds factor income, taxes on products
(product tax, tariff, output tax and both consumer wedges) and the activity tax.
The two agree in every equilibrium, not only at the base year, and the `gap` key
is there so that a disagreement shows. `gni` deducts factor income paid abroad.

`price_indices` has one row per component of final demand, one for GDP (imports
with a minus sign) and one for the terms of trade (the export index over the
import index). The Fisher indices pass the factor-reversal test: price times
quantity is the value ratio. Nominal GDP over the Paasche deflator is
`real_gdp(res)["expenditure"]`, GDP at base-year prices.

`government_account` is recorded by the solver at the solution, with the formulas
the closure solved with, so its `saving` is the government saving that financed
investment.

---

## 6. Closures

```python
closure = cge.Closure(numeraire=1.0,
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

**Government targets.** `government_target="saving"` holds government saving at
its base-year level (or at `government_target_value`, in numeraire units): a
deficit target. `government_target="revenue"` holds total revenue instead, so
that a scenario that cuts one tax has another make the receipts up:
revenue-neutral tax replacement. `government_adjusts` names what moves. With
`"direct_tax"`, every household's direct tax rate is scaled by one common factor.
With `"product_tax"`, every product tax rate (consumer and firm) is scaled the same
way. With `"spending"`, government consumption moves in base-year value shares;
that works under a saving target only, because spending does not change revenue.
The factor is `PeriodResult.tax_scale`.

```python
swap = cge.Closure(government_target="revenue", government_adjusts="direct_tax")
r = cge.solve_period(econ, swap, cge.Scenario(product_tax=econ.ptax * 0.5))
r.tax_scale        # 1.55 on the institutions toy: direct tax rates rise 55%
```

`government="saving_rate"` and `foreign_saving_unit="foreign_currency"` together are
the closure of Hosoe's `stdcge` (`cgelab.examples.hosoe.CLOSURE`). Under them a
tariff abolition cuts public consumption by the lost revenue, and the depreciation
it causes raises what foreign saving finances. The base-year saving rate is
`econ.gov_savrate`: government saving over income net of transfers to households.

**When the exchange rate is not determined.** Under `export_rule="fixed_shares"`
exports sell at the domestic price. With a Cobb–Douglas Armington
(`sigma_armington=1` on every imported commodity), imports are then a fixed
value share of absorption and exports a fixed value share of output, and no
relative price moves the current account. With foreign saving fixed in numeraire
units, every exchange rate is an equilibrium, with the same output and factor
prices. `solve_period` refuses that combination. CET exports, an Armington
elasticity other than one on some imported commodity, foreign saving in dollars
or a fixed exchange rate each restore a channel. Close to one the model is
determined but ill-conditioned: the exchange rate moves as `1 / (sigma - 1)`, and
a sensitivity sweep that crosses that region returns outliers that are the
model's and not the solver's.

`numeraire` exists for the homogeneity test only; no real magnitude may depend on
it. All five combinations in the test matrix reproduce the base year and hold
Walras to `~2e-15` under a 20% product-tax shock.

### Factors that move only partway

A perfectly mobile factor earns one price in every activity. A sector-specific
one is a factor of its own, as `f_kind` is in `toy.build_with_specific_capital`.
`ModelSpec.factor_mobility` sits between them: the factor's endowment is split
among the activities that use it by a transformation with elasticity ω, and
each activity pays its own price.

```python
spec = toy.spec(numeraire_factor="f_cap", factor_mobility={"f_lab": 2.0})
econ = cge.calibrate(sam, spec)
r = cge.solve_period(econ, scenario=cge.Scenario(world_price=np.array([1.0, 1.3])))
r.factor_prices_by_activity[0]     # the wage in each activity
```

Two forms, by `factor_mobility_form`:

- `"cet"`, the default, is the frontier of GTAP's sluggish endowments. A pool
  with endowment `L` and base-year shares `δ_a` has the price index
  `W = [Σ δ_a w_a^(1+ω)]^(1/(1+ω))` and supplies `δ_a L (w_a / W)^ω` to activity
  `a`. Its income is `W L` exactly. The supplies do not add up to `L` away from
  equal prices: moving between activities costs efficiency units.
- `"logit"` keeps the head count: activity `a` gets the share
  `δ_a w_a^ω / Σ δ_b w_b^ω` of `L`, whatever the prices. The pool's price is its
  income over its head count. There is no frontier behind it, so the wage gaps it
  leaves are a gain from moving that the welfare decomposition reports as
  `factor_reallocation`; under the CET that term is zero.

The two agree to first order around equal wages and part at second order. ω = 0
holds the base-year allocation, and the model is then exactly the one with the
factor split into one account per activity. As ω grows the solution approaches
perfect mobility at a rate of 1/ω.

**Pools of origin.** One national pool is the default. `calibrate(...,
factor_pools={"f_lab": table})` takes a proximity matrix instead: a table of
base-year use with pools of origin (occupations, regions, neighbouring sectors)
as rows and activities as columns, each column adding up to what the SAM pays
the factor there. A pool supplies only the activities it reached in the base
year, so a shock to one sector draws first on its neighbours.
`Scenario.pool_supply` moves the pools' endowments; absent, they scale with
the factor's supply, as a growth path needs. `PeriodResult.factor_pools` reports
each pool's price, endowment and supply by activity.

```python
pools = pd.DataFrame([[30.0, 10.0], [10.0, 60.0]], index=["rural", "urban"],
                     columns=["a_agr", "a_ind"])
econ = cge.calibrate(sam, spec, factor_pools={"f_lab": pools})
```

For such a factor, `factor_prices` is its income over its endowment and
`factor_demand` is the endowment, so their product is still its income.
`factor_demand_by_activity` holds the use by activity. The numeraire cannot be
imperfectly mobile: its market is the Walras check, and with a price in every
activity there is no single market to leave out. Labour closures other than
full employment are refused on such a factor. Regional models refuse
`factor_mobility`.

### Formal and informal labour

A SAM often pays labour through more than one account: formal and informal,
covered and uncovered. Each is a factor the value-added nest demands on its
own. `ModelSpec.labour_segments` says that the same people supply all of them,
and `Closure.labour_markets` says how each segment's wage is set and where
whoever a rigid segment does not hire goes.

```python
_, econ = toy.build_with_segmented_labour()      # f_for (industry), f_inf (farms)
floor = cge.LabourMarket(rigid={"f_for": 1.1}, informal="f_inf")
r = cge.solve_period(econ, cge.Closure(labour_markets={"labour": floor}))
r.labour_markets["labour"]["employment"]          # formal 66.2 and informal 43.8, from 70 and 40
```

A rigid segment pays a fixed real wage (`1.1` is ten per cent above the
calibrated one; `None` is the calibrated one) and hires what demand asks for at
it. The segments that are not rigid share one wage, the informal segment's,
and the informal segment employs the rest of the labour force:

| `LabourMarket` | Where the rationed go | Equation |
|---|---|---|
| no rigid segment | nowhere: one labour market | `Σ D_g(w) = N` |
| `unemployment="none"` | the informal segment (a dual market) | `E_inf = N − Σ E_g` |
| `unemployment="harris_todaro"` | a queue for rigid jobs | `w_inf = λ Σ_R w_g E_g / (Σ_R E_g + U)` |

Under Harris–Todaro, workers queue for rigid jobs until the informal wage equals
the wage they expect from queueing. `unemployment_rate` is the base year's,
`U / (E_R + U)`, and `λ = 1 / (1 − rate)` makes the base year reproduce the SAM.
The informal wage can never rise above `λ` times the rigid one; a shock that
asks for that is refused, because nobody would queue. Harris–Todaro does not
tend to the dual market as the base rate goes to zero: it holds the informal
wage at the expected rigid wage, and the dual market lets any gap stand.

**Income follows the workers.** Each household has a fixed workforce in the
group: its base-year jobs in every segment, plus its share of the queue. It
holds its base-year share of each rigid segment's jobs and of the queue (the
queue in proportion to its base-year rigid jobs), and the rest of its workforce
works in the informal segment. A household that loses formal jobs gains
informal ones, so the informal income goes to the people who lost formal work,
not to the households that were already informal. At the base year this
returns the SAM's ownership. `labour_markets[group]` reports, besides wages,
employment and unemployment, `household_workforce`, `household_employment`
(households by segments) and `household_unemployed`.

**A transformation between segments.** `ModelSpec.segment_mobility =
{"labour": ω}` replaces perfect mobility with the transformation of the
previous section: each household's workforce is a pool, split among the
segments by a CET or a logit (`factor_mobility_form`, which takes the group's
name) at the segments' wages. Each flexible segment then has its own wage,
which clears it against that supply. A rigid segment hires what demand asks for,
and the rest of its supply queues in it; `unemployment_rate` is the base-year
queue as a share of the segment's supply, one rate or `{factor: rate}`. There is
no informal buffer and no Harris–Todaro rule here, and both are refused.

```python
spec = toy.spec(factors=("f_cap", "f_for", "f_inf"), numeraire_factor="f_cap",
                labour_segments={"labour": ["f_for", "f_inf"]},
                segment_mobility={"labour": 2.0})
_, econ = toy.build_with_segmented_labour(model_spec=spec)
queue = cge.LabourMarket(rigid={"f_for": 1.1}, unemployment_rate=0.08)
r = cge.solve_period(econ, cge.Closure(labour_markets={"labour": queue}))
r.labour_markets["labour"]["queue"]       # 18.6 queue for formal jobs, from 6.1
```

A rigid segment hires every household's supply in the same proportion, so the
income still follows the workers. At ω = 0 with no base-year queue, a group with
one rigid and one flexible segment is exactly the two factors kept apart under
`Closure(labour="fixed_real_wage")`; as ω grows it approaches one labour market,
at a rate of 1/ω. `labour_markets[group]` reports each segment's `supply` and
`queue` in both modes; under perfect mobility the supply is the employment and
the queue, if any, is the group's.

`Scenario.labour_force` sets the population of a group, the queue included;
absent, it is the base year's, scaled with the supply of the group's factors.
`Growth(labour_factor="labour")` grows a group as a whole. A segment cannot
also be the numeraire, sit in `factor_mobility`, or take `Closure.labour`.

---

## 7. Scenarios

Everything a counterfactual can change lives in one object, so a run can be
logged, compared and swept:

```python
sc = cge.Scenario(
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
W = cge.consumer_wedge(econ, {"c_electricity": 0.031})             # all households
W = cge.consumer_wedge(econ, {"c_electricity": 0.031},
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
growth = cge.Growth(labour=0.005, tfp=0.010,
                    depreciation=0.05, discount=0.05,
                    labour_factor="f_lab", capital_factor="f_cap")

path = cge.run_path(econ, closure, years=range(2021, 2046),
                    scenario=lambda year, t: cge.Scenario(...),
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
path = cge.run_path(econ, years=years, growth=growth,
                    economy=lambda year, t: dataclasses.replace(econ, io=io_path[t]))
```

### Several capital stocks, and who owns them

```python
growth = cge.Growth(capital_factor=("f_cap", "f_cap_h2"),
                    depreciation={"f_cap": 0.05, "f_cap_h2": 0.065},
                    allocation=2.0)          # investment follows relative rents
path = cge.run_path(econ, years=years, growth=growth,
                    scenario=lambda year, t: cge.Scenario(fdi={"f_cap_h2": capex[t]}))
```

Each capital factor is accumulated from its own service flow. Domestic
investment is shared by `allocation`. FDI raises that year's foreign saving —
so it finances investment — and the capital it buys is added to the
**foreign-owned** part of the stock it targets. `path.foreign_capital` records
that part, each year's `foreign_factor_share` is set from it, and its returns
leave through the current account. Counting FDI only as foreign saving would hand
residents a capital stock they did not pay for, and every gain it produced would
be credited to them.

### Capital by sector

```python
spec = cge.ModelSpec(..., factor_mobility={"f_cap": 0.0})
econ = cge.calibrate(sam, spec, factor_pools={"f_cap": "activities"})
path = cge.run_path(econ, years=years, growth=cge.Growth(allocation=2.0))
path.capital_by_pool["f_cap"]            # the stock of each activity, by year
```

A pool per activity makes capital sector-specific: each activity has the
capital it accumulated and pays its own rent for it. `run_path` keeps a stock
per pool and shares each year's investment among them by `Growth.allocation`,
as among several capital factors: `"proportional"` keeps the composition, a
number `e` sends investment toward the activities whose capital earns more, in
proportion to `K (r / r̄)^e`. Year by year, this is the path of capital written
out by hand as one factor per activity. Other pools (owners, regions, with
`ω > 0`) accumulate the same way.

### Putty-clay vintages

```python
sam = cge.split_vintages(sam, 0.08)       # 8% of each activity's capital is new
spec = cge.ModelSpec(..., factors=("f_lab", "f_cap", "f_cap_new"),
                     vintages={"f_cap": "f_cap_new"},
                     vintage_sigma=0.0,          # no substitution ex post
                     sigma_va=0.8)               # substitution ex ante
path = cge.run_path(cge.calibrate(sam, spec), years=years, growth=growth)
path[2030].vintage_value_added           # old and new bundles, by activity
```

Installed capital is clay: it stays in the activity that bought it and works
with the labour its technique was built for. Capital not yet installed is
putty: it goes where it earns most, with the labour intensity the prices of the
year make cheapest. Each activity's value added is then a CES
(`vintage_substitution`) of two bundles. The old bundle is a CES of
`vintage_sigma` (zero by default: fixed proportions) over every factor but new
capital; the new bundle is the usual CES of `sigma_va` over every factor but old
capital. Old capital is sector-specific, a pool per activity; new capital is
mobile.

In `run_path` the new vintage is each year's investment. At the end of the
year, the new capital each activity used is installed there: it joins that
activity's old capital, and the old bundle takes on the technique it was
installed with, averaged with its own by quantity. Without that, capital
installed in 2040 would work with the base year's labour, and an economy whose
capital grows faster than its labour would have old capital no one can man.
`split_vintages`'s `share` is the new vintage's part of each activity's
capital: on a steady path, investment over the capital stock.

LINKAGE has the two bundles as perfect substitutes. That case switches regime
when an activity shrinks faster than its old capital wears out, leaving old
capital idle, and a smooth solver does not switch regimes. Here they are a CES,
20 by default, and the distance to one capital stock, with the same elasticity
in both bundles, falls as one over it. Vintages are not available yet with an
energy nest, with foreign-owned capital or FDI, or in several regions.

### A baseline calibrated to GDP

A baseline built from assumed rates makes GDP whatever the rates give. The
applied practice turns this round, as in the historical simulations of MONASH
and the baselines of LINKAGE: real GDP in each year, observed or projected, is
given, and productivity is the unknown. A policy run then holds productivity at
the calibrated values and lets GDP move.

```python
gdp = pd.Series([0.046, 0.029, 0.032, 0.025], index=range(2022, 2026))   # real growth
base = cge.calibrate_baseline(econ, years=range(2021, 2026), gdp=gdp,
                              growth=cge.Growth(labour=0.01))
base.factor            # Λ_t, the productivity that hits each year
base.achieved          # the GDP the path reached, against base.target
policy = base.run(cge.Scenario(product_tax=new_rates))   # productivity held, GDP moves
```

The years are calibrated in sequence, since each year's capital comes from the
investment of the years before: one scalar root a year, multiplying the
Hicks-neutral productivity of every activity on top of what `growth` and
`scenario` already set. `gdp` is a series of growth rates (`kind="growth"`), an
index (`"index"`) or levels in the SAM's units at base-year prices (`"level"`);
with the first two the first year is solved as `run_path` solves it, and with
levels it is calibrated too. `pattern={"a_agr": 2.0, "a_ind": 1.0}` makes
farming's productivity grow twice as fast as industry's, the path deciding how
fast in all. With `sectors=`, a frame of real value added by year and activity,
each activity has its own unknown and the roots are solved together.

GDP is measured on the expenditure side of `real_gdp` (`measure="expenditure"`)
or as the sum of `real_value_added` (`"production"`), value added by double
deflation: output at base-year prices less the inputs it used at base-year
prices, the volume measure statistical offices publish. The `value_added` of
`real_gdp` cannot be the target. It is factor use at base-year factor prices,
and productivity changes what a factor makes, not the factor.

`base.scenario(policy)` is the `f(year, t)` that `run_path` takes. A policy is
the whole scenario for the year, built from the baseline's as any policy is
built from its reference; a productivity it sets multiplies the calibrated one.
With no policy, running it reproduces the baseline exactly.

### A limited historical closure

`calibrate_historical` generalises the same target-and-shifter mechanism beyond
GDP. Observed variables become roots; technology and relative preferences are
the unknowns. The period equations do not change:

```python
hist = cge.calibrate_historical(
    econ, years=range(2021, 2025),
    targets={
        "gdp": (gdp_observed, lambda r: cge.real_gdp(r)["expenditure"]),
        "food": (food_observed, lambda r: r.hh_quantity[0, 0]),
    },
    shifters={
        "gdp": ("productivity",),
        "food": ("preference", "hh_1", "c_agr"),
    },
    preference_reference={"hh_1": "c_ind"},
    kind="level", scenario=observed_shocks, growth=growth)
hist.target, hist.achieved
hist.run(policy)             # calibrated technology and preferences held
```

Each target is `(observed Series, measure(result))`; each has a distinct
`("productivity",)`, `("productivity", activity)` or `("preference", household,
commodity)` shifter. One positive log multiplier per target is solved jointly
per year, sequentially along a recursive path. `kind` is `"level"`, `"index"`
or `"growth"`, with the baseline's treatment of the first year. `pattern`
exponentiates an aggregate productivity shifter by activity.

Preference multipliers are normalised within each household. The multiplier of
the commodity named in `preference_reference` is fixed at one, which sets the
scale: only relative preferences are identified. Moving that reference
commodity, omitting the reference or reusing a shifter is refused. Every target
must be positive because the root is in logs. This is the limited historical
closure, not the generic closure swap of 0.9. `calibrate_baseline` uses the same
scalar and joint root routines, with its existing API and results.

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
ev = cge.equivalent_variation(reference_path, alternative_path)
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
cge.ev_by_group(ev, {"hh_d1_pro": "d1", "hh_d1_npr": "d1", "hh_d2_pro": "d2", ...})
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
cge.compensating_variation(reference_path, alternative_path)
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
m = cge.accounting_multipliers(sam, spec=spec)    # gov, investment, rest of world exogenous
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

---

## 12. Systematic sensitivity

An elasticity the input–output tables do not supply is a guess, and a result that
rests on one should come with its spread. `sensitivity` solves a run over the
parameters' distributions:

```python
run = cge.spec_sweep(sam, spec, scenario=policy,
                     measure=lambda base, alt: {"gdp_pct": 100 * (
                        cge.real_gdp(alt)["expenditure"] / cge.real_gdp(base)["expenditure"] - 1)})
params = {"sigma_va": cge.Uniform(0.3, 1.3), "sigma_armington[c_ind]": cge.Normal(2.0, 0.5)}

cge.sensitivity(run, params).summary()                                # Stroud quadrature
cge.sensitivity(run, params, method="monte_carlo", draws=500).summary()
```

`spec_sweep` sets `ModelSpec` fields, `"field"` whole or `"field[account]"` for
one entry of a mapping, then recalibrates and solves the base and the scenario at
every point. An elasticity moves the calibrated shares, and a result measured
against a base that was not recalibrated would mix the two effects. Any other
function from `{parameter: value}` to a number, a mapping or a Series is a run
too.

**Quadrature.** Stroud's degree-3 rule, the one Arndt (1996) and DeVuyst and
Preckel (1997) brought to CGE models, solves at `2n` points for `n` independent
parameters with symmetric distributions. The mean is exact whenever the result is
a polynomial of degree three or less in the parameters. The standard deviation is
exact when the result is linear in them. The tests check both on closed forms.
Twenty elasticities cost forty solves. An asymmetric distribution is refused,
because the rule's mean would carry a bias nobody stated.

**Monte Carlo.** Independent draws with a fixed seed, for asymmetric
distributions (`Triangular`), for quantiles, or to check the quadrature. The
summary then adds the 5th, 50th and 95th percentiles and the range. Quantiles
from `2n` quadrature points would be a fiction, so they are not reported.

A spread that is much wider under Monte Carlo than under quadrature is a sign
that the result is far from cubic somewhere in the range. One cause is an
Armington elasticity near one under fixed export shares (see *Closures*). There,
a sweep of `sigma_armington` over `Uniform(1, 3)` on the toy economy had a
median of −0.009% and one draw at −5.8%.

### Where the variance comes from: Sobol indices

The quadrature gives a mean and a standard deviation, and does not say which
parameter the variance comes from when parameters interact. Sobol indices do:

```python
res = cge.sobol(run, params, n=1024)        # n (k + 2) solves
res.indices()            # S1, ST and their half-widths, by output and parameter
res.summary()            # mean, sd and quantiles over the independent draws
```

`S1` is the share of the variance a parameter explains alone; `ST` adds every
interaction it takes part in, so `ST - S1` is how much of its effect depends on
the others. The design is Saltelli's: two matrices `A` and `B` from a scrambled
Sobol sequence, carried to each distribution by its quantile function, and for
each parameter `A` with that column taken from `B`. First-order indices use the
estimator of Saltelli et al. (2010), total ones Jansen's (1999), on outputs
centred on their mean, with bootstrap intervals; given the same seed, SALib
draws the same design and its analyser gives the same indices
(`validation/sobol_salib.py`, to 1e-15). `n` must be a power of two. With twenty
elasticities and `n = 1024` that is 22,528 solves, against forty for the
quadrature: the indices are for when the quadrature has shown a spread worth
explaining.

### Uncertainty in the SAM

The elasticities are not the only estimated inputs: every SAM entry is an estimate
too. `sam_draws` perturbs selected blocks and rebalances every draw, and
`sam_sweep` recalibrates and solves them:

```python
draws = cge.sam_draws(
    sam,
    blocks={"intermediate": (["c_agr", "c_ind"], ["a_agr", "a_ind"])},
    method="dirichlet", concentration=100.0, draws=500, seed=20260724,
    balance_kwargs={"rel_tol": 1e-12, "abs_tol": 1e-10})
res = cge.sam_sweep(draws, spec, scenario=policy, measure=gdp_pct,
                    parameters={"sigma_va": cge.Uniform(0.3, 1.3)})
res.points               # draw, weight, sam_distance, sigma_va, gdp_pct
res.summary()            # mean, sd, quantiles and range over the draws
```

A block is a `(rows, columns)` pair; blocks may not overlap. With `"dirichlet"`,
the positive selected cells of each column receive a Dirichlet shock centred on
their original shares. `concentration` is the precision: larger values stay
closer to the original, structural zeros stay zero, and negative selected cells
are refused. With `"noise"`, nonzero selected cells receive independent
lognormal shocks with mean one and log standard deviation `scale`.

The blocks delimit the **direct shock**, not everything that may move.
`SAM.balance` then restores accounting balance and can adjust cells outside
them, so the final column shares need not retain the Dirichlet distribution.
`balance_kwargs` controls its tolerances; a tight absolute tolerance is useful
on small SAMs whose default absolute floor of one is large relative to their
flows. Every returned draw passes `check_balance` under the same tolerances.

`SAMDraws.points` records the equal sampling weights and
`sam_distance(original, draw)`, the KL divergence `D_KL(original || draw)` over
normalised nonnegative flows. KL is undefined with negative flows, so the column
is NaN in that case rather than a distance invented by shifting the data. For a
forecast comparison, call `sam_distance(observed, predicted)`. `sam_sweep` uses
Monte Carlo summaries and can draw elasticities on the same rows, ready for
coverage and probability-integral-transform calculations. Validation includes a
one-sector Leontief economy: a Dirichlet share in a two-cell column is Beta, and
`E[1 / (1 - a)] = (alpha + beta - 1) / (beta - 1)` for `beta > 1`.

### Under other closures

A closure is a model of who bears an adjustment, and two defensible ones can give
results of opposite sign. `closure_sweep` solves the reference and the scenario
under each closure of a list and measures the change, with the signature of
`spec_sweep`:

```python
grid = cge.closure_grid(investment=["savings_driven", "fixed_real"],
                        foreign=["fixed_saving", "fixed_rate"],
                        labour=["full_employment", "wage_curve"],
                        natural_unemployment=0.08)
sweep = cge.closure_sweep(econ, grid, scenario=policy, measure=gdp_pct)
sweep.table             # one row per closure
sweep.failed            # {label: reason} for the combinations refused or unsolved
sweep.range()           # min, max, spread, the closures at the ends, same sign or not
```

`closure_grid` gives every combination of the `Closure` fields it is given,
labelled by the fields that differ from the default. A combination `Closure`
refuses, or one under which the model does not solve, gets a row of NaN and its
reason in `failed`, and the sweep goes on: which combinations are impossible is
part of the answer. With `years` (and `growth`, `economy`) both are paths of
`run_path`, and `measure` receives the two `PathResult` objects.

### Results in money

A percent change says nothing of size. `in_money` puts a counterfactual in levels:

```python
cge.in_money(base, alt, unit=1e-3, label="R$ billion")   # a SAM in R$ million
```

One row per item, by block: `final_demand` (consumption, government,
investment, exports, imports and GDP), `factor_income`, `households` (the income
a household can spend: net of direct tax and of payments straight to an
activity) and `government` (revenue, consumption, transfers, saving). Columns:
the reference and the alternative at current prices and their change, the same
at base-year prices and their change, and the change in volume in percent.

At base-year prices, final demand is the Laspeyres volume of `price_indices`
and GDP the expenditure side of `real_gdp`; a factor's income becomes its
services, at the base-year price of one; a household's income is deflated by its
own cost-of-living index against the base year's, so its change is the
equivalent variation valued at base-year prices; the government's flows are
deflated by the GDP deflator. Current prices are in units of the numeraire, and
`unit` multiplies every level and change; the package does not know the SAM's
currency, and `label` only names it. Two paths give the table for every year
they share.

### One robustness report

`robustness_report` assembles those analyses for a period without hiding any of
the underlying objects:

```python
rel = cge.robustness_report(
    sam, spec, scenario=policy, measure=gdp_pct,
    sam_blocks={"intermediate": (comms, acts)},
    draws=200, sobol=False, figures=False)
rel.tables["monte_carlo"]
rel.tables["oat"]
rel.parts["closures"].range()
```

The tables cover sourced ranges, one-at-a-time results, quadrature, Monte Carlo,
closures, SAM uncertainty and money. `parts` contains the full
`SensitivityResult`, `ClosureSweep` and other public results. Sobol is opt-in.
The default closure grid crosses savings-driven and fixed-real investment with
fixed-saving and fixed-rate foreign closures; pass `closures=` to replace it.

The default `ranges="rose2023"` uses `0.5x, x, 1.5x` for value-added, Armington
and export-CET elasticities. Rose et al. (2023) is the source of the ±50%
endpoints; the triangular shape is a package convention, not a distribution
estimated in that paper. `ranges="arndt2002"` gives the Armington support
`(0.3, 1.5, 9)` and CET support `(0.3, 1.5, 5)` of Arndt, Robinson and Tarp
(2002). Asymmetric supports skip quadrature with the reason in
`tables["quadrature"].attrs["skipped"]`; Monte Carlo and Sobol still run. Other
active elasticities need `ranges={name: (low, mode, high)}` or
`parameters={name: distribution}` and are named in the error if omitted.

A one-at-a-time point the model refuses is NaN in `tables["oat"]`, with its
parameter, endpoint and reason in `tables["oat_failed"]`. This matters, for
example, when halving an Armington elasticity of two reaches the singular
Cobb–Douglas case under fixed export shares. Other components' failures are not
silently converted into robust results.

Figures are optional and off by default. With `figures=True`, `figures` contains
a tornado and distribution plot per output, and Sobol bars when requested. This
needs `python -m pip install "cgelab[plot]"`; otherwise the error gives that
installation instruction. The base package does not import or depend on
matplotlib.

### Matching external targets

An estimated coefficient can discipline an elasticity without writing a second
model. `match_targets` finds ModelSpec fields whose outcomes are closest to the
external numbers, recalibrating and solving at every evaluation:

```python
fit = cge.match_targets(
    sam, spec,
    targets={"gdp_pct": published_gdp_change},
    parameters={"sigma_armington": (1.0, 4.0)},
    measure=gdp_pct,
    scenario=policy)
fit.parameters        # fitted elasticities
fit.achieved          # coefficients at the fit
fit.distance          # weighted squared distance
fit.success           # optimiser status, not merely a number to trust
fit.evaluations       # every evaluation, including solver errors
```

The objective is `r' Σ⁺ r`, with `r = achieved - target` and `Σ⁺` the
pseudo-inverse of the covariance. `targets` maps coefficient names to values;
`parameters` maps fields accepted by `spec_sweep` to finite `(low, high)` bounds.
`covariance` can be a full matrix or a diagonal of variances, and defaults to the
identity. A positive-semidefinite singular matrix is handled by its
pseudo-inverse; invalid shapes, asymmetry and negative eigenvalues are refused.
`options` is passed to `scipy.optimize.minimize` with L-BFGS-B.

This is a deterministic match, not an econometric estimator and not a confidence
interval. `MatchResult` keeps fitted `parameters`, `achieved`, `target`,
`residuals`, `distance`, `success`, `message`, `nit`, `nfev`, `evaluations` and
the fitted `spec`. A failed optimiser remains visible in the result. Validation
includes the closed-form weighted least-squares solution, a singular covariance
and recovery of an elasticity from a simulated CGE outcome.

### Treated against control

A regional counterfactual can be read as a treatment-effect contrast without
putting an estimator inside the model:

```python
effect = cge.treated_vs_control(
    reference, alternative, treated="r1", controls="r2", measure="output")
effect[["treated_change", "control_change", "effect"]]
```

The effect is the treated region's change minus the controls' weighted change.
Controls are equally weighted by default; `weights={region: value}` is
normalised to one. `measure` is `"output"`, `"real_income"`,
`"factor_demand"`, `"bilateral"`, a callable `f(period, region)`, or a mapping
of result names to any of them. The table also keeps both levels and changes so
the contrast can be audited.

This is arithmetic on solved equilibria, not an identification strategy or a
standard error. The reader supplies the research design and the estimator. The
tests cover a no-op, a shock only in the treated region, a common shock that
cancels, custom measures and bilateral outflows.

### Fixed-real investment in a regional model

`Closure(investment="fixed_real")` also works in the regional solver:

```python
fixed = cge.solve_period(econ, cge.Closure(investment="fixed_real"),
                         cge.Scenario(investment_level=80.0))
fixed.real["inv_real"]       # 80, at base-year purchaser prices
fixed.savings                # nominal saving at the solved prices
fixed.household_saving       # voluntary saving
fixed.forced_saving          # the per-capita levy that closes the account
```

The investment quantity is held, not scaled by current prices. The forced levy
is deducted from households in proportion to population, exactly as in the
one-region model, and `available` equals nominal savings at the solution. With
no scenario level, the calibrated investment is used. `savings_driven` remains
the default. The other regional restrictions (migration, alternative government
closures, margins and bilateral transport) stay explicit rather than being
silently relaxed.

### Government and investment accounts by region

National government and investment accounts remain the default. Additional
accounts can be mapped explicitly instead of splitting their totals by an ad hoc
regional rule:

```python
spec = regions.spec(
    government="r1:gov", investment="r1:s_i",
    regional_government={"r2": "r2:gov"},
    regional_investment={"r2": "r2:s_i"})
```

`regional_government` and `regional_investment` add demand and its financing:
factor income is assigned to each government, household saving includes every
investment account, and final demand enters the origin mix in `bilateral`. A
mapping must name a known region and account and may not reuse one account for
two regions. National accounts remain in place, so a mixed national/regional
model is valid. `real["gov_real"]` and `real["inv_real"]` include all of them,
and `fixed_real` holds the total investment at base-year prices.

---

## 13. Subtotals: which shock did what

A policy rarely moves one thing. When a tariff reform comes with a world-price
change and a tax replacement, the next question is how much of the result came
from which. Removing one shock at a time does not answer it, because the parts
then depend on the order of removal and do not add up once the model is
nonlinear.

```python
reform = cge.Scenario(tariff=np.zeros(n), world_price=new_world, product_tax=econ.ptax * 1.2)
cge.subtotals(econ, cge.Scenario(), reform,
              {"tariff": ["tariff"], "world": ["world_price"], "tax": ["product_tax"]},
              measure=lambda p: {"gdp": cge.real_gdp(p)["expenditure"],
                                 "er": p.exchange_rate})
```

The method is the one GEMPACK reports as subtotals (Harrison, Horridge and
Pearson, 2000). Every shock moves together along the straight line from the
reference to the alternative. At each point the change in a result is the sum,
over shocks, of a partial derivative times that shock's step. Integrating each
term separately splits the total change into contributions that add up to it
exactly, that do not depend on any ordering of the shocks, and that give each
shock its own effect whenever the result is additive in them.

The integral is Gauss–Legendre in the path variable (`nodes=6`). The derivatives
are central differences along each group's direction, each a solve warm-started
from the point on the path. The output has one row per measure: `reference`,
one column per group, their `total`, the `change` solved directly, and the
`residual` between those two. The residual is the numerical error, of order 1e-8
with the default step, and it is reported rather than spread over the groups.

Every field that differs between the two scenarios must be in a group, or its
effect would go unattributed. A group may name a mapping field whole
(`"activity_scale"`) or one entry (`"activity_scale[a_own]"`). Fields that are
not numbers, such as the recycling rule, must agree between the scenarios,
because there is no straight line between two rules. `path_decomposition` is the
same integral for any function of a dict of arrays.

---

## 14. Welfare decomposition

Where a welfare change came from: endowments, technology, each distortion, the
terms of trade, the external account. The accounting is Huff and Hertel's (2000),
written for this model's instruments.

```python
w = cge.welfare_decomposition(econ, cge.Scenario(), reform, closure=closure)
w[["absorption", "endowment", "technology", "allocative_tariff", "terms_of_trade",
   "external_account", "residual"]]
```

The measure is the Divisia volume of domestic absorption: household consumption
(at what households pay), own-account purchases, government consumption and
investment, each priced at current prices along the path from the reference to
the alternative. Differentiating the GDP identity and the balance of payments
splits it into:

| Term | What it is |
|---|---|
| `endowment` | factor prices times the change in factor use |
| `technology` | gross cost times the change in log productivity |
| `allocative_product_tax`, `_consumer_wedge`, `_activity_tax`, `_tariff`, `_output_tax` | for each instrument, its rate times the price of the taxed flow times the change in that flow: Harberger's triangles, one per distortion |
| `terms_of_trade` | the exchange rate times exports times the change in their price in foreign currency, less the same for imports |
| `external_account` | the change in foreign saving and its revaluation by the exchange rate, less factor income paid abroad |

The terms are integrated along the straight path, as in `subtotals`, and
`residual` is the change in absorption minus their sum. It is the test of the
identity. Across ten economies and closures (margins, CET with demand and supply
curves, wedges, own-account activities, LES, the government saving rate, foreign
saving in dollars), it stays below 1e-9.

Two points of reading. Under fixed export shares, exports sell at the producer
price, so their foreign price moves with domestic costs and shows up as a
terms-of-trade effect even for a small economy. And the measure is not the
households' EV. It counts government and investment volumes and prices along the
path. The `absorption_households`, `_own_account`, `_government` and
`_investment` rows split it by component. In Hosoe's `lrgcge` tariff abolition,
households gain while public consumption, which loses the tariff revenue, falls
by more, and the terms of trade take back more than the allocative gain.

Huff and Hertel validated their decomposition inside GTAP. That comparison needs
the GTAP database and is not made here. What is checked is that the identity
closes, that an endowment shock in an economy without distortions is all
endowment, that a small country has no terms-of-trade effect from a home shock,
and that a new distortion costs welfare.

---

## 15. GTAP data and HAR files

The GTAP database and every GEMPACK model store their arrays in header array
(HAR) files. `read_har` reads them without GEMPACK:

```python
base = cge.read_har("GTAP10A/GTAP/2014/basedata.har")
base["VIMS"]                       # Header('VIMS', RE, TRAD_COMM[65] x REG[141] x REG[141]: ...)
base["VIMS"].array                 # numpy, (65, 141, 141)
base["VIMS"].labels[1][:3]         # ['aus', 'nzl', 'xoc']
base["VIMS"].to_series()           # long form indexed by the set labels
prm = cge.read_har("GTAP10A/GTAP/2014/default.prm")
prm["ESBD"].to_series()["pdr"]     # 5.05, the Armington elasticity for paddy rice
```

The types are `1C` (strings), `2I` and `2R` (two-dimensional, unlabelled), `RE`
(reals with a set and labels per dimension, stored in full or sparse) and `RL`.
Values are single precision in the file. `write_har` writes the same types, for
handing data or results to a GEMPACK user.

The reader was written from the files and is checked two ways. On GTAP 10A, 2014,
the 86 headers of `sets.har`, `basedata.har`, `default.prm` and `metadata.har`
agree exactly with GEMPACK's own Python reader, harpy3
(`validation/har_harpy.py`). And four GTAP identities close to single precision:
CIF equals FOB plus margins flow by flow, world supply of transport margins meets
demand, cost equals revenue in every activity, and output is sold. Point
`CGELAB_GTAP_DIR` at a year folder to run those checks with the suite. The data
are licensed and stay out of the repository; `.gitignore` refuses `.har`, `.prm`
and `.pkg` files.

Aggregating GTAP and building a one-region SAM from it are the next steps, with
the regional work of 0.7.

---

## 16. Emissions and a carbon price

An emissions account is data, calibrated from base-year totals like the SAM:

```python
econ = cge.add_emissions(econ,
    intermediate=fuel_by_activity,     # DataFrame, commodities x activities: combustion in production
    household={"c_fuel": 5.0},         # by commodity bought
    government={"c_fuel": 1.0},
    process={"a_cement": 2.0})         # per unit of output, whatever the inputs
cge.emissions(cge.solve_period(econ))  # the totals again, by source
```

Each coefficient is a total divided by the base-year quantity of its flow, so the
base year returns the totals. A total on a flow the SAM records as zero has no
quantity to attach to, and is refused.

`Scenario(carbon_price=p)` is a specific tax, in numeraire units per unit of
emissions:

- **On production,** it is a tax per unit of output equal to `p` times the
  activity's emission intensity at the scenario's input coefficients. With Leontief
  intermediates, taxing the fuel an activity burns and taxing its output by that
  fuel's emissions are the same thing, and the test
  `test_on_production_it_is_an_activity_tax_by_emission_intensity` solves both and
  finds the same equilibrium. With an energy nest (see *Fuel that answers to
  prices* below), fuel use also answers to the price.
- **On households,** it is a specific tax on the fuel they buy.
- **Government's own fuel use** emits and is not taxed, because it would pay the
  tax to itself.

The revenue goes to government, as `carbon_production` and `carbon_households` in
the government account. A revenue-neutral closure recycles it:

```python
neutral = cge.Closure(government_target="revenue", government_adjusts="product_tax")
r = cge.solve_period(econ, neutral, cge.Scenario(carbon_price=0.5))
r.tax_scale        # product tax rates scaled down by what the carbon tax raises
```

GDP from both sides, the welfare decomposition (with an `allocative_carbon_tax`
term, the price times the change in emissions) and subtotals all carry the carbon
price. The JuMP replica carries it too, with the household block written out per
household and commodity when households pay it.

### Fuel that answers to prices

With Leontief intermediates a carbon price can only change how much is produced,
never how it is produced. `ModelSpec.energy` names the energy commodities and
nests them:

```python
spec = cge.ModelSpec(..., energy=["c_coal", "c_oil", "c_gas", "c_ely"],
                     sigma_vae=0.5,      # value added against the energy bundle
                     sigma_energy=1.0)   # between fuels
```

Each activity buys an energy bundle, a CES across fuels, which substitutes for
value added in a value-added-energy bundle, a CES. The bundle's quantity per unit
of output is fixed, and so are the other intermediates. Both bundles are measured
in base-year value and their prices are one at the base year, so the nest
calibrates from the SAM alone. Each activity pays its own fuel prices: the
carbon price on a fuel enters the price that activity pays for it, in proportion
to its own emission coefficient, and carbon on non-energy inputs and process
carbon stay a tax per unit of output. Activities on fixed proportions (`leontief`,
or a negative value added) keep their fuel mix fixed.

**Where energy sits.** `energy_nest="value_added"` (the default) puts the energy
bundle against value added as a whole. `energy_nest="capital"` follows GTAP-E: the
bundle first joins a capital factor (`energy_capital`, default `"f_cap"`) in a
capital-energy composite with elasticity `sigma_ke`, and that composite sits inside
the value-added nest with the other factors, at `sigma_va`. A low `sigma_ke` makes
capital and energy complements while both substitute for labour, which is what a
carbon price that lowers the return to fuel-intensive capital needs. With
`sigma_ke = sigma_va`, the two-level nest is a single CES over the other factors,
capital and energy, and the tests check that it then solves to the same equilibrium
as `energy_nest="value_added"` with `sigma_vae = sigma_va`.

`PeriodResult.input_coefficients` and `value_added_per_unit` are the coefficients
the equilibrium used, and `emissions`, the accounts and the welfare decomposition
read them. With both elasticities at zero, the nest is the Leontief model: the
tests solve both under carbon, tax, productivity and world-price shocks and find
the same equilibrium. With them above zero, the fuel ratios satisfy the CES
first-order conditions at the prices each activity paid, carbon included, and a
higher elasticity cuts emissions more for the same carbon price.

---

## 17. Forward-looking investment, through dsgelab

`run_path` solves one period after another, and no agent looks ahead.
`cgelab.bridge` builds the other object: a representative household that chooses
investment by an Euler equation, while every period's equilibrium is still the SAM
model. [dsgelab](https://github.com/GustavoHM/dsgelab) solves the intertemporal
part; it never reads the SAM. Install it with `pip install "cgelab[dsge]"`.

```python
import dsgelab as dsg
from cgelab import bridge

model, params, guess = bridge.ramsey(econ, depreciation=0.05, discount=0.05)
ss = dsg.steady_state(model, guess, params)             # the model's steady state
path = dsg.perfect_foresight(model, ss, ss, params=params, T=150,
                             initial_states=[0.75 * ss.values["k"]])
```

The model has one state, the capital stock `k` (base-year value), and one jump,
real investment `inv`. Given both, `solve_period` returns the period equilibrium
under `investment="fixed_real"`, with capital services `uc0 * k` at the base-year
user cost `uc0 = discount + depreciation` (the conversion `run_path` uses). What
dsgelab sees is two residuals:

- `k' = (1 - δ) k + inv`;
- `C^-γ P_I / P_C = β C'^-γ (uc0 r' + (1 - δ) P_I') / P_C'`, with `β = 1/(1 + discount)`.

Here `C` is real household consumption, `P_C` its price, `P_I` the price of the
investment bundle and `r` the rental price of capital services. A SAM of a few
hundred accounts does not become a DSGE with a few hundred variables: the static
block is substituted out by the period solver.

**An announced policy** is a sequence of scenarios, one per date.
`bridge.announced(params, scenarios)` gives each date its own scenario and the
next date's, because the Euler equation at `t` solves the period at `t + 1` too.
A tax announced for date 5 moves investment at date 0. `run_path` cannot do that.

**Aggregate uncertainty.** `tfp_persistence=ρ` adds a TFP state with an AR(1) in
logs and a shock `eps_a`. `dsg.solve` then gives the first-order solution and
impulse responses, and `dsg.solve_second_order` gives risk corrections.

**What it is checked against.** In a one-good closed economy with Cobb–Douglas
production (`toy.build_one_good()`), the Ramsey CGE is the stochastic growth
model, and the tests solve both. The steady state is the analytic one (`k = 400`,
`inv = 20`). The first-order transition matrices agree to 1e-7, and so do the
first forty periods of the perfect-foresight path.

**The base year is not, in general, a steady state.** It is one only when the
SAM's investment equals the depreciation of the stock the user cost implies.
`dsg.steady_state` finds the model's own steady state, which is another point, and
a published table has to say which of the two it reports.

---

## 18. Physical units

```python
econ = cge.add_physical_units(econ,
    commodities={"c_ely": ("GWh", 48_000.0)},     # domestic sales plus imports, base year
    activities={"a_h2": ("t", 1_200.0)})          # physical output, base year
p = cge.physical(cge.solve_period(econ, closure, scenario))
p["supply"], p["use"]["c_ely"], p["output"], p["intensity"][("c_ely", "a_h2")]   # GWh per tonne
```

The model's money units differ by flow (producer, border and tax-inclusive values),
so the account anchors on two things. A commodity's physical content is attached
to its Armington composite, the domestic and imported variety that activities and
households buy. An activity's is attached to its activity level. Both are
calibrated from base-year totals, and the link is fixed: a unit of composite
carries the same physical content in every period. That is right for a homogeneous
good, where a kilowatt-hour is a kilowatt-hour, and wrong for a composite of
qualities, which is why the account is opt-in, commodity by commodity. Physical
supply meets physical use in every equilibrium, margins included. The intensity is
constant under Leontief inputs and moves with fuel prices under an energy nest.

`physical_coefficient` runs the other way, for building a SAM. It turns a technical
coefficient stated in physical units (kilowatt-hours of electricity per kilogram
of output, say) into the money entry per unit of output value, at purchaser prices.
Capacity in physical units, as a constraint, needs complementarity and waits for 0.8.

---

## 19. Microsimulation: poverty and inequality

The SAM has a few household accounts; poverty and inequality live in the
households a survey records. `cgelab.micro` carries a solved counterfactual to
them top-down. Each survey household belongs to an account, its income from
each source moves as that account's income from the source moves in the model,
and its cost of living moves with its own budget at the prices its account pays.
The survey households have no behaviour of their own here.

```python
_, econ = toy.build_with_segmented_labour()
floor = cge.LabourMarket(rigid={"f_for": 1.1}, informal="f_inf")
ref = cge.solve_period(econ)
alt = cge.solve_period(econ, cge.Closure(labour_markets={"labour": floor}))
micro = cge.microsimulate(toy.survey(), ref, alt)
cge.inequality(micro)          # Gini, Theil T and L, GE(2), Atkinson, in both periods
cge.poverty(micro, 3.0)        # FGT 0, 1 and 2 at a line of 3 per person
```

**The survey.** `cge.Survey(frame, sources=..., budget=...)` takes one row per
household, with its account (`group`), survey weight and size. `sources` maps
each income column to what moves it: a factor (`"f_cap"`), a factor in one
activity (`"f_lab:a_agr"`, which follows that activity's price under imperfect
mobility), `"transfers"`, `"remittances"`, `"recycled"` (wedge revenue returned
to households) or `"fixed"` (held in nominal terms). `budget` maps spending
columns to commodities; without it, each household uses its account's budget.
`toy.survey()` is a synthetic survey of twelve households, written by hand.

**Reconciliation.** A survey never adds up to the SAM. By default each source of
each account is scaled so that the weighted survey total is the account's, and
`micro.reconciliation` lists the factors; on the toy survey, formal wages of
`hh_1` are scaled by 60/51. A survey and a SAM that disagree on whether an
account has a source at all are refused, by account and source, because no
scaling reconciles them. `reconcile=False` keeps the survey's levels and applies
only the relative changes, and the micro totals then no longer add up to the
model. A source an account did not have in the reference and has in the
alternative, a new transfer, is shared per person among its households.

**Incomes and prices.** `account_sources(period)` is what each account earns,
by source, in a solved period; the factor columns and the last three add up to
`period.hh_income`. Disposable income takes the account's direct tax rate. The
price index is geometric in the household's own budget shares, the exact index
for Cobb–Douglas and the one the model uses for its accounts, so a survey with
one household per account, holding the SAM's incomes, gives back the model's
real incomes. `price_index="laspeyres"` is the alternative; it is never below
the geometric one. Real income is at the reference period's prices, and so is
the poverty line.

**Reweighting.** Under labour segments, scaling every household's wages by its
account's would spread a lost formal job thinly over everyone who had one. With
`reweight=True`, the default, who works where changes through the survey weights
instead. Within each account they are calibrated by raking (Deville and Särndal,
1992), the weights staying positive: the survey's income in each segment, at
base-year wages, moves as the account's employment in the segment does; its
unemployed persons (a `Survey.workers` column mapped to `"unemployed"`) move as
the account's queue does, when the model has one; the number of households and
every other source stay where they were. Each segment's income per worker then
moves with its wage, so the micro totals are still the model's.
`micro.calibration` lists each constraint's base, target and the total reached;
`micro.workers` the persons in each `workers` column with the old weights and
the new; `frame["weight_alt"]` the weights; and every table uses each period's
own weights. A survey whose households in an account cannot reach the targets
with positive weights is refused, naming the constraint: Harris–Todaro
unemployment moves a lot for a small change in the rigid wage, and a small
survey spans only so much of it. One household per account cannot be reweighted
at all; `reweight=False` keeps the weights.

### Behavioural microsimulation

The reweighting changes how many households are like each other; it does not
say which person loses a formal job. Following Bourguignon, Robilliard and
Robinson (2005), `persons=` lets the people decide. A `PersonSurvey` holds one
row per person of the households: the household (its row in the household
survey), a status (a labour segment, `"unemployed"`, `"inactive"`), labour
earnings and characteristics. A `Behaviour` is estimated on it, unless one is
given: earnings equations by segment, `ln y = a_k + X b_k + e`, by weighted
least squares, and the occupational choice by a multinomial logit on the
characteristics, by maximum likelihood (no dependency beyond scipy).

```python
people = cge.microsimulate(toy.survey(), ref, alt, persons=toy.persons())
people.workers                 # persons by status: hh_1 formal 7.5 to 7.09, informal 2.5 to 3.31
people.calibration             # the intercepts' targets, reached
```

A counterfactual reaches the persons through two sets of intercepts, by SAM
account: the choice intercepts move until employment in each segment changes as
the account's does in the model (and unemployment, where the model has a queue),
and the earnings intercepts until earnings per worker change as the account's
income per worker does. The household survey's segment columns are replaced by
the members' earnings, reconciled with the SAM the same way, and the weights
stay. The micro labour income of every account is the model's, segment by
segment.

The choice residuals are what the observed choice reveals. `method="draws"` is
BRR's procedure: Gumbel errors drawn conditional on each person's observed
status, so that the base reproduces the survey; a mover into a segment earns the
equation's prediction plus a normal draw. The results depend on the seed, so
`seed` and `replications` are arguments, the tables report each measure's mean
over the replications and its standard error (`alt_se`), and
`result.replications` keeps every replication's real incomes.
`method="expected"`, the default, is the limit of those replications in closed
form: the probability that a person observed in `j` moves to `k` when the
intercepts shift by `δ`, `transition_probabilities(V, δ)`, which is a logit for
staying and a piecewise analytic integral for moving. It is deterministic and
smooth in `δ`, which is what lets the intercepts hit the model's targets
exactly; draws hit them only up to the granularity of a finite sample.

### Survey data from the IBGE

The IBGE publishes its microdata as fixed-width text with a SAS reading program
that gives each field's position, width and type. `read_ibge(data, program)`
reads the data with that program, so no position is written by hand and a new
layout needs no new code; `read_sas_program` returns the layouts, one per data
step (the POF has sixteen). Text fields stay text, blanks are missing, a `w.d`
informat places the decimal point as SAS does, and `variables=` and
`chunksize=` keep a quarter of the PNAD Contínua within memory.

```python
survey, persons = cge.pnadc("PNADC_032023.txt", "input_PNADC_trimestral.txt",
                            segments={"formal": "f_for", "informal": "f_inf"},
                            group=my_accounts)          # a column, or a function of the households
households = cge.pof("Dados_POF/", "Leitura dos Microdados - SAS.txt",
                     commodities=correspondence, group=my_accounts)
```

`pnadc` gives a household survey and a person survey for the behavioural
microsimulation. A person under 14 or out of the labour force (VD4001 = 2) is
inactive, an unoccupied one (VD4002 = 2) unemployed, and an occupied one
informal by the IBGE's proxy: VD4009 02 (private employee without a signed
card), 04 (domestic worker without one) or 10 (unpaid family worker), or 08 and
09 (employer, own-account) without CNPJ (V4019 = 2); formal otherwise. Earnings
are VD4019, usual monthly earnings from all jobs. V4019 exists from the fourth
quarter of 2015; for earlier quarters `missing_cnpj` must say how employers and
own-account workers are classified. The household is UPA + V1008 + V1014 and its
weight V1028.

`pof` gives a survey of consumption units. Spending comes from the four records
of the IBGE's general spending table, each value `V8000_DEFLA ×
FATOR_ANUALIZACAO / 12`, times `V9011` for the quadros the IBGE treats as monthly
(estimated rent; collective quadros 10 and 19; individual quadros 44, 47 to 50):
monthly, at January 2018 prices. `commodities` maps a prefix of the five-digit
item code to a SAM commodity, the longest prefix winning; what it does not map
stays in `unmapped_spending`, outside the budget. Work income and other income
(by `income`, a prefix map to columns) come out monthly the same way; their
sources are `"fixed"` until mapped to the model's. The size leaves out domestic
employees and their relatives (V0306 18 and 19). The definitions were checked
against the IBGE's dictionaries, reading programs and calculation memo, and
`validation/ibge_layouts.py` compares the layouts with the IBGE's R programs.
The microdata never enter the repository; a check on them runs when
`CGELAB_IBGE_DIR` points at them.

**Measures.** `gini`, `fgt`, `generalized_entropy` (`theil` is GE(1),
`mean_log_deviation` GE(0)) and `atkinson` take any vector and weights.
`ge_decomposition` splits GE(α) into inequality within and between groups,
exactly, and `fgt_decomposition` gives each group's contribution to FGT(α).
`inequality` and `poverty` tabulate them per person, weighting by each period's
survey weight times the household size; `equivalence=θ` divides by `size ** θ`
instead (0.5 is the square-root scale). `poverty(..., by_group=True)` adds each
account's contribution.

---

## 20. Building a SAM from supply and use tables

Most of the work in an applied model happens before `calibrate`. The supply and
use tables give the production block and final demand; who receives the income
comes from the integrated economic accounts, or is assumed. `sut_to_sam`
assembles both in the layout `calibrate` reads and records where each block
came from.

```python
make = pd.DataFrame({"c_x": [100.0]}, index=["a_x"])            # activities by commodities
use = pd.DataFrame({"a_x": [0.0]}, index=["c_x"])               # commodities by activities
final = pd.DataFrame({"hh": [80.0], "s_i": [20.0]}, index=["c_x"])
va = pd.DataFrame({"a_x": [60.0, 40.0]}, index=["f_lab", "f_cap"])
sut = cge.SupplyUse(make=make, use=use, final=final, imports=pd.Series({"c_x": 0.0}),
                    value_added=va)
sam = cge.sut_to_sam(sut, source="hand-written tables")
sam.provenance            # every block: its source, what it is, its total
```

**The tables.** `SupplyUse` holds labelled frames, and the labels become the
accounts:
- `make` is output at basic prices; `use` is intermediate use at purchaser
  prices; `final` is final demand at purchaser prices;
- `imports` are at border prices; `value_added` is factors by activities;
- `product_taxes` and `production_taxes` are net of subsidies;
- `margins` holds a margin commodity's services in every commodity's supply,
  with minus its total on its own column, so each row adds up to zero;
- `tariffs` and `output_taxes` go to accounts of their own when the model reads
  them so;
- `direct_sales` is own-account production sold straight to households.

`final_roles` says what each final-demand column is: households (each becomes a
household account), government, investment (summed into one account) or
exports. A column named like the government, investment or rest-of-world
account needs no role.

**The institutions.** `Institutions` holds what the tables do not: each factor's
income by recipient (households, government, abroad), direct taxes, government
transfers, transfers from and to abroad, and any other flow as
`(receiver, payer, amount, source)`. Without it, all factor income goes to the
one household account, and the provenance says that this was assumed.

**What is refused, and what is residual.** Commodities whose supply and use
disagree, activities whose output and costs disagree, margin rows that do not
add up to zero, factor income paid out that differs from value added, and an
institutional flow that leaves an activity, a commodity or a factor unbalanced
are all refused by name. The savings of each household, of government and of
the rest of the world are residuals, recorded as such. `SupplyUse.gdp()` gives
GDP from production, expenditure and income, which agree when the tables do.
Every SAM of the toy rebuilds from its own tables, cell by cell.

### Disaggregating with household surveys

`split_households(sam, survey, account="hh")` splits one household account into
the groups of a `Survey` (its `group` column: deciles, formal and informal
households, regions). Every cell of the account is shared by each group's share
of the weighted survey total for that cell: income from a factor by the columns
`Survey.sources` maps to it (`"transfers"` stands for government and
`"remittances"` for the rest of the world), spending on a commodity by its budget
columns. `incomes=` and `outlays=` name the columns for other cells, a direct
tax say, and `fallback=` shares a cell the survey has nothing on by
`"population"` or `"households"`; a cell with neither is refused by name. Each
group's saving is the residual, and a negative one is kept: it is data.
`group_population(survey)` gives the persons in each group, which `calibrate`
takes as `population`.

`split_factor(sam, "f_lab", by_activity, owners=...)` splits a factor: in each
activity its payment is shared in proportion to a column of `by_activity` (new
factors by activities), and its income goes to its recipients, government and
abroad keeping their shares and households following `owners` (their savings
recomputed). `segment_shares(survey, persons, segments=[...])` gives both from
the PNAD Contínua read with `activities=` (V4013, the activity of the main job):

```python
by_activity, owners = cge.segment_shares(survey, persons, segments=["f_for", "f_inf"])
split = cge.split_factor(sam, "f_lab", by_activity, owners=owners, source="PNADC 2023q3")
```

Summing the groups or the new factors gives back the original exactly. Two
equivalences check that the split changes nothing it should not: households
split with identical proportions have the aggregate's equilibrium under
Cobb–Douglas demand, and labour split into segments that share one wage (a
group of `labour_segments` with no rigid segment) has the single factor's,
because a CES with two inputs at the same price is the CES with their sum.

### The IBGE's supply and use tables

`read_tru(table1, table2)` reads the IBGE's supply and use tables (Sistema de
Contas Nacionais, tables 1 and 2 at current prices, any level: 12, 20, 51 or 68
activities) into a `SupplyUse`, in R$ million, accounts named by the IBGE's
codes (`a_0191`, `c_01911`) with their descriptions in `sut.labels`. Reading
the `.xls` files needs `xlrd` (`pip install "cgelab[ibge]"`).

```python
sut = cge.read_tru("68_tab1_2021.xls", "68_tab2_2021.xls")
sut.gdp()                  # 9,012,142 by all three sides: the IBGE's published 2021 GDP
sam = cge.sut_to_sam(sut, institutions)
```

Labour is *Remunerações*; capital *Excedente operacional bruto*. Mixed income
is a factor of its own by default (`mixed_income="separate"`, `f_mix`), or joins
labour or capital, explicitly, because which it is moves the distribution.
Taxes on products are the import tax, IPI, ICMS and other taxes net of
subsidies (`tariffs=True` sets the import tax apart); other taxes on
production, net of subsidies, are taxes on production. The trade and transport
margin columns are positive on the goods and negative on the products that
provide them, and each good's margin is shared among its providers in
proportion to what each provides: an allocation the provenance records as
assumed. The NPISH join households, or government with `npish=`. The published
2021 table at 68 activities closes exactly and gives the GDP the IBGE publishes
for that year.
