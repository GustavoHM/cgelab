# cgekit API reference

Everything listed here is importable from the top-level package
(`import cgekit as ck`).

---

## SAM

### `SAM(frame, name="sam")`

Long-form social accounting matrix with columns `row`, `col`, `value` — the
payment **from** the column account **to** the row account.

Constructors: `SAM.from_long(df, row="row", col="col", value="value", name="sam")`,
`SAM.from_square(df, name="sam")`, `SAM.read(path, name=None, **kwargs)`
(`.parquet` or `.csv`).

| Member | Meaning |
|---|---|
| `accounts` | sorted account names |
| `total` | sum of absolute values |
| `matrix(accounts=None)` | dense square frame |
| `block(rows, cols)` | sub-matrix as an array, in the given order |
| `receipts()`, `outlays()`, `imbalance()` | row totals, column totals, their difference |
| `check_balance(rel_tol=1e-9, abs_tol=1.0, raise_on_fail=True)` | the hard gate; returns `ok`, `worst`, `limit`, `offenders`, `imbalance` |
| `balance(rel_tol=1e-9, abs_tol=1.0, max_iter=200, name=None)` | sign-preserving RAS; raises `BalanceError` rather than returning a SAM that only looks balanced |

`BalanceError` is raised by both.

---

## Specification

### `ModelSpec(...)`

| Field | Default | Meaning |
|---|---|---|
| `activity_prefix`, `commodity_prefix`, `household_prefix` | `"a_"`, `"c_"`, `"hh_"` | account classification by prefix |
| `activities`, `commodities`, `households` | `None` | explicit lists; they win over the prefixes |
| `factors` | `("f_lab", "f_cap")` | ordered factor accounts |
| `government`, `investment`, `rest_of_world` | `"gov"`, `"s_i"`, `"row"` | role accounts |
| `numeraire_factor` | first factor | its price is fixed and its market left to Walras |
| `value_added` | `"ces"` | `"ces"`, `"cobb_douglas"` or `"leontief"` |
| `sigma_va`, `sigma_armington` | `0.5`, `2.0` | scalar or mapping by account name |
| `export_rule`, `sigma_cet` | `"fixed_shares"`, `2.0` | `"cet"` is a robustness variant; `sigma_cet` scalar or mapping by commodity |
| `household_demand`, `frisch` | `"cobb_douglas"`, `-1.5` | `"les"` uses `frisch` for subsistence |
| `negative_value_added` | `"leontief"` | or `"net"` |
| `margins` | `True` | read the commodity-to-commodity margin block |

Methods: `classify(sam) -> AccountLayout`, `sigma_va_vector(activities)`,
`sigma_armington_vector(commodities)`.

`AccountLayout`: `activities`, `commodities`, `households`, `factors`,
`government`, `investment`, `rest_of_world`.

---

## Calibration

### `calibrate(sam, spec=None, *, income_elasticities=None, population=None, check_balance=True) -> Economy`

`income_elasticities` maps commodity to income elasticity (LES only).
`population` maps household account to a household count — the weight of a per
capita transfer, and neither the income nor the expenditure share.

### `Economy`

Ordered name lists `acts`, `coms`, `hhs`, `factors`; counts `n_act`, `n_com`,
`n_hh`, `n_fac`; index lookups `activity(name)`, `commodity(name)`,
`household(name)`, `factor(name)`, `numeraire_index`.

| Field | Shape | Meaning |
|---|---|---|
| `X0`, `Q0` | `(na,)`, `(nc,)` | base activity and commodity output |
| `io` | `(nc, na)` | intermediate per unit of activity output, basic prices |
| `msh` | `(na, nc)` | market shares, columns sum to one |
| `theta`, `aunit` | `(nf, na)` | CES value-added shares; Leontief factor per unit output |
| `vunit`, `leontief`, `own_account` | `(na,)` | value added per unit; fixed-proportions flag; own-account flag |
| `sigma_va`, `sigma_arm` | `(na,)`, `(nc,)` | elasticities |
| `ptax`, `atax` | `(nc,)`, `(na,)` | product tax on absorption; specific tax per unit of output |
| `imp0`, `exp0`, `dom0`, `exp_share`, `dom_share` | `(nc,)` | the trade block |
| `A0`, `mq`, `arm_share` | `(nc,)`, `(nc, nc)`, `(nc,)` | absorption, margin per unit, Armington composite per unit |
| `hh_budget`, `hh_committed` | `(nh, nc)` | budget shares; LES subsistence quantities |
| `hh_savrate`, `hh_income0`, `pop_share` | `(nh,)` | saving rate, base income, population weight |
| `hh_facshare` | `(nh, nf)` | ownership, columns sum to one |
| `hh_direct` | `(nh, na)` | payments made straight to an activity |
| `gov_dem`, `inv_dem` | `(nc,)` | government and investment demand, purchaser prices |
| `fac_supply` | `(nf,)` | factor endowments |
| `foreign_sav` | scalar | foreign saving, and so the current-account target under `foreign="fixed_saving"`: the one field to move |
| `trade_balance0` | scalar | base-year trade balance, a record only; the solver does not read it |
| `hh_dtax` | `(nh,)` | direct tax rate on gross income (`gov <- hh`) |
| `hh_gov_transfer`, `hh_remit_net` | `(nh,)` | transfers from government; net transfers from abroad; numeraire units |
| `fac_gov_share`, `fac_row_share` | `(nf,)` | share of each factor's income paid to government and abroad |
| `hh_gross0`, `hh_spend0` | `(nh,)` | base gross income (net of transfers sent abroad); base commodity spending |
| `sigma_cet` | `(nc,)` | CET elasticities, read under `export_rule="cet"` |
| `diagnostics` | dict | negative-value-added activities and treatment, own-account list |

---

## Closure and scenario

### `Closure(numeraire=1.0, investment="savings_driven", government="real_fixed", foreign="fixed_saving", exchange_rate=1.0, tolerance=1e-9)`

`investment` ∈ `{"savings_driven", "fixed_real"}`;
`government` ∈ `{"real_fixed", "nominal_fixed"}`;
`foreign` ∈ `{"fixed_saving", "fixed_rate"}`.
Properties: `solves_exchange_rate`, `describe()`.

### `Scenario(...)`

| Field | Shape | Meaning |
|---|---|---|
| `name` | str | carried into results and summaries |
| `factor_supply` | `(nf,)` | absolute endowments |
| `factor_ownership` | `(nh, nf)` | ownership shares, columns sum to one |
| `productivity` | `(na,)` | Hicks-neutral multiplier per activity |
| `activity_scale` | mapping | own-account activity levels, as a multiple of the base year |
| `product_tax`, `product_tax_firm` | `(nc,)` | consumer rate; non-creditable part facing firms |
| `consumer_wedge`, `government_wedge` | `(nh, nc)` | ad valorem wedges; the first is recycled to households, the second to the budget |
| `wedge_revenue_to`, `wedge_revenue_factor` | str/float/callable, str | who receives the recycled revenue |
| `transfers` | `(nh,)` | lump sum, positive for a receipt |
| `government_absorbs` | scalar | cost charged to the budget |
| `world_price` | `(nc,)` | import price multipliers |
| `export_price` | `(nc,)` | world export price multipliers; `export_rule="cet"` only, refused otherwise |
| `foreign_factor_share` | `(nf,)` | share of each factor owned abroad, overriding the SAM's; `run_path` sets it from the capital it tracks |
| `fdi` | mapping | `{capital factor: amount}` in numeraire units: raises foreign saving, and `run_path` adds the capital it buys to the foreign-owned stock |

`sc.with_(**changes)` returns a copy with fields replaced.

### `consumer_wedge(econ, rates, households=None) -> ndarray`

Build an `(n_hh, n_com)` wedge from a commodity → rate mapping, optionally
restricted to a subset of household accounts.

---

## Solving

### `solve_period(econ, closure=None, scenario=None, *, warm_start=None, raise_on_fail=True) -> PeriodResult`

Unknowns: one price per non-numeraire factor, one for the external closure (the
exchange rate under `foreign="fixed_saving"`, foreign saving under
`"fixed_rate"`), and under `investment="fixed_real"` the forced household saving
that finances the fixed investment.

### `PeriodResult`

| Member | Meaning |
|---|---|
| `factor_prices`, `exchange_rate`, `z` | the solved unknowns |
| `pa`, `pd`, `pq`, `pqf`, `pbase` | activity, domestic, consumer, firm and pre-tax commodity prices |
| `pe` | price received for exports: `exchange_rate × export_price` under CET, `pd` under fixed shares |
| `hh_direct_tax`, `factor_income_abroad`, `foreign_saving` | direct tax by household; factor income paid abroad; the period's foreign saving, FDI included |
| `Q`, `X`, `absorption`, `imports`, `exports` | the quantity system |
| `hh_quantity`, `hh_income`, `hh_commodity_spend`, `hh_direct_outlay`, `hh_wedge_paid` | the household block |
| `gov_quantity`, `investment_quantity`, `savings`, `factor_demand` | the rest of final demand |
| `wedge_revenue`, `government_cost` | recycled revenue; cost falling on the budget |
| `real` | `hh_real`, `gov_real`, `inv_real`, `hh_sav_real`, `gov_sav_real`, `for_sav_real`, `hh_price_index`, `hh_real_income`, `hh_committed_value` |
| `gaps`, `walras`, `residual`, `converged`, `attempt`, `message` | the evidence |
| `price(commodity)`, `household_real_income()`, `summary()` | helpers |

### `value_added_price(econ, w)` / `factor_intensity(econ, w, *, vunit=None, aunit=None)`

The value-added nest on its own. `vunit` and `aunit` override the calibrated
coefficients, which is how a productivity path enters: the CES branch is
scale-free and unaffected, the fixed-proportions branch is a ratio of the two and
must see both.

### Functional forms

`ces_price(shares, prices, sigma)`, `ces_intensity(shares, prices, unit_cost, sigma)`,
`cobb_douglas_price(shares, prices)`,
`armington(domestic_share, p_domestic, p_import, sigma) -> (pq, dsh, msh)`,
`cet(export_share, p_domestic, p_export, sigma) -> (pz, dsh, esh)`,
`cet_domestic_price(export_share, p_output, p_export, sigma) -> p_domestic` (the inverse of `cet` in its domestic price).

Every nest survives the degenerate cases a real SAM contains — an activity with
no value added, a commodity with no imports, a share vector summing to zero — and
returns finite numbers, because one NaN in a 131-commodity system propagates
everywhere and the traceback points at the solver instead of at the account that
caused it.

---

## Dynamics

### `Growth(labour=0.0, tfp=0.0, depreciation=0.05, discount=0.05, labour_factor="f_lab", capital_factor="f_cap", allocation="proportional")`

`capital_factor` is one factor or a sequence of them, each accumulated from its
own service flow; `depreciation` is a rate or a mapping by capital factor.
`allocation` shares domestic investment among them: `"proportional"` to the
stocks, or a number `e` making shares proportional to `K_k (r_k / r_mean)^e`.
`user_cost` is `discount + depreciation`, the rate that converts the SAM's
capital service flow into a stock.

### `run_path(econ, closure=None, *, years, scenario=None, growth=None, economy=None, raise_on_fail=True) -> PathResult`

`scenario` is a single `Scenario` or `f(year, index) -> Scenario`. Factor supplies
set inside a per-year scenario override the accumulated ones. `economy` is
`f(year, index) -> Economy`, for changes that are not scenario fields (input
coefficients, foreign saving, transfer rules); it must keep the accounts of
`econ`, e.g. `dataclasses.replace(econ, io=...)`.

### `PathResult`

`years`, `periods`, `capital_stock` (`(T,)`, or `(T, k)` with several capital
factors), `foreign_capital` (same shape), `capital_factors`, `converged`, `worst_walras`;
`path[year]` gives the `PeriodResult`;
`aggregates()`, `households()`, `commodity(name)` return tidy frames.

---

## Welfare

### `equivalent_variation(reference, alternative, *, base_price_index=None) -> DataFrame`

Columns `year`, `account`, `ev`, `ev_base`, `ev_pct_account`,
`welfare_income_ref/alt`, `real_income_ref/alt`, `wedge_paid_change`,
`direct_outlay_change`.

### `ev_by_group(ev, groups) -> DataFrame`

`groups` maps every account to a group label; a missing account raises. Returns
`year`, `group`, `ev`, `ev_base`, `ev_pct`, `real_income_ref/alt`, computed by
summing real incomes within the group **before** taking the ratio.

### `compensating_variation(reference, alternative) -> DataFrame`

Columns `year`, `account`, `cv`.

---

## Diagnostics

| Function | Answers |
|---|---|
| `base_year_reproduction(econ, closure=None, tol=1e-7)` | does the model return the SAM with no shock? Per-check detail in `checks` |
| `homogeneity_test(econ, closure=None, factor=2.5, tol=1e-7)` | does scaling the numeraire leave every real magnitude untouched? |
| `walras_check(result, tol=1e-8)` | does the market the solver was not told to clear, clear? |
| `report(econ, closure=None)` | all of the above, rendered for a log or a referee appendix |

---

## Bundled examples

```python
from cgekit.examples import toy

sam, econ = toy.build()                        # two sectors, no margins
sam, econ = toy.build_with_margins()           # national-accounts margin block
sam, econ = toy.build_with_own_production()    # an activity selling to households
sam, econ = toy.build_with_twin_accounts()     # identical accounts, for composition tests
sam, econ = toy.build_with_institutions()      # direct taxes, transfers, remittances, foreign capital
sam, econ = toy.build_with_specific_capital()  # industry on a capital factor of its own
spec = toy.spec(household_demand="les", frisch=-2.0)
```

Every entry of these SAMs is written out in the source, so the base year is known
on paper and the reproduction test compares the solver against arithmetic.

## Command line

```
cgekit                              # the bundled toy economy plus diagnostics
cgekit data/processed/sam.parquet --sigma-armington 2.0 --sigma-va 0.5
```
