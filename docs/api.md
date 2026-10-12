# cgelab API reference

Everything listed here is importable from the top-level package
(`import cgelab as ck`).

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
| `export_demand` | `None` | export-demand elasticity η, `E / E0 = (pWe / export_price)^(-η)`; scalar or mapping by commodity, a missing commodity stays a price taker; `export_rule="cet"` only |
| `import_supply` | `None` | import-supply elasticity ψ, `M / M0 = (pWm / world_price)^ψ`; scalar or mapping by commodity |
| `household_demand`, `frisch` | `"cobb_douglas"`, `-1.5` | `"les"` uses `frisch` for subsistence |
| `negative_value_added` | `"leontief"` | or `"net"` |
| `margins` | `True` | read the commodity-to-commodity margin block |
| `regions` | `None` | region names. Accounts are then `region:local`. Absent, the one-region model |
| `sigma_interregional` | `2.0` | CES across regions of origin, under the Armington. Scalar or mapping by local commodity |
| `tariff_account` | `None` | account that collects the import tariff from commodities (Hosoe's TRF). The tariff goes inside the Armington import price |
| `indirect_tax_account` | `None` | account that collects the indirect tax on all of output (Hosoe's IDT). The tax goes inside the gross output price the CET splits, so exports carry it |

Methods: `classify(sam) -> AccountLayout`, `prepare(sam)` (expands regional
accounts), `sigma_va_vector(activities)`, `sigma_armington_vector(commodities)`.

With `regions` set, the numeraire is one factor in one region (`"r1:f_lab"`).
Factors may be passed as local names and are expanded to every region. Buyers
pay a national composite: a CES of origins, then the usual Armington against
foreign imports. One region makes the CES a single variety. `PeriodResult.bilateral`
is `(origin, destination, local commodity)` for households and activities;
government and investment stay national. Margins, CET exports, LES, migration
and non-default closures are refused by name rather than ignored. `to_jump` and
`run_path` refuse more than one region.

`AccountLayout`: `activities`, `commodities`, `households`, `factors`,
`government`, `investment`, `rest_of_world`.

---

## Calibration

### `calibrate(sam, spec=None, *, income_elasticities=None, population=None, tariffs=None, output_taxes=None, check_balance=True) -> Economy`

`income_elasticities` maps commodity to income elasticity (LES only).
`population` maps household account to a household count — the weight of a per
capita transfer, and neither the income nor the expenditure share.
`tariffs` and `output_taxes` are reference rates, one per commodity, when the
SAM has no separate account for them. Their revenue leaves the product tax and
enters the import price and the gross output price; at those rates the base year
is unchanged.

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
| `tm0`, `tauz0` | `(nc,)` or absent | base-year import tariff and indirect tax on output, inside the prices they fall on. Absent, a scenario rate is refused |
| `gov_savrate` | scalar | government saving over income net of transfers to households, read under `government="saving_rate"` |
| `dom0` | `(nc,)` | home sales at their tax-inclusive base value: `(1 + tauz0) × Q0 − exp0` |
| `diagnostics` | dict | negative-value-added activities and treatment, own-account list |

---

## Closure and scenario

### `Closure(numeraire=1.0, investment="savings_driven", government="real_fixed", foreign="fixed_saving", foreign_saving_unit="numeraire", government_target=None, government_target_value=None, government_adjusts="direct_tax", exchange_rate=1.0, tolerance=1e-9)`

`investment` ∈ `{"savings_driven", "fixed_real"}`;
`government` ∈ `{"real_fixed", "nominal_fixed", "saving_rate"}`, the last saving
`econ.gov_savrate` of its income and spending the rest in base-year value shares;
`foreign` ∈ `{"fixed_saving", "fixed_rate"}`;
`foreign_saving_unit` ∈ `{"numeraire", "foreign_currency"}`, the unit in which
`fixed_saving` holds foreign saving;
`government_target` ∈ `{None, "saving", "revenue"}`, held at
`government_target_value` (base year when None, numeraire units), met by
`government_adjusts` ∈ `{"direct_tax", "product_tax", "spending"}` (spending for a
saving target only); the common factor on the adjusting rates is
`PeriodResult.tax_scale`;
`labour` ∈ `{"full_employment", "fixed_real_wage", "wage_curve", "elastic_supply"}`.

`full_employment` is the default and clears every factor market. The other three
apply to `labour_factor` (default `f_lab`) and refuse to use that factor as the
numeraire. `real_wage` pins the real wage (1 is the calibrated wage).
`wage_curve_elasticity` and `natural_unemployment` are the Blanchflower–Oswald
curve; the base year sits at the natural rate, because the SAM records
employment. `labour_supply_elasticity` is ε in `L = L0 * (w/P) ** ε`.
Properties: `solves_exchange_rate`, `describe()`.

### `Scenario(...)`

| Field | Shape | Meaning |
|---|---|---|
| `name` | str | carried into results and summaries |
| `factor_supply` | `(nf,)` | absolute endowments |
| `factor_ownership` | `(nh, nf)` | ownership shares, columns sum to one |
| `productivity` | `(na,)` | Hicks-neutral multiplier per activity |
| `activity_scale` | mapping | own-account activity levels, as a multiple of the base year |
| `product_tax`, `product_tax_firm` | `(nc,)` | consumer rate; non-creditable part facing firms. Does not move the tariff or the output tax |
| `tariff` | `(nc,)` | import tariff, ad valorem. Only if a base-year tariff was calibrated. Zero abolishes it |
| `output_tax` | `(nc,)` | indirect tax on all of output, ad valorem; exports carry it too. Only if a base rate was calibrated (`output_taxes=` or `indirect_tax_account`) |
| `consumer_wedge`, `government_wedge` | `(nh, nc)` | ad valorem wedges; the first is recycled to households, the second to the budget |
| `wedge_revenue_to`, `wedge_revenue_factor` | str/float/callable, str | who receives the recycled revenue |
| `transfers` | `(nh,)` | lump sum, positive for a receipt |
| `government_absorbs` | scalar | cost charged to the budget |
| `world_price` | `(nc,)` | import price multipliers; under `ModelSpec.import_supply`, the shift of the supply curve |
| `export_price` | `(nc,)` | world export price multipliers; `export_rule="cet"` only, refused otherwise; under `ModelSpec.export_demand`, the shift of the demand curve |
| `investment_mix` | `(nc,)` | commodity shares of the investment budget, summing to one; absent, the base-year mix |
| `foreign_factor_share` | `(nf,)` | share of each factor owned abroad, overriding the SAM's; `run_path` sets it from the capital it tracks |
| `carbon_price` | scalar | price per unit of emissions, numeraire units; needs an emissions account |
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
that finances the fixed investment. Under `ModelSpec.export_demand` or
`import_supply`, the log world price of each commodity on a curve.

### `PeriodResult`

| Member | Meaning |
|---|---|
| `factor_prices`, `exchange_rate`, `z` | the solved unknowns |
| `pa`, `pd`, `pq`, `pqf`, `pbase` | activity, domestic, consumer, firm and pre-tax commodity prices |
| `pe` | price received for exports: `exchange_rate × world_export_price` under CET, `pd` under fixed shares |
| `domestic_sales`, `p_domestic` | home sales of domestic output, at tax-inclusive base value, and their price. Off the base year a CET does not add up in quantities, so this is not output less exports |
| `world_export_price`, `world_import_price` | world prices in foreign currency, as multiples of the base year: the scenario's, moved along the curves of `export_demand` and `import_supply` |
| `hh_direct_tax`, `factor_income_abroad`, `foreign_saving` | direct tax by household; factor income paid abroad; the period's foreign saving, FDI included |
| `Q`, `X`, `absorption`, `imports`, `exports` | the quantity system; `imports` at the border, `exports` at their tax-inclusive value |
| `hh_quantity`, `hh_income`, `hh_commodity_spend`, `hh_direct_outlay`, `hh_wedge_paid` | the household block |
| `gov_quantity`, `investment_quantity`, `savings`, `factor_demand` | the rest of final demand |
| `wedge_revenue`, `government_cost` | recycled revenue; cost falling on the budget |
| `government`, `household_saving`, `forced_saving` | the government account at the solution; voluntary household saving; the levy that finances fixed real investment |
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

## Results

Read off a solved period or path. None of them re-solves.

### `factor_demand_by_activity(period) -> DataFrame`

Rows are factors, columns are activities. Row sums equal `period.factor_demand`.
The matrix is the one the solver used, productivity included.

### `real_gdp(period) -> dict`

`expenditure` and `value_added` at base-year prices, and the `gap` between them.
At the base year the gap is `product_tax` plus `activity_tax`. `own_account` is
included in expenditure; without it the gap is short by that activity. `other`
is what remains once quantities leave the base year.

### `national_accounts(period) -> dict`

GDP at current market prices: `consumption`, `government`, `investment`,
`exports`, `imports`, `gdp_expenditure`; `factor_income`, `taxes_on_products`
(product tax, tariff, output tax, both consumer wedges), `taxes_on_production`
(activity tax), `gdp_income`; their `gap`; `factor_income_abroad` and `gni`.

### `price_indices(period, reference=None) -> DataFrame`

Rows `consumption`, `government`, `investment`, `exports`, `imports`, `gdp` and
`terms_of_trade`; columns `laspeyres`, `paasche`, `fisher`, the three
`quantity_` indices and `value`. The reference is another `PeriodResult`, or by
default the calibrated base year.

### `government_account(period) -> Series`

`product_tax`, `activity_tax`, `tariff`, `output_tax`, `direct_tax`,
`factor_income`, `wedge`, `revenue`; `transfers`, `consumption`, `absorbed`,
`saving`. Also `PeriodResult.government`.

### `sectoral_path(path) -> DataFrame`

One row per year and account. Commodities carry `Q`, `pd`, `pq`; activities
carry `X`, `pa`. Also `PathResult.sectoral()`.

---

## Multipliers

Fixed-price models read from the base-year SAM; no equilibrium is solved.
`A[i, j]` is what `j` pays `i` per unit of `j`'s outlay, and `M[i, j]` the
effect on `i` of a unit injected into `j`.

### `accounting_multipliers(sam, *, exogenous=None, endogenous=None, spec=None) -> Multipliers`

Pyatt–Round, `M = (I - A)^-1` with average propensities. With `spec`,
government, investment, the rest of the world and any tariff or indirect-tax
account are exogenous.

### `fixed_price_multipliers(sam, *, households, income_elasticities, exogenous=None, endogenous=None, spec=None, engel="check") -> Multipliers`

Household consumption at marginal propensities `eta_c * a_ch`. Missing
commodities have elasticity one. `engel="rescale"` scales elasticities that
break Engel aggregation; `"check"` refuses them.

### `leontief_multipliers(sam, spec) -> Multipliers`

Activities and commodities endogenous. The activity block of `M` is the domestic
Leontief inverse.

### `Multipliers`

| Member | Meaning |
|---|---|
| `kind`, `accounts` | `"accounting"`, `"fixed_price"` or `"leontief"`; endogenous accounts in order |
| `flows`, `totals` | base-year flows among the endogenous accounts; their column totals |
| `A`, `M` | propensities (average, or marginal where stated) and `(I - A)^-1`, as DataFrames |
| `leakage` | `1 - A.sum(axis=0)`: the share each account pays the exogenous accounts |
| `impact(injection)` | `M @ x` for a mapping `{endogenous account: amount}` |

### `linkages(mult, accounts=None) -> DataFrame`

`backward` (column sum of `M`), `forward` (row sum of the Ghosh inverse
`(I - B)^-1`, `B[i, j] = T[i, j] / total_i`), and `backward_index`,
`forward_index`, each divided by its mean over `accounts`.

### `structural_paths(mult, origin, destination=None, *, max_length=6, threshold=1e-4) -> DataFrame`

Defourny–Thorbecke. One row per elementary path: `path`, `length`, `direct`
(product of propensities), `path_multiplier` (`det(I - A_p) / det(I - A)`),
`total`, `global` (`M[destination, origin]`) and `share` of the global influence.

---

## Subtotals

### `subtotals(econ, reference, alternative, groups, *, measure, closure=None, nodes=6, step=1e-4) -> DataFrame`

Harrison–Horridge–Pearson decomposition of `measure(alternative) -
measure(reference)` over groups of Scenario fields. `groups` maps a label to
field names (`"tariff"`, `"activity_scale"`, `"activity_scale[a_own]"`). Rows
are the measure's outputs; columns `reference`, one per group, `total`,
`change` (solved directly) and `residual`.

### `welfare_decomposition(econ, reference, alternative, *, closure=None, nodes=8, step=1e-4) -> Series`

Huff–Hertel decomposition of the change in real absorption (Divisia, current
prices along the path): `absorption` and its four components
(`absorption_households`, `_own_account`, `_government`, `_investment`);
`endowment`, `technology`, `allocative_product_tax`, `allocative_consumer_wedge`,
`allocative_activity_tax`, `allocative_tariff`, `allocative_output_tax`,
`terms_of_trade`, `external_account`; `sum_of_terms` and `residual`.

### `path_decomposition(f, x0, x1, groups, *, nodes=6, step=1e-4, solve_at=None) -> DataFrame`

The same integral for any `f` of a dict of arrays.

### `scenario_vector(econ, scenario) -> dict`

A scenario's numeric fields as arrays, absent ones at the value they stand for.

---

## Sensitivity

### `sensitivity(run, parameters, *, method="quadrature", draws=1000, seed=20260724) -> SensitivityResult`

`run` maps `{parameter: value}` to a number, a mapping or a Series. `parameters`
maps names to `Uniform(low, high)`, `Normal(mean, sd)` or
`Triangular(low, mode, high)`, independent. `"quadrature"` is Stroud's degree-3
rule at `2n` points (symmetric distributions only); `"monte_carlo"` draws with
the seed.

### `SensitivityResult`

`points` (one row per point: parameters, `weight`, outputs), `parameters`,
`outputs`, and `summary(quantiles=(0.05, 0.5, 0.95))`: `mean` and `sd`, plus
quantiles and range under Monte Carlo.

### `spec_sweep(sam, spec, *, measure, closure=None, scenario=None, calibrate_kwargs=None) -> run`

A run that sets `ModelSpec` fields (`"sigma_va"`, or `"sigma_armington[c_ind]"`
for one entry), recalibrates, solves base and scenario, and returns
`measure(base, alternative)`.

### `stroud_points(n) -> ndarray`

The `(2n, n)` standardised points of the rule.

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

## Emissions

### `add_emissions(econ, *, intermediate=None, household=None, government=None, process=None, unit="tCO2") -> Economy`

An `Economy` with an emissions account calibrated from base-year totals:
`intermediate` commodities × activities, `household` and `government` by
commodity, `process` by activity.

### `emissions(period) -> Series`

`intermediate`, `household`, `government`, `process` and `total` for a solved
period.

### `Emissions`

`intermediate`, `household`, `government`, `process` coefficients per unit of the
base-year quantity of each flow; `intensity(io_eff)` per unit of activity output.

`Scenario.carbon_price` prices one unit of emissions in numeraire units; the
revenue is `carbon_production` and `carbon_households` in `PeriodResult.government`.

---

## HAR files

### `read_har(path, headers=None) -> dict[str, Header]`

GEMPACK header array files, the format of the GTAP database. Types `1C`, `2I`,
`2R`, `RE` (full or sparse) and `RL`. `headers` keeps only some names.

### `write_har(path, headers)`

`{name: Header or array}`. Plain reals become `2R` up to two dimensions and
`RL` above; integers `2I`; strings `1C`; a `Header` with sets `RE`.

### `Header`

`name`, `description`, `kind`, `array` (float64, or int64 for `2I`, object for
`1C`), `sets` and `labels` per dimension, and `to_series()` in long form.

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
from cgelab.examples import toy

sam, econ = toy.build()                        # two sectors, no margins
sam, econ = toy.build_with_margins()           # national-accounts margin block
sam, econ = toy.build_with_own_production()    # an activity selling to households
sam, econ = toy.build_with_twin_accounts()     # identical accounts, for composition tests
sam, econ = toy.build_with_institutions()      # direct taxes, transfers, remittances, foreign capital
sam, econ = toy.build_with_specific_capital()  # industry on a capital factor of its own
sam, econ = toy.build_with_trade_taxes()       # tariff and indirect-tax accounts (TRF, IDT)

from cgelab.examples import hosoe
sam, econ = hosoe.build()                      # Hosoe's stdcge; hosoe.CLOSURE is his closure
sam, econ = hosoe.build(large_country=True)    # lrgcge
sam, econ = hosoe.build_simple()               # splcge, chapter 5
spec = toy.spec(household_demand="les", frisch=-2.0)
```

Every entry of these SAMs is written out in the source, so the base year is known
on paper and the reproduction test compares the solver against arithmetic.

## Command line

```
cgelab                              # the bundled toy economy plus diagnostics
cgelab my_sam.parquet --sigma-armington 2.0 --sigma-va 0.5
```
