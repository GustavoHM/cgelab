# cgelab API reference

Everything listed here is importable from the top-level package
(`import cgelab as cge`), except where the heading names a module:
`cgelab.bridge`, `cgelab.har.Header` and the bundled examples.

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
| `energy`, `sigma_vae`, `sigma_energy` | `None`, `0.5`, `1.0` | energy commodities; a CES between value added and an energy bundle, and across fuels inside it; scalar or mapping by activity |
| `energy_nest`, `energy_capital`, `sigma_ke` | `"value_added"`, `"f_cap"`, `0.5` | `"capital"` puts energy with that capital factor in a composite (`sigma_ke`) inside the value-added nest (`sigma_va`), as in GTAP-E |
| `household_demand`, `frisch` | `"cobb_douglas"`, `-1.5` | `"les"` uses `frisch` for subsistence |
| `negative_value_added` | `"leontief"` | or `"net"` |
| `margins` | `True` | read the commodity-to-commodity margin block |
| `regions` | `None` | region names. Accounts are then `region:local`. Absent, the one-region model |
| `sigma_interregional` | `2.0` | CES across regions of origin, under the Armington. Scalar or mapping by local commodity |
| `tariff_account` | `None` | account that collects the import tariff from commodities (Hosoe's TRF). The tariff goes inside the Armington import price |
| `indirect_tax_account` | `None` | account that collects the indirect tax on all of output (Hosoe's IDT). The tax goes inside the gross output price the CET splits, so exports carry it |
| `factor_mobility` | `None` | `{factor: ω}`, ω ≥ 0: the factor's endowment is transformed across the activities that use it, and each activity pays its own price. Zero holds the base-year allocation. A factor left out is perfectly mobile. Not the numeraire; not with `regions` |
| `labour_segments` | `None` | `{group: [factor, ...]}`: labour accounts that share one population of workers; how they clear is `Closure.labour_markets`. Two or more factors each, not the numeraire, not in `factor_mobility` |
| `segment_mobility` | `None` | `{group: ω}`: each household's workforce in a group of `labour_segments` split among its segments by a CET or a logit (`factor_mobility_form`, by group name); each flexible segment clears, each rigid one has a queue. A group left out has perfect mobility between its segments |
| `factor_mobility_form` | `"cet"` | `"cet"` (a CET frontier, GTAP's sluggish endowments) or `"logit"` (shares of a fixed head count); one form, or a mapping by factor |

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

### `calibrate(sam, spec=None, *, income_elasticities=None, population=None, tariffs=None, output_taxes=None, factor_pools=None, check_balance=True) -> Economy`

`income_elasticities` maps commodity to income elasticity (LES only).
`population` maps household account to a household count — the weight of a per
capita transfer, and neither the income nor the expenditure share.
`tariffs` and `output_taxes` are reference rates, one per commodity, when the
SAM has no separate account for them. Their revenue leaves the product tax and
enters the import price and the gross output price; at those rates the base year
is unchanged.
`factor_pools` maps a factor in `ModelSpec.factor_mobility` to a DataFrame of
base-year use, pools of origin by activities: a proximity matrix. Each column must
add up to what the SAM pays the factor in that activity. Absent, the endowment is
one pool. The calibrated transformations are `econ.mobility`, by factor index.

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

### `Closure(numeraire=1.0, investment="savings_driven", government="real_fixed", foreign="fixed_saving", foreign_saving_unit="numeraire", government_target=None, government_target_value=None, government_adjusts="direct_tax", exchange_rate=1.0, tolerance=1e-9, labour="full_employment", labour_factor="f_lab", real_wage=None, wage_curve_elasticity=0.1, natural_unemployment=0.0, labour_supply_elasticity=0.0, labour_markets={})`

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
`labour_markets` maps each group of `ModelSpec.labour_segments` to a
`LabourMarket`; a group left out is one labour market.
Properties: `solves_exchange_rate`, `describe()`.

### `LabourMarket(rigid={}, informal=None, unemployment="none", unemployment_rate=0.0)`

How one group of labour segments clears. `rigid` maps a segment to its real
wage (1 or `None` is the calibrated one); a rigid segment hires what demand asks
for at that wage.

Under perfect mobility between segments, `informal` is the segment that absorbs
the rest of the labour force (default: the group's last), and the segments that
are neither share its wage. `unemployment="none"` is a dual market;
`"harris_todaro"` puts the rationed in a queue until the informal wage equals the
expected rigid wage, with `unemployment_rate` the base year's (one rate,
strictly between 0 and 1) and `λ = 1 / (1 − rate)`. Each household keeps a fixed
workforce: its base-year share of rigid jobs and of the queue, and the rest
informal.

Under `ModelSpec.segment_mobility`, `informal` and Harris–Todaro are refused:
every flexible segment clears, and `unemployment_rate` is each rigid segment's
base-year queue as a share of its supply, one rate or `{factor: rate}`.

### `Scenario(...)`

| Field | Shape | Meaning |
|---|---|---|
| `name` | str | carried into results and summaries |
| `factor_supply` | `(nf,)` | absolute endowments |
| `labour_force` | mapping | `{group: workers}` for each group of `ModelSpec.labour_segments`, the queue included; absent, the base year's scaled by the supply of the group's factors |
| `pool_supply` | mapping | `{factor: endowment of each pool}` for an imperfectly mobile factor; absent, the pools scale with `factor_supply`, and when both are set they must agree on the total |
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
`import_supply`, the log world price of each commodity on a curve. For a factor
in `ModelSpec.factor_mobility`, its price in each activity that uses it instead
of one price, with one market per activity.

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
| `gov_quantity`, `investment_quantity`, `savings`, `factor_demand` | the rest of final demand; for an imperfectly mobile factor, `factor_demand` is its endowment and `factor_prices` its income over that endowment |
| `factor_demand_by_activity`, `factor_prices_by_activity` | `(nf, na)` factor use and the price each activity pays. Under a CET the use does not add up to the endowment off the base year |
| `labour_markets` | `{group: {...}}` for each group of labour segments: `segments`, `informal` (`None` under `segment_mobility`), `wage`, `real_wage`, `employment`, `supply`, `queue`, `labour_force`, `unemployment`, `unemployment_rate`, `harris_todaro`, `household_workforce`, `household_employment` `(nh, n_seg)`, `household_unemployed`; `None` without segments |
| `factor_pools` | `{factor: {"pools", "price", "endowment", "supply"}}` for each imperfectly mobile factor, `supply` by pool and activity; `None` when every factor is mobile |
| `wedge_revenue`, `government_cost` | recycled revenue; cost falling on the budget |
| `input_coefficients`, `value_added_per_unit` | intermediate input and primary factor use (base-year value) per unit of output as solved; under an energy nest they answer to prices |
| `government`, `household_saving`, `forced_saving` | the government account at the solution; voluntary household saving; the levy that finances fixed real investment |
| `real` | `hh_real`, `gov_real`, `inv_real`, `hh_sav_real`, `gov_sav_real`, `for_sav_real`, `hh_price_index`, `hh_real_income`, `hh_committed_value` |
| `gaps`, `walras`, `residual`, `converged`, `attempt`, `message` | the evidence |
| `price(commodity)`, `household_real_income()`, `summary()` | helpers |

### `value_added_price(econ, w)` / `factor_intensity(econ, w, *, vunit=None, aunit=None)`

The value-added nest on its own. `w` is one price per factor, or `(nf, na)`
when a factor earns a different price in each activity. `vunit` and `aunit` override the calibrated
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
`allocative_activity_tax`, `allocative_carbon_tax`, `allocative_tariff`,
`allocative_output_tax`, `factor_reallocation`, `terms_of_trade`,
`external_account`; `sum_of_terms` and `residual`. For an imperfectly mobile
factor, `endowment` prices each pool's endowment, and `factor_reallocation` is
the activity prices times the change in use less that: zero under a CET, the
gain from moving workers under a logit.

### `path_decomposition(f, x0, x1, groups, *, nodes=6, step=1e-4, solve_at=None) -> DataFrame`

The same integral for any `f` of a dict of arrays.

### `scenario_vector(econ, scenario, *, firm_rate=None) -> dict`

A scenario's numeric fields as arrays, absent ones at the value they stand for.
`product_tax_firm` is left out unless the scenario sets it or `firm_rate=True`;
absent, it follows `product_tax`.

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

`labour_factor` may name a group of `ModelSpec.labour_segments`, which then
grows as a whole.

### `Growth(labour=0.0, tfp=0.0, depreciation=0.05, discount=0.05, labour_factor="f_lab", capital_factor="f_cap", allocation="proportional", productivity="hicks_neutral", investment_deflator="base_price")`

`capital_factor` is one factor or a sequence of them, each accumulated from its
own service flow; `depreciation` is a rate or a mapping by capital factor.
`allocation` shares domestic investment among them: `"proportional"` to the
stocks, or a number `e` making shares proportional to `K_k (r_k / r_mean)^e`.
`user_cost` is `discount + depreciation`, the rate that converts the SAM's
capital service flow into a stock. `productivity` says how `tfp` enters:
`"hicks_neutral"` scales every activity's input requirements, `"factor_augmenting"`
scales the factor endowments instead, and the two give different prices.
`investment_deflator` turns nominal saving into the real investment that adds to
the stock: `"base_price"` values it at base-year purchaser prices, on the same
ruler as the other real magnitudes; `"own_price"` at the current price of the
investment bundle, the volume actually bought.

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

## Forward-looking investment (`cgelab.bridge`, needs dsgelab)

### `ramsey(econ, closure=None, *, capital_factor="f_cap", depreciation=0.05, discount=0.05, risk_aversion=1.0, scenario=None, tfp_persistence=None) -> (model, params, guess)`

A dsgelab `Model` with state `k` (and `a` with `tfp_persistence`), jump `inv`,
and the capital-accumulation and Euler residuals, each period solved by
`solve_period` under `investment="fixed_real"`.

### `announced(params, scenarios) -> list[RamseyParams]`

Per-date parameters for `dsgelab.perfect_foresight`: each date's scenario and the
next date's.

### `period_at(econ, closure, scenario, k, inv, a=1.0, *, capital_factor="f_cap", user_cost, warm=None) -> PeriodResult`

The period equilibrium at a capital stock, an investment level and a TFP level. `warm`, a
previous `PeriodResult`, is the solver's start.

`Scenario.investment_level` is real investment under `investment="fixed_real"`.

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

## Physical units

### `add_physical_units(econ, *, commodities=None, activities=None) -> Economy`

`commodities` maps a commodity to `(unit, physical supply)` (domestic sales plus
imports, base year); `activities` maps an activity to `(unit, physical output)`.

### `PhysicalUnits`

What `add_physical_units` stores in `Economy.physical_units`: `commodities` and
`activities`, each mapping a name to `(unit, physical content per model unit)`.
`None` there means the economy is in money only.

### `physical(period) -> dict`

`supply` and `output` as `(unit, quantity)`; `use` by commodity as a Series over
activities, households, government and investment; `intensity[(commodity,
activity)]` as `(unit, physical input per physical unit of output)`.

### `physical_coefficient(amount, *, input_per_value, output_per_value, input_price=1.0) -> float`

A physical technical coefficient as a SAM entry per unit of output value.

---

## HAR files

### `read_har(path, headers=None) -> dict[str, Header]`

GEMPACK header array files, the format of the GTAP database. Types `1C`, `2I`,
`2R`, `RE` (full or sparse) and `RL`. `headers` keeps only some names.

### `write_har(path, headers)`

`{name: Header or array}`. Plain reals become `2R` up to two dimensions and
`RL` above; integers `2I`; strings `1C`; a `Header` with sets `RE`.

### `Header` (`from cgelab.har import Header`)

`name`, `description`, `kind`, `array` (float64, or int64 for `2I`, object for
`1C`), `sets` and `labels` per dimension, and `to_series()` in long form.

---

## Microsimulation (`cgelab.micro`)

### `Survey(frame, sources, group="group", weight="weight", size="size", budget=None, workers=None)`

One row per household. `sources` maps an income column to a factor, a
`"factor:activity"`, `"transfers"`, `"remittances"`, `"recycled"` or `"fixed"`;
several columns may share a source, and a factor is mapped whole or by activity,
not both. `budget` maps a spending column to a commodity. `workers` maps a
head-count column to a segment or `"unemployed"`: the unemployed are a target of
the reweighting, and every column is reported with both sets of weights. Weights and sizes must be positive.

### `microsimulate(survey, reference, alternative, *, reconcile=True, reweight=True, price_index="geometric", persons=None, behaviour=None, method="expected", seed=0, replications=100) -> MicroResult`

Moves each household's income from a source as its account's does between the
two solved periods, applies the account's direct tax rate, and deflates by the
household's own price index (`"geometric"` or `"laspeyres"`). With `reconcile`,
the survey is first scaled to the reference's account incomes by account and
source; a mismatch in which accounts have which sources is refused. With
`reweight` and labour segments, the weights are calibrated by raking within each
account: segment income at base wages moves as the account's employment in the
segment, unemployed persons as its queue, households and every other source stay
fixed; segment income per worker then moves with the wage. Targets no positive
weights reach are refused by name. With `persons`, the behavioural model moves
each person's status and earnings instead (below), the segment columns of
`survey` are replaced by the members' earnings, and the weights stay;
`behaviour` defaults to `Behaviour.estimate(persons, segments)` with the
household weights; `method` is `"expected"` or `"draws"`.

### `MicroResult`

`frame` (one row per household: `group`, `weight`, `weight_alt`, `size`,
`gross_ref`, `gross_alt`, `disposable_ref`, `disposable_alt`, `price_index`,
`real_ref`, `real_alt`), `income_ref` and `income_alt` (by household and
source), `reconciliation` (`group`, `source`, `survey`, `sam`, `factor`),
`calibration` (`group`, `constraint`, `base`, `target`, `reached`,
`iterations`), `workers` (`group`, `column`, `workers`, `ref`, `alt`: persons
with the old and the new weights), and `welfare(equivalence=1.0)` (real income
per adult equivalent, and person weights `persons_ref` and `persons_alt`).
In `"draws"` mode, `replications` holds real income in the alternative,
households by replications, `real_alt` is its mean, and `inequality` and
`poverty` add an `alt_se` column.

### `PersonSurvey(frame, household="household", status="status", earnings="earnings", characteristics=(), choice_characteristics=None)`

One row per person: the household's index in `Survey.frame`, a status (a
segment, `"unemployed"`, `"inactive"`, ...), non-negative labour earnings, and
the regressors of the earnings equations (`characteristics`) and of the choice
(`choice_characteristics`, by default the same).

### `Behaviour.estimate(persons, segments, *, weights=None, base=None) -> Behaviour`

Earnings equations by weighted least squares on the segment's earners, and a
multinomial logit over all statuses by maximum likelihood; `base` (default
`"inactive"`) has zero coefficients. Fields: `statuses`, `base`, `segments`,
`choice` (one row per status), `earnings` (`{segment: {"coef", "sigma", "n"}}`),
`loglik`, `iterations`; methods `utilities(persons)` and
`predicted_log_earnings(persons)`.

### `transition_probabilities(V, delta) -> ndarray`

`T[i, j, k]`: the probability that person `i` chooses `j` with utilities `V`
and `k` with `V + delta`, under fixed i.i.d. Gumbel errors. Rows add up to the
logit probabilities of the first choice, columns to those of the second.

### `account_sources(period) -> DataFrame`

Each household account's gross income by source in a solved period: every
factor, every `"factor:activity"`, `"transfers"`, `"remittances"`, `"recycled"`.

### `inequality(result, *, equivalence=1.0, epsilons=(0.5, 1.0, 2.0)) -> DataFrame` / `poverty(result, lines, *, alphas=(0, 1, 2), by_group=False, equivalence=1.0) -> DataFrame`

Gini, Theil T and L, GE(2) and Atkinson indices; FGT by line and α. Columns
`ref`, `alt`, `change`; with `by_group`, `(period, account)` with a `total`.

### `gini(x, weights=None)`, `fgt(x, line, alpha=0, weights=None)`, `generalized_entropy(x, alpha=1.0, weights=None)`, `theil`, `mean_log_deviation`, `atkinson(x, epsilon=1.0, weights=None)`

The measures on any vector. The entropy indices and Atkinson with ε > 0 need
positive incomes; the Gini needs a positive mean; the poor are strictly below
the line.

### `ge_decomposition(x, alpha, groups, weights=None)` / `fgt_decomposition(x, line, alpha, groups, weights=None)`

GE(α) within and between groups (rows per group, `within`, `between`,
`total`); FGT(α) by group, the contributions adding up to the total.

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
sam, econ = toy.build_one_good()               # closed, one good: the stochastic growth model

from cgelab.examples import regions
sam, econ = regions.build()                    # two identical regions under one origin nest

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
cgelab --version                    # also: python -m cgelab
```
