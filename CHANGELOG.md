# Changelog

All notable changes to `cgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

## Unreleased

## 0.6.0 — 2026-10-24

Robustness and limited historical calibration. The six original items of the 0.6 milestone
were already released in 0.5.5; this release contains the robustness items and the Armington
Cobb–Douglas fix.

### Added

- `calibrate_historical` is a limited historical closure: observed scalar
  targets become exogenous constraints and technology or relative-preference
  multipliers are solved jointly, in one period or along a recursive path.
  `Historical` replays a policy at the calibrated shifters. Preference weights
  are normalised by household with a declared reference commodity, so only
  relative preferences are identified. `calibrate_baseline` now uses the same
  scalar and joint root routines, with all 15 baseline tests unchanged; new
  validation recovers known simulated histories and hand-written log roots.
- `robustness_report` runs sourced one-at-a-time ranges, quadrature, Monte
  Carlo, closure grids, balanced SAM draws with joint elasticity draws and money
  tables in one call, with optional Sobol indices. `RobustnessReport` keeps both
  summary tables and the full public result objects; refused OAT points are
  reported with their reasons. The `rose2023` preset uses ±50% endpoints from
  Rose et al. and declares the triangular shape a package convention;
  `arndt2002` supplies the published Armington and CET supports and skips
  quadrature when asymmetry makes it invalid. The optional `plot` extra adds
  matplotlib figures only on request, leaving the base installation unchanged.
- `sam_draws` perturbs selected SAM blocks by Dirichlet shares or mean-one
  lognormal noise and rebalances every draw; `sam_sweep` recalibrates and solves
  each SAM, optionally with elasticity draws on the same rows. Its
  `SensitivityResult.points` retain `draw`, `weight` and `sam_distance` for
  coverage and PIT calculations. `sam_distance` is the single public KL
  divergence between SAMs, aligned by account and directed as
  `D_KL(observed || predicted)`; negative flows are refused. Validation includes
  a hand-run noise experiment, base-year reproduction and a one-sector Leontief
  economy whose Beta-distributed share gives
  `E[1 / (1 - a)] = (alpha + beta - 1) / (beta - 1)`.
- `sobol` computes first-order and total Sobol indices on Saltelli's design, from a scrambled
  Sobol sequence carried to each distribution by its quantile function (`ppf`, new on
  `Uniform`, `Normal` and `Triangular`), at `n (k + 2)` solves; `SobolResult.indices()` gives
  them with bootstrap intervals, and its points keep the format of `SensitivityResult.points`.
  The indices match Ishigami's closed form, a linear, an additive and a purely interactive
  model; given the same seed, SALib draws the same design and gives the same indices, to
  8.9e-16, on Ishigami and on the toy economy (`validation/sobol_salib.py`, now in the
  `validation` workflow).

- `closure_sweep` solves the reference and a scenario under each closure of a list, in a
  period or along a path, and measures the change with the signature of `spec_sweep`.
  `closure_grid` gives every combination of `Closure` fields, labelled by what differs from
  the default; a combination `Closure` refuses, or one that does not solve, is reported with
  its reason and the sweep goes on. `ClosureSweep.range` gives each output's spread over the
  closures, the closures at its ends, and whether they all agree on its sign.
- `in_money` puts a counterfactual in levels, at current and at base-year prices, with the
  change in each and the change in volume, in a unit the user picks: final demand, factor
  income, the income households can spend and the government account. Its levels are those
  of `national_accounts`, `real_gdp` and the Laspeyres volume of `price_indices`; a
  household's change at base-year prices is its equivalent variation at base-year prices; a
  doubled numeraire doubles the current-price columns and leaves the others, as the tests
  check.

### Changed

- Table cells on the site wrap their text, so the comparison tables no longer run off the page;
  numbers and inline code still do not break.

### Fixed

- `functions.armington` at `sigma_armington = 1`. The CES formula has the exponent
  `1 / (1 - sigma)`, which is infinite there, and its inner sum is one, so the composite price
  came out as 1 whatever the domestic and import prices, and the intensities were wrong with
  it. Value was still conserved, so no balance check or Walras residual showed it. A
  Cobb–Douglas Armington, on one commodity or on all of them, therefore did not respond to
  prices, in the national and the regional models alike. The nest now takes the Cobb–Douglas
  limit at one, as `ces_price` already did. `tests/test_armington.py` checks the limit, its
  continuity from both sides and a world-price shock that moves a Cobb–Douglas composite.

## 0.5.5 — 2026-10-23

The six items of the roadmap's milestone 0.6 that are done: imperfect factor mobility,
segmented labour, microsimulation, building a SAM from supply and use tables and surveys, a
baseline calibrated to a GDP path, and capital by sector with vintages. They come out as 0.5.5,
ahead of 0.6.0, which waits for the robustness tools the milestone also includes.

### Added

- Imperfect factor mobility. `ModelSpec.factor_mobility = {factor: ω}` transforms
  a factor's endowment across the activities that use it, and each activity pays
  its own price. `factor_mobility_form` is `"cet"` (a CET frontier, GTAP's sluggish
  endowments; pool income is its price index times its endowment) or `"logit"`
  (shares of a fixed head count). `calibrate(..., factor_pools=)` takes a proximity
  matrix, base-year use by pool of origin and activity, and `Scenario.pool_supply`
  moves the pools' endowments. `PeriodResult.factor_prices_by_activity` and
  `factor_pools` report the result; `value_added_price` and `factor_intensity`
  accept a price matrix. At ω = 0, and with pools that each reach one activity,
  the model equals the factor split into separate accounts of a hand-built SAM,
  to 1e-10; the approach to perfect mobility goes as 1/ω. The JuMP replica
  carries the pools and the markets by activity, and six new round-trip cases
  match it to 1e-13.
- Segmented labour. `ModelSpec.labour_segments = {group: [factor, ...]}` says that
  several labour accounts (formal and informal, say) share one population, and
  `Closure.labour_markets = {group: LabourMarket(...)}` how they clear. A rigid
  segment pays a fixed real wage and hires what demand asks for; the informal
  segment employs the rest of the labour force (a dual market), or workers queue
  for rigid jobs until the informal wage equals the expected rigid wage
  (Harris–Todaro, `λ = 1 / (1 − u0)` from the base-year unemployment rate). A
  group with no rigid segment is one labour market. Each household keeps a fixed
  workforce, so a lost formal job moves income to the informal segment within
  the same household. `Scenario.labour_force`, `Growth(labour_factor=group)` and
  `PeriodResult.labour_markets` (wages, employment, unemployment, employment by
  household) go with it, and `toy.build_with_segmented_labour` is the example.
  One labour market equals the toy with one mobile labour account exactly; a
  rigid wage that does not bind rations nobody; Harris–Todaro approaches one
  market as base-year unemployment vanishes, at a rate of u0. Four new JuMP
  round-trip cases match to 1e-13.
- A transformation between labour segments. `ModelSpec.segment_mobility =
  {group: ω}` splits each household's workforce among a group's segments by a
  CET or a logit (`factor_mobility_form` takes the group's name). Each flexible
  segment's wage clears it; a rigid segment hires what demand asks for and the
  rest of its supply queues, with `LabourMarket.unemployment_rate` (one rate or
  one per rigid segment) the base-year queue. At ω = 0 with no queue the group is
  exactly two factors under `Closure(labour="fixed_real_wage")`, to 1e-10; as ω
  grows it approaches one labour market at a rate of 1/ω. Three new JuMP
  round-trip cases match to 1e-13.
- Microsimulation (`cgelab.micro`). `microsimulate(survey, reference, alternative)`
  carries a counterfactual top-down to the households of a `Survey`: income by
  source moves as the household's SAM account's does, disposable income takes
  the account's direct tax, and each household's own geometric (or Laspeyres)
  price index deflates it. The survey is scaled to the SAM by account and source
  first, and the factors are reported. `account_sources` gives each account's
  income by source. The measures `gini`, `fgt`, `generalized_entropy` (`theil`,
  `mean_log_deviation`) and `atkinson`, the exact decompositions of GE by
  within and between groups and of FGT by group, and the tables `inequality` and
  `poverty`. `toy.survey()` is a synthetic survey. A survey that is the SAM
  gives back the model's incomes and real incomes to 1e-12; the measures match
  closed forms and a brute-force Gini.
- Reweighting the survey for employment. Under labour segments,
  `microsimulate(..., reweight=True)`, the default, calibrates the weights within
  each account by raking: segment income at base wages moves as the account's
  employment in the segment, unemployed persons (`Survey.workers`) as its queue,
  households and other sources stay fixed, and segment income per worker moves
  with the wage. `MicroResult.calibration`, `workers` and `frame["weight_alt"]`
  report it, and the tables use each period's weights. Raking on cells that
  partition the sample gives the closed-form ratio per cell; the micro totals
  stay the model's to 1e-10 under a dual market and Harris–Todaro.
- Behavioural microsimulation, in the manner of Bourguignon, Robilliard and
  Robinson (2005). `microsimulate(..., persons=PersonSurvey(...))` estimates
  earnings equations by segment and a multinomial logit of the occupational
  choice (`Behaviour.estimate`, scipy only) and moves each person's status and
  earnings by intercepts shifted per account until employment and earnings per
  worker move as the model's do. `method="draws"` is BRR's conditional Gumbel
  draws, with `seed` and `replications` and standard errors in the tables;
  `method="expected"`, the default, is their limit in closed form
  (`transition_probabilities`), which hits the targets exactly. `toy.persons()`
  is a synthetic person survey. Checked: the logit matches closed forms (shares,
  a saturated binary regressor) and, in `validation/logit_statsmodels.py`,
  statsmodels' `MNLogit` to 1e-8; the transition probabilities add up to both
  choices' logit probabilities to 1e-15 and agree with the draws; the micro labour
  income of every account is the model's to 1e-10.
- Readers for the IBGE's surveys (`cgelab.ibge`). `read_ibge` reads fixed-width
  microdata with the layout of the IBGE's SAS reading program (`read_sas_program`),
  several data steps, `w.d` informats and Latin-1 included. `pnadc` turns a
  quarter of the PNAD Contínua into a household `Survey` and a `PersonSurvey`,
  formal or informal by the IBGE's proxy (VD4009 with V4019). `pof` turns the
  POF 2017–2018 into a survey of consumption units, its spending annualised and
  deflated as the IBGE's tabulation programs do and mapped to SAM commodities by
  item-code prefix. Tests use fixtures written by hand in the IBGE's formats; a
  check on the real microdata runs when `CGELAB_IBGE_DIR` points at them, and
  `validation/ibge_layouts.py` matches the layouts against the IBGE's R
  programs (all sixteen POF records) and a readr reading of the fixture.
- Building a SAM from supply and use tables (`cgelab.build`). `SupplyUse` holds the
  make, use, final-demand, import, value-added, tax and margin tables, labelled
  by the accounts they become; `Institutions` holds factor income by recipient,
  direct taxes, transfers and other flows; `sut_to_sam` assembles the SAM in the
  layout `calibrate` reads, with household, government and foreign saving as
  declared residuals, and records each block's source in `SAM.provenance`.
  Tables that do not add up (a commodity, an activity, a margin row, factor
  income against value added, a flow that unbalances an account) are refused
  by name. Every toy SAM, written by hand, rebuilds from its own tables cell by
  cell, and the three GDPs agree.
- Disaggregating a SAM with household surveys. `split_households` splits a
  household account into a survey's groups, cell by cell, by each group's share
  of the weighted survey total, savings residual; `split_factor` splits a factor
  (labour into formal and informal) by activity and by owner; `segment_shares`
  computes both from the PNAD Contínua, which `pnadc` now reads with each
  worker's activity (V4013, `activities=`). Splitting and summing back is the
  identity; households split with identical proportions keep the aggregate
  equilibrium, and labour split into segments with one wage keeps the single
  factor's.
- `read_tru` reads the IBGE's supply and use tables (tables 1 and 2 of the
  Sistema de Contas Nacionais, any level) into a `SupplyUse`: labour, capital and
  mixed income apart by default, taxes on products and on production net of
  subsidies, the trade and transport margins shared among their providers in
  proportion (recorded as assumed), NPISH with households or government. The
  layout was read off the IBGE's 2021 tables at 68 activities, which close
  exactly and give the GDP the IBGE publishes for 2021, R$ 9,012,142 million; the
  suite reads a hand-written table in that layout. A new extra, `ibge`, installs
  `xlrd` for the `.xls` files.
- Capital by sector. `calibrate(factor_pools={factor: "activities"})` gives each
  activity that uses a factor a pool of its own, so the factor is sector-specific.
  `run_path` accumulates a stock per pool of a capital factor, shares investment
  and FDI among all the stocks by `Growth.allocation` with each pool's price as
  its rent, and reports them in `PathResult.capital_by_pool`. A pool per activity
  gives the path of capital written out by hand as one factor per activity, to
  1e-10.
- Putty-clay vintages. `ModelSpec.vintages = {old: new}` with `split_vintages`
  splits capital into an installed vintage, sector-specific, and a new one, mobile.
  Value added is a CES (`vintage_substitution`, 20 by default) of an old bundle (a
  CES of `vintage_sigma`, fixed proportions by default) and a new bundle (the CES
  of `sigma_va`). In `run_path` the new vintage is each year's investment, and the
  new capital an activity used is installed there at the end of the year, the old
  bundle taking on the technique it was installed with. With one elasticity in
  both bundles and between them the nest is the flat CES of the model without
  vintages, to 1e-10; under fixed proportions the old bundle is its capital over
  the capital share at any prices; a one-good economy on its steady path stays
  there to 1e-12; and as `vintage_substitution` grows the path approaches the one
  with a single capital stock, at a rate of one over it.
  `PeriodResult.vintage_value_added` reports the two bundles. The JuMP replica
  carries the nest, and three new round-trip cases match it to 1e-13.
- `calibrate_baseline` calibrates a recursive baseline to a path of real GDP, in
  the manner of MONASH's historical simulations and LINKAGE's baselines: year by
  year, in sequence, the Hicks-neutral productivity that hits a GDP given as
  growth rates, an index or levels, one scalar a year (spread across activities
  by an optional `pattern`) or one per activity on sectoral value added.
  `Baseline` keeps the calibrated productivity, the path and the target, and
  `scenario(policy)` and `run(policy)` give a policy run at that productivity;
  with no policy it reproduces the baseline exactly. Calibrating on a path
  solved with a known productivity gives it back to 1e-10, by scalar, pattern and
  sector, and a one-good economy matches the Solow recursion worked on paper.
  `real_value_added` measures value added by activity by double deflation, which
  productivity moves and the factor bill of `real_gdp` does not. `run_path`'s
  year step is now an internal stepper the two share; its results are unchanged.
- `welfare_decomposition` has a `factor_reallocation` term: what moving an
  imperfectly mobile factor between activities adds beyond its endowment. It is
  zero under a CET, by the envelope theorem, and the identity still closes.

### Changed

- The site shows, under every example, what it prints and the value of each
  expression on a line of its own, as dsgelab's site does. The outputs are
  written by `tools/check_site_examples.py --write`, and the check, which runs in
  the suite, fails when an output on the page is not what the code prints.
- `real_gdp` values factor services by activity at base-year prices, which for a
  perfectly mobile factor is the same number as before.
- The employment that is labour demand under a fixed real wage or a wage curve
  is now found from a vector of probes, one per factor whose employment is
  demand; with one such factor it is the two probes it was.

## 0.5.0 — 2026-10-18

The first public release, on PyPI together with dsgelab 0.5. It gathers the roadmap's
milestones 0.3 and 0.4, which were not released on their own, and the release engineering of
0.5. Python 3.10 is no longer supported.

### Changed since 0.2.0

- Renamed from `cgekit` to `cgelab`: `cgekit` is taken on PyPI by an
  unrelated project. The import name, distribution name, CLI entry point,
  the `CGELAB_RUN_JULIA` switch and the URLs all move together.

### Added in 0.2, after 0.2.0

- Large-country trade. `ModelSpec.export_demand` puts exports on a demand curve,
  `E / E0 = (pWe / export_price)^(-η)`, and `ModelSpec.import_supply` puts
  imports on a supply curve, `M / M0 = (pWm / world_price)^ψ`: the equations of
  Hosoe's `lrgcge`. A scalar or a mapping by commodity; the rest stay price
  takers. The scenario prices become the shifts of the curves, and
  `PeriodResult.world_export_price` and `world_import_price` hold the solved
  world prices. The base year is unchanged. `export_demand` needs
  `export_rule="cet"`; the regional slice refuses both. The JuMP replica carries
  the curves as constraints, and three new round-trip cases match it to 1e-12.
- Import tariff and indirect tax on output, separate from the product tax on
  absorption and inside the prices they fall on, as in Hosoe's `stdcge`.
  `calibrate(..., tariffs=)` or `ModelSpec.tariff_account` sets the base-year
  tariff (Hosoe's `taum`), and the Armington nest is calibrated on the
  tariff-inclusive value of imports. `calibrate(..., output_taxes=)` or
  `ModelSpec.indirect_tax_account` sets the output tax (Hosoe's `tauz` / IDT),
  and the CET splits output at its tax-inclusive value, so exports carry it.
  Government receives both as levels. `Scenario.tariff` and
  `Scenario.output_tax` move one rate without moving the other; a tariff of
  zero abolishes it. At the calibrated rates the base year is the SAM. The
  JuMP replica carries the same prices.
- `Closure(government="saving_rate")`: government saves the base-year share
  of its income (`Economy.gov_savrate`) and spends the rest in base-year value
  shares. `Closure(foreign_saving_unit="foreign_currency")`: foreign saving
  fixed in dollars under `foreign="fixed_saving"`. Together they are Hosoe's
  closure.
- `cgelab.examples.hosoe`: the SAM, functional forms and closure of Hosoe,
  Gasawa and Hashimoto's `stdcge` and, with `large_country=True`, `lrgcge`.
  The tariff abolition of both matches a line-by-line JuMP transcription of the
  GAMS programs (`validation/hosoe.jl`) in all 49 displayed variables, to
  2.4e-12; `tests/test_hosoe_stdcge.py` checks it against the stored result.
- The JuMP replica converges from a cold start on an applied SAM. Household
  income, spending and the Armington shares now start at base-year levels
  instead of at 1 and at guesses; factor prices and the exchange rate still start
  displaced, so the Python answer never enters the start.
- `PeriodResult.domestic_sales` and `p_domestic`: home sales of domestic output
  and their price. `base_year_reproduction` now also checks imports and
  exports against the SAM.
- Several regions. `ModelSpec.regions` and `sigma_interregional`: a CES across
  regions of origin under the Armington nest. Accounts are `region:local`.
  Factors clear where they are; one national numeraire stays unsolved. A single
  labelled region reproduces the legacy solver, including on the plain toy with
  taxes and intermediates. `examples.regions` is a hand-written two-region SAM.
  Margins, CET, LES, migration, non-default closures, `run_path` and `to_jump`
  are refused by name on this slice.
- `factor_demand_by_activity`, `real_gdp` and `sectoral_path`: employment by
  activity, real GDP from both sides with the tax gap named, and a tidy sectoral
  panel along a path. `PeriodResult.factor_demand_by_activity` holds the matrix
  the solver used.
- A replications page on the site (`site/replications.html`, and the Portuguese pair).
  The benchmark of Hosoe, Gasawa and Hashimoto (2004), chapter 5, as printed in the
  GAMS library file `splcge`, reproduces: output 15 and 35, prices 1. A SAM that
  invests nothing no longer raises on an empty investment basket.
- `Scenario.investment_mix`: commodity composition of the investment budget.
  Absent, the base-year mix. Distinct from `Growth.allocation`, which shares
  investment across capital stocks.
- `Closure.labour`: `full_employment` (unchanged default), `fixed_real_wage`,
  `wage_curve` and `elastic_supply`. The three new closures refuse a labour
  numeraire. Unemployment is not paid. `PeriodResult.employment`,
  `unemployment` and `real_wage`. The JuMP export carries the same equations.

### Added in 0.3

- SAM multipliers and input–output analysis (`cgelab.multipliers`):
  `accounting_multipliers` (Pyatt and Round), `fixed_price_multipliers` with
  marginal propensities from income elasticities and an Engel-aggregation check,
  `leontief_multipliers` (activities and commodities endogenous), `linkages`
  (backward from the Leontief inverse, forward from the Ghosh inverse, with
  Rasmussen indices) and `structural_paths` (Defourny and Thorbecke). The
  Leontief inverse reproduces Miller and Blair's two-sector example; the total
  influences of all elementary paths add up to the multiplier (Lantner's
  identity), to 1e-10.

- National accounts (`national_accounts`): GDP at current prices by expenditure
  and by income, which agree in every equilibrium, and GNI. Price and quantity
  indices (`price_indices`): Laspeyres, Paasche and Fisher by component of final
  demand, for GDP and for the terms of trade, against the base year or another
  solved period. The government account (`government_account`,
  `PeriodResult.government`): revenue by instrument, outlays and saving, recorded
  by the solver at the solution. `PeriodResult.household_saving` and
  `forced_saving`.

- Systematic sensitivity (`sensitivity`, `spec_sweep`): Stroud's degree-3
  quadrature, exact in the mean for results up to cubic in the parameters, and
  Monte Carlo with a fixed seed, over `Uniform`, `Normal` and `Triangular`
  parameters. `spec_sweep` recalibrates at every point.

- Subtotals (`subtotals`, `path_decomposition`): the Harrison, Horridge and
  Pearson decomposition of a result by groups of shocks, integrated along the
  straight path from reference to alternative. The contributions add up to the
  solved change, with a residual of order 1e-8 that is reported. On closed forms
  they are the line integrals.

- Welfare decomposition (`welfare_decomposition`): Huff and Hertel's split of
  the change in real absorption into endowment, technology, one allocative term
  per instrument, terms of trade and the external account, integrated along the
  path. The identity closes to below 1e-9 in ten economies and closures; it is
  not validated against GTAP, which needs the GTAP database.
- `cgelab.examples.hosoe.build_simple()`: Hosoe's chapter 5 (`splcge`).

### Fixed in 0.3

- An undetermined exchange rate is refused. With fixed export shares, a
  Cobb–Douglas Armington on every imported commodity and foreign saving fixed in
  numeraire units, every exchange rate is an equilibrium; the solver used to
  return whichever its start point was.

- The activity tax paid by an own-account activity never reached government:
  households paid it in the price and the tax block counted market activities
  only, so the base year did not reproduce and Walras was off by the tax. The
  JuMP replica had the same omission.

### Added in 0.4

- Government targets (`Closure.government_target`, `government_adjusts`):
  government saving held fixed (a deficit target) or total revenue held fixed
  (revenue-neutral tax replacement), met by scaling every direct tax rate, every
  product tax rate, or government consumption in base-year shares.
  `PeriodResult.tax_scale` is the factor. The JuMP replica carries the same
  closures; `Economy.gov_saving0` and `gov_revenue0` are the base-year targets.

- Emissions accounts (`add_emissions`, `emissions`) calibrated from base-year
  totals by source (combustion in production, household and government fuel
  use, process), and a carbon price (`Scenario.carbon_price`): a tax per unit of
  output by emission intensity on activities, a specific tax on household fuel.
  The revenue is reported by source and recycled by the revenue-neutral
  closure. GDP, the welfare decomposition, subtotals and the JuMP replica carry
  it.

- Value-added-energy nesting (`ModelSpec.energy`, `sigma_vae`,
  `sigma_energy`): an energy bundle, a CES across fuels, substitutes for value
  added in a CES; other intermediates stay Leontief. The carbon price on a fuel
  enters the price each activity pays for it. With zero elasticities the nest is
  the Leontief model exactly; with positive ones the fuel ratios meet the CES
  first-order conditions. The JuMP replica carries the nest.
  `PeriodResult.input_coefficients` and `value_added_per_unit` are the
  coefficients the equilibrium used.

- GTAP-E's arrangement (`ModelSpec.energy_nest="capital"`, `energy_capital`,
  `sigma_ke`): energy joins a capital factor in a capital-energy composite that
  sits inside the value-added nest. With `sigma_ke = sigma_va` it collapses to the
  value-added nest with `sigma_vae = sigma_va`, and the two solve to the same
  equilibrium to 1e-11; capital and energy meet their first-order condition; the
  JuMP replica carries both arrangements.
- The bridge to dsgelab (`cgelab.bridge`, `pip install "cgelab[dsge]"`): a
  Ramsey CGE in which an Euler equation chooses investment and every period is
  the SAM model. dsgelab gives its steady state, first-order solution, impulse
  responses and perfect-foresight transitions; `announced` turns a sequence of
  scenarios into an announced policy. In a one-good closed economy it is the
  stochastic growth model: same steady state, transition matrices and path, to
  1e-7. `toy.build_one_good()` is that economy; `Scenario.investment_level`
  sets real investment under `fixed_real`.

- Physical units (`add_physical_units`, `physical`, `physical_coefficient`):
  physical content per unit of a commodity's Armington composite and of an
  activity's output, calibrated from base-year totals; physical supply, use by
  agent, output and intensity of a solved period; and the translation of a
  physical technical coefficient into a SAM entry. Supply meets use in every
  equilibrium, margins included.

### Fixed in 0.4

- `homogeneity_test` held a fixed exchange rate at one while it scaled the
  numeraire, so every `foreign="fixed_rate"` closure failed the test for a
  reason that was the test's. The rate is nominal and now scales too.

### Added toward 0.7

- `read_har` and `write_har`: GEMPACK header array files, the format of the GTAP
  database, without GEMPACK. On GTAP 10A (2014) the 86 headers harpy3 reads
  agree with it exactly, and four GTAP identities close to single precision.
  The data are licensed and never committed.

### Release engineering (0.5)

- CI runs the suite on Python 3.11–3.14 on Linux, macOS and Windows, and against
  the installed wheel (`CGELAB_TEST_INSTALLED=1`), with `twine check`, both with
  dsgelab installed so the bridge is tested. A
  `validation` workflow runs the JuMP replica and the Hosoe transcription on
  demand and weekly. A `release` workflow, started by pushing a tag `vX.Y.Z`,
  checks that the tag matches the version in `pyproject.toml`, the package and
  `CITATION.cff`, and that the dsgelab the `dsge` extra needs is on PyPI;
  publishes to TestPyPI by trusted publishing, installs what TestPyPI serves and
  runs the suite against it, and only then publishes to PyPI. The site's deploy
  is skipped while the repository is private.
- The sdist carries what its test suite needs (`MANIFEST.in`): the fixtures, the
  site and the tool that runs its examples. `cgelab --version` and
  `python -m cgelab`. The README's links are absolute, so they also work on PyPI.
- `py.typed`; Python-version, OS and typing classifiers; the `dsge` extra, which
  needs `dsgelab>=0.5`. The licence is declared as an SPDX expression (`MIT`),
  with `LICENSE` as the licence file.
- The examples, the docs and the site import the package as `cge` and dsgelab as
  `dsg`, and link to dsgelab's site. The site's language selector names both
  languages with an icon.
- `CONTRIBUTING.md` with the definition of done and a release checklist; issue
  and pull request templates; a known-limitations section in the README.
- `docs/comparison.md` in English; the Portuguese version is
  `docs/comparison.pt-br.md`.

## 0.2.0 — 2026-09-25

The first release as a standalone package. Four gaps and two bugs, found while building a
regional recursive-dynamic CGE on the package, are closed. Every new `Economy` field defaults to
zero, so a SAM without the new flows gives the same results as 0.1.0.

### Fixed

- **CET exports were never solved.** `export_rule="cet"` was accepted by `ModelSpec` and
  advertised, but the solver always used fixed export shares. Output now splits along the CET
  frontier; the domestic price comes from the inverse revenue function (`cet_domestic_price`).
- **The external target was read from two fields.** Under `foreign="fixed_saving"` the trade
  target came from `trade_balance0` and saving from `foreign_sav`, so moving `foreign_sav` alone
  left a Walras residual that grew along a path. `foreign_sav` is now the single target.
- **Institutional flows in the SAM were dropped.** Household income was read as the row total, but
  the solver modelled only factor income: a SAM with a direct tax, a government transfer or a
  remittance missed its own base year (by 12% on `toy.build_with_institutions`).
- **LES marginal budget shares** assumed commodity spending equals income net of saving, which is
  wrong with direct taxes or purchases made straight from an activity. They now use the calibrated
  commodity budget (`Economy.hh_spend0`).

### Added

- `Scenario.export_price` — world export prices, read under CET and refused under fixed shares.
  `sigma_cet` accepts a mapping by commodity.
- Direct taxes (`gov <- hh`), government transfers (`hh <- gov`) and remittances (`hh <-> row`),
  with saving out of disposable income. `PeriodResult.hh_direct_tax`; welfare income is net of
  direct tax.
- Factor income paid to government and abroad (`gov <- f`, `row <- f`); foreign saving is the
  current account. `Scenario.foreign_factor_share` overrides the SAM's foreign shares.
- `Scenario.fdi` — foreign direct investment that raises foreign saving and, in `run_path`, adds
  the capital it buys to a foreign-owned stock whose returns leave the economy.
  `PathResult.foreign_capital` records it.
- Several capital stocks: `Growth.capital_factor` takes a sequence, `depreciation` a mapping, and
  `allocation` shares domestic investment proportionally or by relative rent.
- `run_path(economy=f(year, t))` — a different `Economy` each year, for changes that are not
  scenario fields (input coefficients, a foreign-saving path).
- `econ.diagnostics["unmodelled_flows"]` lists SAM cells the model does not read.
- The JuMP export replicates all of the above; `validation/roundtrip_jump.py` runs 16 cases
  against Ipopt (worst deviation 3e-12). Opt-in pytest cases with `CGELAB_RUN_JULIA=1`.
- `toy.build_with_institutions` and `toy.build_with_specific_capital`.
- A `LICENSE` file, continuous integration, a published site, and the comparison with other
  packages as `docs/comparison.md`.

## 0.1.0 — 2026-09-09

First version, developed inside a research repository: SAM container, balance gate and
sign-preserving RAS; calibration with margins, negative value added, LES and own-account
activities; within-period solution under five closures; scenarios and wedge recycling; recursive
dynamics; equivalent and compensating variation with composition-immune group aggregation;
base-year, homogeneity and Walras diagnostics; export to JuMP.
