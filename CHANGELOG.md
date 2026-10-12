# Changelog

All notable changes to `cgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

## Unreleased

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

### Fixed in 0.3

- An undetermined exchange rate is refused. With fixed export shares, a
  Cobb–Douglas Armington on every imported commodity and foreign saving fixed in
  numeraire units, every exchange rate is an equilibrium; the solver used to
  return whichever its start point was.

- The activity tax paid by an own-account activity never reached government:
  households paid it in the price and the tax block counted market activities
  only, so the base year did not reproduce and Walras was off by the tax. The
  JuMP replica had the same omission.

### Changed

- Renamed from `cgekit` to `cgelab`: `cgekit` is taken on PyPI by an
  unrelated project. The import name, distribution name, CLI entry point,
  the `CGELAB_RUN_JULIA` switch and the URLs all move together.

### Added

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
