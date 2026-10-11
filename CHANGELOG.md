# Changelog

All notable changes to `cgelab`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

## Unreleased

### Changed

- Renamed from `cgekit` to `cgelab`: `cgekit` is taken on PyPI by an
  unrelated project. The import name, distribution name, CLI entry point,
  the `CGELAB_RUN_JULIA` switch and the URLs all move together.

### Added

- Import tariff and indirect tax on domestic sales, separate from the product
  tax on absorption. `calibrate(..., tariffs=)` or `ModelSpec.tariff_account`
  sets the reference tariff (Hosoe's `taum`); `calibrate(..., output_taxes=)`
  or `ModelSpec.indirect_tax_account` sets the output tax (Hosoe's `tauz` /
  IDT). `Scenario.tariff` and `Scenario.output_tax` move one rate without
  moving the other. At the calibrated rate the base year is unchanged, because
  a named account's receipt stays inside the product tax. Government income
  receives the change from that rate. The output tax falls on domestic sales;
  exports sold at the producer price are outside it. The JuMP replica carries
  the same wedge.
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
