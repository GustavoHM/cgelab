# Changelog

All notable changes to `cgekit`. Versions follow [semantic versioning](https://semver.org/);
before 1.0, a minor version may add to the API and a patch version only fixes.

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
  against Ipopt (worst deviation 3e-12). Opt-in pytest cases with `CGEKIT_RUN_JULIA=1`.
- `toy.build_with_institutions` and `toy.build_with_specific_capital`.
- A `LICENSE` file, continuous integration, a published site, and the comparison with other
  packages as `docs/comparison.md`.

## 0.1.0 — 2026-09-09

First version, developed inside a research repository: SAM container, balance gate and
sign-preserving RAS; calibration with margins, negative value added, LES and own-account
activities; within-period solution under five closures; scenarios and wedge recycling; recursive
dynamics; equivalent and compensating variation with composition-immune group aggregation;
base-year, homogeneity and Walras diagnostics; export to JuMP.
