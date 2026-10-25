# cgelab

[![tests](https://github.com/GustavoHM/cgelab/actions/workflows/tests.yml/badge.svg)](https://github.com/GustavoHM/cgelab/actions/workflows/tests.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](https://github.com/GustavoHM/cgelab/blob/main/LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23458851.svg)](https://doi.org/10.5281/zenodo.23458851)
[![site](https://img.shields.io/badge/site-gustavohm.github.io%2Fcgelab-8a4b2a)](https://gustavohm.github.io/cgelab/)

A general toolkit for SAM-based computable general equilibrium models in Python.
You supply a social accounting matrix. The package calibrates a standard nested
CGE from it, solves each period, runs a recursive-dynamic path and measures
welfare. The accounting checks that make any of it believable are built in
rather than left to the user.

A model is assembled from three separable objects, and keeping them separable is
the point:

| Object | Says |
|---|---|
| `SAM` | what the base year **was** |
| `ModelSpec` | what is **assumed** about substitution and functional form |
| `Closure` | which balances are imposed and **who adjusts** |

Results can turn on the last two, so both travel with the results instead of
living in a branch inside a solver.

## Install

```
pip install cgelab                 # from PyPI
pip install "cgelab[parquet]"      # adds pyarrow for .parquet SAMs
pip install "cgelab[dsge]"         # adds dsgelab, for forward-looking investment
pip install "cgelab[ibge]"         # adds xlrd, for the IBGE's supply and use tables
pip install "cgelab[plot]"         # adds matplotlib, for optional robustness figures
pip install "cgelab @ git+https://github.com/GustavoHM/cgelab"   # the current main branch
```

For development:

```
git clone https://github.com/GustavoHM/cgelab && cd cgelab
pip install -e ".[test]" && pytest
```

Requires Python 3.11+, numpy, scipy and pandas. Try it with no data at all:

```
cgelab                    # runs the bundled toy economy and prints the diagnostics
cgelab my_sam.csv --sigma-armington 2.0 --sigma-va 0.5   # the same on your own SAM
```

## Quick start

The bundled toy economy has every SAM entry written out, so the base year is
known on paper and this runs as it is:

```python
import cgelab as cge
from cgelab.examples import toy

sam, econ = toy.build()
print(cge.report(econ))           # base-year reproduction, Walras, homogeneity

base  = cge.solve_period(econ)
shock = cge.solve_period(econ, scenario=cge.Scenario(
    name="tax +50%", product_tax=econ.ptax * 1.5))
print(shock.summary())
```

A counterfactual over a horizon, and its welfare incidence:

```python
growth = cge.Growth(labour=0.005, tfp=0.010, depreciation=0.05, discount=0.05)
years  = range(2021, 2046)

W   = cge.consumer_wedge(econ, {"c_ind": 0.03})
ref = cge.run_path(econ, years=years, growth=growth)
alt = cge.run_path(econ, years=years, growth=growth,
                   scenario=lambda y, t: cge.Scenario(consumer_wedge=W,
                                                      wedge_revenue_to="capital")
                                         if y >= 2026 else cge.Scenario())

ev = cge.equivalent_variation(ref, alt)
print(cge.ev_by_group(ev, {"hh_1": "rich", "hh_2": "poor"}))
```

With your own data, the first lines change (a `.parquet` SAM needs the `parquet` extra):

```python
sam  = cge.SAM.read("my_sam.parquet")       # or SAM.from_long(df) / SAM.from_square(df)
sam.check_balance()                         # a hard gate, not a warning
econ = cge.calibrate(sam, cge.ModelSpec(sigma_armington=2.0, sigma_va=0.5))
```

## Model structure

* **Production:** Leontief between the intermediate bundle and value added, and
  within the intermediate bundle. A configurable nest (CES, Cobb–Douglas or
  Leontief) combines factors inside value added.
* **Make:** commodity market shares are fixed at base-year proportions, so
  activity levels follow from commodity output, `X = MSH @ Q`.
* **Armington:** CES between domestic and imported varieties.
* **Exports:** fixed shares by default, or a CET frontier that answers to world
  export prices. The default is deliberate: the CET elasticity is the one
  parameter of a standard CGE that input–output sources do not supply.
* **Households:** Cobb–Douglas or LES over commodities, out of income net of
  direct tax, saving and payments made straight to an activity. Income includes
  government transfers and net transfers from abroad.
* **Ownership:** factor income is shared among households, government and the
  rest of the world. Returns to foreign-owned capital leave through the current
  account, and foreign saving is the current account, not the trade balance.
* **Factor markets:** a factor is perfectly mobile, or transformed across the
  activities that use it by a CET or a logit, with pools of origin as a proximity
  matrix and a price in each activity. Labour accounts can share one population:
  formal and informal segments with a rigid wage, a dual market or
  Harris–Todaro unemployment, and income that follows each household's workers.
* **Government, investment, foreign:** five closure combinations, each with its
  own adjusting variable (see `Closure`).
* **Dynamics:** recursive, with one or several capital stocks accumulated from
  their service flows, capital by sector, putty-clay vintages, investment shared
  by relative rent, FDI that builds foreign-owned capital, an economy that may
  change year by year, and a baseline calibrated to a path of real GDP.
* **Distribution:** the households of a survey, reconciled with the SAM, with
  poverty and inequality measures and a behavioural microsimulation.

The **numeraire factor's market is deliberately not solved**. It is left to
Walras' law, so its residual is an independent check on the accounting rather
than an identity the solver was told to satisfy. It is recorded on every
period.

## What is in the box

| Capability | Entry point |
|---|---|
| SAM container, balance gate, sign-preserving RAS; GEMPACK/GTAP header array files | `SAM`, `read_har`, `write_har` |
| Calibration: coefficients, nests, taxes (tariff and output tax inside prices), margins, LES subsistence, institutional flows, a list of cells the model does not read | `calibrate` |
| Within-period equilibrium; investment, government (including deficit and revenue targets), foreign and labour closures | `solve_period`, `Closure` |
| Trade: Armington, CET, export demand and import supply curves (large country) | `ModelSpec` |
| Energy: a value-added-energy or capital-energy nest with interfuel substitution, emissions accounts, a carbon price, physical units | `ModelSpec.energy`, `add_emissions`, `add_physical_units` |
| Factor mobility (CET or logit, pools of origin); labour segments, dual markets and Harris–Todaro | `ModelSpec.factor_mobility`, `ModelSpec.labour_segments`, `LabourMarket` |
| Counterfactuals: taxes, wedges, endowments, ownership, productivity, own-account levels, world prices, FDI, the investment basket | `Scenario` |
| Recursive dynamics: several capital stocks, capital by sector, putty-clay vintages, investment allocation, foreign-owned capital, a per-year economy | `run_path`, `Growth`, `ModelSpec.vintages` |
| Baselines calibrated to real GDP and limited historical closures with technology and relative-preference shifters | `calibrate_baseline`, `calibrate_historical`, `Baseline`, `Historical` |
| Microsimulation: survey households reconciled with the SAM; Gini, FGT, generalised entropy, Atkinson and their decompositions; reweighting; a behavioural model (BRR) | `microsimulate`, `inequality`, `poverty`, `Behaviour` |
| Building a SAM from supply and use tables, with the provenance of every block; splitting households and labour by surveys; readers for the IBGE's tables, PNAD Contínua and POF | `sut_to_sam`, `split_households`, `split_factor`, `read_tru`, `pnadc`, `pof` |
| Forward-looking investment (a Ramsey CGE solved by dsgelab) | `cgelab.bridge` |
| Accounts: GDP from both sides, Laspeyres, Paasche and Fisher indices, terms of trade, the government account | `national_accounts`, `price_indices`, `government_account` |
| Decompositions: subtotals by shock, Huff-Hertel welfare decomposition | `subtotals`, `welfare_decomposition` |
| SAM multipliers, linkages, structural path analysis | `accounting_multipliers`, `linkages`, `structural_paths` |
| Systematic sensitivity and one-call robustness reports: quadrature, Monte Carlo, Sobol, SAM draws, closure grids, money and optional figures | `sensitivity`, `spec_sweep`, `sobol`, `sam_draws`, `sam_sweep`, `closure_sweep`, `in_money`, `robustness_report` |
| Minimum-distance matching of model outcomes to external coefficients, with covariance weighting and bounded elasticities | `match_targets`, `MatchResult` |
| Treated-minus-control contrasts of solved regional outcomes, with custom measures and control weights | `treated_vs_control` |
| Regional investment driven by saving or held fixed in real terms, with a per-capita forced-saving levy | `Closure(investment="fixed_real")`, `Scenario.investment_level` |
| Optional government and investment accounts mapped by region, with factor income and household saving included | `ModelSpec.regional_government`, `ModelSpec.regional_investment` |
| Factor migration between regional pools, with CET/logit proximity matrices and origin income preserved | `ModelSpec.factor_migration`, `calibrate(factor_pools=)` |
| Capital stock per region in `run_path`, and a tidy simulated region-by-year panel with optional ex-post noise | `run_path`, `simulated_panel` |
| Welfare: EV, CV and composition-immune group aggregation | `cgelab.welfare` |
| Base-year reproduction, homogeneity, Walras | `cgelab.diagnostics` |
| The same equilibrium as a JuMP model, for independent validation | `to_jump` |

## Documentation

- [Site](https://gustavohm.github.io/cgelab/): an overview with runnable examples, in English and Portuguese, and a replications page
- [`docs/guide.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/guide.md): the user guide, from SAM to welfare
- [`docs/api.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/api.md): API reference
- [`docs/whitepaper.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/whitepaper.md): the mathematics of calibration, the period equilibrium, recursive dynamics and welfare (in Portuguese)
- [`docs/design-notes.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/design-notes.md): the accounting decisions and the errors they prevent
- [`docs/from-scripts.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/from-scripts.md): porting an existing CGE pipeline
- [`docs/comparison.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/comparison.md): how this package compares with what exists, and where it does not win
- [`CHANGELOG.md`](https://github.com/GustavoHM/cgelab/blob/main/CHANGELOG.md), [`ROADMAP.md`](https://github.com/GustavoHM/cgelab/blob/main/ROADMAP.md), [`CONTRIBUTING.md`](https://github.com/GustavoHM/cgelab/blob/main/CONTRIBUTING.md)

## Verification

Every number the package produces is checked against something it did not
compute itself:

- **Hand-built SAMs** whose every entry is written out, so the base year is known on
  paper: base-year reproduction, zero-degree homogeneity and Walras' law in every
  closure, on every variant (margins, own-account activities, institutions, trade
  taxes, CET, energy nests, carbon).
- **Closed forms and published tables**: Miller and Blair's Leontief inverse, the
  stochastic growth model (for the bridge to dsgelab), Stroud quadrature on cubics,
  CES first-order conditions, and the identities that hold in every equilibrium
  (GDP from both sides, the welfare decomposition, subtotals that add up).
- **A published model**: the tariff abolition of Hosoe, Gasawa and Hashimoto's
  `stdcge` and `lrgcge` against a line-by-line JuMP transcription of the GAMS
  programs, in all 49 displayed variables, to 2.4e-12.
- **An independent replica**: `to_jump` writes the same equilibrium as a JuMP model
  in which nothing is nested. `validation/roundtrip_jump.py` runs 50 cases; the
  nested solver and Ipopt agree to 1.2e-11.

```
pytest                                                     # the suite, without external tools
cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
python validation/roundtrip_jump.py                        # the JuMP replica
python validation/replicate_hosoe.py                       # stdcge and lrgcge
```

## Scope and known limitations

`cgelab` covers a single-country CGE at applied scale. It does not have the
maturity of GAMS or GEMPACK, and [`docs/comparison.md`](https://github.com/GustavoHM/cgelab/blob/main/docs/comparison.md) says
where it stands. Things to know before relying on it:

- **Several regions** use destination-specific origin nests under the Armington,
  with bilateral domestic and imported flows, delivered margins, CET exports,
  factor migration, regional capital accumulation, route transport, a global
  service pool and consolidated government closures. LES, trade taxes, energy
  nests and unsupported labour closures are refused by name. A GTAP-style global
  savings bank remains future work.
- **No complementarity**: activities cannot switch off, and there are no quotas,
  price floors or capacity limits. That is 0.8.
- **The solver** uses finite-difference Jacobians on a few unknowns. It is fine up
  to a few hundred accounts; each commodity put on a trade curve adds an unknown,
  and so does each activity that uses an imperfectly mobile factor.
- **Fixed proportions everywhere** (`sigma_va = 0` in every activity) can leave
  the model without a solution under full employment of every factor.
- **Under fixed export shares** with a Cobb-Douglas Armington, the exchange rate is
  not determined, and `solve_period` refuses that combination; near an Armington
  elasticity of one it is badly conditioned.
- **Forward-looking investment** goes through [dsgelab](https://github.com/GustavoHM/dsgelab)
  (`pip install "cgelab[dsge]"`). `run_path` itself has no forward-looking agent.
- **GTAP data** can be read, aggregated with explicit parameter weights and turned
  into a one-region SAM. Comparison with GTAPAgg/FlexAgg and with GTAP's welfare
  decomposition remains open.
- **Platforms**: an editable install (`pip install -e`) can break silently on
  Python 3.14 on macOS; install from the wheel if `import cgelab` fails.

## Related package

[dsgelab](https://github.com/GustavoHM/dsgelab), by the same author, solves dynamic stochastic
general equilibrium (DSGE) models. The module `cgelab.bridge` uses it for the intertemporal part
of a model, a Ramsey CGE in which investment follows an Euler equation
(`pip install "cgelab[dsge]"`).

## Citing

If the package contributed to a published result, cite it as any other methodological input,
with the version, because a number that reproduces only under one version cannot be reproduced
without knowing which. [`CITATION.cff`](https://github.com/GustavoHM/cgelab/blob/main/CITATION.cff)
holds the same metadata in the format GitHub reads.

Moreira, G. H. (2026). cgelab: A general toolkit for SAM-based computable general equilibrium
models in Python (Version 0.7.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23624109

```bibtex
@software{cgelab_2026,
  author  = {Moreira, Gustavo Hoffmann},
  title   = {{cgelab}: a general toolkit for SAM-based computable general equilibrium models in Python},
  year    = {2026},
  version = {0.7.0},
  license = {MIT},
  doi     = {10.5281/zenodo.23624109},
  url     = {https://github.com/GustavoHM/cgelab},
  note    = {Python package}
}
```

Each release is archived on Zenodo with a DOI of its own; the DOI above is that of version
0.6.0 (0.5.0 is [10.5281/zenodo.23458852](https://doi.org/10.5281/zenodo.23458852)). The concept DOI
[10.5281/zenodo.23458851](https://doi.org/10.5281/zenodo.23458851) resolves to the latest release.
