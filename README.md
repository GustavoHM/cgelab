# cgelab

[![tests](https://github.com/GustavoHM/cgelab/actions/workflows/tests.yml/badge.svg)](https://github.com/GustavoHM/cgelab/actions/workflows/tests.yml)
[![site](https://img.shields.io/badge/site-gustavohm.github.io%2Fcgelab-2b6cb0)](https://gustavohm.github.io/cgelab/)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

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
pip install cgelab                 # from PyPI, once 0.5 is released
pip install "cgelab[parquet]"      # adds pyarrow for .parquet SAMs
pip install "cgelab[dsge]"         # adds dsgelab, for forward-looking investment
pip install "cgelab @ git+https://github.com/GustavoHM/cgelab"   # the current main branch
```

For development:

```
git clone https://github.com/GustavoHM/cgelab && cd cgelab
pip install -e ".[test]" && pytest
```

Requires Python 3.10+, numpy, scipy and pandas. Try it with no data at all:

```
cgelab                    # runs the bundled toy economy and prints the diagnostics
```

## Quick start

The bundled toy economy has every SAM entry written out, so the base year is
known on paper and this runs as it is:

```python
import cgelab as ck
from cgelab.examples import toy

sam, econ = toy.build()
print(ck.report(econ))            # base-year reproduction, Walras, homogeneity

base  = ck.solve_period(econ)
shock = ck.solve_period(econ, scenario=ck.Scenario(
    name="tax +50%", product_tax=econ.ptax * 1.5))
print(shock.summary())
```

A counterfactual over a horizon, and its welfare incidence:

```python
growth = ck.Growth(labour=0.005, tfp=0.010, depreciation=0.05, discount=0.05)
years  = range(2021, 2046)

W   = ck.consumer_wedge(econ, {"c_ind": 0.03})
ref = ck.run_path(econ, years=years, growth=growth)
alt = ck.run_path(econ, years=years, growth=growth,
                  scenario=lambda y, t: ck.Scenario(consumer_wedge=W,
                                                    wedge_revenue_to="capital")
                                        if y >= 2026 else ck.Scenario())

ev = ck.equivalent_variation(ref, alt)
ck.ev_by_group(ev, {"hh_1": "rich", "hh_2": "poor"})
```

With your own data, the first lines change:

```python
sam  = ck.SAM.read("my_sam.parquet")        # or SAM.from_long(df) / SAM.from_square(df)
sam.check_balance()                         # a hard gate, not a warning
econ = ck.calibrate(sam, ck.ModelSpec(sigma_armington=2.0, sigma_va=0.5))
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
* **Government, investment, foreign:** five closure combinations, each with its
  own adjusting variable (see `Closure`).
* **Dynamics:** recursive, with one or several capital stocks accumulated from
  their service flows, investment shared by relative rent, FDI that builds
  foreign-owned capital, and an economy that may change year by year.

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
| Counterfactuals: taxes, wedges, endowments, ownership, productivity, own-account levels, world prices, FDI, the investment basket | `Scenario` |
| Recursive dynamics: several capital stocks, investment allocation, foreign-owned capital, a per-year economy | `run_path`, `Growth` |
| Forward-looking investment (a Ramsey CGE solved by dsgelab) | `cgelab.bridge` |
| Accounts: GDP from both sides, Laspeyres, Paasche and Fisher indices, terms of trade, the government account | `national_accounts`, `price_indices`, `government_account` |
| Decompositions: subtotals by shock, Huff-Hertel welfare decomposition | `subtotals`, `welfare_decomposition` |
| SAM multipliers, linkages, structural path analysis | `accounting_multipliers`, `linkages`, `structural_paths` |
| Systematic sensitivity: Stroud quadrature and Monte Carlo over elasticities | `sensitivity`, `spec_sweep` |
| Welfare: EV, CV and composition-immune group aggregation | `cgelab.welfare` |
| Base-year reproduction, homogeneity, Walras | `cgelab.diagnostics` |
| The same equilibrium as a JuMP model, for independent validation | `to_jump` |

## Documentation

- [Site](https://gustavohm.github.io/cgelab/): an overview with runnable examples, in English and Portuguese, and a replications page
- [`docs/guide.md`](docs/guide.md): the user guide, from SAM to welfare
- [`docs/api.md`](docs/api.md): API reference
- [`docs/whitepaper.md`](docs/whitepaper.md): the mathematics of calibration, the period equilibrium, recursive dynamics and welfare (in Portuguese)
- [`docs/design-notes.md`](docs/design-notes.md): the accounting decisions and the errors they prevent
- [`docs/from-scripts.md`](docs/from-scripts.md): porting an existing CGE pipeline
- [`docs/comparison.md`](docs/comparison.md): how this package compares with what exists, and where it does not win
- [`CHANGELOG.md`](CHANGELOG.md), [`ROADMAP.md`](ROADMAP.md), [`CONTRIBUTING.md`](CONTRIBUTING.md)

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
  in which nothing is nested. `validation/roundtrip_jump.py` runs 34 cases; the
  nested solver and Ipopt agree to 1.2e-11.

```
pytest                                                     # the suite, without external tools
cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
python validation/roundtrip_jump.py                        # the JuMP replica
python validation/replicate_hosoe.py                       # stdcge and lrgcge
```

## Scope and known limitations

`cgelab` covers a single-country CGE at applied scale. It does not have the
maturity of GAMS or GEMPACK, and [`docs/comparison.md`](docs/comparison.md) says
where it stands. Things to know before relying on it:

- **Several regions** share one origin nest under the Armington. Factors do not
  migrate, and the regional slice refuses margins, CET, LES, trade taxes, energy
  nests and most closures by name. GTAP-style bilateral trade is 0.7.
- **No complementarity**: activities cannot switch off, and there are no quotas,
  price floors or capacity limits. That is 0.8.
- **The solver** uses finite-difference Jacobians on a few unknowns. It is fine up
  to a few hundred accounts; each commodity put on a trade curve adds an unknown.
- **Fixed proportions everywhere** (`sigma_va = 0` in every activity) can leave
  the model without a solution under full employment of every factor.
- **Under fixed export shares** with a Cobb-Douglas Armington, the exchange rate is
  not determined, and `solve_period` refuses that combination; near an Armington
  elasticity of one it is badly conditioned.
- **Forward-looking investment** goes through dsgelab (`pip install cgelab[dsge]`).
  `run_path` itself has no forward-looking agent.
- **GTAP data** can be read, but not yet aggregated or turned into a SAM, and the
  welfare decomposition has not been compared with GTAP's own.
- **Platforms**: on macOS 27 the SciPy wheels for Python 3.10 do not load; use
  Python 3.11 or later there. An editable install (`pip install -e`) can break
  silently on Python 3.14 on macOS; install from the wheel if `import cgelab`
  fails.

## Citing

If a published result went through this package, cite it. Software is a
methodological input like any other, and the version matters: a number that
only reproduces under one version is not reproducible without it.
[`CITATION.cff`](CITATION.cff) carries the metadata in the format GitHub reads;
fill in the DOI and ORCID before citing.

A repository URL is not an archivable reference. Connect the repository to
Zenodo, cut a release, and cite the DOI that comes out of it.

## License

MIT. See [`LICENSE`](LICENSE).
