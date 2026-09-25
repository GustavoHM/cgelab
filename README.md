# cgekit

[![tests](https://github.com/GustavoHM/cgekit/actions/workflows/tests.yml/badge.svg)](https://github.com/GustavoHM/cgekit/actions/workflows/tests.yml)
[![site](https://img.shields.io/badge/site-gustavohm.github.io%2Fcgekit-2b6cb0)](https://gustavohm.github.io/cgekit/)
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
pip install "cgekit @ git+https://github.com/GustavoHM/cgekit"
pip install "cgekit[parquet] @ git+https://github.com/GustavoHM/cgekit"   # adds pyarrow for .parquet SAMs
```

For development:

```
git clone https://github.com/GustavoHM/cgekit && cd cgekit
pip install -e ".[test]" && pytest
```

Requires Python 3.10+, numpy, scipy and pandas. Try it with no data at all:

```
cgekit                    # runs the bundled toy economy and prints the diagnostics
```

## Quick start

The bundled toy economy has every SAM entry written out, so the base year is
known on paper and this runs as it is:

```python
import cgekit as ck
from cgekit.examples import toy

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
| SAM container, balance gate, sign-preserving RAS | `SAM` |
| Calibration: coefficients, nests, taxes, margins, LES subsistence, institutional flows, a list of cells the model does not read | `calibrate` |
| Within-period equilibrium under five closure combinations | `solve_period` |
| Counterfactuals: taxes, wedges, endowments, ownership, productivity, own-account levels, world prices, FDI | `Scenario` |
| Recursive dynamics: several capital stocks, investment allocation, foreign-owned capital, a per-year economy | `run_path`, `Growth` |
| Welfare: EV, CV and composition-immune group aggregation | `cgekit.welfare` |
| Base-year reproduction, homogeneity, Walras | `cgekit.diagnostics` |
| The same equilibrium as a JuMP model, for independent validation | `to_jump` |

## Documentation

- [Site](https://gustavohm.github.io/cgekit/): an overview with runnable examples, in Portuguese
- [`docs/guide.md`](docs/guide.md): the user guide, from SAM to welfare
- [`docs/api.md`](docs/api.md): API reference
- [`docs/design-notes.md`](docs/design-notes.md): the accounting decisions and the errors they prevent
- [`docs/from-scripts.md`](docs/from-scripts.md): porting an existing CGE pipeline
- [`docs/comparison.md`](docs/comparison.md): how this package compares with the ones that already exist, and where it does not win (in Portuguese)
- [`CHANGELOG.md`](CHANGELOG.md): what changed between versions
- [`ROADMAP.md`](ROADMAP.md): what is still missing, and why it matters

## Verification

`pytest` runs 99 tests against six hand-built SAMs whose every entry is written
out, so the checks compare the solver with arithmetic rather than with its own
previous output:

- base-year reproduction to machine precision, with every price one and every
  quantity at its base-year value: in the plain case, with a national-accounts
  margin block, with an own-account activity and with institutional flows;
- zero-degree homogeneity in prices, in all of them;
- Walras' law under a shock in **every** closure combination, and at every date
  along a growing path;
- the same three gates with CET exports, plus the CET revenue identity and an
  export response that grows with the elasticity;
- a SAM with direct taxes, transfers, remittances and factor income paid to
  government and abroad reproduces its base year, and FDI-financed capital is
  foreign-owned, with its returns leaving the economy;
- several capital stocks keep their composition with no shock, and tilt toward
  the higher rent when allocation responds to it;
- wedge revenue recycled equals wedge revenue paid, exactly;
- group welfare aggregation returns the zero that is there when identical
  households merely migrate between accounts, while the per-account measure
  reports a spurious ±5%;
- every executable example on the site runs.

There is also an independent check of the algebra. `to_jump` writes the same
equilibrium as a JuMP model in which nothing is nested: the price fixed point,
the wedge series and the saving–revenue relation are all plain constraints.
Ipopt solves it from a displaced start. `validation/roundtrip_jump.py` runs 16
cases (margins, own-account activities, CET, direct taxes, transfers,
foreign-owned capital, FDI, LES, alternative closures). The nested solver and
Ipopt agree to at most 3e-12.

```
cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
python validation/roundtrip_jump.py            # or: CGEKIT_RUN_JULIA=1 pytest tests/test_jump_roundtrip.py
```

## Scope

`cgekit` is a single-region, recursive-dynamic model with no forward-looking
agent. It has no multi-region trade, no estimation of elasticities and none of
the maturity of GAMS or GEMPACK. [`docs/comparison.md`](docs/comparison.md) says
where it stands.

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
