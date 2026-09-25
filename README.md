# cgekit

A general toolkit for SAM-based computable general equilibrium models in Python.
You supply a social accounting matrix; the package calibrates a standard nested
CGE from it, solves each period, runs a recursive-dynamic path, and measures
welfare — with the accounting checks that make any of it believable built in
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

```
pip install -e packages/cgekit
pip install -e "packages/cgekit[parquet]"     # adds pyarrow for .parquet SAMs
```

Requires Python 3.10+, numpy, scipy and pandas.

## Quick start

```python
import cgekit as ck

sam  = ck.SAM.read("data/processed/sam_2021.parquet")
sam.check_balance()                                    # a hard gate, not a warning
econ = ck.calibrate(sam, ck.ModelSpec(sigma_armington=2.0, sigma_va=0.5))

print(ck.report(econ))          # base-year reproduction, Walras, homogeneity

base  = ck.solve_period(econ)
shock = ck.solve_period(econ, scenario=ck.Scenario(
    name="tax reform", product_tax=new_rates))
print(shock.summary())
```

A counterfactual over a horizon, and its welfare incidence:

```python
growth = ck.Growth(labour=0.005, tfp=0.010, depreciation=0.05, discount=0.05)
years  = range(2021, 2046)

W = ck.consumer_wedge(econ, {"c_electricity": 0.03})
ref = ck.run_path(econ, years=years, growth=growth)
alt = ck.run_path(econ, years=years, growth=growth,
                  scenario=lambda y, t: ck.Scenario(consumer_wedge=W,
                                                    wedge_revenue_to="capital")
                                        if y >= 2026 else ck.Scenario())

ev = ck.equivalent_variation(ref, alt)
ck.ev_by_group(ev, {"hh_d1_pro": "d1", "hh_d1_npr": "d1", ...})
```

Try it with no data at all:

```
cgekit                    # runs the bundled toy economy and prints the diagnostics
```

## Model structure

* **Production** — Leontief between the intermediate bundle and value added, and
  within the intermediate bundle; a configurable nest (CES, Cobb–Douglas or
  Leontief) over factors inside value added.
* **Make** — commodity market shares fixed at base-year proportions, so activity
  levels follow from commodity output, `X = MSH @ Q`.
* **Armington** — CES between domestic and imported varieties.
* **Exports** — fixed shares by default; CET available as a robustness variant.
  The default is deliberate: the CET elasticity is the one parameter of a
  standard CGE that input–output sources do not supply.
* **Households** — Cobb–Douglas or LES over commodities, on income net of direct
  tax, saving and any payment made straight to an activity. Income includes
  government transfers and net transfers from abroad.
* **Ownership** — factor income is shared among households, government and the
  rest of the world. Returns to foreign-owned capital leave through the current
  account, and foreign saving is the current account, not the trade balance.
* **Government, investment, foreign** — see `Closure`.

The **numeraire factor's market is deliberately not solved**. It is left to
Walras' law so that its residual is an independent check on the accounting rather
than an identity the solver was told to satisfy, and it is recorded on every
period.

## What is in the box

| Capability | Entry point |
|---|---|
| SAM container, balance gate, sign-preserving RAS | `SAM` |
| Calibration: coefficients, nests, taxes, margins, LES subsistence | `calibrate` |
| Within-period equilibrium under five closure combinations | `solve_period` |
| Counterfactuals: taxes, wedges, endowments, ownership, productivity, own-account levels | `Scenario` |
| Recursive dynamics: one or several capital stocks, investment allocation, FDI with foreign ownership, a per-year economy | `run_path`, `Growth` |
| Welfare: EV, CV, and composition-immune group aggregation | `cgekit.welfare` |
| Base-year reproduction, homogeneity, Walras | `cgekit.diagnostics` |

## Documentation

- [`docs/guide.md`](docs/guide.md) — the user guide, from SAM to welfare
- [`docs/api.md`](docs/api.md) — API reference
- [`docs/design-notes.md`](docs/design-notes.md) — the accounting decisions and the errors they prevent
- [`docs/from-scripts.md`](docs/from-scripts.md) — porting an existing CGE pipeline
- [`../COMPARISON.md`](../COMPARISON.md) — how this package compares with the ones that already exist, and where it does not win
- [`site/`](site/) — a GitHub-Pages-ready page with runnable examples

## Verification

`pytest` runs 96 tests against three hand-built SAMs whose every entry is written
out, so the checks compare the solver with arithmetic rather than with its own
previous output:

- base-year reproduction to machine precision — every price one, every quantity
  its base-year value — in the plain case, with a national-accounts margin block,
  and with an own-account activity;
- zero-degree homogeneity in prices, in all three;
- Walras' law under a shock in **every** closure combination, and at every date
  along a growing path;
- the same three gates with CET exports, plus the CET revenue identity and an
  export response that grows with the elasticity;
- moving foreign saving alone keeps Walras, since it is the single
  current-account target;
- a SAM with direct taxes, government transfers, remittances and factor income
  paid to government and abroad reproduces its base year (the previous version
  missed it by 12%); FDI-financed capital is foreign-owned and its returns leave;
- several capital stocks keep their composition with no shock and tilt toward
  the higher rent when allocation responds to it;
- wedge revenue recycled equals wedge revenue paid, exactly;
- group welfare aggregation returning the zero that is there when identical
  households merely migrate between accounts, while the per-account measure
  reports a spurious ±5%.

```
pip install -e "packages/cgekit[test]" && pytest packages/cgekit
```

An independent check of the algebra: `to_jump` writes the same equilibrium as a
JuMP model in which nothing is nested — the price fixed point, the wedge series
and the saving–revenue relation are all plain constraints — and Ipopt solves it
from a displaced start. `validation/roundtrip_jump.py` runs 16 cases (margins,
own-account activities, CET, direct taxes, transfers, foreign-owned capital, FDI,
LES, alternative closures) and agrees with the nested solver to at most 3e-12.

```
cd packages/cgekit/validation
julia --project=. -e 'using Pkg; Pkg.instantiate()'
python roundtrip_jump.py                       # or: CGEKIT_RUN_JULIA=1 pytest tests/test_jump_roundtrip.py
```

## Citing

If a published result went through this package, cite it — software is a methodological
input like any other, and the VERSION matters, since a number that only reproduces under
one version is not reproducible without it. [`CITATION.cff`](CITATION.cff) carries the
metadata in the format GitHub reads; fill in the DOI and ORCID before citing.

A repository URL is not an archivable reference. Connect the repository to Zenodo, cut a
release, and cite the DOI that comes out of it.
