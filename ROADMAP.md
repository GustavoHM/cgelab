# Roadmap

Plan revised on 11 October 2026. The goal is to make `cgelab` **as complete as possible**:
**version 1.0** covers what the reference platforms offer — GAMS/MPSGE, GEMPACK, GTAP and the
IFPRI and PEP standard models — together with the gaps the applied models exposed. The **public
release** is **version 0.5**. Until 11 October this file said that nothing in it was a promise and
that items moved only when an applied model needed them; that rule is withdrawn. The arguments
behind it now stand as design conditions of the items they concerned.

`dsgelab` moves through the same milestones in its own roadmap: "0.3" means the 0.3 of each.
Most items below came up when the package met applied models. What had to be done by hand
there is the clearest evidence of what is missing.

## What counts as done

An item leaves this list when it meets all six conditions. The one guarantee the package offers
is that every number was checked against an independent reference.

1. Implemented, with a public API in `__all__`.
2. **Validated against an independent reference**: another implementation (GAMS, GEMPACK, JuMP,
   a published replication package), a closed form or a published result, in a script anyone
   can run.
3. Tested in the package's suite without external tools; what needs Julia or GAMS lives in
   `validation/`.
4. Documented in the guide and the API reference, with a runnable example on the site.
5. The matching row of `docs/comparison.md` updated.
6. A `CHANGELOG.md` entry.

## Milestones

Each item is described in the sections further down; this is the order.

**0.2 — done.** SAM institutions (direct taxes, transfers, remittances, factor income to
government and abroad, FDI), several capital stocks, per-year economies, CET exports actually
solved, labour closures, the tariff and the indirect tax on domestic sales, the origin nest for
several regions, the result helpers, and the JuMP export replicating all of it. Open from 0.2:

- [ ] the JuMP export converging from a cold start on an applied SAM (it does on the toy
  economies, and on an applied SAM only from a warm start);
- [x] downward-sloping export demand, and its mirror, an upward-sloping import supply
  (`ModelSpec.export_demand`, `import_supply`);
- [ ] the base-year tariff inside the import price, and the indirect tax on all of output.

**0.3 — results, decompositions and accounts.**

- [ ] **Subtotals**: decomposing a result by shock (Harrison, Horridge and Pearson, 2000).
- [ ] **Systematic sensitivity**: Gaussian quadrature and Monte Carlo over elasticities (see
  *Parameter sensitivity*).
- [ ] **Accounts and reports**: GDP by expenditure and by income, Laspeyres, Paasche and Fisher
  price indices, terms of trade, government accounts. `real_gdp` is the first of these.
- [ ] **Welfare decomposition** into allocative efficiency, terms of trade and endowments (Huff
  and Hertel, 2000). Validation against GTAP.
- [ ] **SAM multipliers and input–output analysis**: Leontief and fixed-price multipliers,
  backward and forward linkages, structural path analysis.

**0.4 — energy and the bridge to dsgelab.**

- [ ] **KLEM nesting**, **emissions accounts**, a **carbon tax** with revenue recycling, and
  **physical units for energy** (see *Then: model structure*).
- [ ] **Government closures**: a deficit or saving target, revenue-neutral tax replacement.
- [ ] **The bridge to dsgelab** (see *Extensions that call dsgelab*).

**0.5 — public release.** No new features. Blocking:

- [x] **Rename**: `cgekit` was taken on PyPI; the package is now `cgelab`, free on PyPI on
  11 October 2026. The name still has to be reserved with a first upload to TestPyPI and PyPI.
- [x] **One canonical repository**: `https://github.com/GustavoHM/cgelab`. The copies in the
  research repository are archived.
- [ ] **Language**, **reproducible validation**, release engineering: see *Infrastructure*.

**0.6 — the distributional side and the labour market.** Imperfect factor mobility and the
proximity matrix; segmentation and informality; a link to microsimulation with Gini and FGT
indices; dynamic calibration of the baseline to an observed or projected GDP path; sectoral
capital and vintages (putty-clay); building and disaggregating a SAM from supply-use tables
and household surveys (see *Tools around the model*).

**0.7 — several regions, completed.** Factor migration, regional investment and government
closures, margins and CET inside the origin nest, a capital stock per region in `run_path`;
bilateral trade with transport margins; reading and aggregating GTAP data. Validation against
GTAP and ORANI-G/TERM.

**0.8 — complementarity and solvers.** Activities that can switch off, quotas with rents, price
floors and ceilings, as a mixed complementarity problem; Pyomo with IPOPT and PATH besides
`scipy`. Validation against GAMS/MPSGE with PATH. Analytic derivatives and sparse algebra for
large SAMs.

**0.9 — the frontier.** A generic closure swap in the GEMPACK manner; SAM estimation by cross
entropy; an intertemporal CGE with expectations, on the dsgelab bridge; imperfect competition
and Melitz firms; asymmetric Armington; export taxes; capital utilization; AIDS and CDE demand;
land use allocated by CET; export to GAMS/MPSGE and GEMPACK.

**1.0.** All of the above, the API frozen, strict semantic versioning from then on.

## Next: gaps an applied model hit

**Base-year tariff inside the import price.** `Scenario.tariff` moves the power
`(1 + tm) / (1 + tm0)` around a reference rate. At the calibrated rate that power is 1, the
receipt of a named tariff account stays inside the product tax, and government income receives
the change in the rate. Hosoe, Gasawa and Hashimoto's standard model (`stdcge`, Table 8.1) and
the IFPRI standard model calibrate the Armington shares with `(1 + taum)` inside the import
price, so the base year already contains the wedge and the tariff revenue is a level. Reproducing
an abolition means recalibrating those shares, with base prices still 1, the border payment equal
to SAM imports, and tariff revenue equal to the tariff account at the calibrated rate. The solved
percent changes are computed in `stdcge.gms` and are not stored there.

**Indirect tax on the whole of output.** `Scenario.output_tax` raises the domestic price in the
Armington nest, and the base is domestic sales. Exports sold at the producer price are outside
it. Hosoe's IDT is `tauz` on `pz × Z`, applied inside the CET, so the base is all of output.
With exports, removing one is a different experiment from removing the other.

Done, and no longer gaps: export demand and import supply with finite elasticities
(`ModelSpec.export_demand`, `import_supply`, the equations of `lrgcge`; replicating `lrgcge`
itself waits on the base-year tariff below, since its experiment is the same tariff abolition as
`stdcge`), labour-market closures (`Closure.labour`: fixed real wage, wage curve,
elastic supply), the three result helpers (`factor_demand_by_activity`, `real_gdp`,
`sectoral_path`), `Scenario.investment_mix` for the commodity basket of investment, and
the split of an import tariff (`Scenario.tariff`) from an indirect tax on domestic sales
(`Scenario.output_tax`). At the calibrated rate the base year stays the one the product
tax closed; government income moves when the rate moves. A named tariff or indirect-tax
account keeps its receipt inside that product tax. Investment by destination sector stays
`Growth.allocation`.

## Then: model structure

**Nesting beyond value added.** Intermediates are Leontief, so an energy transition can only
be modelled by changing coefficients by hand. That is what an applied model does by hand: one fuel swapped for another, volume for volume, at a
parity price the user has to pick. A configurable
KLEM nest (capital–energy, then energy between fuels, with CES elasticities) would make fuel
substitution a response to prices.

**Emissions accounts.** A satellite account of emission coefficients (per unit of fuel used and
per unit of activity output), reported along a path, plus a carbon tax as a scenario field. An applied model has to account for CO₂ outside it.

**Physical units for energy.** Linking a sector's monetary output to a physical quantity (kWh of electricity, kg of a fuel) would let a physical input be fixed per unit and a capacity be stated in
tonnes. An applied model has to impose such a coefficient itself.

**Government closures.** Add a fixed public deficit or saving target, and revenue-neutral tax
replacement (one tax adjusts to keep revenue constant). These are needed for any fiscal study.

**Imperfect factor mobility.** Several capital stocks already accumulate separately. Between
fully mobile and fully sector-specific, a CET over sectors for labour or capital would allow
partial mobility. Lofgren and Cicowiez (2017) ask for a further instrument: a proximity matrix,
in which sectors hire from neighbouring occupations rather than from one national nest. That
is what their food-export-price result turns on.

**Asymmetric Armington.** The substitution elasticity is one number per commodity. Cicowiez and
Lofgren (2022) make it depend on the origin of the import. Their Mongolia 2015 workbook also
holds the symmetric tariff experiments; the published result is the asymmetric one.

**Export tax.** An ad valorem tax on exports, distinct from the import tariff and from the
product tax. The same Mongolia model carries it as `te(c)`.

**Capital utilization.** Utilization is fixed at the base-year rate. Galindev and Decaluwé
(2022) compare a standard PEP-1-1 run with one in which utilization responds. The standard run,
a 25% rise in the world price of coal exports, is the nearer replication. The utilization run
is this gap.

**More household demand systems.** AIDS or a CDE system, for income elasticities that change
with income, and a labour–leisure choice for labour supply.

## Later: scope

**Several regions, continued.** The origin nest is in: regional accounts, a CES across
regions of origin under the Armington, factors clearing where they are, one national
numeraire. Still open, and still what a state study asks for next: factor migration, a
regional investment and government closure, margins and CET inside the nest,
`run_path` accumulating a capital stock per region, and a tariff or an indirect tax. The
regional solver refuses a calibrated `tm0` or `tauz0`. A GTAP-style global bank is further
out. Today a single region plus a "rest of the country" account remains the way a
national model stands in for the regions it does not have.

**Market structure beyond a competitive Armington.** The GAMS library files `moncge`, `quocge`,
`irscge` and `twocge` add monopoly rent, an import quota, a fixed cost of capital, or a second
country. The solver has none of these. They wait until an applied model asks.

## Extensions that call dsgelab

`run_path` solves one within-period equilibrium after another. Between periods it updates
capital, labour and productivity, and no agent looks ahead. That function stays as it is. The
extensions in this section are a different object, and they are the reason to keep the two
packages apart rather than fold one into the other: the period equilibrium remains a SAM model
solved here, and the intertemporal residual is solved by dsgelab.

dsgelab never reads a SAM. cgelab would take it as an optional dependency, imported only by the
bridge, so a static model keeps today's install (numpy, scipy, pandas). The bridge is one
residual. Given the predetermined stocks and the forward-looking guesses, cgelab returns the
period equilibrium: rentals, purchaser prices, investment, household consumption. The residual
dsgelab sees is the Euler gap, or Tobin's q. Commodity balances stay inside `solve_period`,
which is where Walras, homogeneity and the base year are already checked. A SAM of a few
hundred accounts does not become a DSGE with a few hundred states. The static block is
substituted out by the period solver, and the perturbation runs in the aggregate states.

**Announced policy.** A tax, a tariff or a carbon price that agents know will change on a
dated schedule leaves year-1 investment where it was, under `run_path`.
`dsgelab.perfect_foresight` already stacks one residual over a horizon and already accepts a
parameter path of length `T+1`. The extension is that residual calling `solve_period`. The
instrument stays a `Scenario` field. An emissions account and a carbon-tax field are still the
gaps under model structure; this item is the anticipation of a path those fields would supply.
A news shock to a world price uses the same bridge.

**Aggregate uncertainty.** The states to perturb are the capital stocks `Growth` already
accumulates, plus a short list of aggregate processes: productivity, a world price, a public
spending shock. Impulse responses come from `solve`. Second-order risk corrections come from
`solve_second_order`. This does not turn cgelab into an estimation package: the likelihood and
the Kalman filter stay in dsgelab.

**Risk-adjusted welfare.** `equivalent_variation` compares two deterministic paths. The
displacement of that variation that comes from aggregate risk, rather than from the mean path,
is `mean_correction` applied to the household consumption the period equilibrium returns. The
utility is the one the household block already has, Cobb–Douglas or LES.

**The forward-looking steady state is another point.** The base year is the accounting
equilibrium the SAM records. The steady state of a Ramsey or Tobin's q system is the point
where the Euler holds and capital is constant. `dsgelab.steady_state` solves that point, with
the period residual inside it. A published table has to say which of the two it reports. The
bridge keeps both, and writes neither over the other.

**Forward-looking migration, after the regional closure.** The open regional item asks for
myopic factor migration, which is a closure inside cgelab and does not need dsgelab. Migration
that answers to expected utility differences is the same bridge as the Euler: the period labour
market stays a closure, and the migration decision is an asset equation dsgelab can stack. It
waits on that closure.

KLEM nesting, tariff accounting, several regions, government closures and a GTAP-style global
bank stay in the sections above. They change the period equilibrium. dsgelab has no
representation for a SAM, and calling it does not close those gaps.

## Tools around the model

**Building a SAM from national-accounts tables.** Most of the work in an applied model happens
before `calibrate`. A regional SAM is usually built by hand from a supply-use table, an
input–output matrix and regional accounts. Useful helpers:

- read a supply-use table into a SAM;
- regionalise national coefficients (location quotients, FLQ);
- balance by cross-entropy as well as RAS;
- report where each block came from.

**Parameter sensitivity.** Run a model over a grid or a distribution of elasticities (Gaussian
quadrature or Monte Carlo) and summarise the results with their spread. Every applied study does
this by hand with its own loop.

**Speed on large SAMs.** The solver uses finite-difference Jacobians on few unknowns, which is
fine up to a few hundred accounts. A multi-region model would need analytic derivatives and
sparse linear algebra.

## Infrastructure

- **Release:** publish to PyPI and connect Zenodo for a DOI. Until then the citation points to
  a repository URL, which is not archivable.
- **Site:** turn the Pages workflow back on when the repository goes public, and render the
  Markdown docs on the site instead of linking to GitHub.
- **Validation in CI:** an optional job that installs Julia and runs
  `validation/roundtrip_jump.py`, which today runs only locally.
- **Language:** the site has English and Portuguese versions; `docs/comparison.md` is still in
  Portuguese only.
- **Packaging:** tests on Python 3.10–3.14 on Linux, macOS and Windows; installation tested from
  the wheel (an editable install can break silently on Python 3.14 on macOS); `python -m build`,
  `twine check`, trusted publishing through TestPyPI first; `CONTRIBUTING.md`, issue and pull
  request templates; `py.typed`; a known-limitations section in the README.
- **External cross-check:** replicate a published standard model against someone else's code.
  Started on the site: the Hosoe, Gasawa and Hashimoto (2004) benchmark from the GAMS library
  file `splcge`. The same authors' `stdcge` is the nearest policy experiment (tariff abolition,
  Table 8.1) and waits on the base-year import price under Next. The IFPRI standard CGE
  (`sim101`, a 50% tariff cut, Zimbabwe and Eswatini data, no solved listing) is the same class
  of gap. Galindev and Decaluwé (2022) is the nearest journal package with a printed result;
  the utilization-rate variant is open. Cicowiez and Lofgren (2022) wait on asymmetric
  Armington, Lofgren and Cicowiez (2017) on the proximity matrix, Jia and Lin (2022) on the
  energy nest and a carbon tax, and Mukashov and Thurlow (2026) on several regions together
  with margins and recursive dynamics. A GTAP paper waits on several regions and on a licence
  for the database.
