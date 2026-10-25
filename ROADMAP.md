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
5. The matching row of `docs/comparison.md` (and `comparison.pt-br.md`) updated.
6. A `CHANGELOG.md` entry.

## Milestones

Each item is described in the sections further down; this is the order.

**0.2 — done.** SAM institutions (direct taxes, transfers, remittances, factor income to
government and abroad, FDI), several capital stocks, per-year economies, CET exports actually
solved, labour closures, the tariff and the indirect tax on domestic sales, the origin nest for
several regions, the result helpers, and the JuMP export replicating all of it. Closed since:

- [x] the JuMP export converging from a cold start on an applied SAM, with prices displaced and
  the rest at base-year levels;
- [x] downward-sloping export demand, and its mirror, an upward-sloping import supply
  (`ModelSpec.export_demand`, `import_supply`);
- [x] the base-year tariff inside the import price, and the indirect tax on all of output,
  with Hosoe's closure (`government="saving_rate"`, foreign saving in dollars). The tariff
  abolition of `stdcge` and `lrgcge` reproduces.

**0.3 — results, decompositions and accounts: done.**

- [x] **Subtotals**: decomposing a result by shock (Harrison, Horridge and Pearson, 2000),
  `subtotals`.
- [x] **Systematic sensitivity**: Gaussian quadrature and Monte Carlo over elasticities
  (`sensitivity`, `spec_sweep`).
- [x] **Accounts and reports**: GDP by expenditure and by income, Laspeyres, Paasche and Fisher
  price indices, terms of trade, government accounts (`national_accounts`, `price_indices`,
  `government_account`).
- [x] **Welfare decomposition** into allocative efficiency, terms of trade and endowments (Huff
  and Hertel, 2000), `welfare_decomposition`. The identity closes to below 1e-9; the comparison
  with GTAP waits on the GTAP database, with the regional items of 0.7.
- [x] **SAM multipliers and input–output analysis**: Leontief and fixed-price multipliers,
  backward and forward linkages, structural path analysis (`cgelab.multipliers`).

**0.4 — energy and the bridge to dsgelab: done.**

- [x] **Emissions accounts** and a **carbon tax** with revenue recycling (`add_emissions`,
  `Scenario.carbon_price`, the revenue-neutral closure).
- [x] **KLEM nesting**: a value-added-energy nest with interfuel substitution
  (`ModelSpec.energy`, `sigma_vae`, `sigma_energy`).
- [x] **Physical units for energy**: `add_physical_units`, `physical`, `physical_coefficient`.
  Capacity as a constraint needs complementarity, in 0.8.
- [x] **Government closures**: a deficit target, revenue-neutral tax replacement
  (`Closure.government_target`, `government_adjusts`), and the fixed saving rate
  (`government="saving_rate"`).
- [x] **The bridge to dsgelab** (see *Extensions that call dsgelab*): `cgelab.bridge`, the
  Ramsey CGE, with announced policy and first-order aggregate uncertainty.

**0.5 — public release.** No new features. Blocking:

- [x] **Rename**: `cgekit` was taken on PyPI; the package is now `cgelab`, free on PyPI on
  11 October 2026. The name still has to be reserved with a first upload to TestPyPI and PyPI.
- [x] **One canonical repository**: `https://github.com/GustavoHM/cgelab`. The copies in the
  research repository are archived.
- [x] **Language**: `docs/comparison.md` in English, with the Portuguese version alongside.
- [x] **Reproducible validation**: the `validation` workflow runs the JuMP replica and the
  Hosoe transcription in CI, on demand and weekly.
- [x] **Packaging**: tests on Python 3.11–3.14 on Linux, macOS and Windows; the suite run
  against the installed wheel; `python -m build` and `twine check` in CI; `py.typed`;
  `CONTRIBUTING.md` with a release checklist; issue and pull request templates; known
  limitations in the README; a `release` workflow for trusted publishing.
- [x] **Release**: 0.5.0 on PyPI on 18 October 2026, archived on Zenodo
  ([10.5281/zenodo.23458852](https://doi.org/10.5281/zenodo.23458852)), the repository public and the site on Pages.

**0.6 — the distributional side and the labour market.** The six items below are done and were
released in 0.5.5, ahead of 0.6.0, which also waits for the robustness items (see
*Robustness*).

- [x] **Imperfect factor mobility and the proximity matrix**: a CET or a logit across the
  activities that use a factor, pools of origin (`ModelSpec.factor_mobility`,
  `calibrate(factor_pools=)`).
- [x] **Segmentation and informality**: labour accounts that share one population, rigid
  segments, a dual market, Harris–Todaro, a transformation between segments
  (`ModelSpec.labour_segments`, `Closure.labour_markets`, `segment_mobility`).
- [x] **Microsimulation, Gini and FGT**: a top-down link reconciled with the SAM, inequality
  and poverty measures with their decompositions, reweighting, a behavioural microsimulation,
  readers for the PNAD Contínua and the POF (`cgelab.micro`, `cgelab.behaviour`,
  `cgelab.ibge`).
- [x] **Building and disaggregating a SAM** from supply and use tables and household surveys
  (`sut_to_sam`, `split_households`, `split_factor`, `read_tru`; see *Tools around the model*).
- [x] **A baseline calibrated to a path of real GDP** (`calibrate_baseline`, `Baseline`).
- [x] **Capital by sector and putty-clay vintages** (`factor_pools={"f_cap": "activities"}`,
  `ModelSpec.vintages`, `split_vintages`).
- [x] **A robustness report**: one call that runs quadrature and Monte Carlo over every
  elasticity, one-at-a-time ranges for a tornado chart, a grid of closures and the
  uncertainty of the SAM itself, and returns tables and optional figures
  (`robustness_report`, with the `plot` extra).
- [x] **Global sensitivity indices** (first-order and total Sobol indices, Saltelli design),
  validated against SALib (`sobol`).
- [x] **Uncertainty in the SAM**: perturbing blocks of the matrix, rebalancing with
  `SAM.balance`, recalibrating and propagating the draws to the results
  (`sam_draws`, `sam_sweep`, `sam_distance`).
- [x] **Closure robustness**: the same scenario under a list of closures (`closure_sweep`,
  `closure_grid`).
- [x] **Results in money**: changes expressed as levels at base-year and at current prices,
  in a unit the user picks (`in_money`).
- [x] **A historical closure**: a limited, early version of the 0.9 closure swap, in which
  observed variables are exogenous and the technology and preference shifters that reproduce
  them are endogenous (`calibrate_historical`, `Historical`). The dynamic calibration of the
  baseline to a GDP path is a special case and uses the same root routines.
- [x] **Release**: 0.6.0 on PyPI and archived on Zenodo
  ([10.5281/zenodo.23624109](https://doi.org/10.5281/zenodo.23624109)).

**0.7 — several regions, completed.** Factor migration, regional investment and government
closures, margins and CET inside the origin nest, a capital stock per region in `run_path`;
bilateral trade with transport margins; reading and aggregating GTAP data. Reading is in:
`read_har` and `write_har`, exact against harpy3 on GTAP 10A. Declared aggregation with explicit
parameter weights and a one-region SAM are in, with a hand-balanced synthetic GTAP example and
optional licensed-data tests. The comparison with GTAPAgg/FlexAgg and ORANI-G/TERM moved to
0.8. On top of the regional model, the
bridge to causal inference (see *General equilibrium and causal inference*):

- [x] **Treated against control** as a result of a regional model (`treated_vs_control`).
- [x] **A simulated panel**: a tidy region-by-year panel from `run_path`, with optional noise,
  on which a reader runs the event-study estimator of their choice (`simulated_panel`).
- [x] **Matching external targets**: elasticities chosen by minimum distance to estimated
  coefficients, weighted by the inverse of their covariance, on the machinery of `spec_sweep`
  (`match_targets`), checked against closed-form weighted least squares.
- [ ] **Release**: the version, the dates and the citations set to 0.7.0, the `validation`
  workflow, a tag, PyPI and Zenodo.

**0.8 — complementarity and solvers.** Activities that can switch off, quotas with rents, price
floors and ceilings, as a mixed complementarity problem; Pyomo with IPOPT and PATH besides
`scipy`. Validation against GAMS/MPSGE with PATH. Analytic derivatives and sparse algebra for
large SAMs. Moved from 0.7: the regional model validated against GTAP and ORANI-G/TERM, and the
GTAP aggregation compared with GTAPAgg/FlexAgg.

**0.9 — the frontier.** A generic closure swap in the GEMPACK manner (the historical closure of
0.6 is its first case); SAM estimation by cross
entropy; an intertemporal CGE with expectations, on the dsgelab bridge; imperfect competition
and Melitz firms; asymmetric Armington; export taxes; capital utilization; AIDS and CDE demand;
land use allocated by CET; export to GAMS/MPSGE and GEMPACK.

**1.0.** All of the above, the API frozen, strict semantic versioning from then on.

## Next: gaps an applied model hit

Nothing from 0.2 to 0.5 is open; the first six items of 0.6 are released in 0.5.5, and 0.6.0
waits for the robustness items and its release.

Done, and no longer gaps:

- **The JuMP replica from a cold start on an applied SAM.** It had started household income at
  1 and the Armington shares at guesses. From base-year levels, as GAMS initialises a model,
  Ipopt solves the applied SAM's base year and a shocked period from a cold start. The SAM is
  not public yet.
- **Trade taxes inside the prices.** The tariff sits inside the Armington import price and the
  indirect tax on all of output inside the gross price the CET splits, as in Hosoe, Gasawa and
  Hashimoto's `stdcge`. With Hosoe's closure (`government="saving_rate"`, foreign saving fixed
  in dollars), the tariff abolition of `stdcge` and of `lrgcge` matches a line-by-line JuMP
  transcription of the GAMS programs in every displayed variable (`validation/hosoe.jl`).
- **Export demand and import supply** with finite elasticities (`ModelSpec.export_demand`,
  `import_supply`), the equations of `lrgcge`.
- **Labour-market closures** (`Closure.labour`: fixed real wage, wage curve, elastic supply).
- **Results:** the three helpers (`factor_demand_by_activity`, `real_gdp`, `sectoral_path`).
- **Investment:** `Scenario.investment_mix` for the commodity basket of investment.
  Investment by destination sector stays `Growth.allocation`.

**Microsimulation.** Done in 0.6: `cgelab.micro`, a top-down link from the household
accounts to the households of a survey, reconciled with the SAM, with Gini, FGT, generalised
entropy and Atkinson indices, the survey reweighted to the model's employment by segment and
unemployment, and a behavioural microsimulation in the manner of Bourguignon, Robilliard and
Robinson (2005), and readers for the IBGE's PNAD Contínua and POF (`cgelab.ibge`).

## Then: model structure

**Nesting beyond value added.** The value-added-energy nest is in: fuel use answers to prices,
which an applied model otherwise does by hand, swapping one fuel for another volume for volume at
a parity price the user picks. GTAP-E's arrangement is in too (`energy_nest="capital"`):
a capital-energy composite inside value added, in which capital and energy can be complements
against labour. Deeper energy trees (electricity against non-electric fuels, then coal against
the rest) would be further nests of the same kind.

**Capacity in physical units.** The physical-units account is in: a coefficient in kWh per kg
becomes a SAM entry with `physical_coefficient`, and a path reports kWh and tonnes. A capacity
stated in tonnes is a bound on an activity, and that is a complementarity condition, in 0.8.

**Government closures.** Done in 0.4: a fixed public saving target and revenue-neutral tax
replacement, met by direct taxes, product taxes or spending (`Closure.government_target`,
`government_adjusts`).

**Imperfect factor mobility.** Done in 0.6: `ModelSpec.factor_mobility`, a CET or a logit
across the activities that use a factor, and `calibrate(factor_pools=)`, pools of origin that
reach only the activities they supplied in the base year. That is the proximity matrix
Lofgren and Cicowiez (2017) ask for, in which sectors hire from neighbouring occupations
rather than from one national nest. Their food-export-price result is the replication still
open.

**Segmentation and informality.** Done in 0.6: `ModelSpec.labour_segments` and
`Closure.labour_markets` (labour accounts that share one population; rigid segments, a dual
market and Harris–Todaro, or `segment_mobility`, a CET or logit between segments with a queue
in each rigid one; income that follows each household's workers). Still open: employment in
persons rather than in base-year wage units, unemployment benefits, and a wage curve by
segment.

**Capital by sector and vintages.** Done in 0.6: a capital stock per activity
(`calibrate(factor_pools={"f_cap": "activities"})`) with investment shared by rents, and
putty-clay vintages (`ModelSpec.vintages`): installed capital sector-specific and with the
technique it was installed with, new capital mobile. Still open: perfect substitution between
the vintages, which needs a regime switch for idle capital, and the sale of old capital by
declining sectors, as in LINKAGE; vintages with an energy nest, with foreign-owned capital
and in several regions.

**Baseline and historical calibration.** Done in 0.6: `calibrate_baseline`, the productivity
path that makes a recursive baseline hit real GDP, by year or by activity, and `Baseline`, which
runs a policy at it. `calibrate_historical` generalises the same root routines to observed scalar
targets with technology and relative-preference shifters, and `Historical` replays a policy at
them. Still open: other unknowns (employment through labour supply, the investment rate through
saving, exports through world prices) and the generic closure swap of 0.9.

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
numeraire. Fixed-real investment with forced saving is in, alongside savings-driven
investment, and optional government and investment accounts can be mapped by
region. CET/logit factor migration with a proximity matrix is in too. Delivered
margins and CET exports inside the origin nest are in, with one-region checks against
the national solver. Destination-specific origin nests reproduce bilateral SAM cells,
with domestic and imported content reported separately. Route transport and the
Cobb–Douglas global service pool are calibrated and solved, with GTAP FOB/CIF and
supply identities tested against a hand-written SAM. Regional government closures
match the national solver, including saving-rate and tax/spending targets. Trade
taxes remain open.
Regional capital accumulation and a tidy simulated panel are in. The
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

dsgelab never reads a SAM. cgelab takes it as an optional dependency (the `dsge` extra),
imported only by the bridge, so a static model keeps the plain install (numpy, scipy, pandas). The bridge is one
residual. Given the predetermined stocks and the forward-looking guesses, cgelab returns the
period equilibrium: rentals, purchaser prices, investment, household consumption. The residual
dsgelab sees is the Euler gap, or Tobin's q. Commodity balances stay inside `solve_period`,
which is where Walras, homogeneity and the base year are already checked. A SAM of a few
hundred accounts does not become a DSGE with a few hundred states. The static block is
substituted out by the period solver, and the perturbation runs in the aggregate states.

Status: the bridge is in (`cgelab.bridge`). Announced policy, the forward-looking steady state
and first-order aggregate uncertainty work through it; risk-adjusted welfare and
forward-looking migration are still open.

**Announced policy.** Done: `bridge.announced`. A tax, a tariff or a carbon price that agents
know will change on a dated schedule leaves year-1 investment where it was, under `run_path`.
`dsgelab.perfect_foresight` already stacks one residual over a horizon and already accepts a
parameter path of length `T+1`. The extension is that residual calling `solve_period`. The
instrument stays a `Scenario` field. The emissions account and the carbon price came in 0.4;
this item is the anticipation of a path those fields supply.
A news shock to a world price uses the same bridge.

**Aggregate uncertainty.** The states to perturb are the capital stocks `Growth` already
accumulates, plus a short list of aggregate processes: productivity, a world price, a public
spending shock. Impulse responses come from `solve`. Second-order risk corrections come from
`solve_second_order`. This does not turn cgelab into an estimation package: the likelihood and
the Kalman filter stay in dsgelab.

**Risk-adjusted welfare.** Open. `equivalent_variation` compares two deterministic paths. The
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

Changes to the period equilibrium stay in the sections above, whether done (energy nests,
tariff accounting, government closures) or open (bilateral trade among regions, a GTAP-style
global bank). dsgelab has no
representation for a SAM, and calling it does not close those gaps.

## Robustness

**What is there.** `sensitivity` and `spec_sweep` (0.3) turn a result into a distribution over
the elasticities; `closure_sweep`, `in_money`, `sam_draws` and `sam_sweep` (0.6) solve it under
a list of closures, put it in money, and propagate balanced SAM draws through recalibration and
solution. `robustness_report` assembles them with sourced ranges, one-at-a-time results and
optional tornado and distribution figures. Its parts stay public, so that a study can run one
of them alone.

**Sobol indices.** The quadrature gives a mean and a standard deviation. It does not say which
parameter the variance comes from when they interact. First-order and total Sobol indices do,
at the cost of many more solves; SALib is the independent reference.

**The SAM is an estimate.** Every entry of a SAM was estimated, and RAS or cross entropy then
moved it. Perturbing blocks of the matrix (a Dirichlet draw on the shares of a column, or
multiplicative noise followed by `SAM.balance`), recalibrating and solving again gives the part
of the uncertainty that the elasticities do not carry. The reference is the same experiment
done by hand, and a case with a closed form (Leontief).

## General equilibrium and causal inference

An event study identifies the effect of a shock on treated units relative to controls. A
general equilibrium model gives the aggregate effect and the counterfactual for a policy that
has not happened. The two meet in three places: estimated coefficients can discipline the
model's elasticities, the model can be asked to predict an effect estimated after the fact,
and the model can generate a panel on which an estimator is run to see what it recovers when
general equilibrium effects reach the controls. The econometrics stays outside the package, in
the estimators the reader already uses. The package provides the regional result, the panel
and the matching, in 0.7, because a treated and a control unit need the regional model.

## Tools around the model

**Building a SAM from national-accounts tables.** Most of the work in an applied model happens
before `calibrate`. A regional SAM is usually built by hand from a supply-use table, an
input–output matrix and regional accounts. Useful helpers:

- read a supply-use table into a SAM (in 0.6: `sut_to_sam`, with the provenance of every
  block, and `read_tru` for the IBGE's tables);
- regionalise national coefficients (location quotients, FLQ);
- balance by cross-entropy as well as RAS;
- report where each block came from (in 0.6: `SAM.provenance`);
- disaggregate households and labour with household surveys (in 0.6: `split_households`,
  `split_factor`, `segment_shares`).

**Speed on large SAMs.** The solver uses finite-difference Jacobians on few unknowns, which is
fine up to a few hundred accounts. A multi-region model would need analytic derivatives and
sparse linear algebra.

## Infrastructure

- **Release:** done for 0.5.0, on PyPI and archived on Zenodo; the steps for the next ones are
  in `CONTRIBUTING.md`, "Releasing".
- **Site:** on Pages since 0.5.0. Still open: render the Markdown docs on the site instead of
  linking to GitHub.
- **Validation in CI:** done, the `validation` workflow.
- **Language:** done for the documents a reader meets first. The whitepaper and the comments in
  the JuMP template and the validation scripts are still in Portuguese.
- **Packaging:** done; see the 0.5 milestone.
- **External cross-check:** replicate a published standard model against someone else's code.
  Done for Hosoe, Gasawa and Hashimoto (2004): the `splcge` benchmark, and the tariff
  abolition of `stdcge` and `lrgcge` against a transcription of the GAMS programs. The GAMS
  output itself is not published; running the `.gms` files in GAMS would close that last
  step. The IFPRI standard CGE
  (`sim101`, a 50% tariff cut, Zimbabwe and Eswatini data, no solved listing) is the same class
  of gap. Galindev and Decaluwé (2022) is the nearest journal package with a printed result;
  the utilization-rate variant is open. Cicowiez and Lofgren (2022) wait on asymmetric
  Armington, Lofgren and Cicowiez (2017) on the proximity matrix, Jia and Lin (2022) on the
  energy nest and a carbon tax, and Mukashov and Thurlow (2026) on several regions together
  with margins and recursive dynamics. A GTAP paper waits on several regions and on a licence
  for the database.
