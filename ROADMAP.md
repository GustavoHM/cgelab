# Roadmap

What `cgekit` does not do yet, in rough order of priority. Each item says why it matters. Most of them came up when the package met applied models. What had to be done by hand there
is the clearest evidence of what is missing.

Nothing here is a promise or a date. Items move when an applied model needs them.

## Next: gaps an applied model hit

**Downward-sloping export demand.** Under CET the world export price is exogenous: the economy
is a price taker abroad. A region selling a large share of a product (a state within a country,
a dominant exporter) needs a demand curve with an elasticity instead. World import prices are
scenario data in the same way. The GAMS library model `lrgcge` clears them, on the same SAM
and the same tariff abolition as `stdcge`.

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

Done, and no longer gaps: labour-market closures (`Closure.labour`: fixed real wage, wage curve,
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

**Forward-looking dynamics.** `run_path` is recursive: no agent looks ahead. Perfect-foresight
investment (a Ramsey or Tobin's q path) would make announced policies matter before they start.

**Market structure beyond a competitive Armington.** The GAMS library files `moncge`, `quocge`,
`irscge` and `twocge` add monopoly rent, an import quota, a fixed cost of capital, or a second
country. The solver has none of these. They wait until an applied model asks.

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
- **Language:** the site and `docs/comparison.md` are in Portuguese and the rest in English.
  Pick one, or keep both, but consistently.
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
