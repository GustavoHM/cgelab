# Roadmap

What `cgekit` does not do yet, in rough order of priority. Each item says why it matters. Most of them came up when the package met applied models. What had to be done by hand there
is the clearest evidence of what is missing.

Nothing here is a promise or a date. Items move when an applied model needs them.

## Next: gaps an applied model hit

**Labour-market closures.** Every factor market clears, so employment is fixed by supply and a
shock shows up in the wage. Many regional and policy studies need unemployment. Candidates are a
fixed real wage (Keynesian closure), a wage curve linking the wage to unemployment, and a labour
supply elasticity. Without them, "jobs created" cannot be a model result, and it is the result a
regional study is most often asked for.

**Results the user now computes by hand.** Three helpers:

- factor demand by activity (employment by sector), now rebuilt from `factor_intensity`;
- real GDP from the expenditure and value-added sides, with the two reconciled;
- a tidy frame of sectoral output and prices along a path.

An applied model computes all three itself, which is where results drift apart between users.

**Investment composition as a scenario field.** Investment demand has fixed base-year
commodity shares. A project that is mostly construction and equipment (a CAPEX programme)
cannot change that mix except through `run_path(economy=...)`. A `Scenario.investment_mix`, or
investment by destination sector, would make the channel explicit.

**Downward-sloping export demand.** Under CET the world export price is exogenous: the economy
is a price taker abroad. A region selling a large share of a product (a state within a country,
a dominant exporter) needs a demand curve with an elasticity instead.

**Import tariffs separate from product taxes.** The SAM's single `gov <- commodity` row mixes
tariffs with domestic product taxes, so a tariff scenario cannot move one without the other.
This needs a tariff account and a rate on the import price.

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
partial mobility.

**More household demand systems.** AIDS or a CDE system, for income elasticities that change
with income, and a labour–leisure choice for labour supply.

## Later: scope

**Several regions.** An interregional model in the tradition of B-MARIA or TERM: regional
SAMs, trade between regions, factor migration. This is the largest item. It is what a
Brazilian state study ultimately wants, and today it is done with a single region plus a "rest
of the country" account.

**Forward-looking dynamics.** `run_path` is recursive: no agent looks ahead. Perfect-foresight
investment (a Ramsey or Tobin's q path) would make announced policies matter before they start.

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
- **External cross-check:** replicate a published standard model, such as IFPRI's standard CGE
  or a GAMS model-library entry, as a check against someone else's code.
