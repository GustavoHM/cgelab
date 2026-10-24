# How cgelab compares with what exists

*Versão em português: [`comparison.pt-br.md`](comparison.pt-br.md).*

The survey of the other packages was made on 9 September 2026. The `cgelab` column was revised
on 22 October 2026, for version 0.6. Rows that changed since 0.1 are marked †. The other
packages' columns were **not** redone, except the recursive-dynamics cell of `PSLmodels/CGE`,
corrected on 22 October 2026 after its code was read again.

The question behind the survey was whether a CGE package in Python already existed. The short
answer is that there are a few, all at textbook scale, and that applied CGE is done in GAMS or
GEMPACK.

## Reading the table

| Mark | Means |
|---|---|
| **✅** | verified: the package was installed, or its code cloned and read |
| **📄** | in the official documentation, **not** verified in the code |
| **—** | absent as far as was found |
| **?** | not checked |

The distinction matters. The code of `pycge`, `PSLmodels/CGE` and `python-cge` was cloned and
read. A **📄** cell may be wrong.

## The table

| Capability | cgelab 0.6 | pycge | PSLmodels/CGE | python-cge | GAMS / GEMPACK (ref.) |
|---|---|---|---|---|---|
| **Origin** | written for an applied SAM | ✅ port of `splcge.gams` from the GAMS library | ✅ "the simplest CGE model of Hosoe, Gawana [sic] and Hashimoto (2010)" | ✅ direct transcription of Hosoe's book, p. 72 | general-purpose platforms |
| **Demonstrated scale** | ✅ the applied SAM it was written for, not public yet; in public, teaching examples | ✅ 3 teaching examples and a regional one | ✅ a **16×18** SAM: 4 sectors, 3 factors | ✅ **2 goods** (`BRD`, `MLK`) | thousands of accounts (GTAP) |
| Backend | ✅ `scipy.optimize.root` on a few unknowns | ✅ Pyomo + IPOPT | ✅ `scipy` `root` | ✅ its own | dedicated solvers |
| Lines of Python | ~11,800 † | ✅ 3,267 | ✅ 2,660 | ✅ 533 | n/a |
| **SAM: reading, balance check, balancing** | ✅ long or square form, a hard gate, sign-preserving RAS | ? | ✅ reads `SAM.xlsx` | ✅ has `balance.py` | ✅ |
| SAM flows the model does not read | ✅ listed at calibration † | ? | ? | ? | the modeller's convention |
| Configurable value-added nest (CES / CD / Leontief) | ✅ | ? | ✅ CES and Leontief | — | ✅ |
| Armington | ✅ | ✅ | ✅ | — | ✅ |
| CET on exports | ✅ † solved, with a world export price; the default fixes shares | ? | ✅ | — | ✅ |
| **Trade and transport margins block** | ✅ | ? | — | — | ✅ |
| **Negative value added handled** | ✅ two rules | ? | — | — | — |
| LES / Stone-Geary demand | ✅ calibrated from the Frisch parameter | ? | — | — | ✅ |
| **Own-account activities** | ✅ detected and scalable | — | — | — | — |
| Creditable / non-creditable split of the product tax | ✅ | — | — | — | ✅ |
| **Direct taxes, transfers and remittances** | ✅ † | ? | ? | ? | ✅ |
| **Foreign-owned capital and FDI** | ✅ † profit remittances in the current account | ? | ? | ? | ✅ |
| **Imperfect factor mobility** | ✅ † CET or logit, pools of origin | ? | ? | ? | ✅ GTAP's sluggish endowments |
| **Segmented labour and informality** | ✅ † dual market, Harris–Todaro, a transformation between segments | ? | ? | ? | the modeller's convention |
| **Microsimulation, poverty and inequality** | ✅ † top-down and behavioural (BRR) | ? | ? | ? | separate tools |
| **A SAM built from supply and use tables** | ✅ † with the provenance of every block; the IBGE's tables | ? | ? | ? | separate tools |
| **Alternative closures** | ✅ 5 combinations, each with its own adjusting variable | ? | — | — | ✅ |
| **Walras as a check, not an equation** | ✅ the numeraire market is left out of the system | ? | — | — | the modeller's convention |
| **Base-year reproduction test** | ✅ automatic | 📄 "compares base and simulation" | — | — | the modeller's convention |
| **Zero-degree homogeneity test** | ✅ automatic | — | — | — | the modeller's convention |
| **Recursive dynamics** | ✅ several capital stocks, capital by sector, putty-clay vintages, allocation by return, a per-year economy, a baseline calibrated to GDP † | — | — the loop in `execute.py` is a damped fixed point within one period, with no time index | — | ✅ |
| **Welfare: EV and CV** | ✅ | — | — | — | the modeller's convention |
| Group aggregation immune to composition | ✅ | — | — | — | — |
| **Independent replica for validation** | ✅ † JuMP/Ipopt, 50 cases | ? | ? | ? | n/a |
| Licence | MIT | ? | CC0-1.0 | ? | proprietary / commercial |
| State declared by the authors | — | — | ✅ "very preliminary… please do not cite" | textbook | mature |

## What the table says

In the new rows (†), the other packages' columns are "?" because they were not checked again.

The three Python CGE packages found implement the textbook model of Hosoe, Gasawa and Hashimoto
(2010) or the GAMS library's `splcge`, with two to four sectors. `PSLmodels/CGE` says of itself,
in its README, that it is "very preliminary" and asks not to be cited. As far as was found, there
is no Python CGE package at applied scale.

That does not make `cgelab` a contribution by itself. It fills a gap that exists because applied
CGE is done in GAMS or GEMPACK, and for good reasons: dedicated solvers, a community, and
databases such as GTAP. The defensible claim is narrow: for an applied-scale CGE in Python,
without a commercial licence and with the accounting checks built in, no alternative was found.

## A caution

`cgelab` was validated in four ways:

- against an independent JuMP/Ipopt replica of the same equilibrium, in 50 cases;
- against a line-by-line JuMP transcription of the GAMS library programs `stdcge` and `lrgcge`
  (Hosoe, Gasawa and Hashimoto, 2004): the tariff abolition agrees in all 49 displayed variables,
  to 2.4e-12;
- against the research pipeline it came from, at version 0.1.0;
- by base-year reproduction, homogeneity and Walras' law to machine precision, on the applied
  SAM it was written for, which is not public yet.

That is evidence that what it does, it does right. It is **not** evidence that it covers what
GAMS covers. The multi-region nest has neither factor migration nor bilateral trade, there is no
complementarity, the package does not estimate elasticities, rational expectations go through
the bridge to `dsgelab`, and it does not have the maturity of a platform with thirty years of
use.

---

## Gaps, measured against platforms in other languages

The table above compares with Python packages. This section widens the yardstick to
GAMS/MPSGE, GEMPACK, GTAP and the IFPRI and PEP standard models, which are not installed here:
the cells come from the literature and the documentation, and are **📄**. Since 11 October 2026
the goal is to cover the whole list by 1.0; each item's version is the one in
[`ROADMAP.md`](../ROADMAP.md).

| Item | Who has it | State in `cgelab` | Version |
|---|---|---|---|
| SAM institutions: direct taxes, transfers, remittances, factor income to government and abroad | 📄 IFPRI, PEP | **done** in 0.2.0 | 0.2 |
| Labour-market closures | 📄 IFPRI, ORANI | **done**: fixed real wage, wage curve, elastic supply | 0.2 |
| Tariff and indirect tax separate from the product tax | 📄 Hosoe (`stdcge`), IFPRI | **done**: the tariff inside the import price and the indirect tax on all of output, as in `stdcge`; the tariff abolition of `stdcge` and `lrgcge` reproduces the transcription of the GAMS programs to 2.4e-12 | 0.2 |
| JuMP replica converging cold at applied scale | — | **done**, on an applied SAM that is not public yet | 0.2 |
| Export demand and import supply with finite elasticities | 📄 Hosoe (`lrgcge`) | **done**: the equations of `lrgcge`; its tariff abolition reproduces the transcription | 0.2 |
| Foreign saving fixed in foreign currency | 📄 Hosoe, IFPRI | **done** | 0.2 |
| Decomposition of results by shock (subtotals) | 📄 GEMPACK | **done**: Harrison, Horridge and Pearson, integrated along the straight path | 0.3 |
| Systematic sensitivity and robustness | 📄 GEMPACK | **done**: Stroud degree-3 quadrature and Monte Carlo, recalibrating at every point; Sobol indices, closure grids, results in money, balanced SAM draws with joint elasticity draws, and a one-call report with sourced ranges and optional figures | 0.3, 0.6 |
| Minimum-distance matching to external targets | 📄 estimation and validation practice | **done**: bounded ModelSpec fields fitted by covariance-weighted distance, with pseudo-inverses and optimiser evidence; exact against closed-form weighted least squares | 0.7 |
| Treated-minus-control regional contrasts | 📄 event-study practice | **done**: treated change less weighted control change, with built-in and custom regional outcomes and hand-checked treated-only and common shocks | 0.7 |
| Accounts and reports: GDP by expenditure and income, price indices, terms of trade | 📄 GEMPACK, GTAP | **done**: GDP from both sides, Laspeyres, Paasche and Fisher indices, terms of trade, the government account | 0.3 |
| Welfare decomposition (Huff and Hertel) | 📄 GTAP | **done** for real absorption, the identity closing below 1e-9; the comparison with GTAP waits on the GTAP database | 0.3 |
| SAM multipliers and input–output | 📄 Pyatt and Round; Defourny and Thorbecke | **done**: accounting, fixed-price, Leontief, linkages (Rasmussen and Ghosh), structural path analysis | 0.3 |
| Energy nests (KLEM), emissions, carbon tax | 📄 ENVISAGE, EPPA, GTAP-E | **done**: a value-added-energy or capital-energy nest (GTAP-E's arrangement) with interfuel substitution, emissions by source, the carbon price inside each activity's fuel price, revenue-neutral recycling, physical units | 0.4 |
| Government closures | 📄 IFPRI | **done**: fixed saving rate (Hosoe's), a saving target and fixed revenue (revenue-neutral replacement), met by the direct tax, the product tax or spending | 0.4 |
| Bridge to `dsgelab` | — | **done**: a Ramsey CGE with investment set by an Euler equation; steady state, first order, perfect foresight and announced policy; the growth model in a one-good economy | 0.4 |
| Imperfect factor mobility; proximity matrix | 📄 GTAP, ORANI | **done**: a CET (GTAP's sluggish endowments) or a logit across activities, pools of origin as a proximity matrix; exact against separate sector-specific factors at ω = 0 and against the JuMP replica | 0.6 |
| Segmentation and informality | 📄 IFPRI; Harris and Todaro | **done**: labour accounts that share one population; one labour market, a dual market and Harris–Todaro, or a CET or logit between segments with a queue in each rigid one; income that follows each household's workers; exact against two factors under a fixed real wage at ω = 0 and against the JuMP replica | 0.6 |
| Link to microsimulation; Gini and FGT | 📄 common practice | **done**: survey households scaled to the SAM by account and source, their own price indices; Gini, FGT, generalised entropy and Atkinson with exact decompositions; the weights recalibrated by raking to the model's employment by segment and unemployment. A behavioural microsimulation (BRR): earnings equations and a multinomial logit of occupational choice, linked by intercepts, with the conditional draws or their closed-form expectation. readers for the IBGE's PNAD Contínua and POF | 0.6 |
| Dynamic and historical calibration; sectoral capital and vintages | 📄 MONASH, GDyn, LINKAGE | **done**: the productivity that hits a path of real GDP, by year or by activity, and a limited historical closure with joint technology and relative-preference shifters, with policy runs at the calibrated path; capital by sector with investment shared by rents; putty-clay vintages against the JuMP replica | 0.6 |
| SAM construction and disaggregation | 📄 IFPRI, PEP | **done**: construction from supply and use tables with the provenance of every block, and disaggregation of households and of labour by household surveys | 0.6 |
| Several regions | 📄 GTAP, TERM, ORANI-G | origin nest, institution maps, fixed-real investment and CET/logit factor migration done; other closure rules and bilateral trade missing | 0.7 |
| GTAP data | 📄 GTAPAgg, FlexAgg | HAR reading and writing done, exact against harpy3 on GTAP 10A; aggregation and a one-region SAM missing | 0.7 |
| Complementarity (MCP); alternative solvers | 📄 GAMS/MPSGE with PATH; pycge (Pyomo) | missing | 0.8 |
| Generic closure swap; cross entropy; intertemporal CGE; imperfect competition; land use; export to GAMS and GEMPACK | 📄 GEMPACK, MPSGE, GTAP-AEZ | missing | 0.9 |

## Sources

- [pycge](https://github.com/juanfung/pycge), [PSLmodels/CGE](https://github.com/PSLmodels/CGE),
  [python-cge](https://github.com/DavoudTaghawiNejad/python-cge): cloned and read
- [World Bank on CGE](https://www.worldbank.org/en/programs/economicpolicy-macro-modeling/cge)
- [GEMPACK](https://en.wikipedia.org/wiki/Gempack)
