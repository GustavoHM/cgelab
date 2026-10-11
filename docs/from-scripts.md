# Porting an existing CGE pipeline

A research CGE pipeline usually grows as a chain of numbered scripts loaded by
`importlib` — `00_config.py`, `06_calibrate.py`, `06_solve.py`, `07_scenarios.py`
— with the model's assumptions spread between a large config module and branches
inside the solver. This note is about moving the model core to `cgelab` and what
that changes.

## What stays where it is

Data collection, source-specific cleaning, SAM construction and the paper's
tables and figures stay in the pipeline. They are about *your* data and *your*
paper and there is nothing general to extract. What moves is the part between the
balanced SAM and the results: calibration, the period solve, closures, scenarios,
dynamics and welfare.

## What the move buys

**Assumptions become objects.** An elasticity in a config module and an
elasticity in a `ModelSpec` are the same number, but only the second travels with
the results, can be swept, and can be printed next to the output that depends on
it. The same for closures, which are otherwise branches inside a solver function.

**The three gates come for free.** Base-year reproduction, homogeneity and Walras
run on any SAM in one call, and a per-year Walras check runs on any path.

**Scenarios become comparable.** A counterfactual written as keyword arguments
scattered over a solver call cannot be logged or diffed; the difference between
two runs then exists only in the code that produced them.

---

## Step 1 — the SAM

```python
sam = ck.SAM.read("data/processed/sam_2021_final.parquet")
sam.check_balance()
```

If your builder already balances, this is a no-op that documents the fact. If it
does not, read the offenders before reaching for `balance()`.

## Step 2 — the specification

Move every substitution parameter and functional-form switch out of the config
module and into a `ModelSpec`. Keep the config module for paths, vintages, source
metadata and the policy calendar — things that are about the data rather than
about the model.

```python
spec = ck.ModelSpec(
    factors=("f_lab", "f_cap"),
    sigma_armington=cfg.SIGMA_ARMINGTON,
    sigma_va=cfg.SIGMA_PRODUCTION,
    negative_value_added=cfg.NEGATIVE_VA_TREATMENT,
    export_rule="fixed_shares",
    numeraire_factor="f_lab",
)
econ = ck.calibrate(sam, spec, population=cfg.HOUSEHOLDS_BY_ACCOUNT)
```

Then, before anything else:

```python
assert ck.base_year_reproduction(econ)["ok"]
assert ck.homogeneity_test(econ)["ok"]
```

If either fails, the port is not done and no later comparison means anything.

## Step 3 — the period solve

Compare the new solver against the old one at the base year and at one shock,
variable by variable, before going further:

```python
old = legacy.solve_period(legacy_econ)          # your existing script
new = ck.solve_period(econ)
assert np.max(np.abs(new.X / old["X"] - 1)) < 1e-8
assert np.max(np.abs(new.pq / old["pq"] - 1)) < 1e-8
```

A base-year agreement is necessary and not sufficient — two different models
routinely agree at the base year and differ under a shock — so repeat it on a
counterfactual.

## Step 4 — scenarios

Each of the old solver's keyword arguments maps to a `Scenario` field:

| Old pattern | New |
|---|---|
| `level=` scaling one activity | `activity_scale={"a_x": level}` |
| `wedge=` household×commodity matrix | `consumer_wedge=` |
| `uplift_to="capital" / "lumpsum" / λ` | `wedge_revenue_to=` |
| `wedge_gov=` for a refund | `government_wedge=` |
| `absorb=("capital", amount)` | `transfers=-amount * shares` |
| `absorb=("government", amount)` | `government_absorbs=amount` |
| `fac_supply=`, `hh_facshare=` | `factor_supply=`, `factor_ownership=` |
| `ptax=`, `ptax_firm=` | `product_tax=`, `product_tax_firm=` |
| `numeraire=` (homogeneity test) | `Closure(numeraire=...)`, or just `homogeneity_test` |

Anything genuinely specific to your policy — a rule that computes a transfer
from your own accounts, an eligibility rule by group — stays in
your own module. Its **output** is a wedge matrix, a transfer vector or a tax
vector, and that is what crosses into `cgelab`. Keeping the policy logic outside
is the right boundary: it is where your paper's contribution is, and it is the
part no general package should own.

## Step 5 — the path

```python
path = ck.run_path(econ, closure, years=range(2021, 2046),
                   scenario=lambda year, t: my_policy_scenario(year, t),
                   growth=ck.Growth(labour=cfg.LABOUR_GROWTH, tfp=cfg.TFP_GROWTH,
                                    depreciation=cfg.DEPRECIATION_RATE,
                                    discount=cfg.DISCOUNT_RATE))
assert path.worst_walras < 1e-9
```

The assertion is the point of the step. A residual that grows along the path is a
coefficient the price block saw and the quantity block did not, and it is
invisible in any single period.

## Step 6 — welfare

```python
ev = ck.equivalent_variation(ref_path, alt_path)
grad = ck.ev_by_group(ev, decile_of_account)
```

If your existing code reports a per-account percentage, check it against the
group measure before trusting it. Where households migrate between accounts, the
two differ by the migration rate, and the per-account version reports composition
as welfare. This is not hypothetical: it is the failure the group aggregation
exists to prevent, and the test suite reproduces it deliberately.

---

## A caution about `importlib`-loaded scripts

Modules whose names begin with a digit cannot be imported normally, which is why
pipelines like this end up loading each other through
`importlib.util.spec_from_file_location`. That works, and it has two costs worth
naming: the load order becomes an implicit dependency graph that nothing
validates, and the modules cannot be unit-tested or installed. Moving the model
core into a package removes both for the part that is general; the numbered
scripts remain the right shape for the data pipeline, where execution order is
genuinely the thing being expressed.

## Order of work

1. SAM reads and balances.
2. `ModelSpec` written; `calibrate` succeeds.
3. **Base-year reproduction and homogeneity pass.** Do not proceed otherwise.
4. Period solve agrees with the old one at the base year and under one shock.
5. Scenarios translated; the wedge-revenue identity holds.
6. Path runs with `worst_walras` at machine precision.
7. Welfare reproduced, and the group measure compared with the per-account one.

Steps 3 and 6 are the ones that catch real errors. They are cheap and they are
the reason to do the port at all.
