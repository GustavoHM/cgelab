# Contributing to cgelab

Thank you for considering it. The package has one guarantee: every number it
produces was checked against something it did not compute itself. A
contribution keeps that guarantee or it does not go in.

## What counts as done

A feature is done when it meets the six conditions in [`ROADMAP.md`](ROADMAP.md):

1. Implemented, with a public API in `__all__`.
2. **Validated against an independent reference**: another implementation (GAMS,
   GEMPACK, JuMP, a published replication package), a closed form or a published
   result, in a script anyone can run.
3. Tested in the package's suite without external tools; what needs Julia or GAMS
   lives in `validation/`.
4. Documented in [`docs/guide.md`](docs/guide.md) and [`docs/api.md`](docs/api.md),
   with a runnable example on the site.
5. The matching row of [`docs/comparison.md`](docs/comparison.md) updated, and of its
   Portuguese version, [`docs/comparison.pt-br.md`](docs/comparison.pt-br.md).
6. A [`CHANGELOG.md`](CHANGELOG.md) entry.

A check the solver could pass by construction is not a validation. Prefer an
identity that would fail if the algebra were wrong (Walras' law on a market the
solver does not clear, GDP from both sides, a nest that must collapse to another
at equal elasticities) over a comparison with the package's own earlier output.

## Development

```
git clone https://github.com/GustavoHM/cgelab && cd cgelab
python -m venv .venv && .venv/bin/pip install -e ".[test,parquet]"
.venv/bin/python -m pytest
.venv/bin/python tools/check_site_examples.py      # every example on the site runs
.venv/bin/python tools/benchmark_solver.py         # time, memory and convergence by model size
```

The suite tests `src/` by default. `CGELAB_TEST_INSTALLED=1` tests the installed
package instead, which is how CI checks the wheel.

Checks that need Julia (`validation/roundtrip_jump.py`, `validation/replicate_hosoe.py`,
`CGELAB_RUN_JULIA=1 pytest tests/test_jump_roundtrip.py`) run in the `validation`
workflow, on demand and weekly. Instantiate the environment once:
`cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()'`.

Licensed data (the GTAP database, a SAM you may not publish) never enters the
repository. `.gitignore` refuses `.har`, `.prm`, `.pkg` and `.sl4` files; tests that
need such data skip unless an environment variable points at it
(`CGELAB_GTAP_DIR`).

## Style

Code, comments and docstrings in English, except the JuMP template and the
validation scripts, which are in Portuguese. Comments say why, not what: the
accounting decision, the error it prevents. An error message names the account,
the field or the closure that caused it and says what to do instead. A value the
model would silently ignore is refused.

## Releasing

Versions follow semantic versioning; before 1.0 a minor version may add to the
API and a patch only fixes.

One-time setup, by a maintainer, outside the repository:

1. On [TestPyPI](https://test.pypi.org) and [PyPI](https://pypi.org), add this
   repository as a trusted publisher for the project `cgelab`: workflow
   `release.yml`, environments `testpypi` and `pypi`.
2. In the repository settings, create the environments `testpypi` and `pypi`
   (adding a required reviewer to `pypi` keeps a person between a tag and a release).
3. Connect the repository to [Zenodo](https://zenodo.org) so each GitHub release
   gets a DOI.

For each release:

1. Set the version in `pyproject.toml`, `src/cgelab/__init__.py` and
   `CITATION.cff` (with `date-released`), and on the site: the version pill, the
   BibTeX, APA and ABNT entries, the example sentence and the footer of
   `site/index.html` and `site/pt-br/index.html`, and the footers of both
   `replications.html`. Give the new section at the top of `CHANGELOG.md` its
   version and date (start one headed `Unreleased` after each release).
2. Run the suite, the site examples and the `validation` workflow.
3. Release dsgelab first: the `dsge` extra needs the matching dsgelab on PyPI.
   Then push a tag `vX.Y.Z`. The `release` workflow refuses a tag that differs
   from the version in `pyproject.toml`, the package or `CITATION.cff`, and stops
   before any upload if that dsgelab is not on PyPI. It builds the wheel and the
   sdist, checks them with `twine`, publishes to TestPyPI, installs what TestPyPI
   serves and runs the suite against it, and only then publishes to PyPI.
4. Publish a GitHub release from the tag; Zenodo archives it and assigns the DOI.
5. Add the Zenodo DOI to `CITATION.cff` and the README.
