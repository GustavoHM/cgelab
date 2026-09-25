"""Every executable example on the site runs, in page order, in every language version."""

import importlib.util
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_site_examples.py"
spec = importlib.util.spec_from_file_location("check_site_examples", TOOL)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


@pytest.mark.parametrize("lang", sorted(checker.PAGES))
def test_every_site_example_runs(lang):
    page = checker.PAGES[lang]
    assert list(checker.blocks(page)), f"the {lang} page has no executable examples"
    assert checker.run(page, verbose=True) == 0


def test_the_language_versions_have_the_same_examples():
    """A translation that drops or adds an example has drifted from the other page."""
    counts = {lang: len(list(checker.blocks(p))) for lang, p in checker.PAGES.items()}
    assert len(set(counts.values())) == 1, counts
