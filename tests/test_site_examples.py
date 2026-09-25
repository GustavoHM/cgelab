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
    for page in checker.PAGES[lang]:
        assert list(checker.blocks(page)), f"{page.name} ({lang}) has no executable examples"
        assert checker.run(page, verbose=True) == 0


def test_the_language_versions_have_the_same_examples():
    """A translation that drops or adds an example has drifted from its pair."""
    en, pt = checker.PAGES["en"], checker.PAGES["pt-br"]
    assert len(en) == len(pt)
    for a, b in zip(en, pt):
        counts = (len(list(checker.blocks(a))), len(list(checker.blocks(b))))
        assert counts[0] == counts[1], (a.name, b.name, counts)
