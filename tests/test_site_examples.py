"""Every executable example on the site runs, in page order."""

import importlib.util
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "tools" / "check_site_examples.py"


def test_every_site_example_runs():
    spec = importlib.util.spec_from_file_location("check_site_examples", TOOL)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    assert list(checker.blocks()), "the site has no executable examples"
    assert checker.run(verbose=True) == 0
