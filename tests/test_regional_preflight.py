"""One report of everything the regional slice refuses, instead of one error at a time."""

import pandas as pd
import pytest

import cgelab as cge
from cgelab.examples import regions
from cgelab.regional import RegionalSliceError, regional_preflight

#: Three refusals at once, each balanced on its own: r1's activity makes 5 of
#: r2's good (r1's household buys it, paid by r1's labour); a new activity r1:b
#: sells 3 straight to r1's household; and r1's good alone pays a product tax of
#: 2, handed back to r1's household as a transfer.
PROBLEMS = [
    ("r1:a", "r2:c", 5.0), ("r2:a", "r2:c", -5.0),
    ("r1:f_lab", "r1:a", 5.0), ("r2:f_lab", "r2:a", -5.0),
    ("r1:hh", "r1:f_lab", 5.0), ("r2:hh", "r2:f_lab", -5.0),
    ("r2:c", "r1:hh", 5.0), ("r2:c", "r2:hh", -5.0),
    ("r1:b", "r1:hh", 3.0), ("r1:f_lab", "r1:b", 3.0), ("r1:hh", "r1:f_lab", 3.0),
    ("gov", "r1:c", 2.0), ("r1:c", "r1:hh", 2.0), ("r1:hh", "gov", 2.0),
]


def _sam(rows):
    return cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))


def test_every_issue_is_listed_at_once():
    sam = _sam(regions.ENTRIES + PROBLEMS)
    sam.check_balance(abs_tol=1e-9)
    spec = regions.spec(activities=["r1:a", "r1:b", "r2:a"])
    table = regional_preflight(sam, spec)
    assert list(table.columns) == ["issue", "account", "detail"]
    assert set(table.issue) == {"make across regions", "own-account activity",
                                "product tax across regions"}
    row = table.set_index("issue").loc["make across regions"]
    assert row.account == "r1:a" and "'r2:c'" in row.detail
    assert table.set_index("issue").loc["own-account activity"].account == "r1:b"
    assert table.set_index("issue").loc["product tax across regions"].account == "c"

    with pytest.raises(RegionalSliceError, match="3 issue") as info:
        cge.calibrate(sam, spec)
    assert isinstance(info.value, ValueError)
    assert info.value.issues == list(table.itertuples(index=False, name=None))


def test_a_sam_inside_the_slice_has_no_issues():
    sam, _ = regions.build()
    assert regional_preflight(sam, regions.spec()).empty
