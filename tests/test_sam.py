"""The SAM container, its balance check and the balancing routine."""

import numpy as np
import pandas as pd
import pytest

from cgelab import SAM, BalanceError


def _long(entries):
    return SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]))


def test_balanced_sam_passes():
    from cgelab.examples import toy
    sam, _ = toy.build()
    assert sam.check_balance()["ok"]
    assert sam.imbalance().abs().max() < 1e-12


def test_unbalanced_sam_is_refused_with_the_offenders_named():
    sam = _long([("a", "b", 10.0), ("b", "a", 7.0)])
    with pytest.raises(BalanceError, match="out of balance"):
        sam.check_balance(abs_tol=0.1)
    rep = sam.check_balance(abs_tol=0.1, raise_on_fail=False)
    assert set(rep["offenders"].index) == {"a", "b"}


def test_sign_preserving_ras_handles_negative_entries():
    """Standard multiplicative RAS breaks on a negative cell; real SAMs have them."""
    sam = _long([("a", "b", 10.0), ("b", "a", 7.0),
                 ("a", "c", -2.0), ("c", "a", 1.0),
                 ("b", "c", 3.0), ("c", "b", 4.0)])
    out = sam.balance(abs_tol=1e-9)
    assert out.check_balance(abs_tol=1e-6)["ok"]
    signs_in = {(r, c): np.sign(v) for r, c, v in sam.frame.itertuples(index=False)}
    signs_out = {(r, c): np.sign(v) for r, c, v in out.frame.itertuples(index=False)}
    assert signs_in == signs_out


def test_square_and_long_forms_agree():
    from cgelab.examples import toy
    sam, _ = toy.build()
    square = sam.matrix()
    again = SAM.from_square(square)
    assert again.imbalance().abs().max() < 1e-12
    assert again.total == pytest.approx(sam.total)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_nonfinite_flows_fail_the_balance_gate_and_ras(value):
    sam = SAM(pd.DataFrame([("a", "b", value), ("b", "a", 3.0)],
                            columns=["row", "col", "value"]))
    with pytest.raises(BalanceError, match="non-finite.*a.*b"):
        sam.check_balance()
    report = sam.check_balance(raise_on_fail=False)
    assert not report["ok"]
    assert report["worst"] == np.inf
    assert set(report["offenders"].index) == {"a", "b"}
    with pytest.raises(BalanceError, match="non-finite"):
        sam.balance()


def test_reading_keeps_missing_flows_for_the_balance_gate():
    long = pd.DataFrame([("a", "b", 1.0), ("a", "b", np.nan), ("b", "a", 3.0)],
                        columns=["row", "col", "value"])
    sam = SAM.from_long(long)
    assert np.isnan(sam.frame.set_index(["row", "col"]).value[("a", "b")])
    with pytest.raises(BalanceError, match="non-finite"):
        sam.check_balance()


def test_an_empty_square_cell_is_a_zero_flow_and_an_infinite_one_is_kept():
    square = pd.DataFrame([[np.nan, 2.0], [2.0, np.nan]], index=["a", "b"],
                          columns=["a", "b"])
    assert SAM.from_square(square).check_balance()["ok"]
    square.loc["a", "a"] = np.inf
    with pytest.raises(BalanceError, match="non-finite"):
        SAM.from_square(square).check_balance()
