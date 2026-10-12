"""SAM multipliers and input-output analysis, against closed forms and a textbook table."""

import itertools

import numpy as np
import pandas as pd
import pytest

import cgelab as ck
from cgelab.examples import toy

# Miller and Blair (2009), Input-Output Analysis, 2nd ed., section 2.3: two sectors,
# interindustry flows Z, gross output x, final demand f. Written as a SAM: activity
# j buys Z[:, j] from commodities and value added from one factor; activity i makes
# commodity i one-for-one; one household owns the factor and buys final demand.
Z = np.array([[150.0, 500.0], [200.0, 100.0]])
X = np.array([1000.0, 2000.0])
F = np.array([350.0, 1700.0])


def miller_blair_sam():
    va = X - Z.sum(axis=0)
    rows = []
    for i, j in itertools.product(range(2), range(2)):
        rows.append((f"c_{i + 1}", f"a_{j + 1}", Z[i, j]))
    for j in range(2):
        rows += [("f_va", f"a_{j + 1}", va[j]), (f"a_{j + 1}", f"c_{j + 1}", X[j]),
                 (f"c_{j + 1}", "hh", F[j])]
    rows.append(("hh", "f_va", va.sum()))
    sam = ck.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    sam.check_balance()
    return sam


SPEC = ck.ModelSpec(activities=["a_1", "a_2"], commodities=["c_1", "c_2"], households=["hh"],
                    factors=("f_va",), government="hh", investment="hh", rest_of_world="hh")


def test_the_leontief_inverse_is_miller_and_blairs():
    m = ck.leontief_multipliers(miller_blair_sam(), SPEC)
    L = m.M.loc[["a_1", "a_2"], ["a_1", "a_2"]].to_numpy()
    A = Z / X[None, :]
    np.testing.assert_allclose(L, np.linalg.inv(np.eye(2) - A), rtol=1e-12)
    # The book prints L to three decimals.
    np.testing.assert_allclose(L, [[1.254, 0.330], [0.264, 1.122]], atol=5e-4)
    # And it returns the output vector from final demand.
    np.testing.assert_allclose(L @ F, X, rtol=1e-12)


def test_impact_of_an_injection():
    m = ck.leontief_multipliers(miller_blair_sam(), SPEC)
    out = m.impact({"c_1": 100.0})
    np.testing.assert_allclose(out[["a_1", "a_2"]], 100 * np.linalg.inv(
        np.eye(2) - Z / X[None, :])[:, 0], rtol=1e-12)
    with pytest.raises(KeyError, match="endogenous"):
        m.impact({"hh": 1.0})


def test_ghosh_forward_linkages_and_the_similarity_identity():
    """G = diag(x)^-1 L diag(x): the Ghosh and Leontief inverses are similar."""
    m = ck.leontief_multipliers(miller_blair_sam(), SPEC)
    acts = ["a_1", "a_2"]
    link = ck.linkages(m, acts)
    L = np.linalg.inv(np.eye(2) - Z / X[None, :])
    G = np.linalg.inv(np.eye(2) - Z / X[:, None])
    np.testing.assert_allclose(G, np.diag(1 / X) @ L @ np.diag(X), rtol=1e-12)
    # Backward on the activity block is the Leontief column sum, forward the Ghosh row sum.
    np.testing.assert_allclose(link.backward, L.sum(axis=0), rtol=1e-12)
    # The activity block of the SAM's Ghosh inverse is the Ghosh inverse of Z: each
    # activity passes through its commodity, which sells its whole output on.
    np.testing.assert_allclose(link.forward, G.sum(axis=1), rtol=1e-12)
    assert link.backward_index.mean() == pytest.approx(1.0)
    assert link.forward_index.mean() == pytest.approx(1.0)


def test_accounting_multipliers_on_the_toy_close_through_households():
    sam, _ = toy.build()
    m = ck.accounting_multipliers(sam, spec=toy.spec())
    assert {"gov", "s_i", "row"}.isdisjoint(m.accounts)
    assert {"hh_1", "hh_2", "f_lab", "a_agr", "c_ind"} <= set(m.accounts)
    # Every endogenous account leaks something, or M would not exist.
    assert (m.leakage > -1e-12).all()
    # Closing households in raises every multiplier over the open Leontief model.
    leo = ck.leontief_multipliers(sam, toy.spec())
    open_ = leo.M.loc[leo.accounts, leo.accounts]
    closed = m.M.loc[leo.accounts, leo.accounts]
    assert (closed.to_numpy() >= open_.to_numpy() - 1e-12).all()
    # M = I + A M, the defining identity.
    np.testing.assert_allclose(m.M, np.eye(len(m.accounts)) + m.A @ m.M, atol=1e-12)


def test_fixed_price_with_unit_elasticities_is_the_accounting_multiplier():
    sam, econ = toy.build()
    acc = ck.accounting_multipliers(sam, spec=toy.spec())
    fp = ck.fixed_price_multipliers(sam, spec=toy.spec(), households=econ.hhs,
                                    income_elasticities={c: 1.0 for c in econ.coms})
    np.testing.assert_allclose(fp.M, acc.M, rtol=1e-12)


def test_fixed_price_marginal_propensities():
    sam, econ = toy.build()
    with pytest.raises(ValueError, match="Engel"):
        ck.fixed_price_multipliers(sam, spec=toy.spec(), households=econ.hhs,
                                   income_elasticities={"c_agr": 0.5, "c_ind": 1.2})
    fp = ck.fixed_price_multipliers(sam, spec=toy.spec(), households=econ.hhs,
                                    income_elasticities={"c_agr": 0.5, "c_ind": 1.2},
                                    engel="rescale")
    acc = ck.accounting_multipliers(sam, spec=toy.spec())
    for h in econ.hhs:
        # same total propensity to consume, shifted toward industry
        assert fp.A.loc[econ.coms, h].sum() == pytest.approx(acc.A.loc[econ.coms, h].sum())
        assert fp.A.loc["c_ind", h] > acc.A.loc["c_ind", h]
        assert fp.A.loc["c_agr", h] < acc.A.loc["c_agr", h]
    # Average propensities everywhere else.
    others = [a for a in acc.accounts if a not in econ.hhs]
    np.testing.assert_allclose(fp.A[others], acc.A[others])


def _all_paths_sum(mult, origin, destination):
    paths = ck.structural_paths(mult, origin, destination,
                                max_length=len(mult.accounts), threshold=0.0)
    return paths, float(paths.total.sum())


def test_structural_paths_add_up_to_the_global_influence():
    """Lantner's theorem: global influence = sum of total influences of all elementary paths."""
    sam, _ = toy.build()
    m = ck.accounting_multipliers(sam, spec=toy.spec())
    for origin, destination in (("c_agr", "hh_2"), ("hh_1", "a_ind"), ("f_cap", "c_agr")):
        paths, total = _all_paths_sum(m, origin, destination)
        assert total == pytest.approx(m.M.loc[destination, origin], rel=1e-10)
        assert paths.share.sum() == pytest.approx(1.0, rel=1e-10)


def test_structural_path_influences():
    sam, _ = toy.build()
    m = ck.accounting_multipliers(sam, spec=toy.spec())
    paths = ck.structural_paths(m, "c_agr", "hh_1", threshold=1e-3)
    first = paths.iloc[0]
    assert first.path[0] == "c_agr" and first.path[-1] == "hh_1"
    # Direct influence is the product of propensities along the arcs.
    arcs = zip(first.path[:-1], first.path[1:])
    assert first.direct == pytest.approx(np.prod([m.A.loc[b, a] for a, b in arcs]))
    assert first.path_multiplier >= 1.0     # circuits off the path only add
    assert (paths.total.abs().diff().dropna() <= 1e-15).all()


def test_misuse_is_refused():
    sam, _ = toy.build()
    with pytest.raises(ValueError, match="exogenous"):
        ck.accounting_multipliers(sam)
    with pytest.raises(ValueError, match="not both"):
        ck.accounting_multipliers(sam, exogenous=["gov"], endogenous=["hh_1"])
    m = ck.accounting_multipliers(sam, spec=toy.spec())
    with pytest.raises(ValueError, match="different"):
        ck.structural_paths(m, "c_agr", "c_agr")
