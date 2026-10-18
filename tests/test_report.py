"""The diagnostic report itself."""

import cgelab as cge
from cgelab.examples import toy


def test_report_renders_and_names_the_numeraire():
    _, econ = toy.build()
    text = cge.report(econ)
    assert "base-year reproduction : PASS" in text
    assert "price homogeneity      : PASS" in text
    assert econ.spec.numeraire_factor in text


def test_report_names_the_activities_put_on_fixed_proportions():
    import pandas as pd
    drop = {("f_cap", "a_agr", 25.0), ("f_lab", "a_agr", 40.0),
            ("hh_1", "f_lab", 70.0), ("hh_1", "f_cap", 45.0)}
    entries = [e for e in toy._BASE if tuple(e) not in drop]
    entries += [("f_lab", "a_agr", 70.0), ("f_cap", "a_agr", -5.0),
                ("hh_1", "f_lab", 100.0), ("hh_1", "f_cap", 15.0)]
    sam = cge.SAM.from_long(pd.DataFrame(entries, columns=["row", "col", "value"]))
    econ = cge.calibrate(sam, toy.spec())
    assert "a_agr" in cge.report(econ)


def test_walras_check_reads_the_unsolved_market():
    _, econ = toy.build()
    res = cge.solve_period(econ, scenario=cge.Scenario(product_tax=econ.ptax * 1.1))
    chk = cge.walras_check(res)
    assert chk["ok"]
    assert chk["factor"] == econ.spec.numeraire_factor
