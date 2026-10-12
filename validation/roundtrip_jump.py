"""roundtrip_jump.py — exportar do cgelab para JuMP, resolver no Julia, e comparar de volta.

Por que este confronto é forte. O cgelab resolve o equilíbrio por ANINHAMENTO: um Newton sobre
poucas incógnitas e, dentro dele, um ponto fixo sobre o preço de entrega, duas somas em forma
fechada — a série infinita da reciclagem do gravame e a relação afim entre poupança e arrecadação —
e uma inversão de Leontief. Cada uma é uma dedução do autor. Em JuMP nada disso é deduzido: cada
bloco é uma restrição e o Ipopt encontra o ponto fixo. Um erro na álgebra do Python não pode,
portanto, ser reproduzido do outro lado por construção.

Roda 19 casos sobre as economias de brinquedo do pacote: sem margens, com bloco de margens, com
atividade de conta própria, com CET, com curvas de demanda de exportação e de oferta de
importação (país grande, como o `lrgcge`), e com instituições (impostos diretos, transferências, remessas,
capital do governo e do exterior, IED), em cenário-base e com choques, em vários fechamentos.

    cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
    python validation/roundtrip_jump.py
"""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import cgelab as ck
from cgelab.examples import toy

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
CAMPOS = ("w", "er", "pa", "pd", "pbase", "Q", "X", "hh_income", "hh_com_spend",
          "savings", "uplift_rev", "exports", "world_pe", "world_pm")


def _rel(a, b, piso=1.0):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), piso)))


def _referencia(res) -> dict:
    return dict(w=list(res.factor_prices), er=res.exchange_rate, pa=list(res.pa),
                pd=list(res.pd), pbase=list(res.pbase), Q=list(res.Q), X=list(res.X),
                Aabs=list(res.absorption), hh_income=list(res.hh_income),
                hh_com_spend=list(res.hh_commodity_spend), savings=res.savings,
                uplift_rev=res.wedge_revenue, exports=list(res.exports),
                world_pe=list(res.world_export_price), world_pm=list(res.world_import_price))


def roda_julia(jl: Path, dados: Path, saida: Path, quente: Path | None = None) -> dict:
    julia = os.environ.get("JULIA", "julia")
    cmd = [julia, f"--project={AQUI}", str(jl), str(dados), str(saida)]
    if quente is not None:
        cmd.append(str(quente))
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-800:]}")
    linhas = [ln for ln in r.stdout.splitlines()
              if ln.startswith(("Ipopt:", "violação", "WALRAS", "variáveis"))]
    return json.loads(saida.read_text()), linhas


def caso(nome: str, econ, closure, scenario) -> dict:
    OUT.mkdir(exist_ok=True)
    jl = OUT / f"{nome}.jl"
    dados = OUT / f"{nome}_data.json"
    ck.to_jump(econ, closure, scenario, path=jl, data_path=dados)
    ref = _referencia(ck.solve_period(econ, closure, scenario))
    (OUT / f"{nome}_ref.json").write_text(json.dumps(ref))

    jlres, linhas = roda_julia(jl, dados, OUT / f"{nome}_out.json")
    print(f"\n{'=' * 78}\n{nome}\n{'=' * 78}")
    for ln in linhas:
        print(f"  {ln}")
    pior = 0.0
    for k in CAMPOS:
        if k not in jlres or k not in ref:
            continue
        d = _rel(jlres[k], ref[k])
        pior = max(pior, d)
        print(f"  {k:<16}{d:>14.2e}")
    print(f"  {'Walras cgelab':<16}{ck.solve_period(econ, closure, scenario).walras:>14.2e}"
          f"   JuMP {jlres['walras']:>10.2e}")
    return {"pior": pior, "viol": jlres["violacao_max"], "ok": pior < 1e-8}


def main() -> int:
    print("=" * 78)
    print("IDA E VOLTA — o cgelab escreve o modelo JuMP, o Ipopt resolve, comparamos de volta")
    print("=" * 78)
    casos = {}
    for etiqueta, build in (("simples", toy.build), ("margens", toy.build_with_margins),
                            ("conta_propria", toy.build_with_own_production)):
        _, econ = build()
        W = ck.consumer_wedge(econ, {"c_ind": 0.08})
        casos[f"{etiqueta}_base"] = caso(f"{etiqueta}_base", econ, ck.Closure(), None)
        casos[f"{etiqueta}_gravame"] = caso(
            f"{etiqueta}_gravame", econ, ck.Closure(),
            ck.Scenario(name="gravame", consumer_wedge=W, wedge_revenue_to="capital",
                        wedge_revenue_factor="f_cap"))
    # CET: a repartição do produto responde ao preço de exportação
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet", sigma_cet=2.0))
    casos["cet_base"] = caso("cet_base", econ, ck.Closure(), None)
    choque_cet = ck.Scenario(name="cet_choque", export_price=np.array([1.15, 0.9]),
                             product_tax=econ.ptax * 1.2)
    casos["cet_choque"] = caso("cet_choque", econ, ck.Closure(), choque_cet)
    casos["cet_fechamento_alternativo"] = caso(
        "cet_fechamento_alternativo", econ,
        ck.Closure(investment="fixed_real", foreign="fixed_rate"), choque_cet)

    # País grande: demanda de exportação e oferta de importação com elasticidade finita
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet", export_demand=2.0,
                                            import_supply=2.0))
    casos["pais_grande_choque"] = caso("pais_grande_choque", econ, ck.Closure(), choque_cet)
    casos["pais_grande_fechamento_alternativo"] = caso(
        "pais_grande_fechamento_alternativo", econ,
        ck.Closure(investment="fixed_real", foreign="fixed_rate"), choque_cet)
    _, econ = toy.build_with_margins(model_spec=toy.spec(
        export_rule="cet", export_demand={"c_ind": 0.8}, import_supply={"c_agr": 3.0}))
    casos["pais_grande_margens_parcial"] = caso(
        "pais_grande_margens_parcial", econ, ck.Closure(),
        ck.Scenario(name="parcial", world_price=np.array([1.2, 1.0, 1.0]),
                    export_price=np.array([1.0, 1.1, 1.0])))

    # Instituições: imposto direto, transferências, remessas, capital do governo e estrangeiro
    _, econ = toy.build_with_institutions()
    casos["instituicoes_base"] = caso("instituicoes_base", econ, ck.Closure(), None)
    casos["instituicoes_imposto"] = caso(
        "instituicoes_imposto", dataclasses.replace(econ, hh_dtax=econ.hh_dtax * 1.5),
        ck.Closure(), ck.Scenario(name="imposto"))
    fsh = econ.fac_row_share.copy()
    fsh[econ.factor("f_cap")] = 0.4
    casos["instituicoes_ied"] = caso(
        "instituicoes_ied", econ, ck.Closure(),
        ck.Scenario(name="ied", foreign_factor_share=fsh, fdi={"f_cap": 5.0}))
    casos["instituicoes_fechamento_alternativo"] = caso(
        "instituicoes_fechamento_alternativo", econ,
        ck.Closure(investment="fixed_real", foreign="fixed_rate", government="nominal_fixed"),
        ck.Scenario(name="tributo", product_tax=econ.ptax * 1.2, foreign_factor_share=fsh))
    _, econ = toy.build_with_institutions(model_spec=toy.spec(household_demand="les",
                                                              frisch=-2.0))
    W = ck.consumer_wedge(econ, {"c_ind": 0.08})
    casos["instituicoes_les_gravame"] = caso(
        "instituicoes_les_gravame", econ, ck.Closure(),
        ck.Scenario(name="gravame", consumer_wedge=W, wedge_revenue_to="capital",
                    wedge_revenue_factor="f_cap"))
    _, econ = toy.build_with_institutions(model_spec=toy.spec(export_rule="cet"))
    casos["instituicoes_cet_ied"] = caso(
        "instituicoes_cet_ied", econ, ck.Closure(),
        ck.Scenario(name="cet_ied", export_price=np.array([1.1, 1.0]), fdi={"f_cap": 3.0}))

    _, econ = toy.build()
    casos["fechamento_alternativo"] = caso(
        "fechamento_alternativo", econ,
        ck.Closure(investment="fixed_real", foreign="fixed_rate"),
        ck.Scenario(name="tributo", product_tax=econ.ptax * 1.2))

    print(f"\n{'=' * 78}\nRESUMO")
    for nome, r in casos.items():
        print(f"  {nome:<26} pior desvio {r['pior']:.1e}   violação {r['viol']:.1e}   "
              f"{'OK' if r['ok'] else 'DIVERGE'}")
    print("=" * 78)
    return 0 if all(r["ok"] for r in casos.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
