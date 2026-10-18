"""roundtrip_jump.py — exportar do cgelab para JuMP, resolver no Julia, e comparar de volta.

Por que este confronto é forte. O cgelab resolve o equilíbrio por ANINHAMENTO: um Newton sobre
poucas incógnitas e, dentro dele, um ponto fixo sobre o preço de entrega, duas somas em forma
fechada — a série infinita da reciclagem do gravame e a relação afim entre poupança e arrecadação —
e uma inversão de Leontief. Cada uma é uma dedução do autor. Em JuMP nada disso é deduzido: cada
bloco é uma restrição e o Ipopt encontra o ponto fixo. Um erro na álgebra do Python não pode,
portanto, ser reproduzido do outro lado por construção.

Roda 34 casos sobre as economias de brinquedo do pacote: sem margens, com bloco de margens, com
atividade de conta própria, com CET, com curvas de demanda de exportação e de oferta de
importação (país grande, como o `lrgcge`), com tarifa e imposto sobre o produto dentro dos preços
(como o `stdcge`), com instituições (impostos diretos, transferências, remessas,
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
import pandas as pd

import cgelab as cge
from cgelab.examples import hosoe, toy

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
    cge.to_jump(econ, closure, scenario, path=jl, data_path=dados)
    ref = _referencia(cge.solve_period(econ, closure, scenario))
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
    print(f"  {'Walras cgelab':<16}{cge.solve_period(econ, closure, scenario).walras:>14.2e}"
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
        W = cge.consumer_wedge(econ, {"c_ind": 0.08})
        casos[f"{etiqueta}_base"] = caso(f"{etiqueta}_base", econ, cge.Closure(), None)
        casos[f"{etiqueta}_gravame"] = caso(
            f"{etiqueta}_gravame", econ, cge.Closure(),
            cge.Scenario(name="gravame", consumer_wedge=W, wedge_revenue_to="capital",
                         wedge_revenue_factor="f_cap"))
    # CET: a repartição do produto responde ao preço de exportação
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet", sigma_cet=2.0))
    casos["cet_base"] = caso("cet_base", econ, cge.Closure(), None)
    choque_cet = cge.Scenario(name="cet_choque", export_price=np.array([1.15, 0.9]),
                              product_tax=econ.ptax * 1.2)
    casos["cet_choque"] = caso("cet_choque", econ, cge.Closure(), choque_cet)
    casos["cet_fechamento_alternativo"] = caso(
        "cet_fechamento_alternativo", econ,
        cge.Closure(investment="fixed_real", foreign="fixed_rate"), choque_cet)

    # País grande: demanda de exportação e oferta de importação com elasticidade finita
    _, econ = toy.build(model_spec=toy.spec(export_rule="cet", export_demand=2.0,
                                            import_supply=2.0))
    casos["pais_grande_choque"] = caso("pais_grande_choque", econ, cge.Closure(), choque_cet)
    casos["pais_grande_fechamento_alternativo"] = caso(
        "pais_grande_fechamento_alternativo", econ,
        cge.Closure(investment="fixed_real", foreign="fixed_rate"), choque_cet)
    _, econ = toy.build_with_margins(model_spec=toy.spec(
        export_rule="cet", export_demand={"c_ind": 0.8}, import_supply={"c_agr": 3.0}))
    casos["pais_grande_margens_parcial"] = caso(
        "pais_grande_margens_parcial", econ, cge.Closure(),
        cge.Scenario(name="parcial", world_price=np.array([1.2, 1.0, 1.0]),
                     export_price=np.array([1.0, 1.1, 1.0])))

    # Tarifa e imposto sobre todo o produto dentro dos preços, como no stdcge; o governo que poupa
    # uma fração da renda; a poupança externa fixa em dólares
    tributos = toy.spec(tariff_account="trf", indirect_tax_account="idt")
    _, econ = toy.build_with_trade_taxes()
    abolicao = cge.Scenario(name="abolicao", tariff=np.zeros(2))
    casos["tarifa_abolicao"] = caso("tarifa_abolicao", econ, cge.Closure(), abolicao)
    casos["tarifa_abolicao_stdcge"] = caso(
        "tarifa_abolicao_stdcge", econ,
        cge.Closure(government="saving_rate", foreign_saving_unit="foreign_currency"), abolicao)
    _, econ = toy.build_with_trade_taxes(model_spec=tributos.__class__(
        **{**tributos.__dict__, "export_rule": "cet"}))
    casos["imposto_produto_cet"] = caso(
        "imposto_produto_cet", econ,
        cge.Closure(investment="fixed_real", government="saving_rate"),
        cge.Scenario(name="idt", output_tax=econ.tauz0 * 3, tariff=econ.tm0 * 0.5))
    _, econ = toy.build_with_trade_taxes(model_spec=tributos.__class__(
        **{**tributos.__dict__, "export_rule": "cet", "export_demand": 2.0,
           "import_supply": 2.0}))
    casos["lrgcge_abolicao"] = caso(
        "lrgcge_abolicao", econ,
        cge.Closure(government="saving_rate", foreign_saving_unit="foreign_currency"), abolicao)

    _, econ = hosoe.build()
    casos["hosoe_stdcge_abolicao"] = caso("hosoe_stdcge_abolicao", econ, hosoe.CLOSURE, abolicao)

    # Instituições: imposto direto, transferências, remessas, capital do governo e estrangeiro
    _, econ = toy.build_with_institutions()
    casos["instituicoes_base"] = caso("instituicoes_base", econ, cge.Closure(), None)
    casos["instituicoes_imposto"] = caso(
        "instituicoes_imposto", dataclasses.replace(econ, hh_dtax=econ.hh_dtax * 1.5),
        cge.Closure(), cge.Scenario(name="imposto"))
    fsh = econ.fac_row_share.copy()
    fsh[econ.factor("f_cap")] = 0.4
    casos["instituicoes_ied"] = caso(
        "instituicoes_ied", econ, cge.Closure(),
        cge.Scenario(name="ied", foreign_factor_share=fsh, fdi={"f_cap": 5.0}))
    casos["instituicoes_fechamento_alternativo"] = caso(
        "instituicoes_fechamento_alternativo", econ,
        cge.Closure(investment="fixed_real", foreign="fixed_rate", government="nominal_fixed"),
        cge.Scenario(name="tributo", product_tax=econ.ptax * 1.2, foreign_factor_share=fsh))
    _, econ = toy.build_with_institutions(model_spec=toy.spec(household_demand="les",
                                                              frisch=-2.0))
    W = cge.consumer_wedge(econ, {"c_ind": 0.08})
    casos["instituicoes_les_gravame"] = caso(
        "instituicoes_les_gravame", econ, cge.Closure(),
        cge.Scenario(name="gravame", consumer_wedge=W, wedge_revenue_to="capital",
                     wedge_revenue_factor="f_cap"))
    _, econ = toy.build_with_institutions(model_spec=toy.spec(export_rule="cet"))
    casos["instituicoes_cet_ied"] = caso(
        "instituicoes_cet_ied", econ, cge.Closure(),
        cge.Scenario(name="cet_ied", export_price=np.array([1.1, 1.0]), fdi={"f_cap": 3.0}))

    # Metas do governo: poupança ou receita fixas, atingidas por um tributo ou pelo gasto
    _, econ = toy.build_with_institutions()
    corte = cge.Scenario(name="corte", product_tax=econ.ptax * 0.5)
    for alvo, ajusta in (("saving", "direct_tax"), ("saving", "spending"),
                         ("revenue", "direct_tax")):
        casos[f"meta_{alvo}_{ajusta}"] = caso(
            f"meta_{alvo}_{ajusta}", econ,
            cge.Closure(government_target=alvo, government_adjusts=ajusta), corte)
    casos["meta_revenue_product_tax"] = caso(
        "meta_revenue_product_tax", econ,
        cge.Closure(government_target="revenue", government_adjusts="product_tax",
                    investment="fixed_real"),
        cge.Scenario(name="dotacao", factor_supply=econ.fac_supply * np.array([1.05, 1.0])))

    # Emissões e preço do carbono: nas atividades (imposto por unidade de produto) e nas famílias
    # (imposto específico, bloco de famílias expandido no JuMP)
    for etiqueta, spec_ in (("cd", toy.spec()), ("les", toy.spec(household_demand="les"))):
        _, econ = toy.build_with_institutions(model_spec=spec_)
        inter = pd.DataFrame(0.0, index=econ.coms, columns=econ.acts)
        inter.loc["c_ind", "a_agr"], inter.loc["c_ind", "a_ind"] = 3.0, 12.0
        econ = cge.add_emissions(econ, intermediate=inter, household={"c_ind": 5.0},
                                 government={"c_ind": 1.0}, process={"a_ind": 2.0})
        casos[f"carbono_{etiqueta}"] = caso(f"carbono_{etiqueta}", econ, cge.Closure(),
                                           cge.Scenario(name="carbono", carbon_price=0.5))
    casos["carbono_neutro"] = caso(
        "carbono_neutro", econ,
        cge.Closure(government_target="revenue", government_adjusts="direct_tax"),
        cge.Scenario(name="carbono", carbon_price=0.5,
                     productivity=np.array([1.0, 1.04])))

    # Ninho valor adicionado-energia: substituição entre combustíveis e entre energia e valor
    # adicionado, com o carbono no preço dos combustíveis
    _, econ = toy.build_with_institutions(model_spec=toy.spec(
        energy=["c_ind"], sigma_vae={"__default__": 0.6, "a_agr": 1.0}, sigma_energy=1.0))
    inter = pd.DataFrame(0.0, index=econ.coms, columns=econ.acts)
    inter.loc["c_ind", "a_agr"], inter.loc["c_ind", "a_ind"] = 3.0, 12.0
    econ = cge.add_emissions(econ, intermediate=inter, household={"c_ind": 5.0},
                             process={"a_ind": 2.0})
    casos["klem_carbono"] = caso("klem_carbono", econ, cge.Closure(), cge.Scenario(
        name="klem", carbon_price=0.5, productivity=np.array([1.02, 1.0])))
    _, econ = toy.build_with_margins(model_spec=toy.spec(
        energy=["c_agr", "c_ind"], sigma_vae=0.4, sigma_energy={"__default__": 2.0, "a_ind": 0.5}))
    inter = pd.DataFrame(0.0, index=econ.coms, columns=econ.acts)
    inter.loc[["c_agr", "c_ind"], ["a_agr", "a_ind"]] = [[2.0, 1.0], [4.0, 9.0]]
    econ = cge.add_emissions(econ, intermediate=inter)
    casos["klem_dois_combustiveis"] = caso(
        "klem_dois_combustiveis", econ,
        cge.Closure(government_target="revenue", government_adjusts="product_tax"),
        cge.Scenario(name="klem2", carbon_price=0.8))

    # Composto capital-energia dentro do ninho do valor adicionado (o arranjo do GTAP-E)
    _, econ = toy.build_with_margins(model_spec=toy.spec(
        energy=["c_agr", "c_ind"], energy_nest="capital", sigma_va=0.8,
        sigma_ke={"__default__": 0.3, "a_ind": 1.0}, sigma_energy=2.0))
    inter = pd.DataFrame(0.0, index=econ.coms, columns=econ.acts)
    inter.loc[["c_agr", "c_ind"], ["a_agr", "a_ind"]] = [[2.0, 1.0], [4.0, 9.0]]
    econ = cge.add_emissions(econ, intermediate=inter)
    casos["klem_capital_energia"] = caso(
        "klem_capital_energia", econ,
        cge.Closure(government_target="revenue", government_adjusts="product_tax"),
        cge.Scenario(name="ke", carbon_price=0.8, productivity=np.array([1.02, 1.0, 1.0])))

    _, econ = toy.build()
    casos["fechamento_alternativo"] = caso(
        "fechamento_alternativo", econ,
        cge.Closure(investment="fixed_real", foreign="fixed_rate"),
        cge.Scenario(name="tributo", product_tax=econ.ptax * 1.2))

    print(f"\n{'=' * 78}\nRESUMO")
    for nome, r in casos.items():
        print(f"  {nome:<26} pior desvio {r['pior']:.1e}   violação {r['viol']:.1e}   "
              f"{'OK' if r['ok'] else 'DIVERGE'}")
    print("=" * 78)
    return 0 if all(r["ok"] for r in casos.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
