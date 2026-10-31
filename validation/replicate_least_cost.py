"""replicate_least_cost.py — menor custo, capacidade, cotas e limites de preço contra o PATH.

`least_cost.jl` lê a MCS e o enunciado do modelo, calibra o modelo nacional por conta própria e
escreve os produtores de menor custo como um problema de complementaridade misto, resolvido
pelo PATH: lucro zero de cada produtor pareado com o seu nível, e o mercado do bem pareado com
o seu preço. O cgelab resolve o mesmo modelo por aninhamento, com a complementaridade escrita
pela função de Fischer–Burmeister. Este script escreve o enunciado de cada caso, sem nenhum
parâmetro calibrado pelo cgelab, e compara preços dos fatores, câmbio, níveis de atividade,
preços ao produtor e ao comprador, custos unitários, renda das famílias e poupança.

Os casos cobrem os dois lados da troca de tecnologia (o ano-base, choques em que as duas
continuam ativas, em que uma delas para e em que a outra para) e os limites de capacidade: um
limite folgado, um que prende o único produtor de um bem, a ordem de mérito (a tecnologia mais
barata no limite e a mais cara de volta) e um limite que só prende depois de um choque) e as
cotas de importação: folgada, que prende com a renda ao governo e com a renda aos donos do
capital; e os limites de preço: teto e piso folgados e que prendem, e o piso salarial folgado,
no ano-base e que prende depois de um choque; e as tecnologias de reserva: parada no ano-base,
tomando o mercado, dividindo-o com a incumbente, entrando quando a incumbente bate no limite de
capacidade e ao lado de duas tecnologias que já competem.

    cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
    python validation/replicate_least_cost.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import cgelab as cge
from cgelab.examples import toy

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
TOL = 1e-8


def _rel(a, b, piso=1.0):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), piso)))


def enunciado(sam, econ, scenario, closure) -> dict:
    """A MCS e o enunciado do modelo, recusando o que least_cost.jl não escreve."""
    spec = econ.spec
    zero = lambda x: x is None or not np.any(np.asarray(x, float) != 0)  # noqa: E731
    fora = {
        "exportações em proporção fixa": spec.export_rule == "fixed_shares",
        "sem margens": zero(econ.mq),
        "sem imposto direto": zero(econ.hh_dtax),
        "sem transferências": zero(econ.hh_gov_transfer) and zero(econ.hh_remit_net),
        "sem renda de fatores ao governo ou ao exterior":
            zero(econ.fac_gov_share) and zero(econ.fac_row_share),
        "sem tarifa nem imposto sobre a produção": econ.tm0 is None and econ.tauz0 is None,
        "famílias Cobb–Douglas": spec.household_demand == "cobb_douglas",
        "sigma_va único": np.ndim(spec.sigma_va) == 0,
        "sem atividade de conta própria": not econ.own_account.any(),
        "mercado de trabalho com pleno emprego ou piso salarial":
            closure.labour in ("full_employment", "wage_floor"),
    }
    ruins = [k for k, ok in fora.items() if not ok]
    if ruins:
        raise ValueError(f"fora do escopo de least_cost.jl: {ruins}")
    tfp = (np.ones(econ.n_act) if scenario.productivity is None
           else np.asarray(scenario.productivity, float))
    world = np.ones(econ.n_com) if scenario.world_price is None else scenario.world_price
    supply = econ.fac_supply if scenario.factor_supply is None else scenario.factor_supply
    backstops = [dict(name=econ.acts[j], commodity=econ.coms[b["commodity"]],
                      like=None if b["like"] is None else econ.acts[b["like"]],
                      markup=b["markup"], costs=b["costs"])
                 for j, b in econ.backstops.items()]
    return dict(
        sam=[[r, c, float(v)] for r, c, v in sam.frame.itertuples(index=False)],
        activities=[a for j, a in enumerate(econ.acts) if j not in econ.backstops],
        commodities=list(econ.coms), factors=list(econ.factors), backstops=backstops,
        households=list(econ.hhs), government=econ.layout.government,
        investment=econ.layout.investment, rest_of_world=econ.layout.rest_of_world,
        least_cost=[econ.coms[i] for i in econ.least_cost], sigma_va=float(spec.sigma_va),
        sigma_armington=dict(zip(econ.coms, map(float, econ.sigma_arm))),
        numeraire_factor=econ.factors[econ.numeraire_index],
        productivity=dict(zip(econ.acts, map(float, tfp))),
        world_price=dict(zip(econ.coms, map(float, world))),
        factor_supply=dict(zip(econ.factors, map(float, supply))),
        capacity={a: float(v) for a, v in scenario.capacity.items()},
        capacity_owner=scenario.capacity_owner,
        import_quota={c: float(v) for c, v in scenario.import_quota.items()},
        quota_rent_to=scenario.quota_rent_to,
        price_ceiling={c: float(v) for c, v in scenario.price_ceiling.items()},
        price_floor={c: float(v) for c, v in scenario.price_floor.items()},
        labour=closure.labour, labour_factor=closure.labour_factor,
        real_wage=1.0 if closure.real_wage is None else float(closure.real_wage))


def roda_julia(dados: Path, saida: Path):
    julia = os.environ.get("JULIA", "julia")
    cmd = [julia, f"--project={AQUI}", str(AQUI / "least_cost.jl"), str(dados), str(saida)]
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    linhas = [ln for ln in r.stdout.splitlines() if ln.startswith(("PATH:", "WALRAS"))]
    return json.loads(saida.read_text()), linhas


def caso(nome, sam, econ, scenario, closure=None) -> dict:
    closure = closure or cge.Closure()
    OUT.mkdir(exist_ok=True)
    dados = OUT / f"least_cost_{nome}.json"
    dados.write_text(json.dumps(enunciado(sam, econ, scenario, closure)))
    jl, linhas = roda_julia(dados, OUT / f"least_cost_{nome}_out.json")
    res = cge.solve_period(econ, closure, scenario)
    bounds = res.price_bounds or {}
    desvios = {
        "w": _rel(jl["w"], res.factor_prices), "er": _rel(jl["er"], res.exchange_rate),
        "X": _rel(jl["X"], res.X), "pd": _rel(jl["pd"], res.pd), "pq": _rel(jl["pq"], res.pq),
        "pa": _rel(jl["pa"], res.pa), "hh_income": _rel(jl["hh_income"], res.hh_income),
        "savings": _rel(jl["savings"], res.savings),
        "rent": _rel(jl["rent"], [(res.capacity or {}).get(a, {}).get("rent", 0.0)
                                  for a in econ.acts]),
        "premium": _rel(jl["premium"], [(res.quota or {}).get(c, {}).get("premium", 0.0)
                                        for c in econ.coms]),
        "subsidy": _rel(jl["subsidy"], [bounds.get("ceiling", {}).get(c, {}).get("subsidy", 0.0)
                                        for c in econ.coms]),
        "purchases": _rel(jl["purchases"], [bounds.get("floor", {}).get(c, {}).get("purchases", 0.0)
                                            for c in econ.coms]),
        "unemployment": _rel(jl["unemployment"], res.unemployment),
    }
    pior = max(desvios.values())
    campo = max(desvios, key=desvios.get)
    ativas = {c: dict(zip(v["activities"], np.round(v["share"], 6)))
              for c, v in (res.least_cost or {}).items()}
    print(f"  {nome:<26} pior desvio {pior:.1e} ({campo})   Walras cgelab {res.walras:.1e} "
          f"JuMP {jl['walras']:.1e}   {'  '.join(linhas)}   participações {ativas}")
    return {"pior": pior, "ok": pior < TOL and jl["status"] == "LOCALLY_SOLVED"}


def main() -> int:
    sam, econ = toy.build_with_technologies()
    alt = econ.activity("a_alt")

    def tfp(value, **capacity):
        out = np.ones(econ.n_act)
        out[alt] = value
        return cge.Scenario(productivity=out, capacity=capacity)

    print("cgelab contra o PATH: menor custo, capacidade, cotas, limites de preço, piso salarial "
          "e reserva")
    casos = {
        "ano_base": caso("ano_base", sam, econ, cge.Scenario()),
        "as_duas_ativas": caso("as_duas_ativas", sam, econ, tfp(1.2)),
        "a_ind_para": caso("a_ind_para", sam, econ, tfp(1.5)),
        "a_alt_para": caso("a_alt_para", sam, econ, tfp(0.8)),
        "a_alt_bem_atras": caso("a_alt_bem_atras", sam, econ, tfp(0.5)),
        "preco_mundial": caso("preco_mundial", sam, econ,
                              cge.Scenario(world_price=np.array([1.0, 1.4]))),
        "limite_folgado": caso("limite_folgado", sam, econ, tfp(1.5, a_alt=300.0)),
        "limite_do_unico": caso("limite_do_unico", sam, econ, tfp(1.0, a_agr=95.0)),
        "ordem_de_merito": caso("ordem_de_merito", sam, econ, tfp(1.5, a_alt=200.0)),
        "limite_depois_do_choque": caso("limite_depois_do_choque", sam, econ,
                                        tfp(0.9, a_ind=125.0)),
        "cota_folgada": caso("cota_folgada", sam, econ,
                             cge.Scenario(import_quota={"c_ind": 40.0})),
        "cota_ao_governo": caso("cota_ao_governo", sam, econ,
                                cge.Scenario(import_quota={"c_agr": 13.5})),
        "cota_ao_capital": caso("cota_ao_capital", sam, econ,
                                cge.Scenario(import_quota={"c_agr": 13.5},
                                             quota_rent_to="f_cap")),
    }
    # Limites de preço, na economia de um produtor por bem.
    sam1, econ1 = toy.build()
    caro = cge.Scenario(productivity=np.array([0.8, 1.0]))
    barato = cge.Scenario(productivity=np.array([1.3, 1.0]))
    casos |= {
        "teto_folgado": caso("teto_folgado", sam1, econ1, caro.with_(price_ceiling={"c_agr": 2.0})),
        "teto_prende": caso("teto_prende", sam1, econ1, caro.with_(price_ceiling={"c_agr": 1.1})),
        "piso_folgado": caso("piso_folgado", sam1, econ1, barato.with_(price_floor={"c_agr": 0.8})),
        "piso_prende": caso("piso_prende", sam1, econ1, barato.with_(price_floor={"c_agr": 0.871})),
        "teto_e_piso": caso("teto_e_piso", sam1, econ1,
                            caro.with_(price_ceiling={"c_agr": 1.1}, price_floor={"c_ind": 0.9})),
    }
    # Piso salarial: o capital é o numerário, para que o mercado de trabalho seja uma equação.
    sam2, econ2 = toy.build(toy.spec(numeraire_factor="f_cap"))
    menos_capital = econ2.fac_supply.copy()
    menos_capital[econ2.factor("f_cap")] *= 0.8
    choque = cge.Scenario(factor_supply=menos_capital)
    # Tecnologias de reserva: custos de fora da MCS.
    def reserva(**backstops):
        s, _ = toy.build()
        return s, cge.calibrate(s, toy.spec(), backstops=backstops)

    def tfp_de(econ, **niveis):
        out = np.ones(econ.n_act)
        for nome, v in niveis.items():
            out[econ.activity(nome)] = v
        return cge.Scenario(productivity=out)

    copia = cge.Backstop("c_ind", like="a_ind", markup=1.25)
    engenharia = cge.Backstop("c_ind", costs={"c_agr": 0.1, "c_ind": 0.2, "f_lab": 0.6, "f_cap": 0.2})
    s3, e3 = reserva(a_new=copia)
    s4, e4 = reserva(a_eng=engenharia)
    s5, e5 = reserva(a_new=cge.Backstop("c_ind", like="a_ind", markup=1.1))
    s6, _ = toy.build_with_technologies()
    e6 = cge.calibrate(s6, toy.spec(least_cost=["c_ind"]),
                       backstops={"a_new": cge.Backstop("c_ind", like="a_alt", markup=1.2)})
    casos |= {
        "reserva_ano_base": caso("reserva_ano_base", s3, e3, cge.Scenario()),
        "reserva_toma": caso("reserva_toma", s3, e3, tfp_de(e3, a_new=1.5)),
        "reserva_divide": caso("reserva_divide", s4, e4, tfp_de(e4, a_eng=1.11)),
        "reserva_capacidade": caso("reserva_capacidade", s5, e5,
                                   tfp_de(e5, a_ind=1.2).with_(capacity={"a_ind": 210.0})),
        "reserva_com_duas": caso("reserva_com_duas", s6, e6, tfp_de(e6, a_new=1.5)),
    }
    casos |= {
        "salario_ano_base": caso("salario_ano_base", sam2, econ2, cge.Scenario(),
                                 cge.Closure(labour="wage_floor")),
        "salario_folgado": caso("salario_folgado", sam2, econ2, choque,
                                cge.Closure(labour="wage_floor", real_wage=0.85)),
        "salario_prende": caso("salario_prende", sam2, econ2, choque,
                               cge.Closure(labour="wage_floor", real_wage=0.95)),
    }
    pior = max(r["pior"] for r in casos.values())
    if all(r["ok"] for r in casos.values()):
        print(f"OK: pior desvio relativo {pior:.1e}")
        return 0
    print(f"FALHOU: pior desvio relativo {pior:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
