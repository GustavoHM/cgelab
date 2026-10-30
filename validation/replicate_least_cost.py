"""replicate_least_cost.py — produtores de menor custo contra um problema de complementaridade no PATH.

`least_cost.jl` lê a MCS e o enunciado do modelo, calibra o modelo nacional por conta própria e
escreve os produtores de menor custo como um problema de complementaridade misto, resolvido
pelo PATH: lucro zero de cada produtor pareado com o seu nível, e o mercado do bem pareado com
o seu preço. O cgelab resolve o mesmo modelo por aninhamento, com a complementaridade escrita
pela função de Fischer–Burmeister. Este script escreve o enunciado de cada caso, sem nenhum
parâmetro calibrado pelo cgelab, e compara preços dos fatores, câmbio, níveis de atividade,
preços ao produtor e ao comprador, custos unitários, renda das famílias e poupança.

Os casos cobrem os dois lados da troca de tecnologia: o ano-base, choques em que as duas
continuam ativas, choques em que uma delas para e choques em que a outra para.

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


def enunciado(sam, econ, scenario) -> dict:
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
    }
    ruins = [k for k, ok in fora.items() if not ok]
    if ruins:
        raise ValueError(f"fora do escopo de least_cost.jl: {ruins}")
    tfp = (np.ones(econ.n_act) if scenario.productivity is None
           else np.asarray(scenario.productivity, float))
    world = np.ones(econ.n_com) if scenario.world_price is None else scenario.world_price
    return dict(
        sam=[[r, c, float(v)] for r, c, v in sam.frame.itertuples(index=False)],
        activities=list(econ.acts), commodities=list(econ.coms), factors=list(econ.factors),
        households=list(econ.hhs), government=econ.layout.government,
        investment=econ.layout.investment, rest_of_world=econ.layout.rest_of_world,
        least_cost=list(spec.least_cost or []), sigma_va=float(spec.sigma_va),
        sigma_armington=dict(zip(econ.coms, map(float, econ.sigma_arm))),
        numeraire_factor=econ.factors[econ.numeraire_index],
        productivity=dict(zip(econ.acts, map(float, tfp))),
        world_price=dict(zip(econ.coms, map(float, world))))


def roda_julia(dados: Path, saida: Path):
    julia = os.environ.get("JULIA", "julia")
    cmd = [julia, f"--project={AQUI}", str(AQUI / "least_cost.jl"), str(dados), str(saida)]
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    linhas = [ln for ln in r.stdout.splitlines() if ln.startswith(("PATH:", "WALRAS"))]
    return json.loads(saida.read_text()), linhas


def caso(nome, sam, econ, scenario) -> dict:
    OUT.mkdir(exist_ok=True)
    dados = OUT / f"least_cost_{nome}.json"
    dados.write_text(json.dumps(enunciado(sam, econ, scenario)))
    jl, linhas = roda_julia(dados, OUT / f"least_cost_{nome}_out.json")
    res = cge.solve_period(econ, scenario=scenario)
    desvios = {
        "w": _rel(jl["w"], res.factor_prices), "er": _rel(jl["er"], res.exchange_rate),
        "X": _rel(jl["X"], res.X), "pd": _rel(jl["pd"], res.pd), "pq": _rel(jl["pq"], res.pq),
        "pa": _rel(jl["pa"], res.pa), "hh_income": _rel(jl["hh_income"], res.hh_income),
        "savings": _rel(jl["savings"], res.savings),
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

    def tfp(value):
        out = np.ones(econ.n_act)
        out[alt] = value
        return cge.Scenario(productivity=out)

    print("cgelab contra o PATH, produtores de c_ind ao menor custo")
    casos = {
        "ano_base": caso("ano_base", sam, econ, cge.Scenario()),
        "as_duas_ativas": caso("as_duas_ativas", sam, econ, tfp(1.2)),
        "a_ind_para": caso("a_ind_para", sam, econ, tfp(1.5)),
        "a_alt_para": caso("a_alt_para", sam, econ, tfp(0.8)),
        "a_alt_bem_atras": caso("a_alt_bem_atras", sam, econ, tfp(0.5)),
        "preco_mundial": caso("preco_mundial", sam, econ,
                              cge.Scenario(world_price=np.array([1.0, 1.4]))),
    }
    pior = max(r["pior"] for r in casos.values())
    if all(r["ok"] for r in casos.values()):
        print(f"OK: pior desvio relativo {pior:.1e}")
        return 0
    print(f"FALHOU: pior desvio relativo {pior:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
