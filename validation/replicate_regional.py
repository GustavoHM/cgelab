"""replicate_regional.py — o modelo de várias regiões contra uma reescrita independente em JuMP.

`regional.jl` lê a MCS e o enunciado do modelo (papéis das contas, elasticidades, rotas de
transporte, migração, fechamento e choque), calibra tudo por conta própria e resolve o
equilíbrio como um sistema quadrado no Ipopt. Este script escreve esse enunciado a partir dos
exemplos do pacote, sem passar nenhum parâmetro calibrado pelo cgelab, e compara o que os dois
lados têm em comum: preços dos fatores, câmbio, produção, exportações, vendas internas e
remessas bilaterais por origem e destino, importações, preços ao comprador relativos ao
ano-base, renda e gasto das famílias, poupança e a oferta de serviços de transporte.

Os casos cobrem o que a validação regional da 0.8 pede: preços diferentes por destino
(importações assimétricas, margens de transporte), transporte com investimento regional,
serviço de transporte exportado, insumos comprados com margem, migração CET e logit e contas
regionais de governo e investimento, cada um no ano-base e sob choque.

    cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
    python validation/replicate_regional.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import cgelab as cge
from cgelab.examples import regions

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
TOL = 1e-8


def _rel(a, b, piso=1.0):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), piso))) if a.size else 0.0


def _fora_do_escopo(econ):
    """O que regional.jl não escreve. Recusado em vez de comparado pela metade."""
    zero = lambda x: x is None or not np.any(np.asarray(x, float) != 0)  # noqa: E731
    checks = {
        "imposto sobre produtos": zero(econ.ptax),
        "imposto sobre a atividade": zero(econ.atax),
        "imposto direto": zero(econ.hh_dtax),
        "transferências do governo": zero(econ.hh_gov_transfer),
        "remessas": zero(econ.hh_remit_net),
        "renda de fatores ao exterior": zero(econ.fac_row_share),
        "margens fora das rotas": econ.regional.transport is not None or zero(econ.regional.mq),
        "famílias Cobb–Douglas": econ.spec.household_demand == "cobb_douglas",
    }
    ruins = [k for k, ok in checks.items() if not ok]
    if ruins:
        raise ValueError(f"fora do escopo de regional.jl: {ruins}")


def enunciado(sam, econ, closure, scenario) -> dict:
    """A MCS e o enunciado do modelo. Nada calibrado pelo cgelab entra aqui."""
    _fora_do_escopo(econ)
    spec, reg = econ.spec, econ.regional
    goods = list(reg.local_commodities)
    factors = []
    for f in econ.factors:
        local = f.split(":", 1)[1]
        if local not in factors:
            factors.append(local)
    make = sam.block(econ.acts, econ.coms)
    activity = {com: econ.acts[int(np.argmax(make[:, i]))] for i, com in enumerate(econ.coms)}
    if np.count_nonzero(make) != len(econ.coms):
        raise ValueError("fora do escopo de regional.jl: uma atividade por bem de origem")
    tfp = (np.ones(econ.n_act) if scenario.productivity is None
           else np.asarray(scenario.productivity, float))
    world = np.ones(len(goods)) if scenario.world_price is None else scenario.world_price
    export = np.ones(len(goods)) if scenario.export_price is None else scenario.export_price
    num = econ.factors[econ.numeraire_index]
    transport = None
    if reg.transport is not None:
        t = reg.transport
        transport = dict(fob=t.fob.tolist(), transport=t.transport.tolist(),
                         supply=t.supply.tolist())
    migration = {}
    for local, omega in (spec.factor_migration or {}).items():
        model = reg.migration[local]["model"]
        migration[local] = dict(omega=float(omega), form=model.form, delta=model.delta.tolist())
    return dict(
        sam=[[r, c, float(v)] for r, c, v in sam.frame.itertuples(index=False)],
        regions=list(spec.regions), goods=goods, factors=factors,
        households=list(econ.hhs),
        household_region={h: h.split(":", 1)[0] for h in econ.hhs},
        activity=activity,
        government=econ.layout.government, investment=econ.layout.investment,
        rest_of_world=econ.layout.rest_of_world,
        regional_government=dict(spec.regional_government or {}),
        regional_investment=dict(spec.regional_investment or {}),
        sigma_va=float(spec.sigma_va),
        sigma_interregional={g: float(s) for g, s in zip(goods, reg.sigma)},
        sigma_armington={g: float(s) for g, s in zip(goods, reg.sigma_arm)},
        sigma_cet={g: float(s) for g, s in zip(goods, reg.sigma_cet[0])},
        export_rule=spec.export_rule,
        transport=transport, migration=migration,
        population_share={h: float(p) for h, p in zip(econ.hhs, econ.pop_share)},
        closure=dict(investment=closure.investment, government=closure.government,
                     numeraire=float(closure.numeraire),
                     numeraire_region=num.split(":")[0], numeraire_factor=num.split(":")[1]),
        scenario=dict(productivity=dict(zip(econ.acts, map(float, tfp))),
                      world_price=dict(zip(goods, map(float, world))),
                      export_price=dict(zip(goods, map(float, export))),
                      investment_level=scenario.investment_level),
    )


def lado_cgelab(econ, res) -> dict:
    reg = econ.regional
    nr, nl = reg.n_reg, reg.n_local
    order = [reg.origin_index[r, c] for r in range(nr) for c in range(nl)]
    activity_of = {}
    for j in range(econ.n_act):
        activity_of[int(np.argmax(econ.msh[j]))] = j
    w = []
    for r in reg.regions:
        for f in econ.factors:
            if f.startswith(r + ":"):
                w.append(res.factor_prices[econ.factors.index(f)])
    pq_rel = res.real["destination_price"] / reg.base_price
    return {
        "w": w, "er": res.exchange_rate,
        "X": [res.X[activity_of[i]] for i in order],
        "exports": [res.exports[i] for i in order],
        "domestic_sales": [res.domestic_sales[i] for i in order],
        "bilateral": res.bilateral,
        "imports_total": res.imports.sum(),
        "pq_rel": pq_rel,
        "hh_income": res.hh_income, "hh_spend": res.hh_commodity_spend,
        "savings": res.savings,
        "qst": None if res.transport is None else res.transport["qst"],
    }


def roda_julia(dados: Path, saida: Path) -> tuple[dict, list]:
    julia = os.environ.get("JULIA", "julia")
    cmd = [julia, f"--project={AQUI}", str(AQUI / "regional.jl"), str(dados), str(saida)]
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    linhas = [ln for ln in r.stdout.splitlines() if ln.startswith(("Ipopt:", "violação"))]
    return json.loads(saida.read_text()), linhas


def compara(econ, jl, cg) -> dict:
    reg = econ.regional
    nr, nd, nl = reg.n_reg, reg.n_dest, reg.n_local
    ativo = np.asarray(jl["ativo"], bool).reshape((nd, nl), order="F")
    lados = {
        "w": (jl["w"], cg["w"]), "er": (jl["er"], cg["er"]),
        "X": (jl["X"], cg["X"]), "exports": (jl["exports"], cg["exports"]),
        "domestic_sales": (jl["domestic_sales"], cg["domestic_sales"]),
        "bilateral": (np.reshape(jl["bilateral"], (nr, nr, nl), order="F"), cg["bilateral"]),
        "imports_total": (jl["imports_total"], cg["imports_total"]),
        "pq_rel": (np.reshape(jl["pq_rel"], (nd, nl), order="F")[ativo], cg["pq_rel"][ativo]),
        "hh_income": (jl["hh_income"], cg["hh_income"]),
        "hh_spend": (jl["hh_spend"], cg["hh_spend"]),
        "savings": (jl["savings"], cg["savings"]),
    }
    if cg["qst"] is not None:
        lados["qst"] = (np.reshape(jl["qst"], (nr, nl), order="F"), cg["qst"])
    return {k: _rel(a, b) for k, (a, b) in lados.items()}


def caso(nome, sam, econ, closure=None, scenario=None) -> dict:
    closure = closure or cge.Closure()
    OUT.mkdir(exist_ok=True)
    pior = 0.0
    print(f"\n{'=' * 78}\n{nome}\n{'=' * 78}")
    for etiqueta, sc in (("ano-base", cge.Scenario()), ("choque", scenario or cge.Scenario())):
        dados = OUT / f"regional_{nome}_{etiqueta}.json"
        dados.write_text(json.dumps(enunciado(sam, econ, closure, sc)))
        jl, linhas = roda_julia(dados, OUT / f"regional_{nome}_{etiqueta}_out.json")
        res = cge.solve_period(econ, closure, sc)
        desvios = compara(econ, jl, lado_cgelab(econ, res))
        pior_aqui = max(desvios.values())
        pior = max(pior, pior_aqui)
        campo = max(desvios, key=desvios.get)
        print(f"  {etiqueta:<9} pior desvio {pior_aqui:.1e} ({campo})   "
              f"Walras cgelab {res.walras:.1e}  JuMP {jl['walras']:.1e}   "
              + "  ".join(linhas))
        if etiqueta == "ano-base":
            # O ano-base da reescrita é a MCS por construção; o do cgelab tem de ser também.
            pior = max(pior, _rel(res.X, sam.block(econ.acts, econ.coms).sum(axis=1)))
    return {"pior": pior, "ok": pior < TOL}


def instituicoes_regionais():
    """As contas de governo e de investimento de cada região, como nos testes."""
    rows = []
    for r, c, value in regions.ENTRIES:
        origin, destination = r.split(":")[0], c.split(":")[0]
        if r in ("gov", "s_i"):
            r = f"{destination}:{r}"
        if c in ("gov", "s_i"):
            c = f"{origin}:{c}"
        rows.append((r, c, value))
    sam = cge.SAM.from_long(pd.DataFrame(rows, columns=["row", "col", "value"]))
    spec = regions.spec(government="r1:gov", investment="r1:s_i",
                        regional_government={"r2": "r2:gov"},
                        regional_investment={"r2": "r2:s_i"})
    return sam, cge.calibrate(sam, spec)


def main() -> int:
    casos = {}
    choque = cge.Scenario(productivity=np.array([1.1, 1.0]), world_price=np.array([1.2]))
    sam, econ = regions.build()
    casos["simetrico"] = caso("simetrico", sam, econ, scenario=choque)

    sam, econ = regions.build(regions.spec(export_rule="cet", sigma_cet=3.0),
                              asymmetric_imports=True)
    casos["importacoes_assimetricas_cet"] = caso(
        "importacoes_assimetricas_cet", sam, econ,
        scenario=cge.Scenario(productivity=np.array([1.0, 0.9]), world_price=np.array([0.9]),
                              export_price=np.array([1.3])))

    tfp = np.ones(4)
    tfp[1] = 1.1
    choque_t = cge.Scenario(productivity=tfp, world_price=np.array([1.2, 1.0]))
    sam, econ = regions.build_with_transport()
    casos["transporte"] = caso("transporte", sam, econ, scenario=choque_t)
    sam, econ = regions.build_with_transport(intermediates=True)
    casos["transporte_insumos"] = caso("transporte_insumos", sam, econ, scenario=choque_t)
    sam, econ = regions.build_with_transport(unrouted_good=True)
    casos["transporte_bem_sem_rota"] = caso(
        "transporte_bem_sem_rota", sam, econ,
        scenario=cge.Scenario(productivity=np.ones(6), world_price=np.array([1.2, 1.0, 0.8])))
    sam, econ = regions.build_with_transport(regional_investment=True)
    casos["transporte_investimento_regional"] = caso(
        "transporte_investimento_regional", sam, econ, scenario=choque_t)
    casos["transporte_investimento_fixo"] = caso(
        "transporte_investimento_fixo", sam, econ, cge.Closure(investment="fixed_real"),
        cge.Scenario(productivity=tfp, investment_level=50.0))
    sam, econ = regions.build_with_transport(service_exports=True, export_rule="cet",
                                             sigma_cet=3.0)
    casos["transporte_servico_exportado_cet"] = caso(
        "transporte_servico_exportado_cet", sam, econ,
        scenario=cge.Scenario(productivity=tfp, world_price=np.array([1.2, 1.0]),
                              export_price=np.array([1.0, 1.3])))

    proximidade = pd.DataFrame([[0.9, 0.1], [0.1, 0.9]], index=["r1", "r2"],
                               columns=["r1", "r2"])
    for forma in ("cet", "logit"):
        spec = regions.spec(factor_migration={"f_lab": 2.0}, factor_mobility_form=forma)
        sam = regions.build()[0]
        econ = cge.calibrate(sam, spec, factor_pools={"f_lab": proximidade})
        casos[f"migracao_{forma}"] = caso(
            f"migracao_{forma}", sam, econ,
            scenario=cge.Scenario(productivity=np.array([0.8, 1.0])))

    sam, econ = instituicoes_regionais()
    casos["instituicoes_poupanca_do_governo"] = caso(
        "instituicoes_poupanca_do_governo", sam, econ,
        cge.Closure(government="saving_rate"), choque)
    casos["instituicoes_investimento_fixo"] = caso(
        "instituicoes_investimento_fixo", sam, econ, cge.Closure(investment="fixed_real"),
        cge.Scenario(productivity=np.array([1.1, 1.0]), investment_level=45.0))

    print(f"\n{'=' * 78}")
    for nome, r in casos.items():
        print(f"  {nome:<36} pior desvio {r['pior']:.1e}   {'OK' if r['ok'] else 'FALHOU'}")
    pior = max(r["pior"] for r in casos.values())
    if all(r["ok"] for r in casos.values()):
        print(f"OK: pior desvio relativo {pior:.1e}")
        return 0
    print(f"FALHOU: pior desvio relativo {pior:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
