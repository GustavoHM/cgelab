"""replicate_trade_instruments.py — imposto sobre exportações e Armington assimétrico contra o PATH.

`trade_instruments.jl` lê a MCS e o enunciado do modelo, calibra o modelo nacional por conta
própria nas unidades do modelo-padrão do IFPRI e escreve o Armington assimétrico como Cicowiez
e Lofgren (2019, tabela 2.1): os desvios DEVP e DEVN da razão PM/PDD em relação à do ano-base,
pareados com a elasticidade SIGMAQ, e os parâmetros do CES recalibrados ao ano-base para a
elasticidade escolhida. O cgelab resolve o mesmo modelo por aninhamento, com a CES na forma de
participações e a elasticidade escolhida pelo sinal da variação do preço relativo. Este script
escreve o enunciado de cada caso, sem nenhum parâmetro calibrado pelo cgelab, e compara preços
dos fatores, câmbio, níveis de atividade, preços ao produtor e ao comprador, importações na
fronteira, exportações FOB e o seu preço, renda das famílias, poupança, a receita do imposto
sobre exportações e a elasticidade escolhida.

Os casos cobrem o imposto sobre exportações com CET e com exportações em proporção fixa (o
ano-base, abolido, elevado, com a tarifa ao lado, com um choque no preço mundial) e o Armington
assimétrico (preço mundial da importação de um bem 25% acima e 25% abaixo, como nos
experimentos do artigo, os dois bens em direções opostas, ao lado dos impostos de comércio e
com exportações em proporção fixa).

    cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
    python validation/replicate_trade_instruments.py
"""
from __future__ import annotations

import dataclasses
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
# As elasticidades-renda com que cada economia foi calibrada, para o enunciado: o Economy
# guarda os parâmetros de expansão, não as elasticidades de que saíram.
INCOME_ELASTICITIES: dict = {}


def _rel(a, b, piso=1.0):
    a = np.atleast_1d(np.asarray(a, float))
    b = np.atleast_1d(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), piso)))


def enunciado(sam, econ, scenario, closure) -> dict:
    """A MCS e o enunciado do modelo, recusando o que trade_instruments.jl não escreve."""
    spec = econ.spec
    zero = lambda x: x is None or not np.any(np.asarray(x, float) != 0)  # noqa: E731
    fora = {
        "sem margens": zero(econ.mq),
        "sem imposto sobre a produção": econ.tauz0 is None,
        "sem imposto direto": zero(econ.hh_dtax),
        "sem transferências": zero(econ.hh_gov_transfer) and zero(econ.hh_remit_net),
        "sem renda de fatores ao governo ou ao exterior":
            zero(econ.fac_gov_share) and zero(econ.fac_row_share),
        "famílias Cobb–Douglas ou CDE": spec.household_demand in ("cobb_douglas", "cde"),
        "sigma_va único": np.ndim(spec.sigma_va) == 0,
        "sem atividade de conta própria": not econ.own_account.any(),
        "fechamento padrão": closure == cge.Closure(),
        "sem choque de produtividade": scenario.productivity is None,
    }
    ruins = [k for k, ok in fora.items() if not ok]
    if ruins:
        raise ValueError(f"fora do escopo de trade_instruments.jl: {ruins}")
    nc = econ.n_com
    vec = lambda v, d: dict(zip(econ.coms, map(float, d if v is None else v)))  # noqa: E731
    falling = spec.sigma_armington_falling_vector(econ.coms)
    return dict(
        sam=[[r, c, float(v)] for r, c, v in sam.frame.itertuples(index=False)],
        activities=list(econ.acts), commodities=list(econ.coms), factors=list(econ.factors),
        households=list(econ.hhs), government=econ.layout.government,
        investment=econ.layout.investment, rest_of_world=econ.layout.rest_of_world,
        tariff_account=spec.tariff_account, export_tax_account=spec.export_tax_account,
        sigma_va=float(spec.sigma_va),
        sigma_armington=dict(zip(econ.coms, map(float, spec.sigma_armington_vector(econ.coms)))),
        sigma_armington_falling={c: (float(v) if np.isfinite(v) else None)
                                 for c, v in zip(econ.coms, falling)},
        export_rule=spec.export_rule, household_demand=spec.household_demand,
        cde_substitution=(None if econ.cde is None
                          else dict(zip(econ.coms, map(float, econ.cde["b"])))),
        income_elasticities=dict(INCOME_ELASTICITIES.get(id(econ), {})),
        sigma_cet=dict(zip(econ.coms, map(float, spec.sigma_cet_vector(econ.coms)))),
        numeraire_factor=econ.factors[econ.numeraire_index],
        world_price=vec(scenario.world_price, np.ones(nc)),
        export_price=vec(scenario.export_price, np.ones(nc)),
        tariff=vec(scenario.tariff, econ.zeros_if_none("tm0", nc)),
        export_tax=vec(scenario.export_tax, econ.zeros_if_none("te0", nc)),
        factor_supply=dict(zip(econ.factors, map(float, econ.fac_supply
                                                 if scenario.factor_supply is None
                                                 else scenario.factor_supply))))


def roda_julia(dados: Path, saida: Path):
    julia = os.environ.get("JULIA", "julia")
    cmd = [julia, f"--project={AQUI}", str(AQUI / "trade_instruments.jl"), str(dados), str(saida)]
    r = subprocess.run(cmd, cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    linhas = [ln for ln in r.stdout.splitlines() if ln.startswith(("PATH:", "WALRAS"))]
    return json.loads(saida.read_text()), linhas


def caso(nome, sam, econ, scenario, closure=None) -> dict:
    closure = closure or cge.Closure()
    OUT.mkdir(exist_ok=True)
    dados = OUT / f"trade_{nome}.json"
    saida = OUT / f"trade_{nome}_out.json"
    saida.unlink(missing_ok=True)
    dados.write_text(json.dumps(enunciado(sam, econ, scenario, closure)))
    jl, linhas = roda_julia(dados, saida)
    res = cge.solve_period(econ, closure, scenario)
    sigma = res.armington_sigma if res.armington_sigma is not None else econ.sigma_arm
    desvios = {
        "w": _rel(jl["w"], res.factor_prices), "er": _rel(jl["er"], res.exchange_rate),
        "X": _rel(jl["X"], res.X), "pd": _rel(jl["pd"], res.pd), "pq": _rel(jl["pq"], res.pq),
        "imports": _rel(jl["imports"], res.imports), "exports": _rel(jl["exports"], res.exports),
        "pe": _rel(jl["pe"], res.pe), "hh_income": _rel(jl["hh_income"], res.hh_income),
        "savings": _rel(jl["savings"], res.savings),
        "export_tax": _rel(jl["export_tax"], res.government["export_tax"]),
        "sigma": _rel(jl["sigma"], sigma),
        "hh_quantity": _rel(jl["hh_quantity"],
                            res.hh_quantity[econ.hh_budget > 0]),
    }
    pior = max(desvios.values())
    campo = max(desvios, key=desvios.get)
    print(f"  {nome:<28} pior desvio {pior:.1e} ({campo:<10}) Walras cgelab {res.walras:.1e} "
          f"JuMP {jl['walras']:.1e}   {'  '.join(linhas)}   sigma {np.round(sigma, 3)}")
    return {"pior": pior, "ok": pior < TOL and jl["status"] == "LOCALLY_SOLVED"}


def main() -> int:
    print("cgelab contra o PATH: imposto sobre exportações e Armington assimétrico")
    sam, econ = toy.build_with_export_tax()
    te0 = econ.te0
    nc = econ.n_com

    def com(**v):
        out = np.ones(nc)
        for nome, x in v.items():
            out[econ.commodity(nome)] = x
        return out

    fixas = toy.spec(tariff_account="trf", export_tax_account="etx")
    _, econ_fixas = toy.build_with_export_tax(fixas)
    casos = {
        "ano_base_cet": caso("ano_base_cet", sam, econ, cge.Scenario()),
        "ano_base_fixas": caso("ano_base_fixas", sam, econ_fixas, cge.Scenario()),
        "abolido_cet": caso("abolido_cet", sam, econ, cge.Scenario(export_tax=np.zeros(nc))),
        "elevado_cet": caso("elevado_cet", sam, econ, cge.Scenario(export_tax=te0 * 4)),
        "um_bem_cet": caso("um_bem_cet", sam, econ,
                           cge.Scenario(export_tax=np.array([0.25, te0[1]]))),
        "com_tarifa_cet": caso("com_tarifa_cet", sam, econ,
                               cge.Scenario(export_tax=te0 * 3, tariff=econ.tm0 * 0.0)),
        "preco_mundial_cet": caso("preco_mundial_cet", sam, econ,
                                  cge.Scenario(export_price=com(c_ind=1.2))),
        "abolido_fixas": caso("abolido_fixas", sam, econ_fixas,
                              cge.Scenario(export_tax=np.zeros(nc))),
        "elevado_fixas": caso("elevado_fixas", sam, econ_fixas,
                              cge.Scenario(export_tax=te0 * 4)),
    }
    # Armington assimétrico: 0,9 quando a importação encarece, 3 quando barateia, como no artigo.
    assim = dataclasses.replace(econ.spec, sigma_armington=0.9, sigma_armington_falling=3.0)
    e_assim = cge.calibrate(sam, assim)
    sam_simples = toy._frame(toy._BASE)
    e_simples = cge.calibrate(sam_simples, toy.spec(export_rule="cet", sigma_armington=0.9,
                                                    sigma_armington_falling=3.0))
    # Com exportações em proporção fixa a conta corrente só se ajusta pelo Armington, e uma
    # elasticidade abaixo de 1 deixa o câmbio mal determinado: aqui as duas ficam acima de 1.
    e_fixas = cge.calibrate(sam_simples, toy.spec(sigma_armington=1.5,
                                                  sigma_armington_falling=4.0))
    e_um = cge.calibrate(sam_simples, toy.spec(export_rule="cet", sigma_armington=0.9,
                                               sigma_armington_falling={"c_ind": 3.0}))
    casos |= {
        "assim_ano_base": caso("assim_ano_base", sam_simples, e_simples, cge.Scenario()),
        "assim_sobe": caso("assim_sobe", sam_simples, e_simples,
                           cge.Scenario(world_price=com(c_ind=1.25))),
        "assim_desce": caso("assim_desce", sam_simples, e_simples,
                            cge.Scenario(world_price=com(c_ind=0.75))),
        "assim_opostos": caso("assim_opostos", sam_simples, e_simples,
                              cge.Scenario(world_price=com(c_agr=1.3, c_ind=0.8))),
        "assim_so_um_bem": caso("assim_so_um_bem", sam_simples, e_um,
                                cge.Scenario(world_price=com(c_agr=0.7, c_ind=0.75))),
        "assim_impostos_sobe": caso("assim_impostos_sobe", sam, e_assim,
                                    cge.Scenario(world_price=com(c_ind=1.25))),
        "assim_impostos_desce": caso("assim_impostos_desce", sam, e_assim,
                                     cge.Scenario(world_price=com(c_ind=0.75),
                                                  export_tax=te0 * 2)),
        "assim_fixas_sobe": caso("assim_fixas_sobe", sam_simples, e_fixas,
                                 cge.Scenario(world_price=com(c_ind=1.25))),
        "assim_fixas_desce": caso("assim_fixas_desce", sam_simples, e_fixas,
                                  cge.Scenario(world_price=com(c_ind=0.75))),
    }
    # Famílias CDE: substituição e elasticidades-renda diferentes por bem.
    eta = {"c_agr": 0.6, "c_ind": 1.2}
    e_cde = cge.calibrate(sam_simples, toy.spec(export_rule="cet", household_demand="cde",
                                                cde_substitution={"c_agr": 0.3, "c_ind": 0.6}),
                          income_elasticities=eta)
    e_cde_imp = cge.calibrate(sam, dataclasses.replace(
        econ.spec, household_demand="cde", cde_substitution=0.5), income_elasticities=eta)
    for e in (e_cde, e_cde_imp):
        INCOME_ELASTICITIES[id(e)] = eta
    casos |= {
        "cde_ano_base": caso("cde_ano_base", sam_simples, e_cde, cge.Scenario()),
        "cde_oferta": caso("cde_oferta", sam_simples, e_cde,
                           cge.Scenario(factor_supply=e_cde.fac_supply * [1.1, 0.95])),
        "cde_preco_mundial": caso("cde_preco_mundial", sam_simples, e_cde,
                                  cge.Scenario(world_price=com(c_agr=1.3, c_ind=0.8))),
        "cde_com_impostos": caso("cde_com_impostos", sam, e_cde_imp,
                                 cge.Scenario(export_tax=np.zeros(nc), tariff=econ.tm0 * 2)),
    }
    pior = max(r["pior"] for r in casos.values())
    if all(r["ok"] for r in casos.values()):
        print(f"OK: {len(casos)} casos, pior desvio relativo {pior:.1e}")
        return 0
    print(f"FALHOU: pior desvio relativo {pior:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
