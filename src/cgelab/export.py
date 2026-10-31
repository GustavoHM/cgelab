"""Exportar uma economia calibrada para uma réplica em JuMP, resolvida SIMULTANEAMENTE.

Por que isto vale a pena, e não é só conveniência. O :mod:`cgelab.solve` resolve o equilíbrio por
ANINHAMENTO: um Newton externo sobre poucas incógnitas e, dentro dele, um ponto fixo sobre o preço
de entrega, duas somas em forma fechada — a série infinita da reciclagem do gravame e a relação
afim entre poupança e arrecadação — e uma inversão de Leontief para a absorção. Cada uma dessas
quatro é uma dedução minha. Em JuMP nada é aninhado, somado ou invertido à mão: cada bloco vira uma
RESTRIÇÃO do sistema, e o resolvedor acha o ponto fixo. Um erro na minha álgebra não pode, por
conseguinte, ser reproduzido do outro lado por construção.

O que é gerado
--------------
``to_jump`` escreve dois arquivos:

* ``<nome>_data.json`` — a economia calibrada, o fechamento e o cenário, já com os coeficientes
  EFETIVOS (produtividade embutida) e as participações de reciclagem calculadas, de modo que o lado
  Julia não precisa reproduzir nenhuma decisão de modelagem;
* ``<nome>.jl`` — o modelo JuMP. É um arquivo FIXO, o mesmo para qualquer economia: ele lê o JSON e
  monta as restrições genericamente. Escrever um ``.jl`` por modelo daria um artefato para auditar
  a cada exercício; assim há um só, verificado uma vez.

Para rodar::

    julia --project=. -e 'using Pkg; Pkg.add(["JuMP","Ipopt","JSON3"])'
    julia --project=. modelo.jl modelo_data.json saida.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .calibrate import Economy
from .closure import Closure
from .mobility import period_endowments
from .segments import period_groups
from .scenario import Scenario, investment_shares
from .solve import _marginal_budget_shares, _recycle_shares, _trade_curves

__all__ = ["to_jump", "replica_data", "JUMP_TEMPLATE"]


def _l(a):
    return np.asarray(a, dtype=float).tolist()


def _instrument(base, scenario_value, n, name):
    """Scenario rate when the base year calibrated one; otherwise a vector of zeros."""
    if scenario_value is not None and base is None:
        raise ValueError(
            f"Scenario.{name} is read only when that instrument was calibrated.")
    if base is None:
        return np.zeros(n)
    if scenario_value is None:
        return base
    value = np.asarray(scenario_value, dtype=float)
    if value.shape != (n,):
        raise ValueError(f"{name} must have length {n}, got shape {value.shape}")
    return value


def to_jump(econ: Economy, closure: Closure | None = None, scenario: Scenario | None = None, *,
            path: str | Path = "modelo_cge.jl", data_path: str | Path | None = None) -> dict:
    """Escreve a réplica em JuMP e o seu JSON de dados. Devolve o dicionário exportado.

    Os coeficientes vão EFETIVOS — a produtividade já dividida para dentro de ``io``, ``vunit`` e
    ``aunit`` —, porque a produtividade Hicks-neutra escala toda exigência de insumo e não apenas o
    custo unitário. Exportar os coeficientes crus e a produtividade em separado convidaria o lado
    Julia a aplicá-la só aos preços, que é exatamente o erro que faz o resíduo de Walras crescer ao
    longo de uma trajetória.
    """
    dados = replica_data(econ, closure, scenario)
    path = Path(path)
    data_path = Path(data_path) if data_path is not None else path.with_name(
        path.stem + "_data.json")
    path.write_text(JUMP_TEMPLATE, encoding="utf-8")
    data_path.write_text(json.dumps(dados), encoding="utf-8")
    return dados


def replica_data(econ: Economy, closure: Closure | None = None,
                 scenario: Scenario | None = None, *, who: str = "to_jump",
                 complementarity: bool = False) -> dict:
    """Os dados da réplica simultânea: o que ``to_jump`` escreve no JSON e o backend Pyomo lê.

    ``who`` nomeia quem recusa o que a réplica não escreve, nas mensagens de erro. Com
    ``complementarity=True`` (o backend PATH) os produtores de menor custo, os limites de
    capacidade, as cotas, os limites de preço e o piso salarial entram nos dados em vez de serem
    recusados; o cenário já foi conferido por ``solve_period``.
    """
    closure = closure or Closure()
    sc = scenario or Scenario()
    dados = _replica_data(econ, closure, sc, who, complementarity)
    if complementarity:
        dados.update(_complementarity_data(econ, closure, sc))
    return dados


def _complementarity_data(econ: Economy, closure: Closure, sc: Scenario) -> dict:
    """As condições de complementaridade, com índices base 0, como ``solve_period`` as lê."""
    ceil = {econ.commodity(k): float(v) for k, v in sc.price_ceiling.items()}
    floor = {econ.commodity(k): float(v) for k, v in sc.price_floor.items()}
    return dict(
        least_cost=[dict(commodity=int(i), makers=[int(j) for j in makers])
                    for i, makers in econ.least_cost.items()],
        capacity=[[econ.activity(k), float(v)] for k, v in sc.capacity.items()],
        capacity_owner=(econ.factor(sc.capacity_owner) if sc.capacity else None),
        import_quota=[[econ.commodity(k), float(v)] for k, v in sc.import_quota.items()],
        quota_rent_to=(None if not sc.import_quota or sc.quota_rent_to == "government"
                       else econ.factor(sc.quota_rent_to)),
        price_ceiling=[[i, b] for i, b in ceil.items()],
        price_floor=[[i, b, float(max(econ.A0[i], 1.0))] for i, b in floor.items()],
    )


def _refuse_complementarity(econ: Economy, closure: Closure, sc: Scenario, who: str) -> None:
    """A réplica quadrada não escreve condições de complementaridade."""
    if sc.capacity:
        raise ValueError(
            f"{who} does not write capacity limits (Scenario.capacity). A limit with "
            "its rent is a complementarity condition, which the replica's square system "
            "cannot hold.")
    if sc.import_quota:
        raise ValueError(
            f"{who} does not write import quotas (Scenario.import_quota). A quota with its "
            "premium is a complementarity condition, which the replica's square system "
            "cannot hold.")
    if sc.price_ceiling or sc.price_floor:
        raise ValueError(
            f"{who} does not write price bounds (Scenario.price_ceiling, price_floor). A "
            "bound with its subsidy or purchases is a complementarity condition, which the "
            "replica's square system cannot hold.")
    if closure.labour == "wage_floor":
        raise ValueError(
            f"{who} does not write closure.labour='wage_floor'. A wage floor is a "
            "complementarity condition, which the replica's square system cannot hold.")
    if econ.least_cost:
        raise ValueError(
            f"{who} does not write least-cost producers (ModelSpec.least_cost). Their "
            "complementarity conditions need a complementarity solver; exporting fixed "
            "market shares instead would describe another model.")


def _replica_data(econ: Economy, closure: Closure, sc: Scenario, who: str,
                  complementarity: bool) -> dict:
    if not complementarity:
        _refuse_complementarity(econ, closure, sc, who)
    if econ.regional is not None and len(econ.regional.regions) > 1:
        raise ValueError(
            f"{who} does not write the interregional nest. Exporting the "
            "regional accounts as ordinary commodities would drop the origin "
            "CES and the replica would be a different model.")
    nf, nc, nh, na = econ.n_fac, econ.n_com, econ.n_hh, econ.n_act
    num = econ.numeraire_index
    lab = (econ.factor(closure.labour_factor)
           if closure.labour_factor in econ.factors else None)
    if closure.labour != "full_employment" and (lab is None or lab == num):
        raise ValueError(
            f"closure.labour={closure.labour!r} cannot use {closure.labour_factor!r} "
            f"as the numeraire. That market is left to Walras' law, so pinning or "
            f"linking its wage would turn the check into an identity or a leak. "
            f"Set ModelSpec.numeraire_factor to another factor.")
    if closure.labour != "full_employment" and lab in econ.mobility:
        raise ValueError(
            f"closure.labour={closure.labour!r} on {closure.labour_factor!r}, which is not "
            f"perfectly mobile; only 'full_employment' is available for it.")
    value0 = (econ.hh_budget * np.asarray(econ.hh_spend0, float)[:, None]).sum(axis=0)
    pq0 = 1.0 + econ.ptax
    q_laspeyres = value0 / np.maximum(pq0, 1e-12)

    supply, endow = period_endowments(econ, sc)
    facshare = (econ.hh_facshare if sc.factor_ownership is None
                else np.asarray(sc.factor_ownership, float))
    grupos = period_groups(econ, closure, sc, supply, facshare)
    if closure.labour != "full_employment" and any(lab in g.group.index for g in grupos):
        raise ValueError(
            f"closure.labour={closure.labour!r} on {closure.labour_factor!r}, which is a "
            f"labour segment; make it rigid in Closure.labour_markets instead.")
    # A renda de um grupo de segmentos segue os trabalhadores de cada domicílio, não a
    # propriedade de cada segmento: as colunas do grupo saem da matriz de propriedade.
    facshare_ef = np.array(facshare, dtype=float)
    for g in grupos:
        facshare_ef[:, g.group.index] = 0.0
    tfp = np.ones(na) if sc.productivity is None else np.asarray(sc.productivity, float)
    ptax_a = econ.ptax if sc.product_tax is None else np.asarray(sc.product_tax, float)
    ptax_f = ptax_a if sc.product_tax_firm is None else np.asarray(sc.product_tax_firm, float)
    world_p = np.ones(nc) if sc.world_price is None else np.asarray(sc.world_price, float)
    W = np.zeros((nh, nc)) if sc.consumer_wedge is None else np.asarray(sc.consumer_wedge, float)
    Wg = (np.zeros((nh, nc)) if sc.government_wedge is None
          else np.asarray(sc.government_wedge, float))
    transfers = np.zeros(nh) if sc.transfers is None else np.asarray(sc.transfers, float)
    world_pe = np.ones(nc) if sc.export_price is None else np.asarray(sc.export_price, float)
    fac_row = (econ.zeros_if_none("fac_row_share", nf) if sc.foreign_factor_share is None
               else np.asarray(sc.foreign_factor_share, float))

    act_scale = np.ones(na)
    for nome, mult in sc.activity_scale.items():
        j = econ.activity(nome) if isinstance(nome, str) else nome
        act_scale[j] = float(mult)
    exog = econ.own_account.copy()
    X_exog = econ.X0 * act_scale * exog

    # Coeficientes CONSTANTES da demanda. `budget/(1+W+Wg)` e `beta/(1+W+Wg)` não dependem de
    # nenhuma variável, e pré-computá-los colapsa o bloco de famílias de n_hh x n_com expressões
    # não lineares para n_com somas lineares. Não é otimização cosmética: com a forma expandida o
    # Ipopt não converge a frio numa MCS aplicada de algumas centenas de contas, e com esta converge.
    Wt = W + Wg
    denom = np.where(1.0 + Wt > 0, 1.0 + Wt, 1.0)
    beta_les = _marginal_budget_shares(econ)
    coef = (beta_les if econ.spec.household_demand == "les" else econ.hh_budget) / denom
    g_cap = ((beta_les if econ.spec.household_demand == "les" else econ.hh_budget)
             * np.where(1.0 + Wt > 0, W / denom, 0.0)).sum(axis=1)
    g_gov = ((beta_les if econ.spec.household_demand == "les" else econ.hh_budget)
             * np.where(1.0 + Wt > 0, Wg / denom, 0.0)).sum(axis=1)
    gamma_sum = econ.hh_committed.sum(axis=0)
    gam_W = (econ.hh_committed * W).sum(axis=0)
    gam_Wg = (econ.hh_committed * Wg).sum(axis=0)

    # Curvas de país grande. JSON não tem infinito: elasticidade 0 marca o tomador de preço, e
    # só as mercadorias com volume no ano-base entram, como no cgelab.
    eta_x, psi_m, curve_x, curve_m = _trade_curves(econ)
    eta_json = np.zeros(nc)
    eta_json[curve_x] = eta_x[curve_x]
    psi_json = np.zeros(nc)
    psi_json[curve_m] = psi_m[curve_m]

    io_eff = econ.io / tfp[None, :]
    # Preço do carbono: por unidade de produto nas atividades (vai no imposto específico) e por
    # unidade comprada nas famílias. Em unidades do numerário.
    if sc.carbon_price is not None and econ.emissions is None:
        raise ValueError("Scenario.carbon_price needs an emissions account: see add_emissions")
    carbono = 0.0 if sc.carbon_price is None else float(sc.carbon_price) * closure.numeraire
    ninho = econ.energy_nest
    emask = ninho.mask if ninho is not None else np.zeros(nc, dtype=bool)
    if carbono:
        # Com ninho de energia o carbono dos combustíveis vai no preço deles; o resto fica por
        # unidade de produto, como no cgelab.
        ef_fixo = np.where(emask[:, None], 0.0, econ.emissions.intermediate)
        carb_atv = carbono * ((ef_fixo * io_eff).sum(axis=0) + econ.emissions.process)
        carb_comb = carbono * np.where(emask[:, None], econ.emissions.intermediate, 0.0)
    else:
        carb_atv, carb_comb = np.zeros(na), np.zeros((nc, na))
    carb_fam = carbono * econ.emissions.household if carbono else np.zeros(nc)
    vunit_eff = econ.vunit / tfp
    aunit_eff = econ.aunit / tfp[None, :]

    nu = closure.numeraire
    gov_q = (econ.gov_dem / (1.0 + econ.ptax) if closure.government == "real_fixed" else None)

    dados = dict(
        nome=str(getattr(econ.layout, "activities", ["?"])[:0] or "cge"),
        na=na, nc=nc, nh=nh, nf=nf, numeraire_index=num + 1, numeraire=float(nu),
        acts=econ.acts, coms=econ.coms, hhs=econ.hhs, factors=econ.factors,
        # tecnologia, já com a produtividade embutida
        io=_l(io_eff), msh=_l(econ.msh), theta=_l(econ.theta), vunit=_l(vunit_eff),
        aunit=_l(aunit_eff), leontief=[bool(b) for b in econ.leontief],
        # Cobb-Douglas é o limite σ = 1 da CES, que é como o modelo JuMP o escreve. Exportar
        # o `sigma_va` da especificação nesse caso resolveria outra função de produção.
        sigma_va=_l(np.ones(na) if econ.spec.value_added == "cobb_douglas" else econ.sigma_va),
        sigma_arm=_l(econ.sigma_arm),
        X0=_l(econ.X0), Q0=_l(econ.Q0), A0=_l(econ.A0),
        # comércio e margens
        dom_share=_l(econ.dom_share), arm_share=_l(econ.arm_share),
        exp_share=_l(econ.exp_share), mq=_l(econ.mq), world_price=_l(world_p),
        cet=bool(econ.spec.export_rule == "cet"), export_price=_l(world_pe),
        sigma_cet=_l(econ.sigma_cet if econ.sigma_cet is not None
                     else econ.spec.sigma_cet_vector(econ.coms)),
        export_demand=_l(eta_json), import_supply=_l(psi_json),
        exp0=_l(econ.exp0), imp0=_l(econ.imp0),
        # tributos
        ptax_a=_l(ptax_a), ptax_f=_l(ptax_f), ptax_base=_l(econ.ptax),
        atax=_l(econ.atax * nu + carb_atv), carbon_household=_l(carb_fam),
        tm=_l(_instrument(econ.tm0, sc.tariff, nc, "tariff")),
        tm0=_l(np.zeros(nc) if econ.tm0 is None else econ.tm0),
        tauz=_l(_instrument(econ.tauz0, sc.output_tax, nc, "output_tax")),
        tauz0=_l(np.zeros(nc) if econ.tauz0 is None else econ.tauz0),
        # famílias
        hh_budget=_l(econ.hh_budget), beta_les=_l(beta_les),
        demand_coef=_l(coef), g_cap=_l(g_cap), g_gov=_l(g_gov),
        gamma_sum=_l(gamma_sum), gam_W=_l(gam_W), gam_Wg=_l(gam_Wg), Wt=_l(Wt),
        hh_committed=_l(econ.hh_committed), hh_savrate=_l(econ.hh_savrate),
        facshare=_l(facshare_ef), hh_direct=_l(econ.hh_direct), transfers=_l(transfers),
        les=bool(econ.spec.household_demand == "les"),
        # política
        wedge=_l(W), wedge_gov=_l(Wg),
        recycle_share=_l(_recycle_shares(econ, sc, facshare, supply)),
        pop_share=_l(econ.pop_share),
        gov_absorbs=float(sc.government_absorbs),
        # blocos exógenos
        fac_supply=_l(supply), gov_dem=_l(econ.gov_dem), gov_q=(None if gov_q is None
                                                                else _l(gov_q)),
        inv_dem=_l(econ.inv_dem),
        inv_w=_l(investment_shares(econ, sc)),
        inv_q_fixed=_l(investment_shares(econ, sc)
                       * (float(econ.inv_dem.sum()) if sc.investment_level is None
                          else float(sc.investment_level))
                       / (1.0 + econ.ptax)),
        act_scale=_l(act_scale), own_account=[bool(b) for b in exog],
        X_exog=_l(X_exog), exog_intermediate=_l(io_eff @ X_exog),
        trade_balance0=float(econ.trade_balance0 * nu),
        # a meta externa é a poupança externa do período, IED incluído; em dólares ela não
        # acompanha o numerário, porque o câmbio já faz isso
        foreign_sav=float((econ.foreign_sav + float(sum(sc.fdi.values())))
                          * (1.0 if closure.foreign_saving_unit == "foreign_currency" else nu)),
        foreign_saving_unit=closure.foreign_saving_unit,
        gov_savrate=float(econ.gov_savrate),
        government_target=closure.government_target,
        government_adjusts=closure.government_adjusts,
        government_target_value=float(
            ({"saving": econ.gov_saving0, "revenue": econ.gov_revenue0}.get(
                closure.government_target, 0.0)
             if closure.government_target_value is None
             else closure.government_target_value) * nu),
        gov_w=_l(econ.gov_dem / max(float(econ.gov_dem.sum()), 1e-300)),
        # instituições: imposto direto, transferências (em unidades do numerário) e a parte da
        # renda de cada fator que vai ao governo e ao exterior
        hh_dtax=_l(econ.zeros_if_none("hh_dtax", nh)),
        gov_transfer=_l(econ.zeros_if_none("hh_gov_transfer", nh) * nu),
        remit_net=_l(econ.zeros_if_none("hh_remit_net", nh) * nu),
        fac_gov_share=_l(econ.zeros_if_none("fac_gov_share", nf)), fac_row_share=_l(fac_row),
        # fechamento
        investment=closure.investment, government=closure.government,
        foreign=closure.foreign, exchange_rate=float(closure.exchange_rate),
        labour=closure.labour,
        labour_index=0 if lab is None else int(lab) + 1,
        real_wage_target=1.0 if closure.real_wage is None else float(closure.real_wage),
        wage_curve_elasticity=float(closure.wage_curve_elasticity),
        natural_unemployment=float(closure.natural_unemployment),
        labour_supply_elasticity=float(closure.labour_supply_elasticity),
        q_laspeyres=_l(q_laspeyres), pq0=_l(pq0),
        # níveis do ano-base, só para o ponto de partida
        hh_income0=_l((econ.hh_income0 if econ.hh_gross0 is None else econ.hh_gross0) * nu),
        hh_spend0=_l((econ.hh_spend0 if econ.hh_spend0 is not None
                      else econ.hh_income0 * (1 - econ.hh_savrate)) * nu),
        scenario_name=sc.name,
        # ninho valor adicionado-energia
        nest=ninho is not None,
        energy_mask=[bool(b) for b in emask],
        vae_q=_l(ninho.vae_unit / tfp if ninho is not None else np.zeros(na)),
        e_share=_l(ninho.share if ninho is not None else np.zeros(na)),
        theta_e=_l(ninho.theta if ninho is not None else np.zeros((nc, na))),
        sigma_vae=_l(ninho.sigma_vae if ninho is not None else np.zeros(na)),
        sigma_energy=_l(ninho.sigma_energy if ninho is not None else np.zeros(na)),
        io_fixed=_l(np.where(emask[:, None], 0.0, io_eff)),
        carb_fuel=_l(carb_comb),
        nest_mode=ninho.mode if ninho is not None else "value_added",
        capital_index=int(ninho.capital) + 1 if ninho is not None else 0,
        nest_free=[bool(b) for b in (ninho.free if ninho is not None and ninho.free is not None
                                      else np.zeros(na, dtype=bool))],
        top_share=_l(ninho.top_share if ninho is not None and ninho.top_share is not None
                     else np.zeros((nf, na))),
        ke_share=_l(ninho.ke_share if ninho is not None and ninho.ke_share is not None
                    else np.zeros(na)),
        k_share=_l(ninho.k_share if ninho is not None and ninho.k_share is not None
                   else np.zeros(na)),
        sigma_ke=_l(ninho.sigma_ke if ninho is not None and ninho.sigma_ke is not None
                    else np.zeros(na)),
        sigma_top=_l(ninho.sigma_top if ninho is not None and ninho.sigma_top is not None
                     else np.zeros(na)),
        # safras de capital (putty-clay): o valor adicionado é uma CES de dois feixes, o velho
        # (sem o capital novo) e o novo (sem o capital velho)
        vintage=econ.vintage is not None,
        vintage_share_old=_l(econ.vintage.share_old if econ.vintage is not None
                             else np.zeros(na)),
        vintage_theta_old=_l(econ.vintage.theta_old if econ.vintage is not None
                             else np.zeros((nf, na))),
        vintage_theta_new=_l(econ.vintage.theta_new if econ.vintage is not None
                             else np.zeros((nf, na))),
        vintage_sigma_old=_l(econ.vintage.sigma_old if econ.vintage is not None
                             else np.zeros(na)),
        vintage_sigma_new=_l(econ.vintage.sigma_new if econ.vintage is not None
                             else np.zeros(na)),
        vintage_sigma_top=_l(econ.vintage.sigma_top if econ.vintage is not None
                             else np.zeros(na)),
        # mobilidade imperfeita: por fator, a forma, a elasticidade, as atividades usuárias, as
        # parcelas de cada pool no ano-base e a dotação de cada pool neste cenário
        mobility=[dict(factor=int(i) + 1, form=m.form, omega=float(m.omega),
                       users=[bool(b) for b in m.users], delta=_l(m.delta),
                       endowment=_l(endow[i]))
                  for i, m in sorted(econ.mobility.items())],
        # segmentos de trabalho: por grupo, os índices (base 1), os rígidos e seus alvos, os
        # flexíveis, o informal, Harris–Todaro, λ, a fila e a força de trabalho de cada domicílio
        segments=[dict(index=[int(i) + 1 for i in g.group.index],
                       rigid=[int(i) + 1 for i in g.rigid], target=_l(g.target),
                       flexible=[int(i) + 1 for i in g.flexible],
                       informal=int(g.informal) + 1, harris_todaro=bool(g.harris_todaro),
                       lam=float(g.lam), U0=float(g.U0), N=float(g.N),
                       shares=_l(g.shares), workforce=_l(g.workforce), queue=_l(g.queue),
                       # transformação entre segmentos: os domicílios com trabalho no grupo
                       # são os pools, com as parcelas de oferta do ano-base
                       transform=g.transform is not None,
                       form=g.transform.form if g.transform is not None else "cet",
                       omega=float(g.transform.omega) if g.transform is not None else 0.0,
                       active=([bool(b) for b in g.active] if g.active is not None
                               else [True] * econ.n_hh),
                       delta=(_l(g.transform.delta) if g.transform is not None
                              else _l(np.zeros((1, len(g.group.index))))),
                       clearing=[int(i) + 1 for i in g.clearing])
                  for g in grupos],
    )
    return dados


JUMP_TEMPLATE = r"""# modelo_cge.jl — réplica em JuMP de uma economia calibrada pelo cgelab.
#
# GERADO por cgelab.export.to_jump. Arquivo FIXO: serve a qualquer economia, lendo o JSON.
#
# O cgelab resolve este mesmo equilíbrio por ANINHAMENTO — um Newton sobre poucas incógnitas e,
# dentro dele, um ponto fixo sobre o preço de entrega, duas somas em forma fechada e uma inversão
# de Leontief. Aqui nada é aninhado: cada um desses blocos é uma restrição, e o Ipopt acha o ponto
# fixo. É por isso que a comparação verifica a ÁLGEBRA e não apenas a aritmética.
#
# O mercado do fator NUMERÁRIO fica deliberadamente de fora, exatamente como no cgelab, de modo que
# o seu resíduo continue a ser verificação pela lei de Walras e não identidade imposta.
#
#   uso:  julia --project=. modelo_cge.jl dados.json [saida.json]

using JuMP, Ipopt, JSON3, Printf

mat(x) = permutedims(reduce(hcat, [Float64.(r) for r in x]))
vecf(x) = Float64.(collect(x))

const ARQ = length(ARGS) >= 1 ? ARGS[1] : "modelo_cge_data.json"
const SAIDA = length(ARGS) >= 2 ? ARGS[2] : "resultado_jump.json"
# Terceiro argumento OPCIONAL: um JSON de partida quente, no formato da saída. Serve a uma
# varredura de cenários — e, na verificação, a separar erro de FORMULAÇÃO de dificuldade de
# convergência: se o modelo assenta com violação nula partindo da resposta conhecida, as equações
# estão certas e o que falta é caminho até lá.
const QUENTE = length(ARGS) >= 3 ? JSON3.read(read(ARGS[3], String)) : nothing
d = JSON3.read(read(ARQ, String))

const na, nc, nh, nf = d.na, d.nc, d.nh, d.nf
const NUM = d.numeraire_index                 # índice 1-based do fator numerário
const nu = Float64(d.numeraire)
const LAB = Int(d.labour_index)               # 1-based; 0 se o fator de trabalho não existe
const LABOUR = String(d.labour)
const labour_slack = LABOUR == "fixed_real_wage" || LABOUR == "wage_curve"
const real_wage_target = Float64(d.real_wage_target)
const beta_w = Float64(d.wage_curve_elasticity)
const u0 = Float64(d.natural_unemployment)
const eps_ls = Float64(d.labour_supply_elasticity)

io = mat(d.io); msh = mat(d.msh); theta = mat(d.theta); aunit = mat(d.aunit); mq = mat(d.mq)
vunit = vecf(d.vunit); leo = collect(Bool, d.leontief)
sva = vecf(d.sigma_va); sarm = vecf(d.sigma_arm)
X0 = vecf(d.X0); Q0 = vecf(d.Q0); A0 = vecf(d.A0)
domsh = vecf(d.dom_share); armsh = vecf(d.arm_share); expsh = vecf(d.exp_share)
wp = vecf(d.world_price)
const CET = Bool(d.cet)
wpe = vecf(d.export_price); scet = vecf(d.sigma_cet)
# Curvas de país grande: elasticidade positiva põe a mercadoria numa curva, zero a deixa
# tomadora de preço. O preço mundial é então uma variável, deslocada pelo cenário.
eta_x = vecf(d.export_demand); psi_m = vecf(d.import_supply)
E0 = vecf(d.exp0); M0 = vecf(d.imp0)
dtax = vecf(d.hh_dtax); gov_tr = vecf(d.gov_transfer); remit = vecf(d.remit_net)
gsh = vecf(d.fac_gov_share); rsh = vecf(d.fac_row_share)
ptax_a = vecf(d.ptax_a); ptax_f = vecf(d.ptax_f); ptax_b = vecf(d.ptax_base); atax = vecf(d.atax)
tm = vecf(d.tm); tm0 = vecf(d.tm0); tauz = vecf(d.tauz); tauz0 = vecf(d.tauz0)
hh_budget = mat(d.hh_budget); beta_les = mat(d.beta_les); gamma = mat(d.hh_committed)
coef = mat(d.demand_coef); Wt = mat(d.Wt)
g_cap = vecf(d.g_cap); g_gov = vecf(d.g_gov)
gamma_sum = vecf(d.gamma_sum); gam_W = vecf(d.gam_W); gam_Wg = vecf(d.gam_Wg)
savrate = vecf(d.hh_savrate); facshare = mat(d.facshare); hh_direct = mat(d.hh_direct)
transfers = vecf(d.transfers); les = Bool(d.les); popsh = vecf(d.pop_share)
W = mat(d.wedge); Wg = mat(d.wedge_gov); alpha = vecf(d.recycle_share)
gov_absorbs = Float64(d.gov_absorbs)
supply = vecf(d.fac_supply); inv_dem = vecf(d.inv_dem); inv_q_fix = vecf(d.inv_q_fixed)
inv_w = vecf(d.inv_w)
q_lasp = vecf(d.q_laspeyres); pq0_lasp = vecf(d.pq0)
hh_income0 = vecf(d.hh_income0); hh_spend0 = vecf(d.hh_spend0)
act_scale = vecf(d.act_scale); X_exog = vecf(d.X_exog); exog_int = vecf(d.exog_intermediate)
own = collect(Bool, d.own_account)
tb0 = Float64(d.trade_balance0); fsav0 = Float64(d.foreign_sav)
gov_dem = vecf(d.gov_dem)

Bmat = io * msh

@printf("cenário=%s  %d atividades, %d mercadorias, %d domicílios, %d fatores\n",
        d.scenario_name, na, nc, nh, nf)
@printf("fechamento: investimento=%s  governo=%s  externo=%s  numerário=%s\n",
        d.investment, d.government, d.foreign, d.factors[NUM])

m = Model(Ipopt.Optimizer)
set_silent(m)
# A bandeira do Ipopt NÃO é o veredito: com objetivo nulo o problema é de pura viabilidade, e o
# teste de convergência sinaliza dificuldade mesmo assentado na solução exata. Pede-se 1e-10, que
# ele certifica, e mede-se depois a violação máxima diretamente.
set_optimizer_attribute(m, "tol", 1e-10)
set_optimizer_attribute(m, "constr_viol_tol", 1e-10)
set_optimizer_attribute(m, "acceptable_tol", 1e-8)
set_optimizer_attribute(m, "max_iter", 8000)
# As magnitudes de uma MCS vão de 1e-3 a 1e6; sem escalonamento o teste de convergência sinaliza
# dificuldade num problema que está resolvido.
set_optimizer_attribute(m, "nlp_scaling_method", "gradient-based")
set_optimizer_attribute(m, "nlp_scaling_max_gradient", 100.0)

# Sem partida quente o arranque é DELIBERADAMENTE deslocado da resposta: arrancar nela verificaria
# que a solução do Python é ponto fixo, e não que este modelo a encontra por conta própria.
q(campo, i, padrao) = QUENTE === nothing ? padrao : Float64(QUENTE[campo][i])
q0(campo, padrao) = QUENTE === nothing ? padrao : Float64(QUENTE[campo])

@variable(m, w[f = 1:nf] >= 1e-8, start = q("w", f, f == NUM ? nu : 0.85 * nu))
fix(w[NUM], nu; force = true)
# Mobilidade imperfeita: o fator tem um preço em cada atividade que o usa. `wf(f, j)` é o preço
# que a atividade j paga pelo fator f; para um fator móvel, o mesmo `w[f]` em toda parte.
const MOB = d.mobility
ismob = falses(nf)
wm = Dict{Tuple{Int,Int},VariableRef}()
for mb in MOB
    f = Int(mb.factor)
    ismob[f] = true
    us = collect(Bool, mb.users)
    for j in 1:na
        us[j] || continue
        wm[(f, j)] = @variable(m, lower_bound = 1e-8,
                               start = QUENTE === nothing ? 0.85 * nu : Float64(QUENTE["w_act"][f][j]))
    end
end
wf(f, j) = haskey(wm, (f, j)) ? wm[(f, j)] : w[f]
# Emprego pago. Sob salário real fixo ou curva salarial a renda usa a DEMANDA, não a força de
# trabalho: pagar o desempregado fabrica renda e o resíduo de Walras deixa de ser verificação.
if labour_slack
    @variable(m, emp >= 0, start = supply[LAB])
end
@variable(m, pva[j = 1:na] >= 0, start = 1.0)
@variable(m, pa[j = 1:na] >= 1e-8, start = q("pa", j, 1.0))
@variable(m, pd[c = 1:nc] >= 1e-8, start = q("pd", c, 1.0))
@variable(m, arm[1:nc] >= 1e-8, start = 1.0)
# Os demais arranques são os níveis do ANO-BASE, como o GAMS inicializa um modelo: não a resposta
# do Python, que só a partida quente usa. Renda de 1.0 numa MCS em que ela vale milhares, ou
# participações de Armington chutadas, bastavam para o Ipopt declarar inviável, numa MCS aplicada,
# um cenário que, a partir do ano-base, ele resolve.
@variable(m, dsh[c = 1:nc] >= 0, start = domsh[c])
@variable(m, mshm[c = 1:nc] >= 0, start = 1 - domsh[c])
@variable(m, pbase[c = 1:nc] >= 1e-8, start = q("pbase", c, 1.0))
@variable(m, Aabs[i = 1:nc] >= 0, start = q("Aabs", i, A0[i]))
@variable(m, Q[i = 1:nc] >= 0, start = q("Q", i, Q0[i]))
@variable(m, Xraw[i = 1:na], start = q("X", i, X0[i]))
@variable(m, hh_income[h = 1:nh], start = q("hh_income", h, hh_income0[h]))
@variable(m, spend[h = 1:nh], start = q("hh_com_spend", h, hh_spend0[h]))
@variable(m, savings, start = q0("savings", sum(inv_dem)))
@variable(m, uplift_rev, start = q0("uplift_rev", 0.0))

# incógnitas que cada fechamento acrescenta
if d.foreign == "fixed_saving"
    @variable(m, er >= 1e-8, start = q0("er", 1.15))
    # Em dólares, a poupança externa em moeda doméstica é o câmbio vezes a meta.
    fsav = d.foreign_saving_unit == "foreign_currency" ? er * fsav0 : fsav0
else
    er_fixo = Float64(d.exchange_rate)
    @expression(m, er, er_fixo)
    @variable(m, fsav, start = fsav0)
end
# Preços mundiais em moeda estrangeira. Fora de uma curva são os do cenário.
@variable(m, pwe[c = 1:nc] >= 1e-8, start = q("world_pe", c, wpe[c]))
@variable(m, pwm[c = 1:nc] >= 1e-8, start = q("world_pm", c, wp[c]))
for c in 1:nc
    eta_x[c] > 0 || fix(pwe[c], wpe[c]; force = true)
    psi_m[c] > 0 || fix(pwm[c], wp[c]; force = true)
end
# Meta do governo atingida por um tributo: um fator comum sobre as alíquotas dele é variável.
const GTARGET = d.government_target === nothing ? "" : String(d.government_target)
const GADJ = String(d.government_adjusts)
const TAX_LEVER = GTARGET != "" && GADJ != "spending"
if TAX_LEVER
    @variable(m, tscale >= 1e-6, start = 1.0)
else
    @expression(m, tscale, 1.0)
end
ta(c) = TAX_LEVER && GADJ == "product_tax" ? ptax_a[c] * tscale : ptax_a[c]
tf(c) = TAX_LEVER && GADJ == "product_tax" ? ptax_f[c] * tscale : ptax_f[c]
td(h) = TAX_LEVER && GADJ == "direct_tax" ? dtax[h] * tscale : dtax[h]
if d.investment == "fixed_real"
    @variable(m, levy, start = 0.0)          # poupança forçada que financia o investimento fixo
else
    @expression(m, levy, 0.0)
end

@expression(m, pq[c = 1:nc], pbase[c] * (1 + ta(c)))     # preço ao consumidor FINAL
# Índice de Laspeyres dos preços ao comprador. No ano-base vale o numerário, e o salário real é
# o salário nominal dividido por ele. Os pesos são as quantidades do ano-base.
@expression(m, P, sum(pq[c] * q_lasp[c] for c in 1:nc) / sum(pq0_lasp[c] * q_lasp[c] for c in 1:nc))
# Segmentos de trabalho. O emprego de um segmento rígido ou flexível é a demanda, uma variável
# E igualada a ela adiante; o informal emprega o resto da força de trabalho, N − ΣE − U.
# Com transformação entre segmentos, cada domicílio é um pool da CET ou da logit: oferta S[h, k]
# a cada segmento, aos salários dos segmentos; o flexível equilibra contra ela e o rígido deixa
# na fila o que não contrata.
const SEG = d.segments
Eseg = Dict{Int,VariableRef}()
Useg = Any[]
SUP = Any[]
pago_seg = Dict{Int,Any}()
for (k, g) in enumerate(SEG)
    for f in vcat(Int.(g.rigid), Int.(g.flexible))
        Eseg[f] = @variable(m, lower_bound = 0.0, start = supply[f])
    end
    contratados = vcat(Int.(g.rigid), Int.(g.flexible))
    for f in contratados
        pago_seg[f] = Eseg[f]
    end
    if Bool(g.transform)
        idx = Int.(g.index); dl = mat(g.delta); ativos = findall(collect(Bool, g.active))
        nw = vecf(g.workforce); om = Float64(g.omega); ns = length(idx)
        S = Dict{Tuple{Int,Int},Any}()
        if String(g.form) == "cet"
            Wp = @variable(m, [a = 1:length(ativos)], lower_bound = 1e-8, start = 1.0)
            for (a, h) in enumerate(ativos)
                @constraint(m, Wp[a]^(1 + om) == sum(dl[a, kk] * w[idx[kk]]^(1 + om)
                                                     for kk in 1:ns if dl[a, kk] > 0))
                for kk in 1:ns
                    dl[a, kk] > 0 || continue
                    S[(h, kk)] = @expression(m, dl[a, kk] * nw[h] * (w[idx[kk]] / Wp[a])^om)
                end
            end
        else
            Zp = @variable(m, [a = 1:length(ativos)], lower_bound = 1e-12, start = 1.0)
            for (a, h) in enumerate(ativos)
                @constraint(m, Zp[a] == sum(dl[a, kk] * w[idx[kk]]^om for kk in 1:ns if dl[a, kk] > 0))
                for kk in 1:ns
                    dl[a, kk] > 0 || continue
                    S[(h, kk)] = @expression(m, nw[h] * dl[a, kk] * w[idx[kk]]^om / Zp[a])
                end
            end
        end
        oferta_k = [@expression(m, sum(S[(h, kk)] for h in 1:nh if haskey(S, (h, kk))))
                    for kk in 1:ns]
        for f in Int.(g.clearing)
            pago_seg[f] = oferta_k[findfirst(==(f), idx)]
        end
        R = [kk for kk in 1:ns if idx[kk] in Int.(g.rigid)]
        push!(Useg, isempty(R) ? 0.0 :
                    @expression(m, sum(oferta_k[kk] - Eseg[idx[kk]] for kk in R)))
        push!(SUP, (S, oferta_k))
        continue
    end
    push!(SUP, nothing)
    if Bool(g.harris_todaro)
        push!(Useg, @variable(m, start = Float64(g.U0)))
    else
        push!(Useg, 0.0)
    end
    pago_seg[Int(g.informal)] = @expression(m, Float64(g.N) - sum(Eseg[f] for f in contratados)
                                               - Useg[k])
end
function paid_supply(f)
    if haskey(pago_seg, f)
        return pago_seg[f]
    elseif labour_slack && f == LAB
        return emp
    elseif LABOUR == "elastic_supply" && f == LAB
        return supply[f] * (w[f] / P) ^ eps_ls
    else
        return supply[f]
    end
end

# O imposto sobre o produto incide sobre TODO o produto, dentro do preço: o produto medido a valor
# com imposto do ano-base custa pd·(1+tauz)/(1+tauz0). A tarifa, do mesmo modo, entra no preço de
# importação, e a importação de Armington é medida a valor com tarifa do ano-base.
@expression(m, pgross[c = 1:nc], pd[c] * (1 + tauz[c]) / (1 + tauz0[c]))
# Preço das vendas DOMÉSTICAS e repartição do produto entre mercado interno e exportação. Sob CET
# a receita da fronteira iguala o preço bruto; sob parcelas fixas, vende-se ao preço bruto na
# proporção do ano-base.
if CET
    @variable(m, pdom[c = 1:nc] >= 1e-8, start = q("pdom", c, 1.0))
    for c in 1:nc
        if 0 < expsh[c] < 1
            @constraint(m, pgross[c]^(1 + scet[c]) == (1 - expsh[c]) * pdom[c]^(1 + scet[c])
                                                  + expsh[c] * (er * pwe[c])^(1 + scet[c]))
        else
            @constraint(m, pdom[c] == pgross[c])
        end
    end
    @expression(m, domfrac[c = 1:nc],
                expsh[c] <= 0 ? 1.0 : expsh[c] >= 1 ? 0.0 :
                (1 - expsh[c]) * (pdom[c] / pgross[c])^scet[c])
    @expression(m, expfrac[c = 1:nc],
                expsh[c] <= 0 ? 0.0 : expsh[c] >= 1 ? 1.0 :
                expsh[c] * (er * pwe[c] / pgross[c])^scet[c])
    @expression(m, pexp[c = 1:nc], er * pwe[c])
else
    @expression(m, pdom[c = 1:nc], pgross[c])
    @expression(m, domfrac[c = 1:nc], 1 - expsh[c])
    @expression(m, expfrac[c = 1:nc], expsh[c])
    @expression(m, pexp[c = 1:nc], pgross[c])
end
@expression(m, pqf[c = 1:nc], pbase[c] * (1 + tf(c)))    # preço à FIRMA (parcela não creditável)

# ---- 1. preço do valor adicionado: CES sobre fatores, Leontief onde o excedente é negativo
const VINT = Bool(d.vintage)
# Preço de uma CES de vários insumos com parcelas `sh` (que somam um): linear com elasticidade
# zero, Cobb-Douglas com um.
function cesprice!(pout, sh, prices, sg)
    k = [i for i in eachindex(sh) if sh[i] > 0]
    if sg == 0
        @constraint(m, pout == sum(sh[i] * prices[i] for i in k))
    elseif abs(sg - 1) < 1e-9
        @constraint(m, log(pout) == sum(sh[i] * log(prices[i]) for i in k))
    else
        @constraint(m, pout^(1 - sg) == sum(sh[i] * prices[i]^(1 - sg) for i in k))
    end
end
vpow(x, y, sg) = sg == 0 ? 1.0 : (x / y)^sg
if VINT
    # Duas safras: o feixe velho, CES de σ_velho sobre os fatores menos o capital novo; o novo,
    # CES de σ_va sobre os fatores menos o capital velho; e uma CES de σ_safra entre os dois.
    vso = vecf(d.vintage_share_old); vth_o = mat(d.vintage_theta_old)
    vth_n = mat(d.vintage_theta_new); vs_o = vecf(d.vintage_sigma_old)
    vs_n = vecf(d.vintage_sigma_new); vs_t = vecf(d.vintage_sigma_top)
    @variable(m, pOld[j = 1:na] >= 1e-8, start = 1.0)
    @variable(m, pNew[j = 1:na] >= 1e-8, start = 1.0)
    for j in 1:na
        cesprice!(pOld[j], vth_o[:, j], [wf(f, j) for f in 1:nf], vs_o[j])
        cesprice!(pNew[j], vth_n[:, j], [wf(f, j) for f in 1:nf], vs_n[j])
        cesprice!(pva[j], [vso[j], 1 - vso[j]], [pOld[j], pNew[j]], vs_t[j])
    end
    @expression(m, bOld[j = 1:na], vso[j] * vpow(pva[j], pOld[j], vs_t[j]))
    @expression(m, bNew[j = 1:na], (1 - vso[j]) * vpow(pva[j], pNew[j], vs_t[j]))
end
for j in 1:na
    VINT && break
    if leo[j]
        @constraint(m, pva[j] * vunit[j] == sum(aunit[f, j] * wf(f, j) for f in 1:nf))
    elseif sum(theta[:, j]) <= 0
        @constraint(m, pva[j] == 0)
    elseif abs(sva[j] - 1.0) < 1e-9
        @constraint(m, pva[j] == prod(wf(f, j)^theta[f, j] for f in 1:nf if theta[f, j] > 0))
    else
        @constraint(m, pva[j]^(1 - sva[j]) ==
                       sum(theta[f, j] * wf(f, j)^(1 - sva[j]) for f in 1:nf if theta[f, j] > 0))
    end
end

# ---- 1b. ninho valor adicionado-energia: a energia é uma CES entre combustíveis, e ela e o valor
# adicionado formam uma CES; os demais insumos ficam Leontief. Preços dos dois agregados = 1 no
# ano-base, quantidades medidas em valor do ano-base.
const NEST = Bool(d.nest)
if NEST
    emask = collect(Bool, d.energy_mask); vaeq = vecf(d.vae_q); esh = vecf(d.e_share)
    thetaE = mat(d.theta_e); svae = vecf(d.sigma_vae); sen = vecf(d.sigma_energy)
    iofix = mat(d.io_fixed); carbf = mat(d.carb_fuel); p0f = vecf(d.ptax_base) .+ 1.0
    @variable(m, pE[j = 1:na] >= 1e-8, start = 1.0)
    fuelrel(c, j) = (pqf[c] + carbf[c, j]) / p0f[c]
    for j in 1:na
        fu = [c for c in 1:nc if thetaE[c, j] > 0]
        if isempty(fu)
            @constraint(m, pE[j] == 1.0)
        elseif sen[j] == 0
            @constraint(m, pE[j] == sum(thetaE[c, j] * fuelrel(c, j) for c in fu))
        elseif abs(sen[j] - 1) < 1e-9
            @constraint(m, log(pE[j]) == sum(thetaE[c, j] * log(fuelrel(c, j)) for c in fu))
        else
            @constraint(m, pE[j]^(1 - sen[j]) == sum(thetaE[c, j] * fuelrel(c, j)^(1 - sen[j]) for c in fu))
        end
    end
    # CES de dois insumos com parcelas a, b: linear com elasticidade zero (aceita parcela negativa).
    function ces2!(pout, a, p1, b, p2, sg)
        if sg == 0
            @constraint(m, pout == a * p1 + b * p2)
        elseif abs(sg - 1) < 1e-9
            @constraint(m, log(pout) == a * log(p1) + b * log(p2))
        else
            @constraint(m, pout^(1 - sg) == a * p1^(1 - sg) + b * p2^(1 - sg))
        end
    end
    pw(x, y, sg) = sg == 0 ? 1.0 : (x / y)^sg
    const NESTMODE = String(d.nest_mode)
    if NESTMODE == "value_added"
        # A energia substitui o valor adicionado inteiro num agregado VAE.
        @variable(m, pVAE[j = 1:na] >= 1e-8, start = 1.0)
        for j in 1:na
            ces2!(pVAE[j], 1 - esh[j], pva[j], esh[j], pE[j], svae[j])
        end
        @expression(m, vaper[j = 1:na], vaeq[j] * (1 - esh[j]) * pw(pVAE[j], pva[j], svae[j]))
        @expression(m, eper[j = 1:na], vaeq[j] * esh[j] * pw(pVAE[j], pE[j], svae[j]))
        @expression(m, topcost[j = 1:na], vaeq[j] * pVAE[j])
        @expression(m, facper[f = 1:nf, j = 1:na],
                    (leo[j] ? (vunit[j] != 0 ? aunit[f, j] / vunit[j] : 0.0)
                            : (theta[f, j] > 0 ? theta[f, j] * (pva[j] / wf(f, j))^sva[j] : 0.0)) * vaper[j])
        @expression(m, fuelfix[c = 1:nc, j = 1:na], 0.0)
        nfree = falses(na)
    else
        # A energia junta-se ao capital num composto KE, que entra no ninho do valor adicionado.
        KI = Int(d.capital_index); nfree = .!collect(Bool, d.nest_free)
        ktop = mat(d.top_share); kesh = vecf(d.ke_share); ksh = vecf(d.k_share)
        sigke = vecf(d.sigma_ke); sigtop = vecf(d.sigma_top)
        @variable(m, pKE[j = 1:na] >= 1e-8, start = 1.0)
        @variable(m, pTOP[j = 1:na] >= 1e-8, start = 1.0)
        for j in 1:na
            if nfree[j]
                @constraint(m, pKE[j] == 1.0); @constraint(m, pTOP[j] == 1.0)
                continue
            end
            ces2!(pKE[j], ksh[j], wf(KI, j), 1 - ksh[j], pE[j], sigke[j])
            fs = [f for f in 1:nf if ktop[f, j] > 0]
            if sigtop[j] == 0
                @constraint(m, pTOP[j] == sum(ktop[f, j] * wf(f, j) for f in fs) + kesh[j] * pKE[j])
            elseif abs(sigtop[j] - 1) < 1e-9
                @constraint(m, log(pTOP[j]) == sum(ktop[f, j] * log(wf(f, j)) for f in fs) + kesh[j] * log(pKE[j]))
            else
                @constraint(m, pTOP[j]^(1 - sigtop[j]) == sum(ktop[f, j] * wf(f, j)^(1 - sigtop[j]) for f in fs)
                                                         + kesh[j] * pKE[j]^(1 - sigtop[j]))
            end
        end
        @expression(m, keper[j = 1:na], vaeq[j] * kesh[j] * pw(pTOP[j], pKE[j], sigtop[j]))
        @expression(m, eper[j = 1:na], nfree[j] ? 0.0 : keper[j] * (1 - ksh[j]) * pw(pKE[j], pE[j], sigke[j]))
        @expression(m, topcost[j = 1:na],
                    nfree[j] ? sum(aunit[f, j] * wf(f, j) for f in 1:nf)
                               + sum((pqf[c] + carbf[c, j]) * io[c, j] for c in 1:nc if emask[c])
                             : vaeq[j] * pTOP[j])
        @expression(m, facper[f = 1:nf, j = 1:na],
                    nfree[j] ? aunit[f, j] :
                    f == KI ? keper[j] * ksh[j] * pw(pKE[j], wf(f, j), sigke[j])
                            : vaeq[j] * ktop[f, j] * pw(pTOP[j], wf(f, j), sigtop[j]))
        @expression(m, fuelfix[c = 1:nc, j = 1:na], nfree[j] && emask[c] ? io[c, j] : 0.0)
    end
    @expression(m, iouse[c = 1:nc, j = 1:na],
                !emask[c] ? iofix[c, j] :
                nfree[j] ? fuelfix[c, j] :
                thetaE[c, j] <= 0 ? 0.0 :
                sen[j] == 0 ? eper[j] * thetaE[c, j] / p0f[c] :
                eper[j] * thetaE[c, j] * (pE[j] / fuelrel(c, j))^sen[j] / p0f[c])
end

# ---- 2-3. custo unitário da atividade e preço doméstico da mercadoria
@constraint(m, [j = 1:na],
            pa[j] == (NEST ? sum(iofix[c, j] * pqf[c] for c in 1:nc) + topcost[j]
                           : sum(io[c, j] * pqf[c] for c in 1:nc) + pva[j] * vunit[j]) + atax[j])
@constraint(m, [c = 1:nc], pd[c] == sum(msh[j, c] * pa[j] for j in 1:na))

# ---- 4-5. Armington
@constraint(m, [c = 1:nc],
            arm[c]^(1 - sarm[c]) == domsh[c] * pdom[c]^(1 - sarm[c])
                                  + (1 - domsh[c]) * (er * pwm[c] * (1 + tm[c]) / (1 + tm0[c]))^(1 - sarm[c]))
@constraint(m, [c = 1:nc], dsh[c] == domsh[c] * (arm[c] / pdom[c])^sarm[c])
@constraint(m, [c = 1:nc], mshm[c] == (1 - domsh[c]) * (arm[c] / (er * pwm[c] * (1 + tm[c]) / (1 + tm0[c])))^sarm[c])

# ---- 6. preço de ENTREGA. As margens são elas mesmas precificadas a preço de entrega, donde o
# ponto fixo que o cgelab itera e que aqui é apenas uma restrição.
@constraint(m, [c = 1:nc],
            pbase[c] == arm[c] * armsh[c] + sum(pqf[mm] * mq[mm, c] for mm in 1:nc))

# ---- 6b. pools de um fator com mobilidade imperfeita. CET: o índice de preço do pool é a
# fronteira dual, W^(1+ω) = Σ δ w^(1+ω), e a oferta à atividade é δ L (w/W)^ω. Logit: a oferta é a
# parcela δ w^ω / Σ δ w^ω de uma dotação fixa em cabeças, e o preço do pool é a renda sobre ela.
oferta = Dict{Tuple{Int,Int},Any}()     # (fator, atividade) => oferta total dos pools
renda_mob = Dict{Int,Any}()             # fator => renda
for mb in MOB
    f = Int(mb.factor); om = Float64(mb.omega); dl = mat(mb.delta); L = vecf(mb.endowment)
    us = collect(Bool, mb.users); np_ = length(L)
    alc(o) = [j for j in 1:na if us[j] && dl[o, j] > 0]
    if String(mb.form) == "cet"
        Wp = @variable(m, [o = 1:np_], lower_bound = 1e-8, start = 1.0)
        for o in 1:np_
            @constraint(m, Wp[o]^(1 + om) == sum(dl[o, j] * wf(f, j)^(1 + om) for j in alc(o)))
        end
        S = (o, j) -> dl[o, j] * L[o] * (wf(f, j) / Wp[o])^om
        renda_mob[f] = @expression(m, sum(Wp[o] * L[o] for o in 1:np_))
    else
        Z = @variable(m, [o = 1:np_], lower_bound = 1e-12, start = 1.0)
        for o in 1:np_
            @constraint(m, Z[o] == sum(dl[o, j] * wf(f, j)^om for j in alc(o)))
        end
        S = (o, j) -> L[o] * dl[o, j] * wf(f, j)^om / Z[o]
        renda_mob[f] = @expression(m, sum(wf(f, j) * S(o, j) for o in 1:np_ for j in alc(o)))
    end
    for j in 1:na
        us[j] || continue
        oferta[(f, j)] = @expression(m, sum(S(o, j) for o in 1:np_ if dl[o, j] > 0))
    end
    # O preço reportado do fator é a renda sobre a dotação, como no cgelab.
    @constraint(m, w[f] * sum(L) == renda_mob[f])
end
# Renda de cada fator: o preço vezes a oferta paga, ou a renda dos pools.
renda_fator(f) = ismob[f] ? renda_mob[f] : w[f] * paid_supply(f)

# ---- 7. famílias, COM a reciclagem do gravame
@expression(m, direct[h = 1:nh], sum(hh_direct[h, j] * pa[j] * act_scale[j] for j in 1:na))
# A poupança forçada do fechamento de investimento real é distribuída PER CAPITA, e a receita
# reciclada do gravame pela regra do cenário: são pesos distintos, e trocá-los inverteria o sinal
# distributivo de um dos dois.
# Renda do trabalho segmentado por domicílio: os empregos rígidos e flexíveis nas participações
# do ano-base, a fila na proporção dos empregos rígidos do ano-base, e o resto no informal.
function renda_seg(h)
    tot = 0.0
    for (k, g) in enumerate(SEG)
        idx = Int.(g.index); sh = mat(g.shares); wf_ = vecf(g.workforce); fila = vecf(g.queue)
        if Bool(g.transform)
            # a oferta do domicílio; no rígido, contratada na proporção E / oferta total
            S, oferta_k = SUP[k]
            for (kk, f) in enumerate(idx)
                haskey(S, (h, kk)) || continue
                parte = f in Int.(g.rigid) ? S[(h, kk)] * Eseg[f] / oferta_k[kk] : S[(h, kk)]
                tot += w[f] * parte * (1 - gsh[f] - rsh[f])
            end
            continue
        end
        contratados = vcat(Int.(g.rigid), Int.(g.flexible))
        emprego = 0.0
        for (kk, f) in enumerate(idx)
            f in contratados || continue
            tot += w[f] * sh[h, kk] * Eseg[f] * (1 - gsh[f] - rsh[f])
            emprego += sh[h, kk] * Eseg[f]
        end
        b = Int(g.informal)
        tot += w[b] * (wf_[h] - emprego - fila[h] * Useg[k]) * (1 - gsh[b] - rsh[b])
    end
    return tot
end
@constraint(m, [h = 1:nh],
            hh_income[h] == sum(facshare[h, f] * renda_fator(f) * (1 - gsh[f] - rsh[f])
                              for f in 1:nf) + renda_seg(h)
                          + transfers[h] - levy * popsh[h] + uplift_rev * alpha[h]
                          + gov_tr[h] + remit[h])
# O gasto sai da renda DISPONÍVEL: imposto direto primeiro, poupança depois.
@constraint(m, [h = 1:nh],
            spend[h] == hh_income[h] * (1 - td(h)) * (1 - savrate[h]) - direct[h])

# O bloco de famílias vai AGREGADO. Os coeficientes `coef = share/(1+W+Wg)` e as frações
# capturadas `g_cap`, `g_gov` são constantes, calculadas na exportação, de modo que a demanda são
# n_com somas lineares e não n_dom x n_com expressões não lineares. Com a forma expandida o Ipopt
# não converge a frio numa MCS aplicada de algumas centenas de contas; com esta, converge.
const SCARB = vecf(d.carbon_household)
const HHCARB = any(SCARB .!= 0)
if HHCARB
    # Com imposto de carbono nas famílias o preço efetivo é ad valorem MAIS específico, e a forma
    # agregada não vale. O bloco vai expandido, por família e mercadoria: preço efetivo, demanda,
    # e as receitas das duas cunhas e do carbono somadas diretamente.
    shr = les ? beta_les : hh_budget
    @expression(m, effp[h = 1:nh, c = 1:nc], pq[c] * (1 + Wt[h, c]) + SCARB[c])
    if les
        @expression(m, committed[h = 1:nh], sum(effp[h, c] * gamma[h, c] for c in 1:nc))
        @expression(m, super[h = 1:nh], spend[h] - committed[h])
    else
        @expression(m, super[h = 1:nh], spend[h])
    end
    @expression(m, hhq[h = 1:nh, c = 1:nc], gamma[h, c] + shr[h, c] * super[h] / effp[h, c])
    @constraint(m, uplift_rev == sum(hhq[h, c] * W[h, c] * pq[c] for h in 1:nh, c in 1:nc))
    @expression(m, gov_cost, -sum(hhq[h, c] * (Wg[h, c] * pq[c] + SCARB[c])
                                  for h in 1:nh, c in 1:nc) + gov_absorbs)
    @expression(m, hh_final[c = 1:nc], sum(hhq[h, c] for h in 1:nh))
else
if les
    @expression(m, committed[h = 1:nh],
                sum(pq[c] * (1 + Wt[h, c]) * gamma[h, c] for c in 1:nc))
    @expression(m, super[h = 1:nh], spend[h] - committed[h])
else
    @expression(m, super[h = 1:nh], spend[h])
end
@expression(m, hh_val[c = 1:nc], sum(coef[h, c] * super[h] for h in 1:nh))

# A série infinita da reciclagem: creditar a receita à renda eleva o gasto, que eleva o gravame,
# que eleva a renda. O cgelab soma isso em forma fechada; aqui é uma restrição e o resolvedor
# encontra o ponto fixo.
@constraint(m, uplift_rev == sum(g_cap[h] * super[h] for h in 1:nh)
                           + sum(gam_W[c] * pq[c] for c in 1:nc))
@expression(m, gov_cost, -(sum(g_gov[h] * super[h] for h in 1:nh)
                           + sum(gam_Wg[c] * pq[c] for c in 1:nc)) + gov_absorbs)

@expression(m, hh_final[c = 1:nc], gamma_sum[c] + hh_val[c] / pq[c])
end
if GTARGET == "saving" && GADJ == "spending"
    # Meta de poupança atingida pelo gasto, nas parcelas de valor do ano-base. Vem antes do
    # fechamento do governo, que ela substitui.
    gov_w = vecf(d.gov_w)
    @variable(m, gbud, start = sum(gov_dem))
    @expression(m, gov_q[c = 1:nc], gov_w[c] * gbud / pq[c])
elseif d.government == "real_fixed"
    gov_q_fix = vecf(d.gov_q)
    @expression(m, gov_q[c = 1:nc], gov_q_fix[c])
elseif d.government == "saving_rate"
    # O governo poupa uma fração fixa da renda e gasta o resto nas parcelas de valor do ano-base.
    gov_w = vecf(d.gov_w)
    @variable(m, gbud, start = sum(gov_dem))
    @expression(m, gov_q[c = 1:nc], gov_w[c] * gbud / pq[c])
else
    @expression(m, gov_q[c = 1:nc], gov_dem[c] * nu / pq[c])
end
@expression(m, gov_spend, sum(pq[c] * gov_q[c] for c in 1:nc))
@expression(m, hh_sav, sum(hh_income[h] * (1 - td(h)) * savrate[h] for h in 1:nh))
# Renda do governo que o bloco tributário não vê: imposto direto e renda de fatores, menos o que
# ele transfere às famílias.
@expression(m, gov_inst, sum(hh_income[h] * td(h) for h in 1:nh)
                       + sum(renda_fator(f) * gsh[f] for f in 1:nf) - sum(gov_tr))

if d.investment == "savings_driven"
    @expression(m, inv_q[c = 1:nc], inv_w[c] * savings / pqf[c])
else
    @expression(m, inv_q[c = 1:nc], inv_q_fix[c])
    @constraint(m, savings == sum(pqf[c] * inv_q[c] for c in 1:nc))
end

# ---- 8. balanço de absorção. Substitui a inversão de Leontief.
@constraint(m, [c = 1:nc],
            Aabs[c] == (NEST ? sum(iouse[c, j] * (Xraw[j] + X_exog[j]) for j in 1:na) - exog_int[c]
                             : sum(Bmat[c, mm] * Q[mm] for mm in 1:nc))
                     + sum(mq[c, mm] * Aabs[mm] for mm in 1:nc)
                     + hh_final[c] + gov_q[c] + inv_q[c] + exog_int[c])
for c in 1:nc
    if expsh[c] >= 1
        @constraint(m, Q[c] == 0)            # sem venda doméstica, como no cgelab
    else
        @constraint(m, Q[c] == dsh[c] * armsh[c] / (domfrac[c] * (1 + tauz0[c])) * Aabs[c])
    end
end
@constraint(m, [j = 1:na], Xraw[j] == sum(msh[j, c] * Q[c] for c in 1:nc))
@expression(m, X[j = 1:na], own[j] ? X_exog[j] : Xraw[j])

# ---- 9. poupança. O cgelab precisa decompor esta relação em parte afim; aqui é a identidade.
@expression(m, fin_c[c = 1:nc], hh_final[c] + gov_q[c])
@expression(m, tax_rev,
            sum(pbase[c] * ((Aabs[c] - fin_c[c]) * tf(c) + fin_c[c] * ta(c))
                for c in 1:nc)
            + sum(atax[j] * X[j] for j in 1:na)        # conta própria incluída
            + (NEST ? sum(carbf[c, j] * iouse[c, j] * X[j] for c in 1:nc, j in 1:na) : 0.0)
            + sum(tm[c] * er * pwm[c] * mshm[c] * armsh[c] * Aabs[c] / (1 + tm0[c]) for c in 1:nc)
            + sum(tauz[c] * pd[c] * Q[c] for c in 1:nc))
@constraint(m, savings == hh_sav + levy + tax_rev + gov_inst - gov_spend - gov_cost + fsav)
if d.government == "saving_rate"
    @constraint(m, gbud == (1 - Float64(d.gov_savrate)) * (tax_rev + gov_inst - gov_cost))
end
const GTVAL = Float64(d.government_target_value)
if GTARGET == "saving" && GADJ == "spending"
    @constraint(m, gbud == tax_rev + gov_inst - gov_cost - GTVAL)
elseif TAX_LEVER && GTARGET == "saving"
    @constraint(m, tax_rev + gov_inst - gov_spend - gov_cost == GTVAL)
elseif TAX_LEVER && GTARGET == "revenue"
    @constraint(m, tax_rev + gov_inst + sum(gov_tr) - gov_cost + gov_absorbs == GTVAL)
end

# ---- 10. demanda de fatores
@expression(m, facint[f = 1:nf, j = 1:na],
            VINT ? ((vth_o[f, j] > 0 ? vth_o[f, j] * vpow(pOld[j], wf(f, j), vs_o[j]) * bOld[j] : 0.0)
                    + (vth_n[f, j] > 0 ? vth_n[f, j] * vpow(pNew[j], wf(f, j), vs_n[j]) * bNew[j] : 0.0))
                 : leo[j] ? (vunit[j] != 0 ? aunit[f, j] / vunit[j] : 0.0)
                   : (theta[f, j] > 0 ? theta[f, j] * (pva[j] / wf(f, j))^sva[j] : 0.0))
@expression(m, facdem_j[f = 1:nf, j = 1:na],
            (NEST ? facper[f, j] : facint[f, j] * vunit[j]) * X[j])
@expression(m, facdem[f = 1:nf], sum(facdem_j[f, j] for j in 1:na))
if labour_slack
    @constraint(m, emp == facdem[LAB])
end

# ---- 11. os mercados resolvidos. O do NUMERÁRIO fica de fora, para a lei de Walras.
for f in 1:nf
    f == NUM && continue
    if ismob[f]
        # um mercado por atividade usuária, contra a oferta dos pools
        for j in 1:na
            haskey(oferta, (f, j)) && @constraint(m, facdem_j[f, j] == oferta[(f, j)])
        end
        continue
    end
    if haskey(pago_seg, f)
        # o segmento emprega o que a demanda pede; o informal, o resto da força de trabalho
        @constraint(m, facdem[f] == pago_seg[f])
        continue
    end
    if LABOUR == "fixed_real_wage" && f == LAB
        continue
    elseif LABOUR == "wage_curve" && f == LAB
        @constraint(m, log(w[f] / P) + beta_w * log((1 - facdem[f] / (supply[f] / (1 - u0))) / u0) == 0)
    elseif LABOUR == "elastic_supply" && f == LAB
        @constraint(m, facdem[f] == supply[f] * (w[f] / P) ^ eps_ls)
    else
        @constraint(m, facdem[f] == supply[f])
    end
end
if LABOUR == "fixed_real_wage"
    @constraint(m, w[LAB] == real_wage_target * P)
end
for (k, g) in enumerate(SEG)
    b = Int(g.informal)
    for (f, alvo) in zip(Int.(g.rigid), vecf(g.target))
        @constraint(m, w[f] == alvo * P)            # salário real fixo
    end
    for f in Int.(g.flexible)
        @constraint(m, w[f] == w[b])                # o salário comum do informal
    end
    if Bool(g.harris_todaro)
        # o salário informal é o salário rígido esperado por quem entra na fila
        R = Int.(g.rigid)
        @constraint(m, w[b] * (sum(Eseg[f] for f in R) + Useg[k])
                       == Float64(g.lam) * sum(w[f] * Eseg[f] for f in R))
    end
end
@expression(m, imports[c = 1:nc], mshm[c] * armsh[c] * Aabs[c] / (1 + tm0[c]))   # na fronteira
@expression(m, exports[c = 1:nc], expfrac[c] * (1 + tauz0[c]) * Q[c])
@expression(m, trade, sum(er * pwm[c] * imports[c] for c in 1:nc)
                    - sum(pexp[c] * exports[c] for c in 1:nc))
# A meta é a CONTA CORRENTE, igual à poupança externa (fixa sob "fixed_saving", incógnita sob
# "fixed_rate"): balança comercial, mais a renda de fatores paga ao exterior, menos as
# transferências líquidas recebidas. `trade_balance0` fica no JSON só como registro do ano-base.
# ---- 12. as curvas: E/E0 = (pWe/deslocamento)^(-η), M/M0 = (pWm/deslocamento)^ψ, como no lrgcge
for c in 1:nc
    if eta_x[c] > 0
        @constraint(m, exports[c] == E0[c] * (pwe[c] / wpe[c])^(-eta_x[c]))
    end
    if psi_m[c] > 0
        @constraint(m, imports[c] == M0[c] * (pwm[c] / wp[c])^psi_m[c])
    end
end
@expression(m, current, trade + sum(renda_fator(f) * rsh[f] for f in 1:nf) - sum(remit))
@constraint(m, current == fsav)

@objective(m, Min, 0.0)
@printf("variáveis=%d  restrições=%d\n", num_variables(m),
        num_constraints(m; count_variable_in_set_constraints = false))
optimize!(m)
@printf("Ipopt: %s (%d iterações)\n", termination_status(m), MOI.get(m, MOI.BarrierIterations()))

function violacao_maxima(mm)
    v = 0.0
    for (F, S) in list_of_constraint_types(mm)
        S <: MOI.EqualTo || continue
        for con in all_constraints(mm, F, S)
            v = max(v, abs(value(con) - MOI.constant(constraint_object(con).set)))
        end
    end
    return v
end
viol = violacao_maxima(m)
walras = (value(facdem[NUM]) - supply[NUM]) / max(abs(supply[NUM]), 1.0)
@printf("violação máxima de restrição = %.3e  (o veredito; a bandeira do Ipopt não é)\n", viol)
@printf("WALRAS (%s, NÃO imposto) = %.3e\n", d.factors[NUM], walras)

out = Dict{String,Any}(
    "status" => string(termination_status(m)),
    "violacao_max" => viol, "walras" => walras,
    "w" => value.(w), "er" => value(er),
    "w_act" => [[value(wf(f, j)) for j in 1:na] for f in 1:nf],
    "desemprego" => [value(u) for u in Useg],
    "pa" => value.(pa), "pd" => value.(pd),
    "pq" => [value(pq[c]) for c in 1:nc], "pqf" => [value(pqf[c]) for c in 1:nc],
    "pbase" => value.(pbase),
    "Q" => value.(Q), "X" => [value(X[j]) for j in 1:na], "Aabs" => value.(Aabs),
    "hh_income" => value.(hh_income), "hh_com_spend" => value.(spend),
    "savings" => value(savings), "uplift_rev" => value(uplift_rev),
    "trade" => value(trade), "facdem" => [value(facdem[f]) for f in 1:nf],
    "exports" => [value(exports[c]) for c in 1:nc],
    "world_pe" => value.(pwe), "world_pm" => value.(pwm), "pdom" => [value(pdom[c]) for c in 1:nc],
)
open(SAIDA, "w") do f
    JSON3.write(f, out)
end
@printf("escrito: %s\n", SAIDA)
"""
