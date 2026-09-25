"""Exportar uma economia calibrada para uma réplica em JuMP, resolvida SIMULTANEAMENTE.

Por que isto vale a pena, e não é só conveniência. O :mod:`cgekit.solve` resolve o equilíbrio por
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
from .scenario import Scenario
from .solve import _marginal_budget_shares, _recycle_shares

__all__ = ["to_jump", "JUMP_TEMPLATE"]


def _l(a):
    return np.asarray(a, dtype=float).tolist()


def to_jump(econ: Economy, closure: Closure | None = None, scenario: Scenario | None = None, *,
            path: str | Path = "modelo_cge.jl", data_path: str | Path | None = None) -> dict:
    """Escreve a réplica em JuMP e o seu JSON de dados. Devolve o dicionário exportado.

    Os coeficientes vão EFETIVOS — a produtividade já dividida para dentro de ``io``, ``vunit`` e
    ``aunit`` —, porque a produtividade Hicks-neutra escala toda exigência de insumo e não apenas o
    custo unitário. Exportar os coeficientes crus e a produtividade em separado convidaria o lado
    Julia a aplicá-la só aos preços, que é exatamente o erro que faz o resíduo de Walras crescer ao
    longo de uma trajetória.
    """
    closure = closure or Closure()
    sc = scenario or Scenario()
    if econ.spec.export_rule == "cet":
        # A réplica escreve as exportações como parcela fixa do produto. Exportar um modelo CET
        # para ela produziria outra economia, e a comparação acusaria um erro que não existe.
        raise NotImplementedError(
            "to_jump ainda não replica export_rule='cet'; a réplica em JuMP usa parcelas fixas "
            "de exportação.")
    nao_replicado = [nome for nome, v in (
        ("impostos diretos", econ.hh_dtax), ("transferências do governo", econ.hh_gov_transfer),
        ("transferências do exterior", econ.hh_remit_net),
        ("renda de fatores do governo", econ.fac_gov_share),
        ("renda de fatores do exterior", econ.fac_row_share),
        ("participação estrangeira nos fatores", sc.foreign_factor_share),
    ) if v is not None and np.any(np.asarray(v) != 0)] + (["IED"] if sc.fdi else [])
    if nao_replicado:
        # Mesmo motivo da CET: a réplica não tem esses blocos, e uma réplica que os omite
        # descreveria outra economia.
        raise NotImplementedError(
            "to_jump ainda não replica: " + ", ".join(nao_replicado) + ".")
    nf, nc, nh, na = econ.n_fac, econ.n_com, econ.n_hh, econ.n_act
    num = econ.numeraire_index

    supply = econ.fac_supply if sc.factor_supply is None else np.asarray(sc.factor_supply, float)
    facshare = (econ.hh_facshare if sc.factor_ownership is None
                else np.asarray(sc.factor_ownership, float))
    tfp = np.ones(na) if sc.productivity is None else np.asarray(sc.productivity, float)
    ptax_a = econ.ptax if sc.product_tax is None else np.asarray(sc.product_tax, float)
    ptax_f = ptax_a if sc.product_tax_firm is None else np.asarray(sc.product_tax_firm, float)
    world_p = np.ones(nc) if sc.world_price is None else np.asarray(sc.world_price, float)
    W = np.zeros((nh, nc)) if sc.consumer_wedge is None else np.asarray(sc.consumer_wedge, float)
    Wg = (np.zeros((nh, nc)) if sc.government_wedge is None
          else np.asarray(sc.government_wedge, float))
    transfers = np.zeros(nh) if sc.transfers is None else np.asarray(sc.transfers, float)

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

    io_eff = econ.io / tfp[None, :]
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
        sigma_va=_l(econ.sigma_va), sigma_arm=_l(econ.sigma_arm),
        X0=_l(econ.X0), Q0=_l(econ.Q0), A0=_l(econ.A0),
        # comércio e margens
        dom_share=_l(econ.dom_share), arm_share=_l(econ.arm_share),
        exp_share=_l(econ.exp_share), mq=_l(econ.mq), world_price=_l(world_p),
        # tributos
        ptax_a=_l(ptax_a), ptax_f=_l(ptax_f), ptax_base=_l(econ.ptax), atax=_l(econ.atax * nu),
        # famílias
        hh_budget=_l(econ.hh_budget), beta_les=_l(beta_les),
        demand_coef=_l(coef), g_cap=_l(g_cap), g_gov=_l(g_gov),
        gamma_sum=_l(gamma_sum), gam_W=_l(gam_W), gam_Wg=_l(gam_Wg), Wt=_l(Wt),
        hh_committed=_l(econ.hh_committed), hh_savrate=_l(econ.hh_savrate),
        facshare=_l(facshare), hh_direct=_l(econ.hh_direct), transfers=_l(transfers),
        les=bool(econ.spec.household_demand == "les"),
        # política
        wedge=_l(W), wedge_gov=_l(Wg),
        recycle_share=_l(_recycle_shares(econ, sc, facshare, supply)),
        pop_share=_l(econ.pop_share),
        gov_absorbs=float(sc.government_absorbs),
        # blocos exógenos
        fac_supply=_l(supply), gov_dem=_l(econ.gov_dem), gov_q=(None if gov_q is None
                                                                else _l(gov_q)),
        inv_dem=_l(econ.inv_dem), inv_q_fixed=_l(econ.inv_dem / (1.0 + econ.ptax)),
        act_scale=_l(act_scale), own_account=[bool(b) for b in exog],
        X_exog=_l(X_exog), exog_intermediate=_l(io_eff @ X_exog),
        trade_balance0=float(econ.trade_balance0 * nu),
        foreign_sav=float(econ.foreign_sav * nu),
        # fechamento
        investment=closure.investment, government=closure.government,
        foreign=closure.foreign, exchange_rate=float(closure.exchange_rate),
        scenario_name=sc.name,
    )

    path = Path(path)
    data_path = Path(data_path) if data_path is not None else path.with_name(
        path.stem + "_data.json")
    path.write_text(JUMP_TEMPLATE, encoding="utf-8")
    data_path.write_text(json.dumps(dados), encoding="utf-8")
    return dados


JUMP_TEMPLATE = r"""# modelo_cge.jl — réplica em JuMP de uma economia calibrada pelo cgekit.
#
# GERADO por cgekit.export.to_jump. Arquivo FIXO: serve a qualquer economia, lendo o JSON.
#
# O cgekit resolve este mesmo equilíbrio por ANINHAMENTO — um Newton sobre poucas incógnitas e,
# dentro dele, um ponto fixo sobre o preço de entrega, duas somas em forma fechada e uma inversão
# de Leontief. Aqui nada é aninhado: cada um desses blocos é uma restrição, e o Ipopt acha o ponto
# fixo. É por isso que a comparação verifica a ÁLGEBRA e não apenas a aritmética.
#
# O mercado do fator NUMERÁRIO fica deliberadamente de fora, exatamente como no cgekit, de modo que
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

io = mat(d.io); msh = mat(d.msh); theta = mat(d.theta); aunit = mat(d.aunit); mq = mat(d.mq)
vunit = vecf(d.vunit); leo = collect(Bool, d.leontief)
sva = vecf(d.sigma_va); sarm = vecf(d.sigma_arm)
X0 = vecf(d.X0); Q0 = vecf(d.Q0); A0 = vecf(d.A0)
domsh = vecf(d.dom_share); armsh = vecf(d.arm_share); expsh = vecf(d.exp_share)
wp = vecf(d.world_price)
ptax_a = vecf(d.ptax_a); ptax_f = vecf(d.ptax_f); ptax_b = vecf(d.ptax_base); atax = vecf(d.atax)
hh_budget = mat(d.hh_budget); beta_les = mat(d.beta_les); gamma = mat(d.hh_committed)
coef = mat(d.demand_coef); Wt = mat(d.Wt)
g_cap = vecf(d.g_cap); g_gov = vecf(d.g_gov)
gamma_sum = vecf(d.gamma_sum); gam_W = vecf(d.gam_W); gam_Wg = vecf(d.gam_Wg)
savrate = vecf(d.hh_savrate); facshare = mat(d.facshare); hh_direct = mat(d.hh_direct)
transfers = vecf(d.transfers); les = Bool(d.les); popsh = vecf(d.pop_share)
W = mat(d.wedge); Wg = mat(d.wedge_gov); alpha = vecf(d.recycle_share)
gov_absorbs = Float64(d.gov_absorbs)
supply = vecf(d.fac_supply); inv_dem = vecf(d.inv_dem); inv_q_fix = vecf(d.inv_q_fixed)
act_scale = vecf(d.act_scale); X_exog = vecf(d.X_exog); exog_int = vecf(d.exog_intermediate)
own = collect(Bool, d.own_account)
tb0 = Float64(d.trade_balance0); fsav0 = Float64(d.foreign_sav)
gov_dem = vecf(d.gov_dem)

Bmat = io * msh
inv_w = inv_dem ./ max(sum(inv_dem), 1e-12)

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
@variable(m, pva[j = 1:na] >= 0, start = 1.0)
@variable(m, pa[j = 1:na] >= 1e-8, start = q("pa", j, 1.0))
@variable(m, pd[c = 1:nc] >= 1e-8, start = q("pd", c, 1.0))
@variable(m, arm[1:nc] >= 1e-8, start = 1.0)
@variable(m, dsh[1:nc] >= 0, start = 1.0)
@variable(m, mshm[1:nc] >= 0, start = 0.1)
@variable(m, pbase[c = 1:nc] >= 1e-8, start = q("pbase", c, 1.0))
@variable(m, Aabs[i = 1:nc] >= 0, start = q("Aabs", i, A0[i]))
@variable(m, Q[i = 1:nc] >= 0, start = q("Q", i, Q0[i]))
@variable(m, Xraw[i = 1:na], start = q("X", i, X0[i]))
@variable(m, hh_income[h = 1:nh], start = q("hh_income", h, 1.0))
@variable(m, spend[h = 1:nh], start = q("hh_com_spend", h, 1.0))
@variable(m, savings, start = q0("savings", sum(inv_dem)))
@variable(m, uplift_rev, start = q0("uplift_rev", 0.0))

# incógnitas que cada fechamento acrescenta
if d.foreign == "fixed_saving"
    @variable(m, er >= 1e-8, start = q0("er", 1.15))
    fsav = fsav0
else
    er_fixo = Float64(d.exchange_rate)
    @expression(m, er, er_fixo)
    @variable(m, fsav, start = fsav0)
end
if d.investment == "fixed_real"
    @variable(m, levy, start = 0.0)          # poupança forçada que financia o investimento fixo
else
    @expression(m, levy, 0.0)
end

@expression(m, pq[c = 1:nc], pbase[c] * (1 + ptax_a[c]))     # preço ao consumidor FINAL
@expression(m, pqf[c = 1:nc], pbase[c] * (1 + ptax_f[c]))    # preço à FIRMA (parcela não creditável)

# ---- 1. preço do valor adicionado: CES sobre fatores, Leontief onde o excedente é negativo
for j in 1:na
    if leo[j]
        @constraint(m, pva[j] * vunit[j] == sum(aunit[f, j] * w[f] for f in 1:nf))
    elseif sum(theta[:, j]) <= 0
        @constraint(m, pva[j] == 0)
    elseif abs(sva[j] - 1.0) < 1e-9
        @constraint(m, pva[j] == prod(w[f]^theta[f, j] for f in 1:nf if theta[f, j] > 0))
    else
        @constraint(m, pva[j]^(1 - sva[j]) ==
                       sum(theta[f, j] * w[f]^(1 - sva[j]) for f in 1:nf if theta[f, j] > 0))
    end
end

# ---- 2-3. custo unitário da atividade e preço doméstico da mercadoria
@constraint(m, [j = 1:na],
            pa[j] == sum(io[c, j] * pqf[c] for c in 1:nc) + pva[j] * vunit[j] + atax[j])
@constraint(m, [c = 1:nc], pd[c] == sum(msh[j, c] * pa[j] for j in 1:na))

# ---- 4-5. Armington
@constraint(m, [c = 1:nc],
            arm[c]^(1 - sarm[c]) == domsh[c] * pd[c]^(1 - sarm[c])
                                  + (1 - domsh[c]) * (er * wp[c])^(1 - sarm[c]))
@constraint(m, [c = 1:nc], dsh[c] == domsh[c] * (arm[c] / pd[c])^sarm[c])
@constraint(m, [c = 1:nc], mshm[c] == (1 - domsh[c]) * (arm[c] / (er * wp[c]))^sarm[c])

# ---- 6. preço de ENTREGA. As margens são elas mesmas precificadas a preço de entrega, donde o
# ponto fixo que o cgekit itera e que aqui é apenas uma restrição.
@constraint(m, [c = 1:nc],
            pbase[c] == arm[c] * armsh[c] + sum(pqf[mm] * mq[mm, c] for mm in 1:nc))

# ---- 7. famílias, COM a reciclagem do gravame
@expression(m, direct[h = 1:nh], sum(hh_direct[h, j] * pa[j] * act_scale[j] for j in 1:na))
# A poupança forçada do fechamento de investimento real é distribuída PER CAPITA, e a receita
# reciclada do gravame pela regra do cenário: são pesos distintos, e trocá-los inverteria o sinal
# distributivo de um dos dois.
@constraint(m, [h = 1:nh],
            hh_income[h] == sum(facshare[h, f] * w[f] * supply[f] for f in 1:nf)
                          + transfers[h] - levy * popsh[h] + uplift_rev * alpha[h])
@constraint(m, [h = 1:nh], spend[h] == hh_income[h] * (1 - savrate[h]) - direct[h])

# O bloco de famílias vai AGREGADO. Os coeficientes `coef = share/(1+W+Wg)` e as frações
# capturadas `g_cap`, `g_gov` são constantes, calculadas na exportação, de modo que a demanda são
# n_com somas lineares e não n_dom x n_com expressões não lineares. Com a forma expandida o Ipopt
# não converge a frio numa MCS aplicada de algumas centenas de contas; com esta, converge.
if les
    @expression(m, committed[h = 1:nh],
                sum(pq[c] * (1 + Wt[h, c]) * gamma[h, c] for c in 1:nc))
    @expression(m, super[h = 1:nh], spend[h] - committed[h])
else
    @expression(m, super[h = 1:nh], spend[h])
end
@expression(m, hh_val[c = 1:nc], sum(coef[h, c] * super[h] for h in 1:nh))

# A série infinita da reciclagem: creditar a receita à renda eleva o gasto, que eleva o gravame,
# que eleva a renda. O cgekit soma isso em forma fechada; aqui é uma restrição e o resolvedor
# encontra o ponto fixo.
@constraint(m, uplift_rev == sum(g_cap[h] * super[h] for h in 1:nh)
                           + sum(gam_W[c] * pq[c] for c in 1:nc))
@expression(m, gov_cost, -(sum(g_gov[h] * super[h] for h in 1:nh)
                           + sum(gam_Wg[c] * pq[c] for c in 1:nc)) + gov_absorbs)

@expression(m, hh_final[c = 1:nc], gamma_sum[c] + hh_val[c] / pq[c])
if d.government == "real_fixed"
    gov_q_fix = vecf(d.gov_q)
    @expression(m, gov_q[c = 1:nc], gov_q_fix[c])
else
    @expression(m, gov_q[c = 1:nc], gov_dem[c] * nu / pq[c])
end
@expression(m, gov_spend, sum(pq[c] * gov_q[c] for c in 1:nc))
@expression(m, hh_sav, sum(hh_income[h] * savrate[h] for h in 1:nh))

if d.investment == "savings_driven"
    @expression(m, inv_q[c = 1:nc], inv_w[c] * savings / pqf[c])
else
    @expression(m, inv_q[c = 1:nc], inv_q_fix[c])
    @constraint(m, savings == sum(pqf[c] * inv_q[c] for c in 1:nc))
end

# ---- 8. balanço de absorção. Substitui a inversão de Leontief.
@constraint(m, [c = 1:nc],
            Aabs[c] == sum(Bmat[c, mm] * Q[mm] for mm in 1:nc)
                     + sum(mq[c, mm] * Aabs[mm] for mm in 1:nc)
                     + hh_final[c] + gov_q[c] + inv_q[c] + exog_int[c])
@constraint(m, [c = 1:nc], Q[c] == dsh[c] * armsh[c] / (1 - expsh[c]) * Aabs[c])
@constraint(m, [j = 1:na], Xraw[j] == sum(msh[j, c] * Q[c] for c in 1:nc))
@expression(m, X[j = 1:na], own[j] ? X_exog[j] : Xraw[j])

# ---- 9. poupança. O cgekit precisa decompor esta relação em parte afim; aqui é a identidade.
@expression(m, fin_c[c = 1:nc], hh_final[c] + gov_q[c])
@expression(m, tax_rev,
            sum(pbase[c] * ((Aabs[c] - fin_c[c]) * ptax_f[c] + fin_c[c] * ptax_a[c])
                for c in 1:nc)
            + sum(atax[j] * Xraw[j] for j in 1:na))
@constraint(m, savings == hh_sav + levy + tax_rev - gov_spend - gov_cost + fsav)

# ---- 10. demanda de fatores
@expression(m, facint[f = 1:nf, j = 1:na],
            leo[j] ? (vunit[j] != 0 ? aunit[f, j] / vunit[j] : 0.0)
                   : (theta[f, j] > 0 ? theta[f, j] * (pva[j] / w[f])^sva[j] : 0.0))
@expression(m, facdem[f = 1:nf], sum(facint[f, j] * vunit[j] * X[j] for j in 1:na))

# ---- 11. os mercados resolvidos. O do NUMERÁRIO fica de fora, para a lei de Walras.
for f in 1:nf
    f == NUM && continue
    @constraint(m, facdem[f] == supply[f])
end
@expression(m, imports[c = 1:nc], mshm[c] * armsh[c] * Aabs[c])
@expression(m, exports[c = 1:nc], expsh[c] * Q[c])
@expression(m, trade, sum(er * wp[c] * imports[c] for c in 1:nc)
                    - sum(pd[c] * exports[c] for c in 1:nc))
# A meta da conta corrente É a poupança externa: fixa sob "fixed_saving", incógnita sob
# "fixed_rate". `trade_balance0` fica no JSON só como registro do ano-base.
@constraint(m, trade == fsav)

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
    "pa" => value.(pa), "pd" => value.(pd),
    "pq" => [value(pq[c]) for c in 1:nc], "pqf" => [value(pqf[c]) for c in 1:nc],
    "pbase" => value.(pbase),
    "Q" => value.(Q), "X" => [value(X[j]) for j in 1:na], "Aabs" => value.(Aabs),
    "hh_income" => value.(hh_income), "hh_com_spend" => value.(spend),
    "savings" => value(savings), "uplift_rev" => value(uplift_rev),
    "trade" => value(trade), "facdem" => [value(facdem[f]) for f in 1:nf],
)
open(SAIDA, "w") do f
    JSON3.write(f, out)
end
@printf("escrito: %s\n", SAIDA)
"""
