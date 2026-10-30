# regional.jl — o modelo de várias regiões do cgelab escrito de novo em JuMP, a partir da MCS.
#
# Nada vem da calibração do cgelab. O script lê a MCS, os papéis das contas, as elasticidades,
# as rotas de transporte e o choque, calibra cada bloco aqui e resolve o equilíbrio como um
# sistema quadrado de restrições, por viabilidade no Ipopt. O cgelab resolve o mesmo modelo por
# aninhamento: um Newton sobre os preços dos fatores e o câmbio e, dentro dele, um ponto fixo de
# preços, uma inversa de Leontief sobre a absorção por destino e, nos fechamentos de governo
# dinâmicos, uma sonda afim. Nenhum desses atalhos existe aqui.
#
# As unidades também são outras. Aqui todo preço vale um no ano-base, inclusive o preço ao
# comprador num destino com margem de transporte; no cgelab esse preço é o da MCS sobre a
# quantidade a preço básico. Por isso a comparação usa níveis de fatores, câmbio, produção,
# remessas bilaterais e valores, e os preços ao comprador relativos ao ano-base.
#
# Escopo, recusado pelo script em Python fora dele: sem impostos, sem margens fora das rotas,
# uma atividade por bem de origem, famílias Cobb–Douglas, valor adicionado CES (Cobb–Douglas em
# sigma = 1), exportações em proporção fixa ou CET, migração CET ou logit, contas regionais de
# governo e de investimento, fechamentos de investimento savings_driven e fixed_real e de
# governo real_fixed e saving_rate.
#
#   uso:  julia --project=validation validation/regional.jl entrada.json saida.json

using JuMP, Ipopt, JSON3, Printf

const E = JSON3.read(read(ARGS[1], String))
const SAIDA = ARGS[2]

S = Dict{Tuple{String,String},Float64}()
for (u, v, x) in E.sam
    S[(String(u), String(v))] = get(S, (String(u), String(v)), 0.0) + Float64(x)
end
sam(u, v) = get(S, (u, v), 0.0)

R = String.(E.regions)
L = String.(E.goods)
FL = String.(E.factors)
H = String.(E.households)
NR, NL, NF, NH = length(R), length(L), length(FL), length(H)
ND = NR + 1                      # o último destino é o canal nacional (governo e investimento)
conta(r, c) = "$(R[r]):$(L[c])"
fator(r, l) = "$(R[r]):$(FL[l])"
atv(r, c) = String(E.activity[Symbol(conta(r, c))])
hreg = [findfirst(==(String(E.household_region[Symbol(h)])), R) for h in H]
gov, inv, row = String(E.government), String(E.investment), String(E.rest_of_world)
govr = Dict(findfirst(==(String(k)), R) => String(v) for (k, v) in E.regional_government)
invr = Dict(findfirst(==(String(k)), R) => String(v) for (k, v) in E.regional_investment)
contas_gov = vcat([gov], collect(values(govr)))
contas_inv = vcat([inv], collect(values(invr)))

compradores_hh(d) = d > NR ? String[] : [H[h] for h in 1:NH if hreg[h] == d]
compradores_atv(d) = d > NR ? String[] : [atv(d, c) for c in 1:NL]
compradores_gov(d) = d > NR ? [gov] : (haskey(govr, d) ? [govr[d]] : String[])
compradores_inv(d) = d > NR ? [inv] : (haskey(invr, d) ? [invr[d]] : String[])
compradores(d) = vcat(compradores_hh(d), compradores_atv(d), compradores_gov(d),
                      compradores_inv(d))
compra(r, d, c, quem) = sum((sam(conta(r, c), b) for b in quem); init = 0.0)

# ---------------------------------------------------------------------------------------------
# Calibração, toda a partir da MCS
# ---------------------------------------------------------------------------------------------
X0 = [sam(atv(r, c), conta(r, c)) for r in 1:NR, c in 1:NL]
E0 = [sam(conta(r, c), row) for r in 1:NR, c in 1:NL]
M0 = [sam(row, conta(r, c)) for r in 1:NR, c in 1:NL]
DS0 = X0 .- E0                                # vendas internas de cada origem
uso = [compra(r, d, c, compradores(d)) for r in 1:NR, d in 1:ND, c in 1:NL]

transporte = E.transport !== nothing
if transporte
    fob = [Float64(E.transport.fob[r][d][c]) for r in 1:NR, d in 1:ND, c in 1:NL]
    trn = [Float64(E.transport.transport[r][d][c]) for r in 1:NR, d in 1:ND, c in 1:NL]
    oferta = [Float64(E.transport.supply[r][c]) for r in 1:NR, c in 1:NL]
    dom = fob
    imp = uso .- fob .- trn
else
    # Sem rotas, cada célula de uma conta de origem divide-se entre produção da região e
    # importação na proporção dessa conta.
    casa = [DS0[r, c] + M0[r, c] > 0 ? DS0[r, c] / (DS0[r, c] + M0[r, c]) : 1.0
            for r in 1:NR, c in 1:NL]
    dom = [uso[r, d, c] * casa[r, c] for r in 1:NR, d in 1:ND, c in 1:NL]
    trn = zeros(NR, ND, NL)
    imp = uso .- dom
    oferta = zeros(NR, NL)
end
@assert minimum(imp) > -1e-9 "a importação calibrada ficou negativa"
tau = [dom[r, d, c] > 0 ? trn[r, d, c] / dom[r, d, c] : 0.0 for r in 1:NR, d in 1:ND, c in 1:NL]
V = dom .+ trn                                 # valor entregue de cada origem, a preço 1 + tau
DD0 = [sum(V[:, d, c]) for d in 1:ND, c in 1:NL]
MM0 = [sum(imp[:, d, c]) for d in 1:ND, c in 1:NL]
A0 = DD0 .+ MM0
ativo = A0 .> 1e-12

sigI = [Float64(E.sigma_interregional[Symbol(l)]) for l in L]
sigA = [Float64(E.sigma_armington[Symbol(l)]) for l in L]
sigT = [Float64(E.sigma_cet[Symbol(l)]) for l in L]
sigVA = Float64(E.sigma_va)
cet = String(E.export_rule) == "cet"

# Origem por destino: preço do composto igual a um no ano-base.
alfa = zeros(NR, ND, NL)
for d in 1:ND, c in 1:NL
    ativo[d, c] || continue
    if DD0[d, c] > 0
        for r in 1:NR
            alfa[r, d, c] = V[r, d, c] * (1 + tau[r, d, c])^(sigI[c] - 1) / DD0[d, c]
        end
    end
end
beta = [ativo[d, c] ? DD0[d, c] / A0[d, c] : 0.0 for d in 1:ND, c in 1:NL]

# Produção: insumos a preço de comprador do destino da atividade, valor adicionado.
io = [sum(sam(conta(r2, c2), atv(r, c)) for r2 in 1:NR) / X0[r, c]
      for c2 in 1:NL, r in 1:NR, c in 1:NL]
F0 = [sam(fator(r, l), atv(r, c)) for l in 1:NF, r in 1:NR, c in 1:NL]
VA0 = [sum(F0[:, r, c]) for r in 1:NR, c in 1:NL]
vcoef = VA0 ./ X0
theta = [F0[l, r, c] / VA0[r, c] for l in 1:NF, r in 1:NR, c in 1:NL]
delta = [X0[r, c] > 0 ? DS0[r, c] / X0[r, c] : 1.0 for r in 1:NR, c in 1:NL]

# Renda dos fatores e seus destinos.
fi0 = [sum(sam(fator(r, l), atv(r, c)) for c in 1:NL) for r in 1:NR, l in 1:NF]
dono = [fi0[r, l] > 0 ? sam(h, fator(r, l)) / fi0[r, l] : 0.0 for h in H, r in 1:NR, l in 1:NF]
ao_gov = [fi0[r, l] > 0 ? sum(sam(g, fator(r, l)) for g in contas_gov) / fi0[r, l] : 0.0
          for r in 1:NR, l in 1:NF]
Y0 = [sum(sam(H[h], fator(r, l)) for r in 1:NR, l in 1:NF) for h in 1:NH]
pou = [sum(sam(i, H[h]) for i in contas_inv) / Y0[h] for h in 1:NH]
gasto0 = [sum(sam(conta(r, c), H[h]) for r in 1:NR) for h in 1:NH, c in 1:NL]
orc = gasto0 ./ sum(gasto0; dims = 2)
pop = [Float64(E.population_share[Symbol(h)]) for h in H]

G0 = [compra(r, d, c, compradores_gov(d)) for r in 1:NR, d in 1:ND, c in 1:NL]
G0 = [sum(G0[:, d, c]) for d in 1:ND, c in 1:NL]
I0 = [compra(r, d, c, compradores_inv(d)) for r in 1:NR, d in 1:ND, c in 1:NL]
I0 = [sum(I0[:, d, c]) for d in 1:ND, c in 1:NL]
rec0 = sum(fi0 .* ao_gov)
sg = rec0 > 0 ? (rec0 - sum(G0)) / rec0 : 0.0
FS0 = sum(M0) - sum(E0)                        # poupança externa, na moeda doméstica

# Migração: pools por região de origem, com a matriz de proximidade.
migra = Dict{Int,Any}()
for (k, m) in E.migration
    l = findfirst(==(String(k)), FL)
    prox = [Float64(m.delta[o][a]) for o in 1:NR, a in 1:NR]
    migra[l] = (omega = Float64(m.omega), forma = String(m.form), delta = prox,
                dotacao = [fi0[o, l] for o in 1:NR])
end

# Choque e fechamento.
nu = Float64(E.closure.numeraire)
num_r = findfirst(==(String(E.closure.numeraire_region)), R)
num_l = findfirst(==(String(E.closure.numeraire_factor)), FL)
fixo_inv = String(E.closure.investment) == "fixed_real"
gov_dinamico = String(E.closure.government) == "saving_rate"
tfp = [Float64(E.scenario.productivity[Symbol(atv(r, c))]) for r in 1:NR, c in 1:NL]
mundo = [Float64(E.scenario.world_price[Symbol(l)]) for l in L]
mundo_x = [Float64(E.scenario.export_price[Symbol(l)]) for l in L]
nivel = E.scenario.investment_level === nothing ? sum(I0) : Float64(E.scenario.investment_level)
dot = fi0                                      # dotação de cada fator a preço 1

# ---------------------------------------------------------------------------------------------
# Equilíbrio
# ---------------------------------------------------------------------------------------------
m = Model(Ipopt.Optimizer)
set_silent(m)
set_optimizer_attribute(m, "tol", 1e-13)
set_optimizer_attribute(m, "constr_viol_tol", 1e-12)
set_optimizer_attribute(m, "max_iter", 3000)
lo = 1e-8

@variable(m, w[r in 1:NR, l in 1:NF] >= lo, start = nu)
@variable(m, er >= lo, start = nu)
@variable(m, pva[r in 1:NR, c in 1:NL] >= lo, start = nu)
@variable(m, pa[r in 1:NR, c in 1:NL] >= lo, start = nu)
@variable(m, ps[r in 1:NR, c in 1:NL] >= lo, start = nu)
@variable(m, P[d in 1:ND, c in 1:NL] >= lo, start = nu)
@variable(m, pq[d in 1:ND, c in 1:NL] >= lo, start = nu)
@variable(m, pt >= lo, start = nu)
@variable(m, X[r in 1:NR, c in 1:NL] >= 0, start = X0[r, c])
@variable(m, DS[r in 1:NR, c in 1:NL] >= 0, start = DS0[r, c])
@variable(m, EX[r in 1:NR, c in 1:NL] >= 0, start = E0[r, c])
@variable(m, D[r in 1:NR, d in 1:ND, c in 1:NL] >= 0, start = dom[r, d, c])
@variable(m, DD[d in 1:ND, c in 1:NL] >= 0, start = DD0[d, c])
@variable(m, MM[d in 1:ND, c in 1:NL] >= 0, start = MM0[d, c])
@variable(m, A[d in 1:ND, c in 1:NL] >= 0, start = A0[d, c])
@variable(m, QT >= 0, start = sum(trn))
@variable(m, QS[r in 1:NR, c in 1:NL] >= 0, start = oferta[r, c])
@variable(m, FD[l in 1:NF, r in 1:NR, c in 1:NL] >= 0, start = F0[l, r, c])
@variable(m, Y[h in 1:NH], start = Y0[h] * nu)
@variable(m, G[d in 1:ND, c in 1:NL] >= 0, start = G0[d, c])
@variable(m, I[d in 1:ND, c in 1:NL] >= 0, start = I0[d, c])
@variable(m, poupanca, start = sum(I0) * nu)
@variable(m, gasto_gov, start = sum(G0) * nu)
@variable(m, levy, start = 0.0)
@variable(m, Wp[l in keys(migra), o in 1:NR] >= lo, start = nu)
@variable(m, Sm[l in keys(migra), o in 1:NR, a in 1:NR] >= 0,
           start = migra[l].delta[o, a] * migra[l].dotacao[o])

fix(w[num_r, num_l], nu; force = true)
fixo_inv || fix(levy, 0.0; force = true)
transporte || fix(pt, nu; force = true)
transporte || fix(QT, 0.0; force = true)

pm(c) = er * mundo[c]
pe(c) = er * mundo_x[c]
pdel(r, d, c) = ps[r, c] + tau[r, d, c] * pt

# Valor adicionado e custo unitário.
for r in 1:NR, c in 1:NL
    usa = [l for l in 1:NF if theta[l, r, c] > 0]
    if abs(sigVA - 1) < 1e-9
        @constraint(m, log(pva[r, c]) == sum(theta[l, r, c] * log(w[r, l]) for l in usa))
        for l in 1:NF
            if theta[l, r, c] > 0
                @constraint(m, FD[l, r, c] * w[r, l] ==
                               theta[l, r, c] * pva[r, c] * vcoef[r, c] / tfp[r, c] * X[r, c])
            else
                fix(FD[l, r, c], 0.0; force = true)
            end
        end
    else
        @constraint(m, pva[r, c]^(1 - sigVA) ==
                       sum(theta[l, r, c] * w[r, l]^(1 - sigVA) for l in usa))
        for l in 1:NF
            if theta[l, r, c] > 0
                @constraint(m, FD[l, r, c] == theta[l, r, c] * (pva[r, c] / w[r, l])^sigVA *
                                              vcoef[r, c] / tfp[r, c] * X[r, c])
            else
                fix(FD[l, r, c], 0.0; force = true)
            end
        end
    end
    @constraint(m, pa[r, c] == sum(io[c2, r, c] / tfp[r, c] * pq[r, c2] for c2 in 1:NL) +
                               vcoef[r, c] / tfp[r, c] * pva[r, c])
end

# Vendas internas e exportações.
for r in 1:NR, c in 1:NL
    if cet && 0 < delta[r, c] < 1
        s = sigT[c]
        @constraint(m, pa[r, c]^(1 + s) == delta[r, c] * ps[r, c]^(1 + s) +
                                          (1 - delta[r, c]) * pe(c)^(1 + s))
        @constraint(m, DS[r, c] == delta[r, c] * (ps[r, c] / pa[r, c])^s * X[r, c])
        @constraint(m, EX[r, c] == (1 - delta[r, c]) * (pe(c) / pa[r, c])^s * X[r, c])
    else
        @constraint(m, ps[r, c] == pa[r, c])
        @constraint(m, DS[r, c] == delta[r, c] * X[r, c])
        @constraint(m, EX[r, c] == (1 - delta[r, c]) * X[r, c])
    end
    # Quem vende internamente vende aos destinos e, se presta o serviço, ao pool.
    @constraint(m, DS[r, c] == sum(D[r, d, c] for d in 1:ND) + QS[r, c])
end

# Pool de transporte Cobb–Douglas.
if transporte
    parte = oferta ./ sum(oferta)
    @constraint(m, log(pt) == sum(parte[r, c] * log(ps[r, c])
                                  for r in 1:NR, c in 1:NL if parte[r, c] > 0))
    @constraint(m, QT == sum(tau[r, d, c] * D[r, d, c] for r in 1:NR, d in 1:ND, c in 1:NL))
end
for r in 1:NR, c in 1:NL
    if transporte && oferta[r, c] > 0
        @constraint(m, QS[r, c] * ps[r, c] == parte[r, c] * pt * QT)
    else
        fix(QS[r, c], 0.0; force = true)
    end
end

# Destinos: composto de origens, Armington, demandas.
for d in 1:ND, c in 1:NL
    if !ativo[d, c]
        fix(P[d, c], nu; force = true); fix(pq[d, c], nu; force = true)
        fix(DD[d, c], 0.0; force = true); fix(MM[d, c], 0.0; force = true)
        for r in 1:NR
            fix(D[r, d, c], 0.0; force = true)
        end
        continue
    end
    orig = [r for r in 1:NR if alfa[r, d, c] > 0]
    s = sigI[c]
    if DD0[d, c] > 0
        @constraint(m, P[d, c]^(1 - s) == sum(alfa[r, d, c] * pdel(r, d, c)^(1 - s) for r in orig))
    else
        fix(P[d, c], nu; force = true)
    end
    for r in 1:NR
        if alfa[r, d, c] > 0
            @constraint(m, D[r, d, c] == alfa[r, d, c] * (P[d, c] / pdel(r, d, c))^s * DD[d, c])
        else
            fix(D[r, d, c], 0.0; force = true)
        end
    end
    sa = sigA[c]
    b = beta[d, c]
    if 0 < b < 1
        @constraint(m, pq[d, c]^(1 - sa) == b * P[d, c]^(1 - sa) + (1 - b) * pm(c)^(1 - sa))
        @constraint(m, DD[d, c] == b * (pq[d, c] / P[d, c])^sa * A[d, c])
        @constraint(m, MM[d, c] == (1 - b) * (pq[d, c] / pm(c))^sa * A[d, c])
    elseif b >= 1
        @constraint(m, pq[d, c] == P[d, c])
        @constraint(m, DD[d, c] == A[d, c])
        fix(MM[d, c], 0.0; force = true)
    else
        @constraint(m, pq[d, c] == pm(c))
        fix(DD[d, c], 0.0; force = true)
        @constraint(m, MM[d, c] == A[d, c])
    end
end

# Renda: fatores, migração, famílias, governo.
renda_f = Dict{Tuple{Int,Int},Any}()
for r in 1:NR, l in 1:NF
    if haskey(migra, l)
        mg = migra[l]
        renda_f[(r, l)] = @expression(m, Wp[l, r] * mg.dotacao[r])
    else
        renda_f[(r, l)] = @expression(m, w[r, l] * dot[r, l])
    end
end
for (l, mg) in migra, o in 1:NR
    om = mg.omega
    alvo = [a for a in 1:NR if mg.delta[o, a] > 0]
    if mg.forma == "cet"
        @constraint(m, Wp[l, o]^(1 + om) == sum(mg.delta[o, a] * w[a, l]^(1 + om) for a in alvo))
        for a in 1:NR
            if mg.delta[o, a] > 0
                @constraint(m, Sm[l, o, a] == mg.delta[o, a] * mg.dotacao[o] *
                                              (w[a, l] / Wp[l, o])^om)
            else
                fix(Sm[l, o, a], 0.0; force = true)
            end
        end
    else
        for a in 1:NR
            if mg.delta[o, a] > 0
                @constraint(m, Sm[l, o, a] * sum(mg.delta[o, b] * w[b, l]^om for b in alvo) ==
                               mg.dotacao[o] * mg.delta[o, a] * w[a, l]^om)
            else
                fix(Sm[l, o, a], 0.0; force = true)
            end
        end
        @constraint(m, Wp[l, o] * mg.dotacao[o] == sum(w[a, l] * Sm[l, o, a] for a in 1:NR))
    end
end
@constraint(m, [h in 1:NH], Y[h] == sum(dono[h, r, l] * renda_f[(r, l)]
                                        for r in 1:NR, l in 1:NF) - levy * pop[h])
receita = @expression(m, sum(ao_gov[r, l] * renda_f[(r, l)] for r in 1:NR, l in 1:NF))
if gov_dinamico
    @constraint(m, gasto_gov == (1 - sg) * receita)
    for d in 1:ND, c in 1:NL
        if G0[d, c] > 0
            @constraint(m, G[d, c] * pq[d, c] == G0[d, c] / sum(G0) * gasto_gov)
        else
            fix(G[d, c], 0.0; force = true)
        end
    end
else
    for d in 1:ND, c in 1:NL
        fix(G[d, c], G0[d, c]; force = true)
    end
    @constraint(m, gasto_gov == sum(pq[d, c] * G[d, c] for d in 1:ND, c in 1:NL))
end

# Poupança e investimento. Fixa a quantidade, um gravame sobre as famílias fecha a poupança;
# senão a poupança fixa o valor, com participações fixas por bem e, dentro de um bem,
# quantidades na proporção do ano-base entre os canais.
@constraint(m, poupanca == sum(pou[h] * Y[h] for h in 1:NH) + (receita - gasto_gov) +
                           FS0 * nu + levy)
if fixo_inv
    for d in 1:ND, c in 1:NL
        fix(I[d, c], I0[d, c] * nivel / sum(I0); force = true)
    end
    @constraint(m, poupanca == sum(pq[d, c] * I[d, c] for d in 1:ND, c in 1:NL))
else
    vs = [sum(I0[:, c]) / sum(I0) for c in 1:NL]
    for c in 1:NL
        canais = [d for d in 1:ND if I0[d, c] > 0]
        if isempty(canais)
            for d in 1:ND
                fix(I[d, c], 0.0; force = true)
            end
            continue
        end
        k = canais[1]
        @constraint(m, sum(pq[d, c] * I[d, c] for d in canais) == vs[c] * poupanca)
        for d in 1:ND
            if I0[d, c] > 0 && d != k
                @constraint(m, I[d, c] * I0[k, c] == I[k, c] * I0[d, c])
            elseif I0[d, c] <= 0
                fix(I[d, c], 0.0; force = true)
            end
        end
    end
end

# Absorção em cada destino.
for d in 1:ND, c in 1:NL
    ativo[d, c] || continue
    consumo = d > NR ? 0.0 :
              sum((orc[h, c] * (1 - pou[h]) * Y[h] / pq[d, c] for h in 1:NH if hreg[h] == d);
                  init = 0.0)
    insumos = d > NR ? 0.0 :
              sum(io[c, d, c2] / tfp[d, c2] * X[d, c2] for c2 in 1:NL)
    @constraint(m, A[d, c] == consumo + G[d, c] + I[d, c] + insumos)
end
# Quem comprava algo que não existe no ano-base não compra depois: o bem fica fora.
for d in 1:NR, c in 1:NL
    ativo[d, c] && continue
    @assert all(io[c, d, c2] == 0 for c2 in 1:NL) && all(orc[h, c] == 0 for h in 1:NH if hreg[h] == d)
    fix(A[d, c], 0.0; force = true)
end
for c in 1:NL
    ativo[ND, c] || fix(A[ND, c], 0.0; force = true)
end

# Mercados de fatores; o do numerário fica de fora e é o teste de Walras.
demanda(r, l) = sum(FD[l, r, c] for c in 1:NL)
for r in 1:NR, l in 1:NF
    (r == num_r && l == num_l) && continue
    if haskey(migra, l)
        @constraint(m, demanda(r, l) == sum(Sm[l, o, r] for o in 1:NR))
    else
        @constraint(m, demanda(r, l) == dot[r, l])
    end
end
# Conta corrente: a poupança externa fixa na moeda doméstica.
preco_x(r, c) = cet ? pe(c) : pa[r, c]
@constraint(m, sum(pm(c) * MM[d, c] for d in 1:ND, c in 1:NL) -
               sum(preco_x(r, c) * EX[r, c] for r in 1:NR, c in 1:NL) == FS0 * nu)

@objective(m, Min, 0.0)
optimize!(m)
st = termination_status(m)
println("Ipopt: ", st)

v(x) = value.(x)
oferta_num = haskey(migra, num_l) ? sum(value(Sm[num_l, o, num_r]) for o in 1:NR) : dot[num_r, num_l]
walras = value(demanda(num_r, num_l)) - oferta_num
viol = maximum(abs, values(primal_feasibility_report(m)); init = 0.0)
@printf("violação %.2e   WALRAS %.2e\n", viol, walras)

pares(x, f) = [f(r, c) for r in 1:NR for c in 1:NL]
out = Dict(
    "status" => string(st), "walras" => walras, "violacao_max" => viol,
    "w" => [value(w[r, l]) for r in 1:NR for l in 1:NF],
    "er" => value(er),
    "X" => pares(X, (r, c) -> value(X[r, c])),
    "exports" => pares(EX, (r, c) -> value(EX[r, c])),
    "domestic_sales" => pares(D, (r, c) -> sum(value(D[r, d, c]) for d in 1:ND)),
    "bilateral" => [value(D[r, d, c]) for r in 1:NR, d in 1:NR, c in 1:NL][:],
    "imports_total" => sum(v(MM)),
    "pq_rel" => [value(pq[d, c]) for d in 1:ND, c in 1:NL][:],
    "hh_income" => [value(Y[h]) for h in 1:NH],
    "hh_spend" => [(1 - pou[h]) * value(Y[h]) for h in 1:NH],
    "savings" => value(poupanca),
    "qst" => [value(QS[r, c]) for r in 1:NR, c in 1:NL][:],
    "ativo" => [ativo[d, c] for d in 1:ND, c in 1:NL][:],
)
open(SAIDA, "w") do f
    JSON3.write(f, out)
end
