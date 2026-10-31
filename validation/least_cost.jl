# least_cost.jl — produtores que competem ao menor custo, limites de capacidade, cotas de
# importação, tetos e pisos de preço, piso salarial e tecnologias de reserva (backstop), como
# problema de complementaridade misto resolvido pelo PATH.
#
# Nada vem do cgelab. O script lê a MCS e o enunciado (papéis das contas, elasticidades, as
# commodities de menor custo, o choque), calibra o modelo nacional aqui e escreve cada
# condição pareada com a sua variável, como no GAMS/MPSGE:
#
#     custo unitário_j − preço_i ≥ 0   ⟂   X_j ≥ 0     (cada produtor de i)
#     Σ_j X_j − Q_i = 0                ⟂   preço_i      (o mercado do bem)
#     capacidade_j − X_j ≥ 0           ⟂   renda_j ≥ 0  (cada atividade com limite)
#     cota_i − M_i ≥ 0                 ⟂   chi_i ≥ 0    (cada bem com cota de importação)
#     teto_i·P·pq0_i − pq_i ≥ 0        ⟂   sub_i ≥ 0    (cada bem com teto de preço)
#     pq_i − piso_i·P·pq0_i ≥ 0        ⟂   compra_i ≥ 0 (cada bem com piso de preço)
#     w_L/P − piso salarial ≥ 0        ⟂   U ≥ 0        (com o piso salarial)
#
# Uma tecnologia de reserva não produz no ano-base e não está na MCS: os seus custos por unidade
# vêm do enunciado, ou como os de outra atividade (insumos e fatores vezes o markup, o imposto
# sobre a atividade por unidade como está), ou como valores de cada insumo, a preço do comprador,
# e de cada fator. Ela entra como mais um produtor do bem, ao menor custo.
#
# P é o índice de Laspeyres dos preços ao comprador, com as quantidades das famílias no ano-base.
# O subsídio sai da alíquota do imposto sobre produtos, para firmas e famílias; as compras de
# sustentação entram na absorção e saem da poupança do governo; os desempregados não recebem.
#
# chi é um prêmio ad valorem sobre o preço de importação na fronteira; a renda chi·pm·M vai ao
# governo ou aos donos do fator indicado.
#
# A renda de escassez entra no custo unitário da atividade e é paga aos donos do fator indicado,
# como renda × produção.
#
# e as demais equações do equilíbrio como igualdades pareadas com variáveis livres. O PATH
# resolve o sistema inteiro de uma vez. O cgelab resolve o mesmo modelo por aninhamento e
# escreve a complementaridade com a função de Fischer–Burmeister dentro do Newton; aqui não há
# nem aninhamento nem suavização.
#
# Escopo, recusado pelo script em Python fora dele: imposto sobre produtos com a mesma alíquota
# para firmas e famílias, imposto sobre a atividade por unidade, sem margens, exportações em
# proporção fixa, famílias Cobb–Douglas sem imposto direto nem transferências, governo com
# consumo real fixo, investimento savings_driven e poupança externa fixa no numerário.
#
#   uso:  julia --project=validation validation/least_cost.jl entrada.json saida.json

using JuMP, PATHSolver, JSON3, Printf

const E = JSON3.read(read(ARGS[1], String))
const SAIDA = ARGS[2]

S = Dict{Tuple{String,String},Float64}()
for (u, v, x) in E.sam
    S[(String(u), String(v))] = get(S, (String(u), String(v)), 0.0) + Float64(x)
end
sam(u, v) = get(S, (u, v), 0.0)

ATV = String.(E.activities)
COM = String.(E.commodities)
FAT = String.(E.factors)
FAM = String.(E.households)
gov, inv, row = String(E.government), String(E.investment), String(E.rest_of_world)
NA, NC, NF, NH = length(ATV), length(COM), length(FAT), length(FAM)

# ---------------------------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------------------------
X0 = [sum(sam(ATV[j], COM[i]) for i in 1:NC) for j in 1:NA]
faz = [sam(ATV[j], COM[i]) for j in 1:NA, i in 1:NC]          # make
Q0 = [sum(faz[:, i]) for i in 1:NC]
E0 = [sam(COM[i], row) for i in 1:NC]
M0 = [sam(row, COM[i]) for i in 1:NC]
imp_prod = [sam(gov, COM[i]) for i in 1:NC]                     # imposto sobre produtos
D0 = Q0 .- E0                                                   # vendas internas
A0 = D0 .+ M0                                                   # absorção a preço básico
ptax = imp_prod ./ A0
delta = D0 ./ A0
es = E0 ./ Q0
io = [sam(COM[i], ATV[j]) / (1 + ptax[i]) / X0[j] for i in 1:NC, j in 1:NA]
F0 = [sam(FAT[f], ATV[j]) for f in 1:NF, j in 1:NA]
VA0 = [sum(F0[:, j]) for j in 1:NA]
vunit = VA0 ./ X0
theta = F0 ./ VA0'
atax = [sam(gov, ATV[j]) / X0[j] for j in 1:NA]
L0 = [sum(F0[f, :]) for f in 1:NF]
dono = [sam(FAM[h], FAT[f]) / L0[f] for h in 1:NH, f in 1:NF]
Y0 = [sum(sam(FAM[h], FAT[f]) for f in 1:NF) for h in 1:NH]
pou = [sam(inv, FAM[h]) / Y0[h] for h in 1:NH]
orc = [sam(COM[i], FAM[h]) for h in 1:NH, i in 1:NC]
orc = orc ./ sum(orc; dims = 2)
gq = [sam(COM[i], gov) / (1 + ptax[i]) for i in 1:NC]
vs = [sam(COM[i], inv) for i in 1:NC]
vs = vs ./ sum(vs)
FS0 = sum(M0) - sum(E0)
quem = Dict(i => [j for j in 1:NA if faz[j, i] > 0] for i in 1:NC)
for b in E.backstops
    i = findfirst(==(String(b.commodity)), COM)
    if b.like !== nothing
        j = findfirst(==(String(b.like)), ATV)
        m = Float64(b.markup)
        io_b, f_b, t_b = io[:, j] .* m, F0[:, j] ./ X0[j] .* m, atax[j]
    else
        io_b = [Float64(get(b.costs, Symbol(COM[k]), 0.0)) / (1 + ptax[k]) for k in 1:NC]
        f_b = [Float64(get(b.costs, Symbol(FAT[f]), 0.0)) for f in 1:NF]
        t_b = 0.0
    end
    global io = hcat(io, io_b)
    global theta = hcat(theta, f_b ./ sum(f_b))
    push!(ATV, String(b.name)); push!(X0, 0.0); push!(vunit, sum(f_b)); push!(atax, t_b)
    push!(quem[i], length(ATV))
    global NA = length(ATV)
end
for i in 1:NC
    @assert !isempty(quem[i])
end
menor_custo = Set(findfirst(==(String(c)), COM) for c in E.least_cost)
for i in 1:NC
    i in menor_custo || @assert length(quem[i]) == 1 "$(COM[i]) tem mais de um produtor fora do menor custo"
end

sva = Float64(E.sigma_va)
sarm = [Float64(E.sigma_armington[Symbol(c)]) for c in COM]
num = findfirst(==(String(E.numeraire_factor)), FAT)
tfp = [Float64(E.productivity[Symbol(a)]) for a in ATV]
limite = Dict(findfirst(==(String(a)), ATV) => Float64(v) for (a, v) in E.capacity)
cota = Dict(findfirst(==(String(c)), COM) => Float64(v) for (c, v) in E.import_quota)
cota_ao_governo = String(E.quota_rent_to) == "government"
dono_cota = cota_ao_governo ? 0 : findfirst(==(String(E.quota_rent_to)), FAT)
dono_renda = findfirst(==(String(E.capacity_owner)), FAT)
mundo = [Float64(E.world_price[Symbol(c)]) for c in COM]
teto = Dict(findfirst(==(String(c)), COM) => Float64(v) for (c, v) in E.price_ceiling)
piso = Dict(findfirst(==(String(c)), COM) => Float64(v) for (c, v) in E.price_floor)
piso_salarial = String(E.labour) == "wage_floor"
lab = piso_salarial ? findfirst(==(String(E.labour_factor)), FAT) : 0
salario_minimo = Float64(E.real_wage)
q0 = [sum(sam(COM[i], FAM[h]) for h in 1:NH) / (1 + ptax[i]) for i in 1:NC]
oferta = [Float64(E.factor_supply[Symbol(f)]) for f in FAT]       # oferta dos fatores no cenário

# ---------------------------------------------------------------------------------------------
# O problema de complementaridade
# ---------------------------------------------------------------------------------------------
m = Model(PATHSolver.Optimizer)
set_silent(m)
set_optimizer_attribute(m, "convergence_tolerance", 1e-13)
lo = 1e-8
@variable(m, w[f in 1:NF] >= lo, start = 1.0)
@variable(m, er >= lo, start = 1.0)
@variable(m, pd[i in 1:NC] >= lo, start = 1.0)
@variable(m, pq[i in 1:NC] >= lo, start = 1 + ptax[i])
@variable(m, A[i in 1:NC] >= 0, start = A0[i])
@variable(m, Q[i in 1:NC] >= 0, start = Q0[i])
@variable(m, X[j in 1:NA] >= 0, start = X0[j])
@variable(m, Y[h in 1:NH], start = Y0[h])
@variable(m, poup, start = sum(sam(COM[i], inv) for i in 1:NC))
@variable(m, renda[j in 1:NA] >= 0, start = 0.0)
@variable(m, chi[i in 1:NC] >= 0, start = 0.0)
for i in 1:NC
    haskey(cota, i) || fix(chi[i], 0.0; force = true)
end
for j in 1:NA
    haskey(limite, j) || fix(renda[j], 0.0; force = true)
end
@variable(m, sub[i in 1:NC] >= 0, start = 0.0)
@variable(m, compra[i in 1:NC] >= 0, start = 0.0)
@variable(m, U >= 0, start = 0.0)
for i in 1:NC
    haskey(teto, i) || fix(sub[i], 0.0; force = true)
    haskey(piso, i) || fix(compra[i], 0.0; force = true)
end
piso_salarial || fix(U, 0.0; force = true)
emprego(f) = f == lab ? oferta[f] - U : oferta[f]
fix(w[num], 1.0; force = true)

@expression(m, pva[j in 1:NA],
            sum(theta[f, j] * w[f]^(1 - sva) for f in 1:NF if theta[f, j] > 0)^(1 / (1 - sva)))
@expression(m, pa[j in 1:NA],
            sum(io[i, j] / tfp[j] * pq[i] for i in 1:NC) + vunit[j] / tfp[j] * pva[j] + atax[j]
            + renda[j])
@expression(m, fd[f in 1:NF, j in 1:NA],
            theta[f, j] > 0 ? vunit[j] / tfp[j] * X[j] * theta[f, j] * (pva[j] / w[f])^sva : 0.0)
pm(i) = er * mundo[i]                                         # na fronteira
pmc(i) = pm(i) * (1 + chi[i])                                 # com o prêmio da cota
tx(i) = ptax[i] - sub[i]                                      # alíquota depois do subsídio
pb(i) = pq[i] / (1 + tx(i))                                   # preço Armington, básico
@expression(m, DD[i in 1:NC], delta[i] * (pb(i) / pd[i])^sarm[i] * A[i])
@expression(m, MM[i in 1:NC], (1 - delta[i]) * (pb(i) / pmc(i))^sarm[i] * A[i])
@expression(m, renda_cota, sum(chi[i] * pm(i) * MM[i] for i in keys(cota); init = 0.0))
@expression(m, receita, sum(pb(i) * tx(i) * A[i] for i in 1:NC) + sum(atax[j] * X[j] for j in 1:NA)
                        + (cota_ao_governo ? renda_cota : 0.0))
@expression(m, gasto_gov, sum(pq[i] * (gq[i] + compra[i]) for i in 1:NC))
@expression(m, P, sum(pq[i] * q0[i] for i in 1:NC) / sum((1 + ptax[i]) * q0[i] for i in 1:NC))

# Preço ao comprador: o composto de Armington, com o imposto.
@constraint(m, [i in 1:NC],
            pq[i] - (1 + tx(i)) * (delta[i] * pd[i]^(1 - sarm[i])
                                     + (1 - delta[i]) * pmc(i)^(1 - sarm[i]))^(1 / (1 - sarm[i])) ⟂ pq[i])
# Produção: lucro zero e quantidades.
for i in 1:NC
    if i in menor_custo
        for j in quem[i]
            @constraint(m, pa[j] - pd[i] ⟂ X[j])                 # custo ≥ preço ⟂ X ≥ 0
        end
        @constraint(m, sum(X[j] for j in quem[i]) - Q[i] ⟂ pd[i])
    else
        j = only(quem[i])
        @constraint(m, pa[j] - pd[i] ⟂ pd[i])
        @constraint(m, X[j] - Q[i] ⟂ X[j])
    end
end
# Absorção e produção do bem.
@constraint(m, [i in 1:NC],
            A[i] - (sum(orc[h, i] * (1 - pou[h]) * Y[h] for h in 1:NH) / pq[i] + gq[i] + compra[i]
                    + vs[i] * poup / pq[i] + sum(io[i, j] / tfp[j] * X[j] for j in 1:NA)) ⟂ A[i])
@constraint(m, [i in 1:NC], Q[i] - DD[i] / (1 - es[i]) ⟂ Q[i])
# Renda, poupança, fatores, conta corrente.
@expression(m, renda_total, sum(renda[j] * X[j] for j in keys(limite); init = 0.0))
@constraint(m, [h in 1:NH], Y[h] - sum(dono[h, f] * w[f] * emprego(f) for f in 1:NF)
                            - dono[h, dono_renda] * renda_total
                            - (cota_ao_governo ? 0.0 : dono[h, dono_cota] * renda_cota) ⟂ Y[h])
for (j, cap) in limite
    @constraint(m, cap - X[j] ⟂ renda[j])                   # capacidade − X ≥ 0 ⟂ renda ≥ 0
end
for (i, q) in cota
    @constraint(m, q - MM[i] ⟂ chi[i])                      # cota − M ≥ 0 ⟂ chi ≥ 0
end
for (i, b) in teto
    @constraint(m, b * P * (1 + ptax[i]) - pq[i] ⟂ sub[i])  # teto − pq ≥ 0 ⟂ subsídio ≥ 0
end
for (i, b) in piso
    @constraint(m, pq[i] - b * P * (1 + ptax[i]) ⟂ compra[i])   # pq − piso ≥ 0 ⟂ compras ≥ 0
end
if piso_salarial
    @constraint(m, w[lab] / P - salario_minimo ⟂ U)         # w/P − piso ≥ 0 ⟂ U ≥ 0
end
@constraint(m, poup - (sum(pou[h] * Y[h] for h in 1:NH) + receita - gasto_gov + FS0) ⟂ poup)
for f in 1:NF
    f == num && continue
    @constraint(m, sum(fd[f, j] for j in 1:NA) - emprego(f) ⟂ w[f])
end
@constraint(m, sum(pm(i) * MM[i] for i in 1:NC) - sum(pd[i] * es[i] * Q[i] for i in 1:NC) - FS0 ⟂ er)

optimize!(m)
st = termination_status(m)
println("PATH: ", st)
walras = value(sum(fd[num, j] for j in 1:NA)) - oferta[num]
@printf("WALRAS %.2e\n", walras)
out = Dict("status" => string(st), "walras" => walras,
           "w" => value.(w)[:], "er" => value(er), "X" => value.(X)[:], "pd" => value.(pd)[:],
           "pq" => value.(pq)[:], "pa" => [value(pa[j]) for j in 1:NA],
           "hh_income" => value.(Y)[:], "savings" => value(poup), "rent" => value.(renda)[:],
           "premium" => value.(chi)[:], "subsidy" => value.(sub)[:],
           "purchases" => value.(compra)[:], "unemployment" => value(U))
open(SAIDA, "w") do f
    JSON3.write(f, out)
end
