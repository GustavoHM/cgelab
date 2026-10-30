# least_cost.jl — produtores que competem ao menor custo, como problema de complementaridade
# misto resolvido pelo PATH.
#
# Nada vem do cgelab. O script lê a MCS e o enunciado (papéis das contas, elasticidades, as
# commodities de menor custo, o choque), calibra o modelo nacional aqui e escreve cada
# condição pareada com a sua variável, como no GAMS/MPSGE:
#
#     custo unitário_j − preço_i ≥ 0   ⟂   X_j ≥ 0     (cada produtor de i)
#     Σ_j X_j − Q_i = 0                ⟂   preço_i      (o mercado do bem)
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
mundo = [Float64(E.world_price[Symbol(c)]) for c in COM]

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
fix(w[num], 1.0; force = true)

@expression(m, pva[j in 1:NA],
            sum(theta[f, j] * w[f]^(1 - sva) for f in 1:NF if theta[f, j] > 0)^(1 / (1 - sva)))
@expression(m, pa[j in 1:NA],
            sum(io[i, j] / tfp[j] * pq[i] for i in 1:NC) + vunit[j] / tfp[j] * pva[j] + atax[j])
@expression(m, fd[f in 1:NF, j in 1:NA],
            theta[f, j] > 0 ? vunit[j] / tfp[j] * X[j] * theta[f, j] * (pva[j] / w[f])^sva : 0.0)
pm(i) = er * mundo[i]
pb(i) = pq[i] / (1 + ptax[i])                                 # preço Armington, básico
@expression(m, DD[i in 1:NC], delta[i] * (pb(i) / pd[i])^sarm[i] * A[i])
@expression(m, MM[i in 1:NC], (1 - delta[i]) * (pb(i) / pm(i))^sarm[i] * A[i])
@expression(m, receita, sum(pb(i) * ptax[i] * A[i] for i in 1:NC) + sum(atax[j] * X[j] for j in 1:NA))
@expression(m, gasto_gov, sum(pq[i] * gq[i] for i in 1:NC))

# Preço ao comprador: o composto de Armington, com o imposto.
@constraint(m, [i in 1:NC],
            pq[i] - (1 + ptax[i]) * (delta[i] * pd[i]^(1 - sarm[i])
                                     + (1 - delta[i]) * pm(i)^(1 - sarm[i]))^(1 / (1 - sarm[i])) ⟂ pq[i])
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
            A[i] - (sum(orc[h, i] * (1 - pou[h]) * Y[h] for h in 1:NH) / pq[i] + gq[i]
                    + vs[i] * poup / pq[i] + sum(io[i, j] / tfp[j] * X[j] for j in 1:NA)) ⟂ A[i])
@constraint(m, [i in 1:NC], Q[i] - DD[i] / (1 - es[i]) ⟂ Q[i])
# Renda, poupança, fatores, conta corrente.
@constraint(m, [h in 1:NH], Y[h] - sum(dono[h, f] * w[f] * L0[f] for f in 1:NF) ⟂ Y[h])
@constraint(m, poup - (sum(pou[h] * Y[h] for h in 1:NH) + receita - gasto_gov + FS0) ⟂ poup)
for f in 1:NF
    f == num && continue
    @constraint(m, sum(fd[f, j] for j in 1:NA) - L0[f] ⟂ w[f])
end
@constraint(m, sum(pm(i) * MM[i] for i in 1:NC) - sum(pd[i] * es[i] * Q[i] for i in 1:NC) - FS0 ⟂ er)

optimize!(m)
st = termination_status(m)
println("PATH: ", st)
walras = value(sum(fd[num, j] for j in 1:NA)) - L0[num]
@printf("WALRAS %.2e\n", walras)
out = Dict("status" => string(st), "walras" => walras,
           "w" => value.(w)[:], "er" => value(er), "X" => value.(X)[:], "pd" => value.(pd)[:],
           "pq" => value.(pq)[:], "pa" => [value(pa[j]) for j in 1:NA],
           "hh_income" => value.(Y)[:], "savings" => value(poup))
open(SAIDA, "w") do f
    JSON3.write(f, out)
end
