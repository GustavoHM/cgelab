# cross_entropy.jl — a MCS balanceada mais próxima de uma MCS a priori, pelo primal.
#
# Nada vem do cgelab. O script lê a MCS a priori e a informação (totais de contas, células
# fixas, células-alvo, agregados lineares) e resolve o problema primal com o Ipopt:
#
#     min  Σ_ij T_ij log(T_ij / P_ij) − T_ij + P_ij
#     s.a. Σ_j T_ij = Σ_j T_ji                  para cada conta i (receitas = gastos)
#          Σ_j T_ij = total_i                    para cada conta com total dado
#          T_ij = P_ij                           para cada célula fixa
#          T_ij = alvo_ij                        para cada célula-alvo
#          Σ peso_ij T_ij = valor                para cada agregado
#          T_ij ≥ 0, e T_ij = 0 onde P_ij = 0.
#
# Uma entrada negativa da MCS a priori vai para a célula transposta com o sinal trocado antes
# de resolver, e volta depois. O cgelab resolve o dual por Newton; aqui não há dual.
#
#   uso:  julia --project=validation validation/cross_entropy.jl entrada.json saida.json

using JuMP, Ipopt, JSON3, Printf

const E = JSON3.read(read(ARGS[1], String))
const SAIDA = ARGS[2]

contas = String.(E.accounts)
idx = Dict(a => k for (k, a) in enumerate(contas))
celulas = Tuple{Int,Int}[]
P = Float64[]
sinal = Float64[]
origem = Dict{Tuple{Int,Int},Int}()
for (u, v, x) in E.prior
    x = Float64(x)
    x == 0 && continue
    i, j = idx[String(u)], idx[String(v)]
    if x < 0
        push!(celulas, (j, i)); push!(sinal, -1.0)
    else
        push!(celulas, (i, j)); push!(sinal, 1.0)
    end
    push!(P, abs(x))
    origem[(i, j)] = length(P)
end
M = length(P)
loc(u, v) = origem[(idx[String(u)], idx[String(v)])]

m = Model(Ipopt.Optimizer)
set_silent(m)
set_optimizer_attribute(m, "tol", 1e-12)
set_optimizer_attribute(m, "acceptable_tol", 1e-12)
@variable(m, T[k in 1:M] >= 1e-12, start = P[k])
@objective(m, Min, sum(T[k] * log(T[k] / P[k]) - T[k] + P[k] for k in 1:M))
receitas(i) = sum(T[k] for k in 1:M if celulas[k][1] == i; init = 0.0)
gastos(i) = sum(T[k] for k in 1:M if celulas[k][2] == i; init = 0.0)
for i in eachindex(contas)
    @constraint(m, receitas(i) == gastos(i))
end
for (a, v) in E.totals
    @constraint(m, receitas(idx[String(a)]) == Float64(v))
end
for (u, v) in E.fixed
    k = loc(u, v)
    fix(T[k], P[k]; force = true)
end
for (u, v, x) in E.targets
    k = loc(u, v)
    @constraint(m, sinal[k] * T[k] == Float64(x))
end
for ag in E.aggregates
    @constraint(m, sum(sinal[loc(u, v)] * Float64(w) * T[loc(u, v)] for (u, v, w) in ag.weights)
                   == Float64(ag.value))
end
optimize!(m)
st = termination_status(m)
println("Ipopt: ", st)
t = value.(T)
saida = [[contas[sinal[k] > 0 ? celulas[k][1] : celulas[k][2]],
          contas[sinal[k] > 0 ? celulas[k][2] : celulas[k][1]], sinal[k] * t[k]] for k in 1:M]
div = sum(t[k] * log(t[k] / P[k]) - t[k] + P[k] for k in 1:M)
@printf("DIVERGENCIA %.12e\n", div)
open(SAIDA, "w") do f
    JSON3.write(f, Dict("status" => string(st), "sam" => saida, "divergence" => div))
end
