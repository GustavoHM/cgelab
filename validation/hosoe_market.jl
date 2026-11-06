# hosoe_market.jl — transcrição em JuMP de moncge.gms e irscge.gms, da biblioteca de modelos do
# GAMS.
#
# Hosoe, Gasawa e Hashimoto (2004), Handbook of Computable General Equilibrium Modeling, caps.
# 10.4 (monopólio) e 10.6 (economias de escala). O GAMS não está instalado aqui, então o código
# publicado foi transcrito linha a linha: a mesma MCS, a mesma calibração e as mesmas equações,
# com os mesmos nomes. Nada vem do cgelab. O GAMS maximiza uma utilidade fictícia sujeita a um
# sistema quadrado; aqui o sistema é resolvido por viabilidade, que é o mesmo ponto.
#
# moncge: o produtor doméstico de cada bem é um monopolista que cobra o markup de Lerner sobre as
# vendas internas. Os compradores pagam pd/eta pela variedade doméstica; a renda de monopólio
# RT = (1 − eta)/eta·pd·D é renda da família. sigma = 3.
#
# irscge: um custo fixo FC = nu·(pagamento ao capital), nu = 0,1, entra no custo unitário como
# FC/Z e é renda da família. sigma = 2.
#
# Resolve o ano-base e a abolição das tarifas (taum = 0) de cada um e escreve os níveis.
#
#   uso:  julia --project=validation validation/hosoe_market.jl [saida.json]

using JuMP, Ipopt, JSON3, Printf

const SAIDA = length(ARGS) >= 1 ? ARGS[1] : "hosoe_market_out.json"

const U = ["BRD", "MLK", "CAP", "LAB", "IDT", "TRF", "HOH", "GOV", "INV", "EXT"]
const I = ["BRD", "MLK"]
const H = ["CAP", "LAB"]
SAM = Dict{Tuple{String,String},Float64}()
for (u, v, x) in [
        ("BRD", "BRD", 21), ("BRD", "MLK", 8), ("BRD", "HOH", 20), ("BRD", "GOV", 19),
        ("BRD", "INV", 16), ("BRD", "EXT", 8),
        ("MLK", "BRD", 17), ("MLK", "MLK", 9), ("MLK", "HOH", 30), ("MLK", "GOV", 14),
        ("MLK", "INV", 15), ("MLK", "EXT", 4),
        ("CAP", "BRD", 20), ("CAP", "MLK", 30),
        ("LAB", "BRD", 15), ("LAB", "MLK", 25),
        ("IDT", "BRD", 5), ("IDT", "MLK", 4),
        ("TRF", "BRD", 1), ("TRF", "MLK", 2),
        ("HOH", "CAP", 50), ("HOH", "LAB", 40),
        ("GOV", "IDT", 9), ("GOV", "TRF", 3), ("GOV", "HOH", 23),
        ("INV", "HOH", 17), ("INV", "GOV", 2), ("INV", "EXT", 12),
        ("EXT", "BRD", 13), ("EXT", "MLK", 11)]
    SAM[(u, v)] = x
end
sam(u, v) = get(SAM, (u, v), 0.0)
for u in U
    @assert abs(sum(sam(u, v) for v in U) - sum(sam(v, u) for v in U)) < 1e-12 u
end

"""Calibração e solução de moncge (`modelo = :mon`) ou irscge (`modelo = :irs`)."""
function modelo_hosoe(modelo::Symbol, zera_tarifa::Bool)
    mon = modelo == :mon
    sigma = Dict(i => (mon ? 3.0 : 2.0) for i in I)
    psi = Dict(i => 2.0 for i in I)
    eta = Dict(i => (sigma[i] - 1) / sigma[i] for i in I)
    phi = Dict(i => (psi[i] + 1) / psi[i] for i in I)
    nu = Dict(i => 0.1 for i in I)

    # * Loading the initial values
    Td0 = sam("GOV", "HOH")
    Tz0 = Dict(j => sam("IDT", j) for j in I)
    Tm0 = Dict(j => sam("TRF", j) for j in I)
    X0 = Dict((i, j) => sam(i, j) for i in I, j in I)
    M0 = Dict(i => sam("EXT", i) for i in I)
    Xp0 = Dict(i => sam(i, "HOH") for i in I)
    Xg0 = Dict(i => sam(i, "GOV") for i in I)
    Xv0 = Dict(i => sam(i, "INV") for i in I)
    E0 = Dict(i => sam(i, "EXT") for i in I)
    taum0 = Dict(j => Tm0[j] / M0[j] for j in I)
    Q0 = Dict(i => Xp0[i] + Xg0[i] + Xv0[i] + sum(X0[(i, j)] for j in I) for i in I)
    RT0 = Dict(j => 0.0 for j in I)
    FC = Dict(j => 0.0 for j in I)
    if mon
        D0 = Dict(i => (Q0[i] - (1 + taum0[i]) * M0[i]) / (1 / eta[i]) for i in I)
        RT0 = Dict(j => (1 - eta[j]) / eta[j] * D0[j] for j in I)
        F0 = Dict((h, j) => sam(h, j) - sam(h, j) / sum(sam(k, j) for k in H) * RT0[j]
                  for h in H, j in I)
        Y0 = Dict(j => sum(F0[(h, j)] for h in H) for j in I)
        Z0 = Dict(j => Y0[j] + sum(X0[(i, j)] for i in I) for j in I)
    else
        F0 = Dict((h, j) => (h == "CAP" ? (1 - nu[j]) * sam(h, j) : sam(h, j)) for h in H, j in I)
        FC = Dict(j => nu[j] * sam("CAP", j) for j in I)
        Y0 = Dict(j => sum(F0[(h, j)] for h in H) for j in I)
        Z0 = Dict(j => Y0[j] + sum(X0[(i, j)] for i in I) + FC[j] for j in I)
    end
    tauz = Dict(j => Tz0[j] / Z0[j] for j in I)
    if !mon
        D0 = Dict(i => (1 + tauz[i]) * Z0[i] - E0[i] for i in I)
    end
    FF = Dict(h => sum(F0[(h, j)] for j in I) for h in H)
    Sp0 = sam("INV", "HOH")
    Sg0 = sam("INV", "GOV")
    Sf = sam("INV", "EXT")
    # O valor da variedade doméstica para o comprador: pd/eta no monopólio, pd sem ele.
    kd = Dict(i => (mon ? 1 / eta[i] : 1.0) for i in I)

    # * Calibration
    alpha = Dict(i => Xp0[i] / sum(Xp0[j] for j in I) for i in I)
    beta = Dict((h, j) => F0[(h, j)] / sum(F0[(k, j)] for k in H) for h in H, j in I)
    b = Dict(j => Y0[j] / prod(F0[(h, j)]^beta[(h, j)] for h in H) for j in I)
    ax = Dict((i, j) => X0[(i, j)] / Z0[j] for i in I, j in I)
    ay = Dict(j => Y0[j] / Z0[j] for j in I)
    mu = Dict(i => Xg0[i] / sum(Xg0[j] for j in I) for i in I)
    lambda = Dict(i => Xv0[i] / (Sp0 + Sg0 + Sf) for i in I)
    deltam = Dict(i => (1 + taum0[i]) * M0[i]^(1 - eta[i]) /
                       ((1 + taum0[i]) * M0[i]^(1 - eta[i]) + kd[i] * D0[i]^(1 - eta[i])) for i in I)
    deltad = Dict(i => kd[i] * D0[i]^(1 - eta[i]) /
                       ((1 + taum0[i]) * M0[i]^(1 - eta[i]) + kd[i] * D0[i]^(1 - eta[i])) for i in I)
    gamma = Dict(i => Q0[i] / (deltam[i] * M0[i]^eta[i] + deltad[i] * D0[i]^eta[i])^(1 / eta[i])
                 for i in I)
    xie = Dict(i => E0[i]^(1 - phi[i]) / (E0[i]^(1 - phi[i]) + D0[i]^(1 - phi[i])) for i in I)
    xid = Dict(i => D0[i]^(1 - phi[i]) / (E0[i]^(1 - phi[i]) + D0[i]^(1 - phi[i])) for i in I)
    theta = Dict(i => Z0[i] / (xie[i] * E0[i]^phi[i] + xid[i] * D0[i]^phi[i])^(1 / phi[i]) for i in I)
    extra0 = sum(RT0[j] for j in I) + sum(FC[j] for j in I)
    ssp = Sp0 / (sum(FF[h] for h in H) + extra0)
    ssg = Sg0 / (Td0 + sum(Tz0[j] for j in I) + sum(Tm0[j] for j in I))
    taud = Td0 / (sum(FF[h] for h in H) + extra0)

    taum = zera_tarifa ? Dict(i => 0.0 for i in I) : taum0
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "tol", 1e-12)
    set_optimizer_attribute(m, "constr_viol_tol", 1e-12)
    lo = 0.00001
    @variable(m, Y[j in I] >= lo, start = Y0[j])
    @variable(m, F[h in H, j in I] >= lo, start = F0[(h, j)])
    @variable(m, X[i in I, j in I] >= lo, start = X0[(i, j)])
    @variable(m, Z[j in I] >= lo, start = Z0[j])
    @variable(m, Xp[i in I] >= lo, start = Xp0[i])
    @variable(m, Xg[i in I] >= lo, start = Xg0[i])
    @variable(m, Xv[i in I] >= lo, start = Xv0[i])
    @variable(m, E[i in I] >= lo, start = E0[i])
    @variable(m, M[i in I] >= lo, start = M0[i])
    @variable(m, Q[i in I] >= lo, start = Q0[i])
    @variable(m, D[i in I] >= lo, start = D0[i])
    @variable(m, pf[h in H] >= lo, start = 1)
    @variable(m, py[j in I] >= lo, start = 1)
    @variable(m, pz[j in I] >= lo, start = 1)
    @variable(m, pq[i in I] >= lo, start = 1)
    @variable(m, pe[i in I] >= lo, start = 1)
    @variable(m, pm[i in I] >= lo, start = 1)
    @variable(m, pd[i in I] >= lo, start = 1)
    @variable(m, epsilon >= lo, start = 1)
    @variable(m, Sp >= lo, start = Sp0)
    @variable(m, Sg >= lo, start = Sg0)
    @variable(m, Td >= lo, start = Td0)
    @variable(m, Tz[j in I] >= 0, start = Tz0[j])
    @variable(m, Tm[i in I] >= 0, start = Tm0[i])
    @variable(m, RT[j in I], start = RT0[j])
    pWe = Dict(i => 1.0 for i in I)
    pWm = Dict(i => 1.0 for i in I)
    renda = @expression(m, sum(pf[h] * FF[h] for h in H) + sum(RT[j] for j in I)
                           + sum(FC[j] for j in I))

    # * domestic production
    @constraint(m, [j in I], Y[j] == b[j] * prod(F[h, j]^beta[(h, j)] for h in H))
    @constraint(m, [i in I, j in I], X[i, j] == ax[(i, j)] * Z[j])
    @constraint(m, [j in I], Y[j] == ay[j] * Z[j])
    @constraint(m, [h in H, j in I], F[h, j] == beta[(h, j)] * py[j] * Y[j] / pf[h])
    @constraint(m, [j in I], pz[j] == ay[j] * py[j] + sum(ax[(i, j)] * pq[i] for i in I)
                                      + FC[j] / Z[j])
    # * government behavior
    @constraint(m, Td == taud * renda)
    @constraint(m, [j in I], Tz[j] == tauz[j] * pz[j] * Z[j])
    @constraint(m, [i in I], Tm[i] == taum[i] * pm[i] * M[i])
    @constraint(m, [i in I],
                Xg[i] == mu[i] * (Td + sum(Tz[j] for j in I) + sum(Tm[j] for j in I) - Sg) / pq[i])
    # * investment behavior
    @constraint(m, [i in I], Xv[i] == lambda[i] * (Sp + Sg + epsilon * Sf) / pq[i])
    # * savings
    @constraint(m, Sp == ssp * renda)
    @constraint(m, Sg == ssg * (Td + sum(Tz[j] for j in I) + sum(Tm[j] for j in I)))
    # * household consumption
    @constraint(m, [i in I], Xp[i] == alpha[i] * (renda - Sp - Td) / pq[i])
    @constraint(m, [j in I], RT[j] == (mon ? (1 - eta[j]) / eta[j] * pd[j] * D[j] : 0.0))
    # * international trade
    @constraint(m, [i in I], pe[i] == epsilon * pWe[i])
    @constraint(m, [i in I], pm[i] == epsilon * pWm[i])
    @constraint(m, sum(pWe[i] * E[i] for i in I) + Sf == sum(pWm[i] * M[i] for i in I))
    # * Armington function
    @constraint(m, [i in I],
                Q[i] == gamma[i] * (deltam[i] * M[i]^eta[i] + deltad[i] * D[i]^eta[i])^(1 / eta[i]))
    @constraint(m, [i in I],
                M[i] == (gamma[i]^eta[i] * deltam[i] * pq[i] / ((1 + taum[i]) * pm[i]))^(1 / (1 - eta[i])) * Q[i])
    @constraint(m, [i in I],
                D[i] == (gamma[i]^eta[i] * deltad[i] * pq[i] / (kd[i] * pd[i]))^(1 / (1 - eta[i])) * Q[i])
    # * transformation function
    @constraint(m, [i in I],
                Z[i] == theta[i] * (xie[i] * E[i]^phi[i] + xid[i] * D[i]^phi[i])^(1 / phi[i]))
    @constraint(m, [i in I],
                E[i] == (theta[i]^phi[i] * xie[i] * (1 + tauz[i]) * pz[i] / pe[i])^(1 / (1 - phi[i])) * Z[i])
    @constraint(m, [i in I],
                D[i] == (theta[i]^phi[i] * xid[i] * (1 + tauz[i]) * pz[i] / pd[i])^(1 / (1 - phi[i])) * Z[i])
    # * market clearing condition
    @constraint(m, [i in I], Q[i] == Xp[i] + Xg[i] + Xv[i] + sum(X[i, j] for j in I))
    @constraint(m, [h in H], sum(F[h, j] for j in I) == FF[h])
    # * numeraire
    fix(pf["LAB"], 1.0; force = true)

    @objective(m, Max, prod(Xp[i]^alpha[i] for i in I))
    optimize!(m)
    st = termination_status(m)
    viol = maximum(abs(value(c) - MOI.constant(constraint_object(c).set))
                   for (Fn, S) in list_of_constraint_types(m) if S <: MOI.EqualTo
                   for c in all_constraints(m, Fn, S))
    @printf("%-7s taum=%s  %s  violação %.2e\n", mon ? "moncge" : "irscge",
            string(round.([taum[i] for i in I]; digits = 4)), st, viol)
    v(x) = Dict(i => value(x[i]) for i in I)
    return Dict(
        "status" => string(st), "violacao" => viol,
        "Y" => v(Y), "Z" => v(Z), "Xp" => v(Xp), "Xg" => v(Xg), "Xv" => v(Xv),
        "E" => v(E), "M" => v(M), "Q" => v(Q), "D" => v(D),
        "F" => Dict("$(h),$(j)" => value(F[h, j]) for h in H, j in I),
        "pf" => Dict(h => value(pf[h]) for h in H), "py" => v(py), "pz" => v(pz),
        "pq" => v(pq), "pe" => v(pe), "pm" => v(pm), "pd" => v(pd),
        "epsilon" => value(epsilon), "Sp" => value(Sp), "Sg" => value(Sg), "Td" => value(Td),
        "Tz" => v(Tz), "Tm" => v(Tm), "RT" => v(RT), "FC" => FC, "UU" => objective_value(m),
    )
end

saida = Dict{String,Any}()
for (nome, modelo) in (("moncge", :mon), ("irscge", :irs))
    saida[nome] = Dict("base" => modelo_hosoe(modelo, false),
                       "abolicao" => modelo_hosoe(modelo, true))
end
open(SAIDA, "w") do f
    JSON3.write(f, saida)
end
@printf("escrito: %s\n", SAIDA)
