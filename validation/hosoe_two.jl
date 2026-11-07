# hosoe_two.jl — transcrição em JuMP de twocge.gms, da biblioteca de modelos do GAMS.
#
# Hosoe, Gasawa e Hashimoto (2004), Handbook of Computable General Equilibrium Modeling, cap.
# 10.3: dois países, Japão e EUA, cada um um stdcge, que comerciam entre si. O preço de
# exportação de um bem em dólares num país é o de importação no outro, e as exportações de um
# são as importações do outro. O câmbio dos EUA é fixo em um: com as duas balanças de pagamentos
# e todos os mercados, uma equação é redundante. O GAMS não está instalado aqui, então o código
# publicado foi transcrito linha a linha: a mesma MCS, a mesma calibração e as mesmas equações,
# com os mesmos nomes. Nada vem do cgelab. O GAMS maximiza uma soma fictícia de utilidades
# sujeita a um sistema quadrado; aqui o sistema é resolvido por viabilidade.
#
# Resolve o ano-base e a abolição das tarifas nos dois países (taum = 0) e escreve os níveis.
#
#   uso:  julia --project=validation validation/hosoe_two.jl [saida.json]

using JuMP, Ipopt, JSON3, Printf

const SAIDA = length(ARGS) >= 1 ? ARGS[1] : "hosoe_two_out.json"
const U = ["BRD", "MLK", "CAP", "LAB", "IDT", "TRF", "HOH", "GOV", "INV", "EXT"]
const I = ["BRD", "MLK"]
const H = ["CAP", "LAB"]
const R = ["JPN", "USA"]

SAM = Dict{Tuple{String,String,String},Float64}()
for (u, v, x) in [
        ("BRD", "BRD", 21), ("BRD", "MLK", 8), ("MLK", "BRD", 17), ("MLK", "MLK", 9),
        ("CAP", "BRD", 20), ("CAP", "MLK", 30), ("LAB", "BRD", 15), ("LAB", "MLK", 25),
        ("IDT", "BRD", 5), ("IDT", "MLK", 4), ("TRF", "BRD", 1), ("TRF", "MLK", 2),
        ("HOH", "CAP", 50), ("HOH", "LAB", 40), ("GOV", "IDT", 9), ("EXT", "BRD", 13),
        ("EXT", "MLK", 11),
        ("BRD", "HOH", 20), ("BRD", "GOV", 19), ("BRD", "INV", 16), ("BRD", "EXT", 8),
        ("MLK", "HOH", 30), ("MLK", "GOV", 14), ("MLK", "INV", 15), ("MLK", "EXT", 4),
        ("GOV", "TRF", 3), ("GOV", "HOH", 23), ("INV", "HOH", 17), ("INV", "GOV", 2),
        ("INV", "EXT", 12)]
    SAM[(u, v, "JPN")] = x
end
for (u, v, x) in [
        ("BRD", "BRD", 40), ("BRD", "MLK", 1), ("MLK", "BRD", 17), ("MLK", "MLK", 29),
        ("CAP", "BRD", 33), ("CAP", "MLK", 30), ("LAB", "BRD", 15), ("LAB", "MLK", 31),
        ("IDT", "BRD", 10), ("IDT", "MLK", 20), ("TRF", "BRD", 1), ("TRF", "MLK", 1),
        ("HOH", "CAP", 63), ("HOH", "LAB", 46), ("GOV", "IDT", 30), ("EXT", "BRD", 8),
        ("EXT", "MLK", 4),
        ("BRD", "HOH", 30), ("BRD", "GOV", 20), ("BRD", "INV", 20), ("BRD", "EXT", 13),
        ("MLK", "HOH", 30), ("MLK", "GOV", 14), ("MLK", "INV", 15), ("MLK", "EXT", 11),
        ("GOV", "TRF", 2), ("GOV", "HOH", 29), ("INV", "HOH", 20), ("INV", "GOV", 27),
        ("INV", "EXT", -12)]
    SAM[(u, v, "USA")] = x
end
sam(u, v, r) = get(SAM, (u, v, r), 0.0)
for r in R, u in U
    @assert abs(sum(sam(u, v, r) for v in U) - sum(sam(v, u, r) for v in U)) < 1e-12 (r, u)
end

# * loading the initial values
Td0 = Dict(r => sam("GOV", "HOH", r) for r in R)
Tz0 = Dict((j, r) => sam("IDT", j, r) for j in I, r in R)
Tm0 = Dict((j, r) => sam("TRF", j, r) for j in I, r in R)
F0 = Dict((h, j, r) => sam(h, j, r) for h in H, j in I, r in R)
Y0 = Dict((j, r) => sum(F0[(h, j, r)] for h in H) for j in I, r in R)
X0 = Dict((i, j, r) => sam(i, j, r) for i in I, j in I, r in R)
Z0 = Dict((j, r) => Y0[(j, r)] + sum(X0[(i, j, r)] for i in I) for j in I, r in R)
M0 = Dict((i, r) => sam("EXT", i, r) for i in I, r in R)
tauz = Dict((j, r) => Tz0[(j, r)] / Z0[(j, r)] for j in I, r in R)
taum0 = Dict((j, r) => Tm0[(j, r)] / M0[(j, r)] for j in I, r in R)
Xp0 = Dict((i, r) => sam(i, "HOH", r) for i in I, r in R)
FF = Dict((h, r) => sam("HOH", h, r) for h in H, r in R)
Xg0 = Dict((i, r) => sam(i, "GOV", r) for i in I, r in R)
Xv0 = Dict((i, r) => sam(i, "INV", r) for i in I, r in R)
E0 = Dict((i, r) => sam(i, "EXT", r) for i in I, r in R)
Q0 = Dict((i, r) => Xp0[(i, r)] + Xg0[(i, r)] + Xv0[(i, r)] + sum(X0[(i, j, r)] for j in I)
          for i in I, r in R)
D0 = Dict((i, r) => (1 + tauz[(i, r)]) * Z0[(i, r)] - E0[(i, r)] for i in I, r in R)
Sp0 = Dict(r => sam("INV", "HOH", r) for r in R)
Sg0 = Dict(r => sam("INV", "GOV", r) for r in R)
Sf = Dict(r => sam("INV", "EXT", r) for r in R)

# * calibration
sigma = Dict(i => 2.0 for i in I)
psi = Dict(i => 2.0 for i in I)
eta = Dict(i => (sigma[i] - 1) / sigma[i] for i in I)
phi = Dict(i => (psi[i] + 1) / psi[i] for i in I)
alpha = Dict((i, r) => Xp0[(i, r)] / sum(Xp0[(j, r)] for j in I) for i in I, r in R)
beta = Dict((h, j, r) => F0[(h, j, r)] / sum(F0[(k, j, r)] for k in H) for h in H, j in I, r in R)
b = Dict((j, r) => Y0[(j, r)] / prod(F0[(h, j, r)]^beta[(h, j, r)] for h in H) for j in I, r in R)
ax = Dict((i, j, r) => X0[(i, j, r)] / Z0[(j, r)] for i in I, j in I, r in R)
ay = Dict((j, r) => Y0[(j, r)] / Z0[(j, r)] for j in I, r in R)
mu = Dict((i, r) => Xg0[(i, r)] / sum(Xg0[(j, r)] for j in I) for i in I, r in R)
lambda = Dict((i, r) => Xv0[(i, r)] / (Sp0[r] + Sg0[r] + Sf[r]) for i in I, r in R)
deltam = Dict((i, r) => (1 + taum0[(i, r)]) * M0[(i, r)]^(1 - eta[i]) /
                        ((1 + taum0[(i, r)]) * M0[(i, r)]^(1 - eta[i]) + D0[(i, r)]^(1 - eta[i]))
              for i in I, r in R)
deltad = Dict((i, r) => D0[(i, r)]^(1 - eta[i]) /
                        ((1 + taum0[(i, r)]) * M0[(i, r)]^(1 - eta[i]) + D0[(i, r)]^(1 - eta[i]))
              for i in I, r in R)
gamma = Dict((i, r) => Q0[(i, r)] / (deltam[(i, r)] * M0[(i, r)]^eta[i] +
                                     deltad[(i, r)] * D0[(i, r)]^eta[i])^(1 / eta[i]) for i in I, r in R)
xie = Dict((i, r) => E0[(i, r)]^(1 - phi[i]) / (E0[(i, r)]^(1 - phi[i]) + D0[(i, r)]^(1 - phi[i]))
           for i in I, r in R)
xid = Dict((i, r) => D0[(i, r)]^(1 - phi[i]) / (E0[(i, r)]^(1 - phi[i]) + D0[(i, r)]^(1 - phi[i]))
           for i in I, r in R)
theta = Dict((i, r) => Z0[(i, r)] / (xie[(i, r)] * E0[(i, r)]^phi[i] +
                                     xid[(i, r)] * D0[(i, r)]^phi[i])^(1 / phi[i]) for i in I, r in R)
ssp = Dict(r => Sp0[r] / sum(FF[(h, r)] for h in H) for r in R)
ssg = Dict(r => Sg0[r] / (Td0[r] + sum(Tz0[(j, r)] for j in I) + sum(Tm0[(j, r)] for j in I)) for r in R)
taud = Dict(r => Td0[r] / sum(FF[(h, r)] for h in H) for r in R)

function resolve(taum)
    m = Model(Ipopt.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "tol", 1e-12)
    set_optimizer_attribute(m, "constr_viol_tol", 1e-12)
    lo = 0.00001
    @variable(m, Y[j in I, r in R] >= lo, start = Y0[(j, r)])
    @variable(m, F[h in H, j in I, r in R] >= lo, start = F0[(h, j, r)])
    @variable(m, X[i in I, j in I, r in R] >= lo, start = X0[(i, j, r)])
    @variable(m, Z[j in I, r in R] >= lo, start = Z0[(j, r)])
    @variable(m, Xp[i in I, r in R] >= lo, start = Xp0[(i, r)])
    @variable(m, Xg[i in I, r in R] >= lo, start = Xg0[(i, r)])
    @variable(m, Xv[i in I, r in R] >= lo, start = Xv0[(i, r)])
    @variable(m, E[i in I, r in R] >= lo, start = E0[(i, r)])
    @variable(m, M[i in I, r in R] >= lo, start = M0[(i, r)])
    @variable(m, Q[i in I, r in R] >= lo, start = Q0[(i, r)])
    @variable(m, D[i in I, r in R] >= lo, start = D0[(i, r)])
    @variable(m, pf[h in H, r in R] >= lo, start = 1)
    @variable(m, py[j in I, r in R] >= lo, start = 1)
    @variable(m, pz[i in I, r in R] >= lo, start = 1)
    @variable(m, pq[i in I, r in R] >= lo, start = 1)
    @variable(m, pe[i in I, r in R] >= lo, start = 1)
    @variable(m, pm[i in I, r in R] >= lo, start = 1)
    @variable(m, pd[i in I, r in R] >= lo, start = 1)
    @variable(m, epsilon[r in R] >= lo, start = 1)
    @variable(m, pWe[i in I, r in R] >= lo, start = 1)
    @variable(m, pWm[i in I, r in R] >= lo, start = 1)
    @variable(m, Sp[r in R] >= lo, start = Sp0[r])
    @variable(m, Sg[r in R] >= lo, start = Sg0[r])
    @variable(m, Td[r in R] >= lo, start = Td0[r])
    @variable(m, Tz[i in I, r in R] >= 0, start = Tz0[(i, r)])
    @variable(m, Tm[i in I, r in R] >= 0, start = Tm0[(i, r)])

    # * domestic production
    @constraint(m, [j in I, r in R], Y[j, r] == b[(j, r)] * prod(F[h, j, r]^beta[(h, j, r)] for h in H))
    @constraint(m, [i in I, j in I, r in R], X[i, j, r] == ax[(i, j, r)] * Z[j, r])
    @constraint(m, [j in I, r in R], Y[j, r] == ay[(j, r)] * Z[j, r])
    @constraint(m, [h in H, j in I, r in R], F[h, j, r] == beta[(h, j, r)] * py[j, r] * Y[j, r] / pf[h, r])
    @constraint(m, [j in I, r in R], pz[j, r] == ay[(j, r)] * py[j, r] + sum(ax[(i, j, r)] * pq[i, r] for i in I))
    # * government behavior
    @constraint(m, [r in R], Td[r] == taud[r] * sum(pf[h, r] * FF[(h, r)] for h in H))
    @constraint(m, [i in I, r in R], Tz[i, r] == tauz[(i, r)] * pz[i, r] * Z[i, r])
    @constraint(m, [i in I, r in R], Tm[i, r] == taum[(i, r)] * pm[i, r] * M[i, r])
    @constraint(m, [i in I, r in R], Xg[i, r] == mu[(i, r)] * (Td[r] + sum(Tz[j, r] for j in I) +
                                                              sum(Tm[j, r] for j in I) - Sg[r]) / pq[i, r])
    # * investment behavior
    @constraint(m, [i in I, r in R], Xv[i, r] == lambda[(i, r)] * (Sp[r] + Sg[r] + epsilon[r] * Sf[r]) / pq[i, r])
    # * savings
    @constraint(m, [r in R], Sp[r] == ssp[r] * sum(pf[h, r] * FF[(h, r)] for h in H))
    @constraint(m, [r in R], Sg[r] == ssg[r] * (Td[r] + sum(Tz[j, r] for j in I) + sum(Tm[j, r] for j in I)))
    # * household consumption
    @constraint(m, [i in I, r in R], Xp[i, r] == alpha[(i, r)] * (sum(pf[h, r] * FF[(h, r)] for h in H) -
                                                                Sp[r] - Td[r]) / pq[i, r])
    # * international trade
    @constraint(m, [i in I, r in R], pe[i, r] == epsilon[r] * pWe[i, r])
    @constraint(m, [i in I, r in R], pm[i, r] == epsilon[r] * pWm[i, r])
    @constraint(m, [r in R], sum(pWe[i, r] * E[i, r] for i in I) + Sf[r] == sum(pWm[i, r] * M[i, r] for i in I))
    # * Armington function
    @constraint(m, [i in I, r in R], Q[i, r] == gamma[(i, r)] * (deltam[(i, r)] * M[i, r]^eta[i] +
                                                                deltad[(i, r)] * D[i, r]^eta[i])^(1 / eta[i]))
    @constraint(m, [i in I, r in R], M[i, r] == (gamma[(i, r)]^eta[i] * deltam[(i, r)] * pq[i, r] /
                                                 ((1 + taum[(i, r)]) * pm[i, r]))^(1 / (1 - eta[i])) * Q[i, r])
    @constraint(m, [i in I, r in R], D[i, r] == (gamma[(i, r)]^eta[i] * deltad[(i, r)] * pq[i, r] /
                                                 pd[i, r])^(1 / (1 - eta[i])) * Q[i, r])
    # * transformation function
    @constraint(m, [i in I, r in R], Z[i, r] == theta[(i, r)] * (xie[(i, r)] * E[i, r]^phi[i] +
                                                                xid[(i, r)] * D[i, r]^phi[i])^(1 / phi[i]))
    @constraint(m, [i in I, r in R], E[i, r] == (theta[(i, r)]^phi[i] * xie[(i, r)] * (1 + tauz[(i, r)]) *
                                                 pz[i, r] / pe[i, r])^(1 / (1 - phi[i])) * Z[i, r])
    @constraint(m, [i in I, r in R], D[i, r] == (theta[(i, r)]^phi[i] * xid[(i, r)] * (1 + tauz[(i, r)]) *
                                                 pz[i, r] / pd[i, r])^(1 / (1 - phi[i])) * Z[i, r])
    # * market clearing condition
    @constraint(m, [i in I, r in R], Q[i, r] == Xp[i, r] + Xg[i, r] + Xv[i, r] + sum(X[i, j, r] for j in I))
    @constraint(m, [h in H, r in R], FF[(h, r)] == sum(F[h, j, r] for j in I))
    # * international market clearing condition
    for i in I, r in R, rr in R
        r == rr && continue
        @constraint(m, pWe[i, r] == pWm[i, rr])
        @constraint(m, E[i, r] == M[i, rr])
    end
    # * numeraire, and the redundant variable
    for r in R
        fix(pf["LAB", r], 1.0; force = true)
    end
    fix(epsilon["USA"], 1.0; force = true)

    @objective(m, Max, sum(prod(Xp[i, r]^alpha[(i, r)] for i in I) for r in R))
    optimize!(m)
    st = termination_status(m)
    viol = maximum(abs(value(c) - MOI.constant(constraint_object(c).set))
                   for (Fn, S) in list_of_constraint_types(m) if S <: MOI.EqualTo
                   for c in all_constraints(m, Fn, S))
    @printf("twocge taum=%s  %s  violação %.2e\n", string([taum[(i, r)] for i in I, r in R]), st, viol)
    out = Dict{String,Any}("status" => string(st), "violacao" => viol)
    for r in R
        v(x) = Dict(i => value(x[i, r]) for i in I)
        out[r] = Dict(
            "Y" => v(Y), "Z" => v(Z), "Xp" => v(Xp), "Xg" => v(Xg), "Xv" => v(Xv),
            "E" => v(E), "M" => v(M), "Q" => v(Q), "D" => v(D),
            "F" => Dict("$(h),$(j)" => value(F[h, j, r]) for h in H, j in I),
            "pf" => Dict(h => value(pf[h, r]) for h in H), "py" => v(py), "pz" => v(pz),
            "pq" => v(pq), "pe" => v(pe), "pm" => v(pm), "pd" => v(pd),
            "pWe" => v(pWe), "pWm" => v(pWm),
            "epsilon" => value(epsilon[r]), "Sp" => value(Sp[r]), "Sg" => value(Sg[r]),
            "Td" => value(Td[r]), "Tz" => v(Tz), "Tm" => v(Tm))
    end
    return out
end

saida = Dict("base" => resolve(taum0),
             "abolicao" => resolve(Dict((i, r) => 0.0 for i in I, r in R)))
open(SAIDA, "w") do f
    JSON3.write(f, saida)
end
@printf("escrito: %s\n", SAIDA)
