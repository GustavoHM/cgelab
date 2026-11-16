# melitz.jl — o equilíbrio de Melitz a partir das primitivas, com as integrais por quadratura.
#
# Nada vem do cgelab, e nenhuma forma fechada da distribuição de Pareto entra aqui: nem a
# probabilidade de sobrevivência (φmin/φ*)^k, nem a razão k/(k − σ + 1) das médias, nem a massa
# de entrantes, nem a igualdade entre a despesa e a renda do trabalho. O script lê só os
# parâmetros e escreve o problema da firma e as condições de equilíbrio:
#
#   preço           p(φ) = σ/(σ − 1) · 1/φ               (salário = 1, o numerário)
#   demanda         q(φ) = E · P^(σ − 1) · p^(−σ)        (τp no mercado externo)
#   lucro           π_d(φ) = p q − q/φ − f_d
#                   π_x(φ) = τp q_x − τ q_x/φ − f_x      (τ unidades produzidas por unidade entregue)
#   densidade       g(φ) = k φmin^k φ^(−k − 1),  φ ≥ φmin
#
#   lucro zero      π_d(φ_d) = 0 e π_x(φ_x) = 0
#   livre entrada   ∫_{φ_d} π_d g dφ + ∫_{φ_x} π_x g dφ = f_e
#   índice de preço P^(1 − σ) = M_e [∫_{φ_d} p^(1 − σ) g dφ + ∫_{φ_x} (τp)^(1 − σ) g dφ]
#   trabalho        M_e [f_e + ∫_{φ_d} (f_d + q/φ) g dφ + ∫_{φ_x} (f_x + τ q_x/φ) g dφ] = L
#
# As incógnitas são φ_d, φ_x, M_e, P e a despesa E. Os dois países são idênticos, então as
# importações de um são as exportações do outro e o salário é o mesmo. A restrição orçamentária
# E = L não é imposta: sai da solução, e o script que compara confere.
#
# Cada integral de c a ∞ é feita em s = ln(φ/c), de 0 a S, com Gauss-Legendre composto; os nós
# vêm de Golub-Welsch. S trunca a cauda onde o integrando mais lento já caiu abaixo de 1e-17 do
# seu valor em s = 0. As massas, as receitas e as produtividades médias também são integrais
# por quadratura, calculadas depois da solução.
#
#   uso: julia --project=validation validation/melitz.jl entrada.json saida.json

using JuMP, Ipopt, JSON3, LinearAlgebra

const E_ = JSON3.read(read(ARGS[1], String))
const SAIDA = ARGS[2]

const sigma = Float64(E_.sigma)
const k = Float64(E_.pareto_shape)
const phimin = Float64(E_.productivity_min)
const L = Float64(E_.labour)
const fe = Float64(E_.entry_cost)
const fd = Float64(E_.domestic_fixed_cost)
const fx = Float64(E_.export_fixed_cost)
const tau = Float64(E_.iceberg_cost)

# Nós e pesos de Gauss-Legendre em [−1, 1], pelos autovalores da matriz de Jacobi.
function gauss_legendre(n)
    beta = [i / sqrt(4.0 * i^2 - 1.0) for i in 1:n-1]
    F = eigen(SymTridiagonal(zeros(n), beta))
    return F.values, 2.0 .* F.vectors[1, :] .^ 2
end

# Regra composta em [0, S]: painéis de largura h, n nós em cada um.
function regra(S, h, n)
    x, w = gauss_legendre(n)
    paineis = ceil(Int, S / h)
    h = S / paineis
    s, peso = Float64[], Float64[]
    for j in 0:paineis-1, i in 1:n
        push!(s, h * (j + (x[i] + 1.0) / 2.0))
        push!(peso, w[i] * h / 2.0)
    end
    return s, peso
end

# O integrando mais lento cai como exp(−(k − σ + 1) s) e o mais rápido como exp(−k s).
const DECAIMENTO = k - sigma + 1.0
const S_MAX = 40.0 / DECAIMENTO
const NOS, PESOS = regra(S_MAX, 1.5 / k, 12)
const ESCALA = exp.(NOS)

preco(phi) = sigma / (sigma - 1.0) / phi
dens(phi) = k * phimin^k * phi^(-k - 1.0)
# ∫_c^∞ h(φ) g(φ) dφ, com φ = c·exp(s) e dφ = φ ds.
integral(h, c) = sum(PESOS[i] * h(c * ESCALA[i]) * dens(c * ESCALA[i]) * (c * ESCALA[i])
                     for i in eachindex(NOS))

m = Model(Ipopt.Optimizer)
set_silent(m)
set_optimizer_attribute(m, "tol", 1e-13)
set_optimizer_attribute(m, "acceptable_tol", 1e-12)
set_optimizer_attribute(m, "max_iter", 500)
@variable(m, phid >= phimin, start = 1.2 * phimin)
@variable(m, phix >= phimin, start = 2.0 * phimin)
@variable(m, me >= 1e-8, start = L / (2.0 * sigma * fe))
@variable(m, P >= 1e-8, start = 1.0 / phimin)
@variable(m, Ex >= 1e-8, start = L)

qd(phi) = Ex * P^(sigma - 1.0) * preco(phi)^(-sigma)
qx(phi) = Ex * P^(sigma - 1.0) * (tau * preco(phi))^(-sigma)
lucro_d(phi) = preco(phi) * qd(phi) - qd(phi) / phi - fd
lucro_x(phi) = tau * preco(phi) * qx(phi) - tau * qx(phi) / phi - fx

@constraint(m, lucro_d(phid) == 0)
@constraint(m, lucro_x(phix) == 0)
@constraint(m, integral(lucro_d, phid) + integral(lucro_x, phix) == fe)
@constraint(m, P^(1.0 - sigma) == me * (integral(phi -> preco(phi)^(1.0 - sigma), phid)
                                       + integral(phi -> (tau * preco(phi))^(1.0 - sigma), phix)))
@constraint(m, me * (fe + integral(phi -> fd + qd(phi) / phi, phid)
                     + integral(phi -> fx + tau * qx(phi) / phi, phix)) == L)

optimize!(m)
status = termination_status(m)

# Depois da solução, tudo em Float64 e de novo por quadratura.
pd, px, ne, price, desp = value(phid), value(phix), value(me), value(P), value(Ex)
vqd(phi) = desp * price^(sigma - 1.0) * preco(phi)^(-sigma)
vqx(phi) = desp * price^(sigma - 1.0) * (tau * preco(phi))^(-sigma)
sd = integral(phi -> 1.0, pd)
sx = integral(phi -> 1.0, px)
md, mx = ne * sd, ne * sx
rd = ne * integral(phi -> preco(phi) * vqd(phi), pd)
rx = ne * integral(phi -> tau * preco(phi) * vqx(phi), px)
# A média de Melitz: a firma representativa cuja produtividade reproduz o agregado CES.
media_d = (integral(phi -> phi^(sigma - 1.0), pd) / sd)^(1.0 / (sigma - 1.0))
media_x = (integral(phi -> phi^(sigma - 1.0), px) / sx)^(1.0 / (sigma - 1.0))
residuo = maximum(abs, [
    preco(pd) * vqd(pd) - vqd(pd) / pd - fd,
    tau * preco(px) * vqx(px) - tau * vqx(px) / px - fx,
    integral(phi -> preco(phi) * vqd(phi) - vqd(phi) / phi - fd, pd) +
    integral(phi -> tau * preco(phi) * vqx(phi) - tau * vqx(phi) / phi - fx, px) - fe,
])

out = Dict(
    "status" => string(status),
    "nodes" => length(NOS),
    "domestic_cutoff" => pd,
    "export_cutoff" => px,
    "entrants" => ne,
    "domestic_firms" => md,
    "exporters" => mx,
    "price_index" => price,
    "welfare" => desp / price,
    "expenditure" => desp,
    "domestic_revenue" => rd,
    "export_revenue" => rx,
    "average_domestic_productivity" => media_d,
    "average_export_productivity" => media_x,
    "export_share" => mx / md,
    "entry_probability" => sd,
    "real_wage" => 1.0 / price,
    "domestic_share" => rd / (rd + rx),
    "residual" => residuo,
)
open(SAIDA, "w") do io
    JSON3.write(io, out)
end
println("MELITZ status=$(status) nos=$(length(NOS)) despesa-L=$(desp - L) residuo=$(residuo)")
