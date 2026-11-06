# trade_instruments.jl — imposto sobre exportações e Armington assimétrico, como problema de
# complementaridade misto resolvido pelo PATH.
#
# Nada vem do cgelab. O script lê a MCS e o enunciado (papéis das contas, as contas da tarifa e
# do imposto sobre exportações, elasticidades, a regra de exportação, o choque), calibra o
# modelo nacional aqui, nas unidades do modelo-padrão do IFPRI (Lofgren et al., 2002), e não
# nas do cgelab:
#
#     importações em unidades do valor na fronteira, PM = pwm·(1 + tm)·EXR, PM0 = 1 + tm0;
#     exportações em unidades do valor FOB, PE = pwe·(1 − te)·EXR, PE0 = 1 − te0;
#     vendas internas QD a preço do produtor, PDD0 = 1; produção QX, PX0 = 1.
#
# O Armington assimétrico é o de Cicowiez e Lofgren (2019, tabela 2.1), equação por equação:
#
#     (1)  PM/PDD − DEVP + DEVN = PM0/PDD0
#     (2)  SIGMAQ − σl ≥ 0  ⟂  DEVP ≥ 0
#     (3)  σh − SIGMAQ ≥ 0  ⟂  DEVN ≥ 0
#     (4)  QQ = PHIQ·(DELTAM·QM^−RHOQ + DELTADD·QD^−RHOQ)^(−1/RHOQ)
#     (5)  QM/QD = (PDD/PM · DELTAM/DELTADD)^SIGMAQ
#     (6)  RHOQ = 1/SIGMAQ − 1
#     (7)  DELTAM = PM0·QM0^(1/SIGMAQ) / (PM0·QM0^(1/SIGMAQ) + PDD0·QD0^(1/SIGMAQ))
#     (8)  DELTADD = PDD0·QD0^(1/SIGMAQ) / (PM0·QM0^(1/SIGMAQ) + PDD0·QD0^(1/SIGMAQ))
#     (9)  PHIQ = QQ0 / (DELTAM·QM0^−RHOQ + DELTADD·QD0^−RHOQ)^(−1/RHOQ)
#
# com (1) pareada com SIGMAQ, e os parâmetros (7)–(9) recalibrados ao ano-base para a
# elasticidade escolhida. O preço do composto sai da identidade de valor PQ·QQ = PDD·QD +
# PM·QM. Um bem com uma só elasticidade tem SIGMAQ fixo e não tem (1)–(3).
#
# Exportações: CET, com o preço do produtor PE contra o das vendas internas PDS, ou em
# proporção fixa da produção, em que o produtor recebe PX por unidade a preço do produtor e o
# preço FOB é PX·(1 − te0)/(1 − te). O governo recebe te·preço FOB·QE.
#
# Escopo, recusado pelo script em Python fora dele: um produtor por bem e um bem por atividade,
# imposto sobre produtos com a mesma alíquota para firmas e famílias, imposto sobre a atividade
# por unidade, sem margens nem imposto sobre a produção, famílias Cobb–Douglas sem imposto
# direto nem transferências, governo com consumo real fixo, investimento savings_driven e
# poupança externa fixa no numerário.
#
#   uso:  julia --project=validation validation/trade_instruments.jl entrada.json saida.json

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
trf = E.tariff_account === nothing ? "" : String(E.tariff_account)
etx = E.export_tax_account === nothing ? "" : String(E.export_tax_account)
NA, NC, NF, NH = length(ATV), length(COM), length(FAT), length(FAM)

# ---------------------------------------------------------------------------------------------
# Calibração
# ---------------------------------------------------------------------------------------------
faz = [sam(ATV[j], COM[i]) for j in 1:NA, i in 1:NC]
prod = Dict{Int,Int}()                                          # bem -> a atividade que o faz
for i in 1:NC
    js = [j for j in 1:NA if faz[j, i] > 0]
    @assert length(js) == 1 "$(COM[i]) precisa de um só produtor"
    @assert count(>(0), faz[js[1], :]) == 1 "$(ATV[js[1]]) faz mais de um bem"
    prod[i] = js[1]
end
QX0 = [sum(faz[:, i]) for i in 1:NC]
X0 = [sum(faz[j, :]) for j in 1:NA]
QE0 = [sam(COM[i], row) for i in 1:NC]                          # FOB
QM0 = [sam(row, COM[i]) for i in 1:NC]                          # na fronteira
TRF = [trf == "" ? 0.0 : sam(trf, COM[i]) for i in 1:NC]
ETX = [etx == "" ? 0.0 : sam(etx, COM[i]) for i in 1:NC]
tm0 = [QM0[i] > 0 ? TRF[i] / QM0[i] : 0.0 for i in 1:NC]
te0 = [QE0[i] > 0 ? ETX[i] / QE0[i] : 0.0 for i in 1:NC]
PM0 = 1 .+ tm0
PE0 = 1 .- te0
QD0 = QX0 .- PE0 .* QE0                                         # vendas internas, a preço do produtor
QQ0 = QD0 .+ PM0 .* QM0                                         # composto a preço básico, PQ0 = 1
ptax = [sam(gov, COM[i]) / QQ0[i] for i in 1:NC]
io = [sam(COM[i], ATV[j]) / (1 + ptax[i]) / X0[j] for i in 1:NC, j in 1:NA]
F0 = [sam(FAT[f], ATV[j]) for f in 1:NF, j in 1:NA]
vunit = [sum(F0[:, j]) for j in 1:NA] ./ X0
theta = F0 ./ [sum(F0[:, j]) for j in 1:NA]'
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
FS0 = sum(QM0) - sum(QE0)

sva = Float64(E.sigma_va)
sl = [Float64(E.sigma_armington[Symbol(c)]) for c in COM]
sh = [E.sigma_armington_falling[Symbol(c)] === nothing ? NaN :
      Float64(E.sigma_armington_falling[Symbol(c)]) for c in COM]
assim = [isfinite(sh[i]) && QM0[i] > 0 && QD0[i] > 0 for i in 1:NC]
cet = String(E.export_rule) == "cet"
st = [Float64(E.sigma_cet[Symbol(c)]) for c in COM]
num = findfirst(==(String(E.numeraire_factor)), FAT)
pwm = [Float64(E.world_price[Symbol(c)]) for c in COM]
pwe = [Float64(E.export_price[Symbol(c)]) for c in COM]
tm = [Float64(E.tariff[Symbol(c)]) for c in COM]
te = [Float64(E.export_tax[Symbol(c)]) for c in COM]
oferta = [Float64(E.factor_supply[Symbol(f)]) for f in FAT]

# CET: QX = αt·(δt·QE^ρt + (1 − δt)·QD^ρt)^(1/ρt), com ρt = 1 + 1/σt.
rt = 1 .+ 1 ./ st
dt = [QE0[i] > 0 ? 1 / (1 + (1 / PE0[i]) * (QE0[i] / QD0[i])^(rt[i] - 1)) : 0.0 for i in 1:NC]
at = [QE0[i] > 0 ? QX0[i] / (dt[i] * QE0[i]^rt[i] + (1 - dt[i]) * QD0[i]^rt[i])^(1 / rt[i]) : 1.0
      for i in 1:NC]

# ---------------------------------------------------------------------------------------------
# O problema de complementaridade
# ---------------------------------------------------------------------------------------------
m = Model(PATHSolver.Optimizer)
set_silent(m)
set_optimizer_attribute(m, "convergence_tolerance", 1e-11)
lo = 1e-8
@variable(m, w[f in 1:NF] >= lo, start = 1.0)
@variable(m, er >= lo, start = 1.0)
@variable(m, px[i in 1:NC] >= lo, start = 1.0)
@variable(m, pds[i in 1:NC] >= lo, start = 1.0)
@variable(m, pq[i in 1:NC] >= lo, start = 1.0)                   # composto, preço básico
@variable(m, QQ[i in 1:NC] >= 0, start = QQ0[i])
@variable(m, QD[i in 1:NC] >= 0, start = QD0[i])
@variable(m, QM[i in 1:NC] >= 0, start = QM0[i])
@variable(m, QE[i in 1:NC] >= 0, start = QE0[i])
@variable(m, QX[i in 1:NC] >= 0, start = QX0[i])
@variable(m, Y[h in 1:NH], start = Y0[h])
@variable(m, poup, start = sum(sam(COM[i], inv) for i in 1:NC))
@variable(m, sigmaq[i in 1:NC], start = sl[i])
@variable(m, devp[i in 1:NC] >= 0, start = 0.0)
@variable(m, devn[i in 1:NC] >= 0, start = 0.0)
for i in 1:NC
    if !assim[i]
        fix(sigmaq[i], sl[i]; force = true)
        fix(devp[i], 0.0; force = true)
        fix(devn[i], 0.0; force = true)
    end
end
fix(w[num], 1.0; force = true)

PM(i) = pwm[i] * (1 + tm[i]) * er
pfob(i) = cet ? pwe[i] * er : px[i] * PE0[i] / (1 - te[i])
PE(i) = (1 - te[i]) * pfob(i)                                    # o que o produtor recebe
pc(i) = pq[i] * (1 + ptax[i])                                    # ao comprador
X(j) = QX[only(i for (i, jj) in prod if jj == j)]
@expression(m, pva[j in 1:NA],
            sum(theta[f, j] * w[f]^(1 - sva) for f in 1:NF if theta[f, j] > 0)^(1 / (1 - sva)))
@expression(m, pa[j in 1:NA], sum(io[i, j] * pc(i) for i in 1:NC) + vunit[j] * pva[j] + atax[j])
@expression(m, fd[f in 1:NF, j in 1:NA],
            theta[f, j] > 0 ? vunit[j] * X(j) * theta[f, j] * (pva[j] / w[f])^sva : 0.0)

# (6)–(9) como funções de SIGMAQ.
rho(i) = 1 / sigmaq[i] - 1
dm(i) = PM0[i] * QM0[i]^(1 / sigmaq[i]) /
        (PM0[i] * QM0[i]^(1 / sigmaq[i]) + QD0[i]^(1 / sigmaq[i]))
dd(i) = QD0[i]^(1 / sigmaq[i]) / (PM0[i] * QM0[i]^(1 / sigmaq[i]) + QD0[i]^(1 / sigmaq[i]))
phi(i) = QQ0[i] / (dm(i) * QM0[i]^(-rho(i)) + dd(i) * QD0[i]^(-rho(i)))^(-1 / rho(i))

for i in 1:NC
    j = prod[i]
    # Produção: lucro zero e o nível.
    @constraint(m, pa[j] - px[i] ⟂ px[i])
    # Exportações e vendas internas.
    if QE0[i] > 0 && cet
        @constraint(m, QX[i] - at[i] * (dt[i] * QE[i]^rt[i] + (1 - dt[i]) * QD[i]^rt[i])^(1 / rt[i]) ⟂ pds[i])
        @constraint(m, QE[i] - QD[i] * (PE(i) / pds[i] * (1 - dt[i]) / dt[i])^(1 / (rt[i] - 1)) ⟂ QE[i])
        @constraint(m, px[i] * QX[i] - pds[i] * QD[i] - PE(i) * QE[i] ⟂ QX[i])
    elseif QE0[i] > 0
        @constraint(m, pds[i] - px[i] ⟂ pds[i])
        @constraint(m, QE[i] - QE0[i] / QX0[i] * QX[i] ⟂ QE[i])
        @constraint(m, QD[i] - QD0[i] / QX0[i] * QX[i] ⟂ QX[i])
    else
        fix(QE[i], 0.0; force = true)
        @constraint(m, pds[i] - px[i] ⟂ pds[i])
        @constraint(m, QD[i] - QX[i] ⟂ QX[i])
    end
    # Armington.
    if QM0[i] > 0 && QD0[i] > 0
        @constraint(m, QQ[i] - phi(i) * (dm(i) * QM[i]^(-rho(i)) + dd(i) * QD[i]^(-rho(i)))^(-1 / rho(i)) ⟂ pq[i])
        @constraint(m, QM[i] - QD[i] * (pds[i] / PM(i) * dm(i) / dd(i))^sigmaq[i] ⟂ QM[i])
        @constraint(m, pq[i] * QQ[i] - pds[i] * QD[i] - PM(i) * QM[i] ⟂ QD[i])
    else
        @assert QM0[i] == 0 "importar sem vender no mercado interno está fora do escopo"
        fix(QM[i], 0.0; force = true)
        @constraint(m, pq[i] - pds[i] ⟂ pq[i])
        @constraint(m, QQ[i] - QD[i] ⟂ QD[i])
    end
    if assim[i]
        @constraint(m, PM(i) / pds[i] - devp[i] + devn[i] - PM0[i] ⟂ sigmaq[i])    # (1)
        @constraint(m, sigmaq[i] - sl[i] ⟂ devp[i])                               # (2)
        @constraint(m, sh[i] - sigmaq[i] ⟂ devn[i])                               # (3)
    end
end
# Absorção.
@constraint(m, [i in 1:NC],
            QQ[i] - (sum(orc[h, i] * (1 - pou[h]) * Y[h] for h in 1:NH) / pc(i) + gq[i]
                     + vs[i] * poup / pc(i) + sum(io[i, j] * X(j) for j in 1:NA)) ⟂ QQ[i])
@expression(m, receita,
            sum(pq[i] * ptax[i] * QQ[i] for i in 1:NC) + sum(atax[j] * X(j) for j in 1:NA)
            + sum(tm[i] * pwm[i] * er * QM[i] for i in 1:NC)
            + sum(te[i] * pfob(i) * QE[i] for i in 1:NC))
@expression(m, gasto_gov, sum(pc(i) * gq[i] for i in 1:NC))
@constraint(m, [h in 1:NH], Y[h] - sum(dono[h, f] * w[f] * oferta[f] for f in 1:NF) ⟂ Y[h])
@constraint(m, poup - (sum(pou[h] * Y[h] for h in 1:NH) + receita - gasto_gov + FS0) ⟂ poup)
for f in 1:NF
    f == num && continue
    @constraint(m, sum(fd[f, j] for j in 1:NA) - oferta[f] ⟂ w[f])
end
@constraint(m, sum(pwm[i] * er * QM[i] for i in 1:NC) - sum(pfob(i) * QE[i] for i in 1:NC) - FS0 ⟂ er)

optimize!(m)
status = termination_status(m)
println("PATH: ", status)
walras = value(sum(fd[num, j] for j in 1:NA)) - oferta[num]
@printf("WALRAS %.2e\n", walras)
out = Dict("status" => string(status), "walras" => walras,
           "w" => value.(w)[:], "er" => value(er), "X" => [value(X(j)) for j in 1:NA],
           "pd" => value.(px)[:], "pq" => [value(pc(i)) for i in 1:NC],
           "imports" => value.(QM)[:], "exports" => value.(QE)[:],
           "pe" => [value(pfob(i)) for i in 1:NC],
           "hh_income" => value.(Y)[:], "savings" => value(poup),
           "export_tax" => value(sum(te[i] * pfob(i) * QE[i] for i in 1:NC)),
           "sigma" => value.(sigmaq)[:])
open(SAIDA, "w") do f
    JSON3.write(f, out)
end
