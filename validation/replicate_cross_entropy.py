"""replicate_cross_entropy.py — a entropia cruzada do cgelab contra o primal resolvido pelo Ipopt.

`cross_entropy.jl` lê a MCS a priori e a informação e resolve o problema primal (a divergência
de Csiszár sobre os fluxos, sujeita às balanças e à informação) com o Ipopt. O cgelab resolve o
dual por Newton. Este script escreve cada caso, compara célula a célula e a divergência.

Os casos: só a balança; totais de todas as contas (o RAS); totais de algumas; uma célula fixa;
uma célula-alvo; um agregado (as exportações); tudo junto; e uma MCS a priori com entradas
negativas.

    cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
    python validation/replicate_cross_entropy.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from cgelab import SAM
from cgelab.estimation import cross_entropy
from cgelab.examples import toy

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
TOL = 1e-8


def perturbada(semente=7, negativo=False) -> SAM:
    sam = toy._frame(toy._BASE)
    f = sam.frame.copy()
    f["value"] = f.value * np.random.default_rng(semente).uniform(0.8, 1.25, len(f))
    if negativo:
        # A poupança do governo vira despoupança: −5 de s_i para gov.
        f.loc[(f.row == "s_i") & (f.col == "gov"), "value"] = -5.0
    return SAM(frame=f, name="a_priori")


def caso(nome, prior, **info) -> dict:
    OUT.mkdir(exist_ok=True)
    dados, saida = OUT / f"ce_{nome}.json", OUT / f"ce_{nome}_out.json"
    saida.unlink(missing_ok=True)
    dados.write_text(json.dumps(dict(
        accounts=prior.accounts,
        prior=[[r, c, float(v)] for r, c, v in prior.frame.itertuples(index=False)],
        totals={a: float(v) for a, v in info.get("totals", {}).items()},
        fixed=[list(rc) for rc in info.get("fixed", [])],
        targets=[[r, c, float(v)] for (r, c), v in info.get("targets", {}).items()],
        aggregates=[dict(weights=[[r, c, float(w)] for (r, c), w in pesos.items()], value=valor)
                    for pesos, valor in info.get("aggregates", [])])))
    julia = os.environ.get("JULIA", "julia")
    r = subprocess.run([julia, f"--project={AQUI}", str(AQUI / "cross_entropy.jl"), str(dados),
                        str(saida)], cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    jl = json.loads(saida.read_text())
    est = cross_entropy(prior, **info)
    ref = SAM.from_long(pd.DataFrame(jl["sam"], columns=["row", "col", "value"]))
    acc = prior.accounts
    a, b = est.sam.matrix(acc).to_numpy(), ref.matrix(acc).to_numpy()
    pior = float(np.max(np.abs(a - b)) / max(1.0, float(np.abs(b).max())))
    dd = abs(est.divergence - jl["divergence"]) / max(1.0, abs(jl["divergence"]))
    pior = max(pior, dd)
    print(f"  {nome:<18} pior desvio {pior:.1e}   divergência {est.divergence:.6f}   "
          f"Ipopt {jl['status']}   Newton {est.iterations} passos   "
          f"desequilíbrio {est.sam.imbalance().abs().max():.1e}")
    return {"pior": pior, "ok": pior < TOL and jl["status"] == "LOCALLY_SOLVED"}


def main() -> int:
    print("cgelab contra o Ipopt: entropia cruzada")
    prior = perturbada()
    base = toy._frame(toy._BASE)
    todos = base.receipts().to_dict()
    exportacoes = ({("c_agr", "row"): 1.0, ("c_ind", "row"): 1.0}, 31.0)
    casos = {
        "so_balanca": caso("so_balanca", prior),
        "todos_os_totais": caso("todos_os_totais", prior, totals=todos),
        "alguns_totais": caso("alguns_totais", prior,
                              totals={"a_agr": 100.0, "c_ind": 300.0, "hh_1": 115.0}),
        "celula_fixa": caso("celula_fixa", prior, fixed=[("f_lab", "a_agr"), ("c_ind", "s_i")]),
        "celula_alvo": caso("celula_alvo", prior, targets={("c_agr", "hh_1"): 33.0}),
        "agregado": caso("agregado", prior, aggregates=[exportacoes]),
        "tudo_junto": caso("tudo_junto", prior, totals={"a_agr": 100.0},
                           fixed=[("f_lab", "a_agr")], targets={("c_agr", "hh_1"): 33.0},
                           aggregates=[exportacoes]),
        "negativo": caso("negativo", perturbada(negativo=True), totals={"a_ind": 200.0}),
    }
    pior = max(r["pior"] for r in casos.values())
    if all(r["ok"] for r in casos.values()):
        print(f"OK: {len(casos)} casos, pior desvio relativo {pior:.1e}")
        return 0
    print(f"FALHOU: pior desvio relativo {pior:.1e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
