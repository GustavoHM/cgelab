"""replicate_melitz.py — as formas fechadas do cgelab contra o equilíbrio resolvido das primitivas.

`melitz.jl` recebe só os parâmetros. Escreve o preço, a demanda e o lucro de cada firma, a
densidade de Pareto e cinco condições (lucro zero nos dois cortes, livre entrada, o índice de
preços e o mercado de trabalho), faz toda integral por quadratura de Gauss-Legendre e resolve
com o Ipopt. Não usa nenhuma forma fechada da Pareto e não impõe que a despesa seja a renda do
trabalho: essa igualdade sai da solução e é conferida aqui. O cgelab calcula tudo em forma
fechada, sem resolver nada.

Os casos: a base; custo iceberg menor e maior; custo fixo de exportação menor e maior; outras
elasticidades e custo de entrada; outra escala; cauda pesada (k − σ + 1 = 0,5); e seleção quase
no canto em que toda firma ativa exporta.

    cd validation && julia --project=. -e 'using Pkg; Pkg.instantiate()' && cd ..
    python validation/replicate_melitz.py
    python validation/replicate_melitz.py --write-fixture
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import cgelab as cge

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
TOL = 1e-9
CAMPOS = (
    "domestic_cutoff", "export_cutoff", "entrants", "domestic_firms", "exporters",
    "price_index", "welfare", "domestic_revenue", "export_revenue",
    "average_domestic_productivity", "average_export_productivity", "export_share",
    "entry_probability", "real_wage", "domestic_share",
)
CASOS = {
    "base": cge.MelitzSpec(),
    "lower_iceberg": replace(cge.MelitzSpec(), iceberg_cost=1.2),
    "higher_iceberg": replace(cge.MelitzSpec(), iceberg_cost=2.0),
    "lower_export_fixed": replace(cge.MelitzSpec(), export_fixed_cost=0.7),
    "higher_export_fixed": replace(cge.MelitzSpec(), export_fixed_cost=1.8),
    "entry_and_elasticities": cge.MelitzSpec(
        sigma=3.0, pareto_shape=4.5, entry_cost=0.5, iceberg_cost=1.6,
    ),
    "scale": cge.MelitzSpec(
        sigma=5.0, pareto_shape=6.5, productivity_min=2.0, labour=250.0,
        entry_cost=2.0, domestic_fixed_cost=1.5, export_fixed_cost=2.0,
        iceberg_cost=1.7,
    ),
    "heavy_tail": cge.MelitzSpec(sigma=4.0, pareto_shape=3.5, entry_cost=2.0),
    "near_all_exporters": replace(cge.MelitzSpec(), iceberg_cost=1.05, entry_cost=0.8),
}


def julia(nome: str, spec) -> dict:
    OUT.mkdir(exist_ok=True)
    dados, saida = OUT / f"melitz_{nome}.json", OUT / f"melitz_{nome}_out.json"
    saida.unlink(missing_ok=True)
    dados.write_text(json.dumps(asdict(spec)))
    r = subprocess.run([os.environ.get("JULIA", "julia"), f"--project={AQUI}",
                        str(AQUI / "melitz.jl"), str(dados), str(saida)],
                       cwd=OUT, capture_output=True, text=True)
    if not saida.exists():
        raise RuntimeError(f"Julia falhou:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    return json.loads(saida.read_text())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-fixture", action="store_true")
    args = parser.parse_args()
    fixture, pior, falhou = {}, 0.0, False
    for nome, spec in CASOS.items():
        ref = julia(nome, spec)
        nosso = cge.solve_melitz(spec)
        desvios = {c: abs(float(getattr(nosso, c)) - ref[c]) / max(1.0, abs(ref[c]))
                   for c in CAMPOS}
        campo = max(desvios, key=desvios.get)
        # A restrição orçamentária que o Julia não impôs.
        orcamento = abs(ref["expenditure"] - spec.labour) / spec.labour
        ok = (ref["status"] == "LOCALLY_SOLVED" and desvios[campo] < TOL and orcamento < TOL
              and ref["residual"] < TOL)
        pior = max(pior, desvios[campo])
        falhou |= not ok
        print(f"  {nome:<24} pior desvio {desvios[campo]:.1e} ({campo})  "
              f"despesa − trabalho {orcamento:.1e}  nós {ref['nodes']}  "
              f"{'OK' if ok else 'FALHOU'}")
        fixture[nome] = {"spec": asdict(spec), "result": {c: ref[c] for c in CAMPOS}}
    print(f"{len(CASOS)} casos, pior desvio {pior:.1e}")
    if falhou:
        return 1
    if args.write_fixture:
        destino = AQUI.parent / "tests" / "data" / "melitz.json"
        destino.write_text(json.dumps(fixture, indent=2) + "\n")
        print(f"Fixture gravada em {destino}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
