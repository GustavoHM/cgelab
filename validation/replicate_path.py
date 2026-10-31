"""replicate_path.py — o backend Pyomo/PATH contra o solver do cgelab, com complementaridade.

`solve_period(..., backend="path")` escreve o equilíbrio do período em Pyomo, as equações da
réplica de `to_jump` mais as condições de complementaridade (`pyomo.mpec`), cada uma pareada com
a sua variável, e o PATH resolve o problema de complementaridade misto inteiro de uma vez. O
cgelab escreve as mesmas condições com a função de Fischer–Burmeister dentro do seu Newton
aninhado. Este script roda os 26 casos de `replicate_least_cost.py` (menor custo, capacidade,
cotas, limites de preço, piso salarial e reserva) pelos dois caminhos e compara preços dos
fatores, câmbio, níveis de atividade, preços, custos unitários, renda das famílias, poupança,
rendas de escassez, prêmios, subsídios, compras e desemprego. Os mesmos casos já batem com a
formulação independente de `least_cost.jl`; aqui não é preciso Julia.

    pip install amplpy pyomo && python -m amplpy.modules install path
    python validation/replicate_path.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import replicate_least_cost as lc  # noqa: E402

import cgelab as cge  # noqa: E402

TOL = 1e-8
tempos = {"scipy": 0.0, "path": 0.0}


def _campos(res, econ) -> dict:
    bounds = res.price_bounds or {}
    return {
        "w": res.factor_prices, "er": res.exchange_rate, "X": res.X, "pd": res.pd,
        "pq": res.pq, "pa": res.pa, "hh_income": res.hh_income, "savings": res.savings,
        "rent": [(res.capacity or {}).get(a, {}).get("rent", 0.0) for a in econ.acts],
        "premium": [(res.quota or {}).get(c, {}).get("premium", 0.0) for c in econ.coms],
        "subsidy": [bounds.get("ceiling", {}).get(c, {}).get("subsidy", 0.0) for c in econ.coms],
        "purchases": [bounds.get("floor", {}).get(c, {}).get("purchases", 0.0)
                      for c in econ.coms],
        "unemployment": res.unemployment,
    }


def caso(nome, sam, econ, scenario, closure=None) -> dict:
    closure = closure or cge.Closure()
    t = time.perf_counter()
    ref = cge.solve_period(econ, closure, scenario)
    tempos["scipy"] += time.perf_counter() - t
    t = time.perf_counter()
    res = cge.solve_period(econ, closure, scenario, backend="path", raise_on_fail=False)
    tempos["path"] += time.perf_counter() - t
    a, b = _campos(res, econ), _campos(ref, econ)
    desvios = {k: lc._rel(a[k], b[k]) for k in a}
    pior = max(desvios.values())
    campo = max(desvios, key=desvios.get)
    ok = res.converged and pior < TOL
    print(f"  {nome:<26} pior desvio {pior:.1e} ({campo:<12}) resíduo cgelab {res.residual:.1e}"
          f"   {res.message}   {'OK' if ok else 'DIVERGE'}")
    return {"pior": pior, "ok": ok}


def main() -> int:
    lc.caso = caso
    lc.print = lambda *a, **k: None        # o resumo de replicate_least_cost fala do JuMP
    try:
        print("backend Pyomo/PATH contra o scipy, nos casos de replicate_least_cost.py")
        codigo = lc.main()
    finally:
        del lc.print
    print(f"tempo total: scipy {tempos['scipy']:.1f} s, path {tempos['path']:.1f} s")
    print("OK" if codigo == 0 else "FALHOU")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
