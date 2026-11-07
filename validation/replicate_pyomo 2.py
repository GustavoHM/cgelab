"""replicate_pyomo.py — o backend Pyomo/IPOPT contra o solver do cgelab, nos casos do roundtrip.

`solve_period(..., backend="ipopt")` escreve o equilíbrio do período como um sistema simultâneo
em Pyomo, as mesmas equações da réplica em JuMP de `to_jump`, e o IPOPT o resolve; o cgelab
mapeia a solução nas suas incógnitas e só a aceita se o seu próprio resíduo valer ali. Este
script roda os 50 casos de `roundtrip_jump.py` (as mesmas economias, fechamentos e cenários) pelo
backend e pelo `scipy` e compara preços dos fatores, câmbio, preços, quantidades, renda das
famílias, poupança, exportações e preços mundiais. Não precisa de Julia.

    pip install amplpy pyomo && python -m amplpy.modules install ipopt
    python validation/replicate_pyomo.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roundtrip_jump as rt  # noqa: E402

import cgelab as cge  # noqa: E402

TOL = 1e-8
tempos = {"scipy": 0.0, "ipopt": 0.0}


def caso(nome: str, econ, closure, scenario) -> dict:
    t = time.perf_counter()
    ref = cge.solve_period(econ, closure, scenario)
    tempos["scipy"] += time.perf_counter() - t
    t = time.perf_counter()
    res = cge.solve_period(econ, closure, scenario, backend="ipopt", raise_on_fail=False)
    tempos["ipopt"] += time.perf_counter() - t
    a, b = rt._referencia(res), rt._referencia(ref)
    desvios = {k: rt._rel(a[k], b[k]) for k in rt.CAMPOS}
    pior = max(desvios.values())
    campo = max(desvios, key=desvios.get)
    ok = res.converged and pior < TOL
    print(f"  {nome:<46} pior desvio {pior:.1e} ({campo:<12}) resíduo cgelab {res.residual:.1e}"
          f"   {res.message}   {'OK' if ok else 'DIVERGE'}")
    return {"pior": pior, "viol": res.residual, "ok": ok}


def main() -> int:
    rt.caso = caso
    rt.print = lambda *a, **k: None        # o resumo de roundtrip_jump fala do JuMP
    try:
        print("backend Pyomo/IPOPT contra o scipy, nos casos de roundtrip_jump.py")
        codigo = rt.main()
    finally:
        del rt.print
    print(f"tempo total: scipy {tempos['scipy']:.1f} s, ipopt {tempos['ipopt']:.1f} s")
    print("OK" if codigo == 0 else "FALHOU")
    return codigo


if __name__ == "__main__":
    sys.exit(main())
