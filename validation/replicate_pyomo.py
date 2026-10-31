"""replicate_pyomo.py — os backends Pyomo contra o solver do cgelab, nos casos do roundtrip.

`solve_period(..., backend="ipopt")` escreve o equilíbrio do período como um sistema simultâneo
em Pyomo, as mesmas equações da réplica em JuMP de `to_jump`, e o IPOPT o resolve; o cgelab
mapeia a solução nas suas incógnitas e só a aceita se o seu próprio resíduo valer ali. Este
script roda os 50 casos de `roundtrip_jump.py` (as mesmas economias, fechamentos e cenários) pelo
backend e pelo `scipy` e compara preços dos fatores, câmbio, preços, quantidades, renda das
famílias, poupança, exportações e preços mundiais. Não precisa de Julia. Com `--backend path`
o mesmo sistema vai ao PATH, como problema de complementaridade misto sem nenhuma condição de
complementaridade: só igualdades, que ele tem de resolver como o IPOPT.

    pip install amplpy pyomo && python -m amplpy.modules install ipopt path
    python validation/replicate_pyomo.py [--backend path]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import roundtrip_jump as rt  # noqa: E402

import cgelab as cge  # noqa: E402

TOL = 1e-8
BACKEND = "ipopt"
tempos = {"scipy": 0.0, "backend": 0.0}


def caso(nome: str, econ, closure, scenario) -> dict:
    t = time.perf_counter()
    ref = cge.solve_period(econ, closure, scenario)
    tempos["scipy"] += time.perf_counter() - t
    t = time.perf_counter()
    res = cge.solve_period(econ, closure, scenario, backend=BACKEND, raise_on_fail=False)
    tempos["backend"] += time.perf_counter() - t
    a, b = rt._referencia(res), rt._referencia(ref)
    desvios = {k: rt._rel(a[k], b[k]) for k in rt.CAMPOS}
    pior = max(desvios.values())
    campo = max(desvios, key=desvios.get)
    ok = res.converged and pior < TOL
    print(f"  {nome:<46} pior desvio {pior:.1e} ({campo:<12}) resíduo cgelab {res.residual:.1e}"
          f"   {res.message}   {'OK' if ok else 'DIVERGE'}")
    return {"pior": pior, "viol": res.residual, "ok": ok}


def main(backend: str = "ipopt") -> int:
    global BACKEND
    BACKEND = backend
    rt.caso = caso
    rt.print = lambda *a, **k: None        # o resumo de roundtrip_jump fala do JuMP
    try:
        print(f"backend Pyomo/{backend.upper()} contra o scipy, nos casos de roundtrip_jump.py")
        codigo = rt.main()
    finally:
        del rt.print
    print(f"tempo total: scipy {tempos['scipy']:.1f} s, {backend} {tempos['backend']:.1f} s")
    print("OK" if codigo == 0 else "FALHOU")
    return codigo


if __name__ == "__main__":
    sys.exit(main(sys.argv[sys.argv.index("--backend") + 1] if "--backend" in sys.argv
                  else "ipopt"))
