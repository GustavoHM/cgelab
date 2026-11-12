"""replicate_hosoe_two.py — twocge contra a transcrição do programa do GAMS.

`hosoe_two.jl` transcreve linha a linha o twocge da biblioteca do GAMS (Hosoe, Gasawa e
Hashimoto, 2004, cap. 10.3): Japão e EUA, cada um um stdcge, que comerciam entre si. Nada vem do
cgelab. Este script o roda, resolve os mesmos dois países com `cgelab.solve_two_countries` (cada
um com `solve_period`, os preços mundiais achados por fora) e compara cada variável que o
programa original exibe, nos dois países, no ano-base e com a abolição das tarifas nos dois.

    python validation/replicate_hosoe_two.py            # compara
    python validation/replicate_hosoe_two.py --write    # e grava a referência que a suíte lê

A referência gravada (`tests/data/hosoe_twocge.json`) deixa a suíte checar a réplica sem Julia.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

import cgelab as cge
from cgelab.examples import hosoe

AQUI = Path(__file__).resolve().parent
OUT = AQUI / "out"
REFERENCIA = AQUI.parent / "tests" / "data" / "hosoe_twocge.json"
CASOS = {"base": None, "abolicao": cge.Scenario(name="abolicao", tariff=np.zeros(2))}


def cgelab_niveis() -> dict:
    (_, jpn), (_, usa) = hosoe.build_two_countries()
    out = {}
    for caso, sc in CASOS.items():
        r = cge.solve_two_countries(jpn, usa, closure_a=hosoe.CLOSURE, closure_b=hosoe.CLOSURE,
                                    scenario_a=sc, scenario_b=sc, names=("JPN", "USA"))
        out[caso] = {pais: hosoe.variables(e, r.periods[pais])
                     for pais, e in (("JPN", jpn), ("USA", usa))}
    return out


def _achata(d, prefixo=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _achata(v, f"{prefixo}{k}.")
        else:
            yield f"{prefixo}{k}", float(v)


def compara(ref: dict, nosso: dict) -> float:
    pior = 0.0
    for caso in ref:
        for pais in ("JPN", "USA"):
            r = dict(_achata(ref[caso][pais]))
            n = dict(_achata(nosso[caso][pais]))
            comuns = sorted(set(r) & set(n))
            desvio = {k: abs(n[k] - r[k]) / max(abs(r[k]), 1.0) for k in comuns}
            k_pior = max(desvio, key=desvio.get)
            pior = max(pior, desvio[k_pior])
            print(f"  {caso:<9}{pais:<5}{len(comuns):>4} variáveis   pior desvio "
                  f"{desvio[k_pior]:.1e} ({k_pior})")
    return pior


def main() -> int:
    OUT.mkdir(exist_ok=True)
    saida = OUT / "hosoe_two_out.json"
    julia = os.environ.get("JULIA", "julia")
    r = subprocess.run([julia, f"--project={AQUI}", str(AQUI / "hosoe_two.jl"), str(saida)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not saida.exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    transcricao = json.loads(saida.read_text())
    for caso in transcricao.values():
        for chave in ("status", "violacao"):
            caso.pop(chave)
    print("cgelab contra a transcrição de twocge.gms")
    pior = compara(transcricao, cgelab_niveis())
    ok = pior < 1e-9
    print(f"{'OK' if ok else 'DIVERGE'}: pior desvio relativo {pior:.1e}")
    if ok and "--write" in sys.argv:
        REFERENCIA.write_text(json.dumps(transcricao, indent=1, sort_keys=True) + "\n")
        print(f"gravado: {REFERENCIA}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
