"""replicate_hosoe_market.py — moncge e irscge contra a transcrição dos programas do GAMS.

`hosoe_market.jl` transcreve linha a linha dois programas da biblioteca do GAMS (Hosoe, Gasawa e
Hashimoto, 2004, caps. 10.4 e 10.6): o monopólio com markup de Lerner nas vendas internas e o
custo fixo de um décimo do capital. Nada vem do cgelab. Este script o roda, resolve os mesmos
modelos com o cgelab (`ModelSpec.monopoly` e `fixed_costs`, em `cgelab.examples.hosoe`) e compara
cada variável que o programa original exibe, no ano-base e com a abolição das tarifas.

    python validation/replicate_hosoe_market.py            # compara
    python validation/replicate_hosoe_market.py --write    # e grava a referência que a suíte lê

A referência gravada (`tests/data/hosoe_market.json`) deixa a suíte checar a réplica sem Julia.
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
REFERENCIA = AQUI.parent / "tests" / "data" / "hosoe_market.json"
CASOS = {"base": None, "abolicao": cge.Scenario(name="abolicao", tariff=np.zeros(2))}


def cgelab_niveis() -> dict:
    out = {}
    for modelo, construtor in (("moncge", hosoe.build_monopoly), ("irscge", hosoe.build_scale)):
        _, econ = construtor()
        out[modelo] = {caso: hosoe.variables(econ, cge.solve_period(econ, hosoe.CLOSURE, sc))
                       for caso, sc in CASOS.items()}
    return out


def _achata(d, prefixo=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from _achata(v, f"{prefixo}{k}.")
        else:
            yield f"{prefixo}{k}", float(v)


def compara(ref: dict, nosso: dict) -> float:
    pior = 0.0
    for modelo in ref:
        for caso in ref[modelo]:
            r = dict(_achata(ref[modelo][caso]))
            n = dict(_achata(nosso[modelo][caso]))
            comuns = sorted(set(r) & set(n))
            desvio = {k: abs(n[k] - r[k]) / max(abs(r[k]), 1.0) for k in comuns}
            k_pior = max(desvio, key=desvio.get)
            pior = max(pior, desvio[k_pior])
            print(f"  {modelo:<7}{caso:<9}{len(comuns):>4} variáveis   pior desvio "
                  f"{desvio[k_pior]:.1e} ({k_pior})")
    return pior


def main() -> int:
    OUT.mkdir(exist_ok=True)
    saida = OUT / "hosoe_market_out.json"
    julia = os.environ.get("JULIA", "julia")
    r = subprocess.run([julia, f"--project={AQUI}", str(AQUI / "hosoe_market.jl"), str(saida)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not saida.exists():
        print(r.stdout[-2000:], r.stderr[-2000:])
        return 1
    transcricao = json.loads(saida.read_text())
    for modelo in transcricao.values():
        for caso in modelo.values():
            for chave in ("status", "violacao", "FC"):
                caso.pop(chave)
            if modelo is transcricao["irscge"]:
                caso.pop("RT")
    print("cgelab contra a transcrição de moncge.gms e irscge.gms")
    pior = compara(transcricao, cgelab_niveis())
    ok = pior < 1e-9
    print(f"{'OK' if ok else 'DIVERGE'}: pior desvio relativo {pior:.1e}")
    if ok and "--write" in sys.argv:
        REFERENCIA.write_text(json.dumps(transcricao, indent=1, sort_keys=True) + "\n")
        print(f"gravado: {REFERENCIA}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
