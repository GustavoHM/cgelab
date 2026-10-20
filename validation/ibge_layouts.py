"""ibge_layouts.py — os leitores do IBGE no cgelab contra outras leituras dos mesmos arquivos.

1. PNAD Contínua: o `cgelab.ibge.read_ibge` lê o fixture `tests/data/ibge` e o R lê o mesmo
   fixture como o pacote PNADcIBGE lê os microdados (o programa SAS com `readr::read_table`, os
   dados com `readr::read_fwf`); com o PNADcIBGE instalado, usa-se o próprio `read_pnadc`. Os
   campos têm de coincidir. O PNADcIBGE ignora os decimais implícitos de um informat `w.d`, que a
   PNAD Contínua não usa; esses campos ficam fora do confronto.
2. POF 2017–2018: com `CGELAB_IBGE_DIR` apontando para os programas de leitura do IBGE
   (`Leitura dos Microdados - SAS.txt` e `Leitura dos Microdados - R.R`), as larguras que o cgelab
   extrai do programa SAS têm de ser as do `read.fwf` do programa R, registro por registro, e os
   campos têm de ser contíguos.

    python validation/ibge_layouts.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from cgelab.ibge import read_ibge, read_sas_program

RAIZ = Path(__file__).resolve().parents[1]
FIXTURE = RAIZ / "tests" / "data" / "ibge"

R_LEITURA = r"""
args <- commandArgs(trailingOnly = TRUE)
dados <- args[1]; input <- args[2]; saida <- args[3]
if (requireNamespace("PNADcIBGE", quietly = TRUE)) {
  x <- PNADcIBGE::read_pnadc(microdata = dados, input_txt = input)
} else {
  lay <- suppressWarnings(readr::read_table(input, col_names = FALSE,
                                            locale = readr::locale(encoding = "latin1")))
  lay <- subset(lay, substr(X1, 1, 1) == "@")
  lay$type <- ifelse(substr(lay$X3, 1, 1) == "$", "c", "d")
  lay$start <- as.numeric(gsub("@", "", lay$X1))
  lay$width <- as.integer(gsub("\\..*$", "", chartr("$", " ", lay$X3)))
  lay$end <- lay$start + lay$width - 1
  x <- readr::read_fwf(dados, readr::fwf_positions(lay$start, lay$end, lay$X2),
                       col_types = paste0(lay$type, collapse = ""),
                       locale = readr::locale(encoding = "latin1"))
}
utils::write.csv(as.data.frame(x), saida, row.names = FALSE, na = "")
"""


def pnadc() -> bool:
    rscript = shutil.which("Rscript")
    if rscript is None:
        print("PNAD Contínua: sem Rscript; nada a confrontar")
        return True
    dados, input_ = FIXTURE / "PNADC_teste.txt", FIXTURE / "pnadc_input.txt"
    with tempfile.TemporaryDirectory() as tmp:
        prog = Path(tmp) / "leitura.R"
        prog.write_text(R_LEITURA)
        saida = Path(tmp) / "r.csv"
        r = subprocess.run([rscript, str(prog), str(dados), str(input_), str(saida)],
                           capture_output=True, text=True)
        if not saida.exists():
            print("PNAD Contínua: o R falhou\n" + r.stderr[-1500:])
            return False
        deles = pd.read_csv(saida, dtype=str, keep_default_na=False)
    nosso = read_ibge(dados, input_)
    campos = read_sas_program(input_)["pnadc_teste"].fields.set_index("name")
    ok = True
    for nome, f in campos.iterrows():
        if f["decimals"]:
            continue
        a = nosso[nome]
        b = deles[nome].replace("", np.nan)
        if f["char"]:
            iguais = (a.fillna("").astype(str) == b.fillna("").astype(str)).all()
        else:
            iguais = np.allclose(a.astype(float), b.astype(float), equal_nan=True, rtol=1e-14)
        ok &= bool(iguais)
        if not iguais:
            print(f"  {nome}: cgelab {a.tolist()}  R {b.tolist()}")
    print(f"PNAD Contínua: {len(nosso)} pessoas, {len(campos)} campos  "
          f"{'OK' if ok else 'DIVERGE'}")
    return ok


def pof() -> bool:
    raiz = os.environ.get("CGELAB_IBGE_DIR")
    if not raiz:
        print("POF: CGELAB_IBGE_DIR não definido; nada a confrontar")
        return True
    sas = next(Path(raiz).rglob("*Leitura dos Microdados - SAS.txt"), None)
    r_prog = next(Path(raiz).rglob("*Leitura dos Microdados - R.R"), None)
    if sas is None or r_prog is None:
        print("POF: os programas de leitura não estão em CGELAB_IBGE_DIR")
        return True
    registros = {r.infile.lower(): r for r in read_sas_program(sas).values()}
    texto = r_prog.read_text(encoding="latin-1")
    blocos = re.findall(r'read\.fwf\("([^"]+)"\s*,\s*widths\s*=\s*c\(([^)]*)\)', texto, re.S)
    ok = True
    for arquivo, larguras in blocos:
        w = [int(x) for x in re.findall(r"\d+", larguras)]
        lay = registros.get(arquivo.lower())
        if lay is None:
            print(f"  {arquivo}: sem registro no programa SAS")
            ok = False
            continue
        s, ini = lay.fields["width"].tolist(), lay.fields["start"].tolist()
        contiguos = all(b == a + la for a, la, b in zip(ini, s, ini[1:]))
        igual = s == w and contiguos
        ok &= igual
        print(f"  {arquivo:<42} R {len(w):>3}  SAS {len(s):>3}  {'OK' if igual else 'DIVERGE'}")
    print(f"POF: {len(blocos)} registros  {'OK' if ok else 'DIVERGE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([pnadc(), pof()]) else 1)
