"""logit_statsmodels.py — a escolha ocupacional do cgelab contra o logit multinomial do statsmodels.

O `cgelab.behaviour.Behaviour.estimate` ajusta o logit multinomial por máxima verossimilhança
com o gradiente analítico e o BFGS do scipy; a suíte confere formas fechadas (só interceptos, e
um regressor binário saturado). Aqui o confronto é com outra implementação, num modelo com
regressores contínuos, onde não há forma fechada: os coeficientes e a log-verossimilhança têm de
coincidir com os do `statsmodels.MNLogit`. Também as equações de rendimento contra o `OLS`.

O statsmodels não é dependência do cgelab; instale-o só para rodar este script:

    pip install statsmodels
    python validation/logit_statsmodels.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:                        # pragma: no cover - depende do ambiente
    print("statsmodels não está instalado; nada a validar aqui")
    raise SystemExit(0)

from cgelab.behaviour import Behaviour, PersonSurvey


def pesquisa(n=4000, semente=20261019):
    rng = np.random.default_rng(semente)
    educ = rng.normal(9.0, 3.0, size=n)
    idade = rng.uniform(16, 65, size=n)
    mulher = rng.integers(0, 2, size=n).astype(float)
    # Utilidades verdadeiras: formal e informal contra a inatividade.
    V = np.column_stack([-4.0 + 0.35 * educ + 0.02 * idade - 0.5 * mulher,
                         -1.0 + 0.10 * educ + 0.01 * idade - 0.2 * mulher,
                         np.zeros(n)])
    escolha = np.argmax(V - np.log(-np.log(rng.uniform(size=V.shape))), axis=1)
    status = np.array(["f_for", "f_inf", "inactive"])[escolha]
    lny = np.where(escolha == 0, 0.5 + 0.08 * educ, -0.2 + 0.05 * educ) + rng.normal(0, 0.4, n)
    rend = np.where(escolha < 2, np.exp(lny), 0.0)
    quadro = pd.DataFrame({"household": 0, "status": status, "earnings": rend,
                           "educ": educ, "idade": idade, "mulher": mulher})
    return PersonSurvey(quadro, characteristics=["educ", "mulher"],
                        choice_characteristics=["educ", "idade", "mulher"])


def main() -> int:
    p = pesquisa()
    b = Behaviour.estimate(p, ["f_for", "f_inf"])
    f = p.frame
    Z = sm.add_constant(f[["educ", "idade", "mulher"]].to_numpy())
    # O statsmodels usa a primeira categoria como base: codifique a inatividade como 0.
    codigo = f["status"].map({"inactive": 0, "f_for": 1, "f_inf": 2}).to_numpy()
    ref = sm.MNLogit(codigo, Z).fit(method="newton", tol=1e-12, maxiter=200, disp=False)
    pior = 0.0
    for col, s in ((0, "f_for"), (1, "f_inf")):
        nosso = b.choice[b.statuses.index(s)]
        deles = np.asarray(ref.params)[:, col]
        d = float(np.max(np.abs(nosso - deles)))
        pior = max(pior, d)
        print(f"logit {s:<6} cgelab {np.round(nosso, 6)}  statsmodels {np.round(deles, 6)}"
              f"  desvio {d:.2e}")
    dll = abs(b.loglik - float(ref.llf))
    print(f"log-verossimilhança: cgelab {b.loglik:.8f}  statsmodels {float(ref.llf):.8f}"
          f"  desvio {dll:.2e}")
    for s in ("f_for", "f_inf"):
        m = (f["status"] == s).to_numpy()
        X = sm.add_constant(f.loc[m, ["educ", "mulher"]].to_numpy())
        ols = sm.OLS(np.log(f.loc[m, "earnings"].to_numpy()), X).fit()
        d = float(np.max(np.abs(b.earnings[s]["coef"] - ols.params)))
        pior = max(pior, d)
        print(f"rendimento {s:<6} desvio dos coeficientes {d:.2e}")
    ok = pior < 1e-5 and dll < 1e-6
    print("OK" if ok else "DIVERGE")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
