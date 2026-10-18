"""Exportação para JuMP: estrutura dos dados e do modelo gerado.

A verificação de VALOR — exportar, resolver no Julia e comparar de volta — está em
``validation/roundtrip_jump.py``, porque precisa de Julia, JuMP e Ipopt instalados. Estes testes
cobrem o que se pode cobrir sem eles: que os dados exportados são os coeficientes EFETIVOS, que o
fechamento e o cenário viajam junto, e que o modelo gerado deixa de fora exatamente o mercado que
tem de ficar de fora.
"""

import json

import numpy as np
import pytest

import cgelab as cge
from cgelab.examples import toy


def _exporta(tmp_path, econ, closure=None, scenario=None):
    jl = tmp_path / "m.jl"
    dados = cge.to_jump(econ, closure, scenario, path=jl)
    return dados, jl.read_text(encoding="utf-8"), json.loads((tmp_path / "m_data.json").read_text(encoding="utf-8"))


def test_exporta_os_coeficientes_com_a_produtividade_embutida(tmp_path):
    """A produtividade Hicks-neutra escala toda exigência de insumo, não só o custo unitário.

    Exportar os coeficientes crus e a produtividade em separado convidaria o lado Julia a aplicá-la
    apenas aos preços, que é o erro cujo sintoma é um resíduo de Walras que CRESCE ao longo de uma
    trajetória. Por isso vão efetivos.
    """
    _, econ = toy.build()
    tfp = np.full(econ.n_act, 1.25)
    dados, _, _ = _exporta(tmp_path, econ, scenario=cge.Scenario(productivity=tfp))
    assert np.allclose(np.array(dados["io"]), econ.io / tfp[None, :])
    assert np.allclose(np.array(dados["vunit"]), econ.vunit / tfp)


def test_o_fechamento_e_o_cenario_viajam_com_os_dados(tmp_path):
    _, econ = toy.build()
    W = cge.consumer_wedge(econ, {"c_ind": 0.05})
    clo = cge.Closure(investment="fixed_real", foreign="fixed_rate", government="nominal_fixed")
    sc = cge.Scenario(name="teste", consumer_wedge=W, wedge_revenue_to="lumpsum")
    dados, _, disco = _exporta(tmp_path, econ, clo, sc)
    assert dados["investment"] == "fixed_real"
    assert dados["foreign"] == "fixed_rate"
    assert dados["government"] == "nominal_fixed"
    assert dados["scenario_name"] == "teste"
    assert np.allclose(np.array(dados["wedge"]), W)
    # sob "lumpsum" a reciclagem segue a POPULAÇÃO, não a propriedade do capital
    assert np.allclose(np.array(dados["recycle_share"]), econ.pop_share)
    assert disco["nc"] == econ.n_com


def test_o_modelo_gerado_deixa_o_mercado_do_numerario_de_fora(tmp_path):
    """É o desenho inteiro: o resíduo dele é verificação, não identidade imposta."""
    _, econ = toy.build()
    _, texto, _ = _exporta(tmp_path, econ)
    assert "f == NUM && continue" in texto
    assert "WALRAS" in texto
    assert "violacao_maxima" in texto
    assert "a bandeira do Ipopt não é" in texto


def test_o_modelo_gerado_e_o_mesmo_para_qualquer_economia(tmp_path):
    """Um `.jl` por modelo seria um artefato a auditar por exercício; há um só."""
    _, e1 = toy.build()
    _, e2 = toy.build_with_margins()
    _, t1, _ = _exporta(tmp_path / "a", e1) if (tmp_path / "a").mkdir() or True else (None, "", {})
    _, t2, _ = _exporta(tmp_path / "b", e2) if (tmp_path / "b").mkdir() or True else (None, "", {})
    assert t1 == t2


def test_atividades_de_conta_propria_saem_marcadas(tmp_path):
    _, econ = toy.build_with_own_production()
    dados, _, _ = _exporta(tmp_path, econ,
                           scenario=cge.Scenario(activity_scale={"a_own": 2.0}))
    j = econ.activity("a_own")
    assert dados["own_account"][j] is True
    assert dados["act_scale"][j] == 2.0
    assert dados["X_exog"][j] == pytest.approx(2.0 * econ.X0[j])
    # a demanda intermediária desse bloco entra na absorção, e não pelo balanço de mercadorias
    assert len(dados["exog_intermediate"]) == econ.n_com


def test_to_jump_writes_cobb_douglas_value_added_as_unit_elasticity(tmp_path):
    """The spec's sigma_va is not read under value_added='cobb_douglas'; exporting
    it made the replica solve a CES the package never solved."""
    _, econ = toy.build(model_spec=toy.spec(value_added="cobb_douglas", sigma_va=0.5))
    dados = cge.to_jump(econ, path=tmp_path / "m.jl")
    assert dados["sigma_va"] == [1.0] * econ.n_act
