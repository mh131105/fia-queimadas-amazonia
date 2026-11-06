"""Testes unitários da Etapa 5: Rede Bayesiana, d-separação, inferência por enumeração e pgmpy."""

import unittest

import numpy as np
from pgmpy.inference import VariableElimination

from rede_bayesiana import (
    ARESTAS_PADRAO,
    NOS_PADRAO,
    RedeBayesianaBinaria,
    extrair_variaveis_binarias,
)
from utils import carregar_dados, iterar_folds, preparar_particao


class TestEstruturaEDSeparacao(unittest.TestCase):
    def setUp(self):
        self.rede = RedeBayesianaBinaria()

    def test_validacao_dag_e_ciclo(self):
        self.assertEqual(set(self.rede.nos), set(NOS_PADRAO))
        self.assertEqual(self.rede.arestas, ARESTAS_PADRAO)
        # M e H não têm pais; F deve aparecer após S e H na ordem topológica
        pos = {no: i for i, no in enumerate(self.rede.ordem_topologica)}
        for u, v in ARESTAS_PADRAO:
            self.assertLess(pos[u], pos[v])

        with self.assertRaises(ValueError):
            RedeBayesianaBinaria(nos=("A", "B", "C"), arestas=(("A", "B"), ("B", "C"), ("C", "A")))
        with self.assertRaises(ValueError):
            RedeBayesianaBinaria(nos=("A", "B"), arestas=(("A", "A"),))
        with self.assertRaises(ValueError):
            RedeBayesianaBinaria(nos=("A", "B"), arestas=(("A", "Z"),))

    def test_economia_de_parametros(self):
        info = self.rede.economia_parametros()
        self.assertEqual(info["n_variaveis"], 5)
        self.assertEqual(info["parametros_conjunta"], 31)
        self.assertEqual(info["parametros_por_no"], {"M": 1, "R": 2, "S": 4, "H": 1, "F": 4})
        self.assertEqual(info["parametros_rede"], 12)
        self.assertEqual(info["economia_absoluta"], 19)
        self.assertAlmostEqual(info["economia_relativa"], 19 / 31)

    def test_d_separacao_e_colisor_compatibilidade_pgmpy(self):
        # Define CPTs válidas apenas para instanciar o modelo pgmpy e comparar d-separação
        self.rede.definir_cpts({
            "M": [0.5, 0.5],
            "H": [0.5, 0.5],
            "R": [[0.8, 0.7], [0.2, 0.3]],
            "S": [[[0.3, 0.8], [0.5, 0.9]], [[0.7, 0.2], [0.5, 0.1]]],
            "F": [[[0.8, 0.7], [0.6, 0.5]], [[0.2, 0.3], [0.4, 0.5]]],
        })
        modelo_pgmpy = self.rede.para_pgmpy()

        # 1) Colisor S -> F <- H: bloqueado sem F, aberto dado F
        self.assertTrue(self.rede.d_separados("S", "H", dado=()))
        self.assertFalse(self.rede.d_separados("S", "H", dado=("F",)))

        # 2) Cadeia mediada por S: M e F são d-separados dado S; R e F são d-separados dado S
        self.assertFalse(self.rede.d_separados("M", "F", dado=()))
        self.assertTrue(self.rede.d_separados("M", "F", dado=("S",)))
        self.assertTrue(self.rede.d_separados("R", "F", dado=("S",)))
        self.assertTrue(self.rede.d_separados("R", "H", dado=("S",)))
        self.assertFalse(self.rede.d_separados("R", "H", dado=("F",)))

        # Compara com is_dconnected do pgmpy em diversos pares e condicionantes
        casos = [
            ("S", "H", ()),
            ("S", "H", ("F",)),
            ("M", "F", ()),
            ("M", "F", ("S",)),
            ("M", "F", ("R",)),
            ("M", "F", ("R", "S")),
            ("R", "F", ("S",)),
            ("R", "H", ()),
            ("R", "H", ("F",)),
            ("R", "H", ("S", "F")),
            ("M", "H", ()),
            ("M", "H", ("F",)),
        ]
        for x, y, z in casos:
            d_sep_proprio = self.rede.d_separados(x, y, dado=z)
            try:
                d_con_pgmpy = modelo_pgmpy.is_dconnected(x, y, observed=list(z))
            except TypeError:
                d_con_pgmpy = modelo_pgmpy.is_dconnected(x, y, Z=list(z))
            self.assertEqual(d_sep_proprio, not d_con_pgmpy, msg=f"Divergência em ({x}, {y} | {z})")

    def test_colisor_com_descendente_observado(self):
        # S -> F <- H e F -> A (alerta descendente do colisor)
        rede_desc = RedeBayesianaBinaria(
            nos=("S", "H", "F", "A"),
            arestas=(("S", "F"), ("H", "F"), ("F", "A")),
        )
        self.assertTrue(rede_desc.d_separados("S", "H", dado=()))
        self.assertFalse(rede_desc.d_separados("S", "H", dado=("F",)))
        self.assertFalse(rede_desc.d_separados("S", "H", dado=("A",)))


class TestInferenciaPorEnumeracao(unittest.TestCase):
    def test_inferencia_sintetica_e_explaining_away(self):
        rede = RedeBayesianaBinaria()
        # CPTs sintéticas com números redondos
        rede.definir_cpts({
            "M": [0.4, 0.6],
            "H": [0.5, 0.5],
            "R": [[0.9, 0.7], [0.1, 0.3]],
            "S": [
                [[0.3, 0.8], [0.1, 0.6]],  # S=0
                [[0.7, 0.2], [0.9, 0.4]],  # S=1
            ],
            "F": [
                [[0.9, 0.7], [0.6, 0.2]],  # F=0
                [[0.1, 0.3], [0.4, 0.8]],  # F=1
            ],
        })
        conj = rede.distribuicao_conjunta_rede()
        self.assertEqual(conj.shape, (2, 2, 2, 2, 2))
        self.assertAlmostEqual(conj.sum(), 1.0, places=12)

        # Sem observar F, S e H são independentes na rede: P(S=1 | H=1) == P(S=1 | H=0) == P(S=1)
        p_s = rede.inferencia_por_enumeracao("S")
        p_s_h1 = rede.inferencia_por_enumeracao("S", {"H": 1})
        p_s_h0 = rede.inferencia_por_enumeracao("S", {"H": 0})
        np.testing.assert_allclose(p_s, p_s_h1, atol=1e-12)
        np.testing.assert_allclose(p_s, p_s_h0, atol=1e-12)

        # Observando o colisor F=1, o caminho abre: P(S=1 | F=1, H=0) != P(S=1 | F=1, H=1)
        p_s_f1_h0 = rede.inferencia_por_enumeracao("S", {"F": 1, "H": 0})
        p_s_f1_h1 = rede.inferencia_por_enumeracao("S", {"F": 1, "H": 1})
        self.assertNotAlmostEqual(p_s_f1_h0[1], p_s_f1_h1[1], places=3)

        # Compara todas as consultas com VariableElimination do pgmpy
        ve = VariableElimination(rede.para_pgmpy())
        consultas = [
            ("F", {}),
            ("F", {"S": 1}),
            ("F", {"M": 1, "R": 0}),
            ("F", {"M": 1, "R": 0, "S": 1, "H": 1}),
            ("S", {"F": 1}),
            ("S", {"F": 1, "H": 0}),
            ("M", {"F": 1, "R": 0}),
            ("R", {"S": 1, "F": 1}),
        ]
        for alvo, ev in consultas:
            q_proprio = rede.inferencia_por_enumeracao(alvo, ev)
            q_pgmpy = ve.query(variables=[alvo], evidence=ev, show_progress=False).values
            np.testing.assert_allclose(q_proprio, q_pgmpy, atol=1e-12)

    def test_dados_reais_fold0_igualdade_pgmpy(self):
        dados = carregar_dados(2024)
        _, treino, validacao = next(iterar_folds(dados, "espacial"))
        particao = preparar_particao(treino, validacao, "numerica")
        med = particao.parametros["medianas"]
        vars_tr = extrair_variaveis_binarias(particao.X_treino, particao.y_cls_treino, med)
        vars_va = extrair_variaveis_binarias(particao.X_avaliacao, particao.y_cls_avaliacao, med)

        rede = RedeBayesianaBinaria().fit(vars_tr, alpha=0.0)
        ve = VariableElimination(rede.para_pgmpy())

        for alvo, ev in [
            ("F", {}),
            ("F", {"S": 1}),
            ("F", {"S": 1, "H": 0}),
            ("F", {"M": 1, "R": 0, "S": 1, "H": 0}),
            ("S", {"F": 1}),
            ("S", {"F": 1, "H": 0}),
            ("S", {"F": 1, "H": 1}),
            ("M", {"F": 1}),
            ("M", {"F": 1, "S": 1}),
        ]:
            q_enum = rede.inferencia_por_enumeracao(alvo, ev)
            q_ve = ve.query(variables=[alvo], evidence=ev, show_progress=False).values
            np.testing.assert_allclose(q_enum, q_ve, atol=1e-12)

        probs_va = rede.predict_proba(vars_va)
        self.assertEqual(probs_va.shape, (len(validacao), 2))
        np.testing.assert_allclose(probs_va.sum(axis=1), 1.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
