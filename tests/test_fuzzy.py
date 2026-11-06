"""Testes unitários da Etapa 6: funções triangulares, sistema Mamdani e comparação com scikit-fuzzy."""

import unittest
import warnings

import numpy as np

from fuzzy import SistemaMamdani, VariavelFuzzy, triangular


class TestTriangular(unittest.TestCase):
    def test_pico_bordas_e_fora_do_suporte(self):
        self.assertAlmostEqual(triangular(0.5, 0.0, 0.5, 1.0), 1.0)
        self.assertAlmostEqual(triangular(0.0, 0.0, 0.5, 1.0), 0.0)
        self.assertAlmostEqual(triangular(1.0, 0.0, 0.5, 1.0), 0.0)
        self.assertAlmostEqual(triangular(-0.1, 0.0, 0.5, 1.0), 0.0)
        self.assertAlmostEqual(triangular(1.1, 0.0, 0.5, 1.0), 0.0)

    def test_interpolacao_linear(self):
        self.assertAlmostEqual(triangular(0.25, 0.0, 0.5, 1.0), 0.5)
        self.assertAlmostEqual(triangular(0.75, 0.0, 0.5, 1.0), 0.5)

    def test_triangulos_degenerados(self):
        self.assertAlmostEqual(triangular(0.0, 0.0, 0.0, 0.5), 1.0)
        self.assertAlmostEqual(triangular(0.25, 0.0, 0.0, 0.5), 0.5)
        self.assertAlmostEqual(triangular(1.0, 0.5, 1.0, 1.0), 1.0)
        self.assertAlmostEqual(triangular(0.75, 0.5, 1.0, 1.0), 0.5)

    def test_entrada_vetorial(self):
        x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        np.testing.assert_allclose(triangular(x, 0.0, 0.5, 1.0),
                                   [0.0, 0.5, 1.0, 0.5, 0.0])

    def test_parametros_fora_de_ordem(self):
        with self.assertRaises(ValueError):
            triangular(0.5, 1.0, 0.5, 0.0)
        with self.assertRaises(ValueError):
            triangular(0.5, 0.0, 1.0, 0.5)


class TestVariavelFuzzy(unittest.TestCase):
    def setUp(self):
        self.universo = np.linspace(0, 1, 101)
        self.risco = VariavelFuzzy("risco", self.universo, {
            "baixo": (0.0, 0.0, 0.4),
            "alto": (0.6, 1.0, 1.0),
        })

    def test_pertinencia_por_termo(self):
        self.assertAlmostEqual(self.risco.pertinencia(0.2, "baixo"), 0.5)
        self.assertAlmostEqual(self.risco.pertinencia(0.8, "alto"), 0.5)
        self.assertAlmostEqual(self.risco.pertinencia(0.5, "baixo"), 0.0)

    def test_pertinencia_sem_termo_devolve_dict(self):
        resultado = self.risco.pertinencia(0.2)
        self.assertIsInstance(resultado, dict)
        self.assertEqual(set(resultado), {"baixo", "alto"})
        self.assertAlmostEqual(resultado["baixo"], 0.5)
        self.assertAlmostEqual(resultado["alto"], 0.0)

    def test_termo_inexistente(self):
        with self.assertRaises(KeyError):
            self.risco.pertinencia(0.5, "medio")

    def test_universo_invalido(self):
        with self.assertRaises(ValueError):
            VariavelFuzzy("x", np.array([]), {"a": (0, 0, 1)})
        with self.assertRaises(ValueError):
            VariavelFuzzy("x", np.array([0.5]), {"a": (0, 0, 1)})

    def test_termo_com_numero_errado_de_parametros(self):
        with self.assertRaises(ValueError):
            VariavelFuzzy("x", self.universo, {"a": (0, 0)})
        with self.assertRaises(ValueError):
            VariavelFuzzy("x", self.universo, {"a": (0, 0, 1, 1)})


class TestSistemaMamdaniUmaEntrada(unittest.TestCase):
    """Reproduz o exemplo da seção 9.3 do PDF com uma variável de entrada."""

    def setUp(self):
        universo = np.linspace(0, 1, 101)
        self.risco = VariavelFuzzy("risco", universo, {
            "baixo": (0.0, 0.0, 0.4),
            "alto": (0.6, 1.0, 1.0),
        })
        self.urgencia = VariavelFuzzy("urgencia", np.linspace(0, 100, 201), {
            "baixa": (0.0, 0.0, 50.0),
            "alta": (50.0, 100.0, 100.0),
        })
        self.sistema = SistemaMamdani(
            [self.risco], self.urgencia,
            [([("risco", "alto")], "alta"), ([("risco", "baixo")], "baixa")])

    def test_exemplo_do_pdf_risco_alto(self):
        u, _ = self.sistema.avaliar({"risco": 0.75})
        self.assertGreater(u, 50)
        self.assertLess(u, 100)

    def test_risco_baixo_produz_urgencia_baixa(self):
        u, _ = self.sistema.avaliar({"risco": 0.2})
        self.assertLess(u, 50)

    def test_resultado_dentro_do_universo(self):
        for valor in (0.0, 0.3, 0.5, 0.7, 1.0):
            u, _ = self.sistema.avaliar({"risco": valor})
            self.assertGreaterEqual(u, 0.0)
            self.assertLessEqual(u, 100.0)

    def test_entrada_faltante(self):
        with self.assertRaises(ValueError):
            self.sistema.avaliar({})

    def test_termo_consequente_invalido(self):
        with self.assertRaises(ValueError):
            SistemaMamdani([self.risco], self.urgencia,
                           [([("risco", "alto")], "media")])

    def test_variavel_antecedente_invalida(self):
        with self.assertRaises(ValueError):
            SistemaMamdani([self.risco], self.urgencia,
                           [([("outra", "alto")], "alta")])

    def test_termo_antecedente_invalido(self):
        with self.assertRaises(ValueError):
            SistemaMamdani([self.risco], self.urgencia,
                           [([("risco", "medio")], "alta")])


class TestSistemaMamdaniDuasEntradas(unittest.TestCase):
    """Sistema com duas entradas, operador E (mínimo) e agregação por máximo."""

    def setUp(self):
        universo = np.linspace(0, 1, 101)
        self.risco = VariavelFuzzy("risco", universo, {
            "baixo": (0.0, 0.0, 0.3),
            "alto": (0.7, 1.0, 1.0),
        })
        self.seca = VariavelFuzzy("seca", universo, {
            "baixa": (0.0, 0.0, 0.3),
            "alta": (0.7, 1.0, 1.0),
        })
        self.urgencia = VariavelFuzzy("urgencia", np.linspace(0, 100, 201), {
            "baixa": (0.0, 0.0, 50.0),
            "alta": (50.0, 100.0, 100.0),
        })
        self.sistema = SistemaMamdani(
            [self.risco, self.seca], self.urgencia,
            [([("risco", "alto"), ("seca", "alta")], "alta"),
             ([("risco", "baixo")], "baixa"),
             ([("seca", "baixa")], "baixa")])

    def test_ambos_altos_produz_urgencia_alta(self):
        u, _ = self.sistema.avaliar({"risco": 0.9, "seca": 0.9})
        self.assertGreater(u, 50)

    def test_operador_e_usa_minimo(self):
        u, _ = self.sistema.avaliar({"risco": 1.0, "seca": 0.0})
        self.assertLess(u, 50)

    def test_ativacoes_registradas(self):
        _, ativacoes = self.sistema.avaliar({"risco": 0.9, "seca": 0.9})
        self.assertEqual(len(ativacoes), len(self.sistema.regras))
        for forca in ativacoes.values():
            self.assertGreaterEqual(forca, 0.0)
            self.assertLessEqual(forca, 1.0)

    def test_avaliar_lote_consistente_com_avaliar(self):
        valores = {"risco": np.array([0.9, 0.1, 0.5]),
                   "seca": np.array([0.9, 0.1, 0.5])}
        lote = self.sistema.avaliar_lote(valores)
        for i in range(3):
            individual, _ = self.sistema.avaliar({
                "risco": valores["risco"][i],
                "seca": valores["seca"][i]})
            self.assertAlmostEqual(lote[i], individual, places=10)

    def test_avaliar_lote_em_zona_morta_nao_emite_aviso(self):
        """Sem regra ativada, a implementação própria devolve o centro do universo sem warning."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            resultado = self.sistema.avaliar_lote(
                {"risco": np.array([0.5]), "seca": np.array([0.5])})
        self.assertAlmostEqual(resultado[0], self.urgencia.universo.mean(), places=6)

    def test_avaliar_lote_comprimentos_incompativeis(self):
        with self.assertRaises(ValueError):
            self.sistema.avaliar_lote({"risco": np.array([0.5, 0.5]),
                                        "seca": np.array([0.5])})


class TestEquivalenciaScikitFuzzy(unittest.TestCase):
    """Compara a implementação própria com scikit-fuzzy em pontos em que ao menos uma regra dispara.

    A comparação é limitada a essas entradas porque o scikit-fuzzy não popula
    ``sim.output`` quando todas as forças de ativação são zero. A implementação
    própria devolve o centro do universo como fallback nesse caso, o que não
    tem contraparte no scikit-fuzzy. As duas implementações usam grades de
    saída com resoluções diferentes (201 pontos no universo da variável de saída
    vs. 1001 pontos internos do SistemaMamdani), de modo que o centroide pode
    diferir ligeiramente; a tolerância de 0,5 cobre essa diferença de
    discretização sem mascarar erros reais na fuzzificação ou na inferência.
    """

    def setUp(self):
        universo = np.linspace(0, 1, 101)
        self.risco = VariavelFuzzy("risco", universo, {
            "baixo": (0.0, 0.0, 0.3),
            "medio": (0.2, 0.5, 0.8),
            "alto": (0.7, 1.0, 1.0),
        })
        self.seca = VariavelFuzzy("seca", universo, {
            "baixa": (0.0, 0.0, 0.3),
            "alta": (0.7, 1.0, 1.0),
        })
        self.urgencia = VariavelFuzzy("urgencia", np.linspace(0, 100, 201), {
            "baixa": (0.0, 0.0, 50.0),
            "alta": (50.0, 100.0, 100.0),
        })
        self.sistema = SistemaMamdani(
            [self.risco, self.seca], self.urgencia,
            [([("risco", "alto"), ("seca", "alta")], "alta"),
             ([("risco", "baixo")], "baixa")])

    def test_equivalencia_em_grade_aleatoria(self):
        sim = self.sistema.construir_scikitfuzzy()
        gerador = np.random.default_rng(42)
        for _ in range(20):
            if gerador.random() < 0.5:
                # risco baixo garante a ativação da regra 2
                pr = gerador.uniform(0.0, 0.25)
                ps = gerador.uniform(0.0, 1.0)
            else:
                # ambos altos garantem a ativação da regra 1
                pr = gerador.uniform(0.75, 1.0)
                ps = gerador.uniform(0.75, 1.0)
            sim.input["risco"] = pr
            sim.input["seca"] = ps
            sim.compute()
            u_skf = sim.output["urgencia"]
            u_propria, _ = self.sistema.avaliar({"risco": pr, "seca": ps})
            self.assertAlmostEqual(u_skf, u_propria, delta=0.5,
                                   msg=f"divergência em risco={pr:.3f}, seca={ps:.3f}")


if __name__ == "__main__":
    unittest.main()
