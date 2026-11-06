"""Etapa 6: Sistema de Inferência Fuzzy Mamdani, implementado do zero com NumPy.

Inclui funções de pertinência triangulares, fuzzificação, inferência (min-máx),
defuzzificação por centroide e uma ponte para scikit-fuzzy como referência.
"""

import numpy as np


def triangular(x, a, b, c):
    """Função de pertinência triangular com pico em b e suporte [a, c]."""
    if not (a <= b <= c):
        raise ValueError("É preciso a <= b <= c.")
    escalar = np.isscalar(x)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.zeros_like(x)
    if b > a:
        mask = (x >= a) & (x <= b)
        y[mask] = (x[mask] - a) / (b - a)
    else:
        y[x == a] = 1.0
    if c > b:
        mask = (x > b) & (x <= c)
        y[mask] = (c - x[mask]) / (c - b)
    else:
        y[x == b] = 1.0
    return float(y[0]) if escalar else y


class VariavelFuzzy:
    """Variável linguística com funções de pertinência triangulares."""

    def __init__(self, nome, universo, termos):
        self.nome = nome
        self.universo = np.asarray(universo, dtype=float)
        if self.universo.ndim != 1 or len(self.universo) < 2:
            raise ValueError("Universo deve ser um array 1D com ao menos 2 pontos.")
        self.termos = {}
        for termo, params in termos.items():
            if len(params) != 3:
                raise ValueError(f"Termo '{termo}' deve ter três parâmetros (a,b,c).")
            self.termos[termo] = tuple(float(v) for v in params)

    def pertinencia(self, x, termo=None):
        """Retorna a pertinência de x no termo (ou um dict com todos os termos)."""
        if termo is not None:
            return triangular(x, *self.termos[termo])
        return {t: triangular(x, *p) for t, p in self.termos.items()}


class SistemaMamdani:
    """Sistema Mamdani com regras SE-ENTÃO (E = mínimo), agregação máximo e centroide."""

    def __init__(self, entradas, saida, regras, resolucao=1001):
        self.entradas = {v.nome: v for v in entradas}
        self.saida = saida
        self.regras = []
        for antecedentes, consequente in regras:
            if consequente not in saida.termos:
                raise ValueError(f"Termo de saída desconhecido: {consequente}.")
            for var, termo in antecedentes:
                if var not in self.entradas:
                    raise ValueError(f"Variável de entrada desconhecida: {var}.")
                if termo not in self.entradas[var].termos:
                    raise ValueError(f"Termo '{termo}' desconhecido em '{var}'.")
            self.regras.append((tuple(antecedentes), consequente))
        self._grade_saida = np.linspace(saida.universo[0], saida.universo[-1], resolucao)

    def avaliar(self, valores):
        """Avalia o sistema para um dicionário de valores de entrada."""
        for nome in self.entradas:
            if nome not in valores:
                raise ValueError(f"Falta valor para a entrada '{nome}'.")
        agregado = np.zeros_like(self._grade_saida)
        ativacoes = {}
        for antecedentes, consequente in self.regras:
            forcas = []
            for var, termo in antecedentes:
                forcas.append(float(self.entradas[var].pertinencia(valores[var], termo)))
            forca = min(forcas) if forcas else 1.0
            ativacoes[(antecedentes, consequente)] = forca
            if forca > 0:
                mu_saida = triangular(self._grade_saida, *self.saida.termos[consequente])
                agregado = np.maximum(agregado, np.minimum(forca, mu_saida))
        soma = agregado.sum()
        if soma == 0:
            return float(self._grade_saida.mean()), ativacoes
        centroide = float((self._grade_saida * agregado).sum() / soma)
        return centroide, ativacoes
        
    def avaliar_lote(self, valores):
        nomes = list(self.entradas)
        n = len(valores[nomes[0]])
        for nome in nomes:
            if len(valores[nome]) != n:
                raise ValueError("Todos os arrays de entrada devem ter o mesmo comprimento.")
        agregado = np.zeros((n, len(self._grade_saida)))
        for antecedentes, consequente in self.regras:
            forcas = np.ones(n)
            for var, termo in antecedentes:
                forcas = np.minimum(forcas, self.entradas[var].pertinencia(valores[var], termo))
            mu_saida = triangular(self._grade_saida, *self.saida.termos[consequente])
            contrib = np.minimum(forcas[:, None], mu_saida[None, :])
            agregado = np.maximum(agregado, contrib)
        somas = agregado.sum(axis=1)
        somas_seguras = np.where(somas > 0, somas, 1.0)
        centroides = np.where(
            somas > 0,
            (agregado * self._grade_saida).sum(axis=1) / somas_seguras,
            self._grade_saida.mean(),
        )
        return centroides
        
    def construir_scikitfuzzy(self):
        """Constrói o ControlSystemSimulation equivalente usando scikit-fuzzy."""
        import skfuzzy as fuzz
        from skfuzzy import control as ctrl

        antecedentes_ctrl = {}
        for nome, var in self.entradas.items():
            univ = var.universo
            antecedentes_ctrl[nome] = ctrl.Antecedent(univ, nome)
            for termo, (a, b, c) in var.termos.items():
                antecedentes_ctrl[nome][termo] = fuzz.trimf(univ, [a, b, c])

        saida_ctrl = ctrl.Consequent(self.saida.universo, self.saida.nome)
        for termo, (a, b, c) in self.saida.termos.items():
            saida_ctrl[termo] = fuzz.trimf(self.saida.universo, [a, b, c])

        regras_ctrl = []
        for antecedentes, consequente in self.regras:
            antecedente = None
            for var, termo in antecedentes:
                clausula = antecedentes_ctrl[var][termo]
                antecedente = clausula if antecedente is None else antecedente & clausula
            regras_ctrl.append(ctrl.Rule(antecedente, saida_ctrl[consequente]))

        sistema = ctrl.ControlSystem(regras_ctrl)
        return ctrl.ControlSystemSimulation(sistema)
