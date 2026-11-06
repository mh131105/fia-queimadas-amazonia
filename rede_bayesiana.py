"""Etapa 5: Redes Bayesianas sobre variáveis binárias, implementadas com NumPy.

Inclui validação do DAG (acíclico e ordenação topológica), estimação de CPTs a partir
de contagens empíricas (com opção de suavização de Laplace), verificação exata de
d-separação por análise de caminhos (cadeias, bifurcações e colisores), cálculo de
economia de parâmetros e inferência exata por enumeração (ENUMERATION-ASK).
"""

from collections import deque
from itertools import product

import numpy as np
import pandas as pd

from probabilidade import binarizar, contagens_conjuntas


NOS_PADRAO = ("M", "R", "S", "H", "F")
ARESTAS_PADRAO = (
    ("M", "R"),
    ("M", "S"),
    ("R", "S"),
    ("S", "F"),
    ("H", "F"),
)


def extrair_variaveis_binarias(X, y_cls, medianas):
    """Constrói o dicionário das 5 variáveis binárias do domínio a partir dos cortes do treino.

    M: mes >= mediana(mes)
    R: precipitacao > mediana(precipitacao)  [estrito, pois a mediana é 0 mm]
    S: numero_dias_sem_chuva >= mediana(numero_dias_sem_chuva)
    H: hora_utc >= mediana(hora_utc)
    F: frp >= Q75(frp) (já fornecido em y_cls)
    """
    f = np.asarray(y_cls, dtype=int)
    if f.ndim != 1 or f.size == 0 or not np.isin(f, (0, 1)).all():
        raise ValueError("y_cls deve ser um vetor binário (0 ou 1) não vazio.")
    return {
        "M": binarizar(X["mes"], medianas["mes"]).astype(int),
        "R": binarizar(X["precipitacao"], medianas["precipitacao"], estrito=True).astype(int),
        "S": binarizar(X["numero_dias_sem_chuva"], medianas["numero_dias_sem_chuva"]).astype(int),
        "H": binarizar(X["hora_utc"], medianas["hora_utc"]).astype(int),
        "F": f,
    }


def _validar_dados_binarios(dados, nos):
    if isinstance(dados, pd.DataFrame):
        faltantes = [n for n in nos if n not in dados.columns]
        if faltantes:
            raise ValueError(f"Variáveis ausentes em dados: {faltantes}")
        colunas = {n: np.asarray(dados[n], dtype=int) for n in nos}
    elif isinstance(dados, dict):
        faltantes = [n for n in nos if n not in dados]
        if faltantes:
            raise ValueError(f"Variáveis ausentes em dados: {faltantes}")
        colunas = {n: np.asarray(dados[n], dtype=int) for n in nos}
    else:
        raise ValueError("dados deve ser um dict ou DataFrame mapeando cada nó ao seu vetor binário.")

    comprimentos = {len(v) for v in colunas.values()}
    if len(comprimentos) != 1 or next(iter(comprimentos)) == 0:
        raise ValueError("Todas as variáveis devem ter o mesmo comprimento não nulo.")
    for nome, vetor in colunas.items():
        if vetor.ndim != 1 or not np.isin(vetor, (0, 1)).all():
            raise ValueError(f"A variável '{nome}' deve ser um vetor 1D binário (0 ou 1).")
    return colunas


class RedeBayesianaBinaria:
    """Rede Bayesiana discreta sobre variáveis binárias (valores 0 e 1).

    Para um nó X com k pais (P_1, ..., P_k), a CPT armazena um array NumPy de formato
    ``(2,) * (k + 1)``, onde o eixo 0 corresponde aos valores de X (0 ou 1) e os eixos
    seguintes correspondem aos pais na ordem declarada em ``self.pais[X]``.
    """

    def __init__(self, nos=NOS_PADRAO, arestas=ARESTAS_PADRAO):
        nos = tuple(nos)
        if not nos or len(set(nos)) != len(nos):
            raise ValueError("A lista de nós deve ser não vazia e sem duplicatas.")
        self.nos = nos
        self.arestas = tuple((u, v) for u, v in arestas)
        if len(set(self.arestas)) != len(self.arestas):
            raise ValueError("Há arestas duplicadas.")

        conjunto_nos = set(self.nos)
        self.pais = {n: [] for n in self.nos}
        self.filhos = {n: [] for n in self.nos}
        for u, v in self.arestas:
            if u not in conjunto_nos or v not in conjunto_nos:
                raise ValueError(f"Aresta ({u!r}, {v!r}) referencia nó inexistente.")
            if u == v:
                raise ValueError(f"Auto-laço não permitido no nó {u!r}.")
            self.pais[v].append(u)
            self.filhos[u].append(v)

        self.pais = {n: tuple(ps) for n, ps in self.pais.items()}
        self.filhos = {n: tuple(fs) for n, fs in self.filhos.items()}
        self.ordem_topologica = self._ordenar_topologicamente()
        self.ancestrais = {n: self._calcular_ancestrais(n) for n in self.nos}
        self.descendentes = {n: self._calcular_descendentes(n) for n in self.nos}
        self.cpts_ = {}
        self.contagens_ = {}
        self.n_treino_ = 0
        self.alpha_ = 0.0

    def _ordenar_topologicamente(self):
        grau_entrada = {n: len(self.pais[n]) for n in self.nos}
        fila = deque([n for n in self.nos if grau_entrada[n] == 0])
        ordem = []
        while fila:
            u = fila.popleft()
            ordem.append(u)
            for v in self.filhos[u]:
                grau_entrada[v] -= 1
                if grau_entrada[v] == 0:
                    fila.append(v)
        if len(ordem) != len(self.nos):
            raise ValueError("O grafo fornecido contém ciclo e não é um DAG válido.")
        return tuple(ordem)

    def _calcular_ancestrais(self, no):
        visitados = set()
        pilha = list(self.pais[no])
        while pilha:
            atual = pilha.pop()
            if atual not in visitados:
                visitados.add(atual)
                pilha.extend(self.pais[atual])
        return frozenset(visitados)

    def _calcular_descendentes(self, no):
        visitados = set()
        pilha = list(self.filhos[no])
        while pilha:
            atual = pilha.pop()
            if atual not in visitados:
                visitados.add(atual)
                pilha.extend(self.filhos[atual])
        return frozenset(visitados)

    def fit(self, dados, alpha=0.0):
        """Estima as CPTs P(X_i | Pa(X_i)) a partir dos dados binários de treino.

        ``alpha`` é a suavização aditiva de Laplace por célula (0.0 = frequência empírica pura).
        Se uma configuração de pais tiver contagem total zero com alpha=0, utiliza uniforme (0.5, 0.5).
        """
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError("alpha deve ser um número finito não negativo.")
        colunas = _validar_dados_binarios(dados, self.nos)
        self.n_treino_ = len(next(iter(colunas.values())))
        self.alpha_ = float(alpha)
        self.cpts_ = {}
        self.contagens_ = {}

        for no in self.ordem_topologica:
            pais_no = self.pais[no]
            variaveis = [colunas[no]] + [colunas[p] for p in pais_no]
            cont = contagens_conjuntas(*variaveis)
            self.contagens_[no] = cont
            numerador = cont.astype(float) + self.alpha_
            denominador = numerador.sum(axis=0, keepdims=True)
            with np.errstate(divide="ignore", invalid="ignore"):
                cpt = np.where(denominador > 0, numerador / denominador, 0.5)
            self.cpts_[no] = cpt
        return self

    def definir_cpts(self, cpts):
        """Permite fixar CPTs manualmente (útil em testes sintéticos de resposta exata)."""
        novas = {}
        for no in self.ordem_topologica:
            if no not in cpts:
                raise ValueError(f"CPT ausente para o nó {no!r}.")
            arr = np.asarray(cpts[no], dtype=float)
            formato_esperado = (2,) * (1 + len(self.pais[no]))
            if arr.shape != formato_esperado:
                raise ValueError(f"CPT de {no!r} deve ter formato {formato_esperado}, recebeu {arr.shape}.")
            if np.any(arr < 0) or not np.allclose(arr.sum(axis=0), 1.0, atol=1e-9):
                raise ValueError(f"CPT de {no!r} deve ter probabilidades não negativas somando 1 no eixo 0.")
            novas[no] = arr.copy()
        self.cpts_ = novas
        return self

    def _verificar_ajuste(self):
        if not self.cpts_:
            raise ValueError("As CPTs ainda não foram estimadas. Chame fit() ou definir_cpts() antes.")

    def prob_condicional(self, no, valor, evidencia):
        """Devolve P(no = valor | Pa(no)) a partir da CPT armazenada e do dicionário ``evidencia``."""
        self._verificar_ajuste()
        if valor not in (0, 1):
            raise ValueError("O valor da variável binária deve ser 0 ou 1.")
        indice = (int(valor),) + tuple(int(evidencia[p]) for p in self.pais[no])
        return float(self.cpts_[no][indice])

    def tabela_cpt(self, no):
        """Retorna um DataFrame legível com a CPT do nó solicitado e as contagens de treino."""
        self._verificar_ajuste()
        if no not in self.cpts_:
            raise ValueError(f"Nó desconhecido: {no!r}.")
        pais_no = self.pais[no]
        cpt = self.cpts_[no]
        cont = self.contagens_.get(no)
        if not pais_no:
            df = pd.DataFrame({
                f"P({no}=0)": [cpt[0]],
                f"P({no}=1)": [cpt[1]],
            }, index=["marginal"])
            if cont is not None:
                df["n_treino"] = [int(cont.sum())]
            return df

        linhas = []
        for valores_pais in product((0, 1), repeat=len(pais_no)):
            item = {p: v for p, v in zip(pais_no, valores_pais)}
            item[f"P({no}=0 | pais)"] = float(cpt[(0,) + valores_pais])
            item[f"P({no}=1 | pais)"] = float(cpt[(1,) + valores_pais])
            if cont is not None:
                item["n_treino"] = int(cont[(slice(None),) + valores_pais].sum())
            linhas.append(item)
        return pd.DataFrame(linhas).set_index(list(pais_no))

    def inferencia_por_enumeracao(self, var_consulta, evidencia=None):
        """Implementa ENUMERATION-ASK(X, e, bn) de Russell & Norvig do zero.

        Retorna um array NumPy ``[P(X=0 | e), P(X=1 | e)]`` somando 1.
        """
        self._verificar_ajuste()
        if var_consulta not in self.pais:
            raise ValueError(f"Variável de consulta desconhecida: {var_consulta!r}.")
        evidencia = {} if evidencia is None else dict(evidencia)
        for k, v in evidencia.items():
            if k not in self.pais:
                raise ValueError(f"Variável de evidência desconhecida: {k!r}.")
            if v not in (0, 1):
                raise ValueError(f"Valor de evidência para {k!r} deve ser 0 ou 1.")

        if var_consulta in evidencia:
            q = np.zeros(2, dtype=float)
            q[int(evidencia[var_consulta])] = 1.0
            return q

        q = np.zeros(2, dtype=float)
        for x_val in (0, 1):
            ev_estendida = dict(evidencia)
            ev_estendida[var_consulta] = x_val
            q[x_val] = self._enumerate_all(self.ordem_topologica, ev_estendida)

        soma = q.sum()
        if soma <= 0:
            raise ValueError("A evidência fornecida tem probabilidade zero na rede.")
        return q / soma

    def _enumerate_all(self, variaveis, evidencia):
        """Implementa ENUMERATE-ALL(vars, e) recursivamente na ordem topológica."""
        if not variaveis:
            return 1.0
        primeiro = variaveis[0]
        restante = variaveis[1:]
        if primeiro in evidencia:
            p = self.prob_condicional(primeiro, evidencia[primeiro], evidencia)
            return p * self._enumerate_all(restante, evidencia)
        soma = 0.0
        for y_val in (0, 1):
            ev_y = dict(evidencia)
            ev_y[primeiro] = y_val
            p = self.prob_condicional(primeiro, y_val, ev_y)
            soma += p * self._enumerate_all(restante, ev_y)
        return soma

    def distribuicao_conjunta_rede(self):
        """Reconstrói a conjunta completa P(X_1, ..., X_n) induzida pela fatoração da rede.

        Os eixos do array retornado ``(2,) * n`` seguem a ordem de ``self.nos``.
        """
        self._verificar_ajuste()
        conj = np.zeros((2,) * len(self.nos), dtype=float)
        for valores in product((0, 1), repeat=len(self.nos)):
            atribuicao = dict(zip(self.nos, valores))
            prod = 1.0
            for no in self.ordem_topologica:
                prod *= self.prob_condicional(no, atribuicao[no], atribuicao)
            conj[valores] = prod
        return conj

    def predict_proba(self, dados_evidencia, alvo="F"):
        """Calcula [P(alvo=0 | e), P(alvo=1 | e)] para cada linha de evidência.

        Como as variáveis de evidência são binárias, calcula cada configuração única uma só vez
        via ``inferencia_por_enumeracao`` e mapeia para as N linhas.
        """
        self._verificar_ajuste()
        vars_ev = [n for n in self.nos if n != alvo]
        colunas = _validar_dados_binarios(dados_evidencia, vars_ev)
        matriz = np.column_stack([colunas[n] for n in vars_ev])
        configuracoes_unicas, inverso = np.unique(matriz, axis=0, return_inverse=True)
        probs_unicas = np.zeros((len(configuracoes_unicas), 2), dtype=float)
        for idx, config in enumerate(configuracoes_unicas):
            ev = {n: int(val) for n, val in zip(vars_ev, config)}
            probs_unicas[idx] = self.inferencia_por_enumeracao(alvo, ev)
        return probs_unicas[inverso]

    def predict(self, dados_evidencia, alvo="F", limiar=0.5):
        probs = self.predict_proba(dados_evidencia, alvo=alvo)[:, 1]
        return (probs >= limiar).astype(int)

    def economia_parametros(self):
        """Calcula o número de parâmetros livres da rede vs. conjunta completa (2^n - 1)."""
        n = len(self.nos)
        conjunta_completa = (2 ** n) - 1
        por_no = {no: 2 ** len(self.pais[no]) for no in self.nos}
        total_rede = sum(por_no.values())
        economia = conjunta_completa - total_rede
        return {
            "n_variaveis": n,
            "parametros_conjunta": conjunta_completa,
            "parametros_por_no": por_no,
            "parametros_rede": total_rede,
            "economia_absoluta": economia,
            "economia_relativa": economia / conjunta_completa,
        }

    def caminhos_simples(self, origem, destino):
        """Lista todos os caminhos simples não direcionados entre ``origem`` e ``destino``."""
        if origem not in self.pais or destino not in self.pais:
            raise ValueError("Origem e destino devem pertencer aos nós da rede.")
        if origem == destino:
            return [[origem]]
        adjacencia = {n: set(self.pais[n]) | set(self.filhos[n]) for n in self.nos}
        caminhos = []

        def buscar(atual, visitados, caminho):
            if atual == destino:
                caminhos.append(list(caminho))
                return
            for vizinho in sorted(adjacencia[atual]):
                if vizinho not in visitados:
                    visitados.add(vizinho)
                    caminho.append(vizinho)
                    buscar(vizinho, visitados, caminho)
                    caminho.pop()
                    visitados.remove(vizinho)

        buscar(origem, {origem}, [origem])
        return caminhos

    def _tripla_ativa(self, a, b, c, observados):
        """Avalia se a tripla (a - b - c) ao longo de um caminho está ativa dado ``observados``.

        - Colisor (a -> b <- c): ativo se e somente se b ou algum descendente de b é observado.
        - Cadeia (a -> b -> c ou a <- b <- c) ou bifurcação (a <- b -> c): ativa se e somente se
          o nó central b NÃO é observado.
        """
        eh_colisor = (a in self.pais[b]) and (c in self.pais[b])
        if eh_colisor:
            abertura_colisor = {b} | set(self.descendentes[b])
            return bool(abertura_colisor & observados), "colisor"
        tipo = "bifurcacao" if (b in self.pais[a] and b in self.pais[c]) else "cadeia"
        return (b not in observados), tipo

    def analisar_caminhos_d_separacao(self, x, y, dado=()):
        """Inspeciona cada caminho simples entre ``x`` e ``y`` dado o conjunto ``dado``."""
        observados = set(dado)
        if x in observados or y in observados or x == y:
            raise ValueError("x e y devem ser distintos e disjuntos do conjunto condicionante.")
        invalidos = ({x, y} | observados) - set(self.nos)
        if invalidos:
            raise ValueError(f"Nós desconhecidos: {invalidos}")

        relatorio = []
        for caminho in self.caminhos_simples(x, y):
            ativo = True
            detalhes_triplas = []
            for i in range(1, len(caminho) - 1):
                a, b, c = caminho[i - 1], caminho[i], caminho[i + 1]
                tripla_ativa, tipo = self._tripla_ativa(a, b, c, observados)
                detalhes_triplas.append({
                    "tripla": (a, b, c),
                    "no_central": b,
                    "tipo": tipo,
                    "ativa": tripla_ativa,
                })
                if not tripla_ativa:
                    ativo = False
            relatorio.append({
                "caminho": tuple(caminho),
                "ativo": ativo,
                "triplas": detalhes_triplas,
            })
        return relatorio

    def d_separados(self, x, y, dado=()):
        """Retorna True se e somente se todos os caminhos entre ``x`` e ``y`` estão bloqueados por ``dado``."""
        relatorio = self.analisar_caminhos_d_separacao(x, y, dado=dado)
        return not any(item["ativo"] for item in relatorio)

    def para_pgmpy(self):
        """Constrói o modelo equivalente no pgmpy (DiscreteBayesianNetwork + TabularCPD)."""
        self._verificar_ajuste()
        try:
            from pgmpy.models import DiscreteBayesianNetwork as BNModel
        except ImportError:
            from pgmpy.models import BayesianNetwork as BNModel
        from pgmpy.factors.discrete import TabularCPD

        modelo = BNModel(list(self.arestas))
        for no in self.nos:
            if no not in modelo.nodes():
                modelo.add_node(no)

        cpds = []
        for no in self.ordem_topologica:
            pais_no = list(self.pais[no])
            cpt = self.cpts_[no]
            if not pais_no:
                valores = cpt.reshape(2, 1)
                cpd = TabularCPD(variable=no, variable_card=2, values=valores)
            else:
                valores = cpt.reshape(2, 2 ** len(pais_no))
                cpd = TabularCPD(
                    variable=no,
                    variable_card=2,
                    values=valores,
                    evidence=pais_no,
                    evidence_card=[2] * len(pais_no),
                )
            cpds.append(cpd)

        modelo.add_cpds(*cpds)
        if not modelo.check_model():
            raise ValueError("Falha na verificação de consistência do modelo pgmpy.")
        return modelo
