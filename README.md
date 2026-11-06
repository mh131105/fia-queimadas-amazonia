# Fundamentos de IA: focos de fogo no bioma Amazônia

Trabalho integrado **Do Simbólico à Incerteza**, Prof. Edjard Mota, especificação de 15/10/2026.

O projeto estuda a potência radiativa (FRP) de focos **já detectados** no bioma Amazônia brasileiro. A fonte principal é o Programa Queimadas do INPE, a partir do [catálogo indicado](https://dados.gov.br/dados/conjuntos-dados/monitoramento-de-queimadas).

## Estado atual

**Dataset auditado e etapas 1 a 5 implementadas e verificadas.** Os notebooks contêm as implementações próprias, comparações com bibliotecas de referência (`scikit-learn` e `pgmpy`), testes sintéticos, resultados e discussão. Os 33 testes unitários automatizados do repositório (`dataset`, `probabilidade`, `etapa4` e `rede_bayesiana`) passam integralmente.

A etapa 6 (Lógica Fuzzy), a integração final do pipeline, a avaliação externa em 2025, o relatório LaTeX de 6–8 páginas e a apresentação ainda estão pendentes. Os cálculos e a discussão exigidos para o relatório nas etapas 1 a 5 estão preparados nos notebooks; o relatório será escrito ao final.

Recorte metodológico: bioma Amazônia no Brasil, sensor AQUA_M-T, desenvolvimento em 2024 e teste em 2025. A regressão estima FRP em MW. A classificação destaca FRP igual ou superior ao percentil 75 do treino corrente, com limiar congelado para sua validação. Na avaliação final, esse limiar deverá ser aprendido em 2024 e aplicado a 2025. O rótulo é estatístico, sem equivalência automática com gravidade ambiental.

## Implementações disponíveis

| Arquivo | Conteúdo |
|---|---|
| [arvore.py](arvore.py) | Entropia, ganho de informação, aprendizagem ID3 e previsão, implementados com NumPy |
| [etapa1.ipynb](etapa1.ipynb) | Comparação com `DecisionTreeClassifier`, cinco folds, contas dos três primeiros níveis e discussão de monotonicidade |
| [linear.py](linear.py) | Regressão univariada por fórmula fechada, multivariada/Ridge por gradiente e logística com sigmoide estável |
| [etapa2.ipynb](etapa2.ipynb) | Comparações com `LinearRegression`, `Ridge` e `LogisticRegression`, convergência e curva de coeficientes × lambda |
| [probabilidade.py](probabilidade.py) | Conjunta de variáveis binárias, marginais, condicionais, regra da cadeia, Bayes, independência condicional e bootstrap por município, com NumPy |
| [etapa3.ipynb](etapa3.ipynb) | Conjunta FRP elevada × período longo sem chuva, Bayes causal/diagnóstico, regra da cadeia com 3 variáveis, independência condicional dada a classe, estabilidade nos 5 folds e sensibilidade |
| [aprendizado_probabilistico.py](aprendizado_probabilistico.py) e [em.py](em.py) | Estimadores MLE e MAP (Beta-Bernoulli), Bernoulli Naive Bayes do zero e Mistura Gaussiana Diagonal via EM por classe |
| [etapa4.ipynb](etapa4.ipynb) | MLE vs. MAP, Bernoulli Naive Bayes, EM com variáveis latentes ($K=2, 3$), curvas de log-verossimilhança e comparação nos 5 folds espaciais |
| [rede_bayesiana.py](rede_bayesiana.py) | DAG sobre variáveis binárias, estimação de CPTs, d-separação por análise de caminhos, economia de parâmetros, inferência exata por enumeração (`ENUMERATION-ASK`) e exportação para `pgmpy` |
| [etapa5.ipynb](etapa5.ipynb) | DAG de 5 nós ($M, R, S, H, F$), CPTs em 2024, inferência por enumeração vs. `VariableElimination` (`pgmpy`), d-separação, demonstração de colisor ($S \to F \leftarrow H$), avaliação nos 5 folds espaciais e saídas para a Etapa 6 |

Os algoritmos próprios ficam nos módulos; os ajustes das bibliotecas ficam nos notebooks. A comparação adicional com `Ridge` e os diagnósticos temporais são escolhas do projeto. A especificação exige explicitamente as comparações com a árvore, a regressão linear e a logística do scikit-learn.

## Resultados revisados de 2024

Resultados da versão [`a333fbb`](https://github.com/mh131105/fia-queimadas-amazonia/commit/a333fbb5bc577ba50e947067c12ecd749a3aba33), confirmados em 31/10/2026. As tabelas mostram **média ± desvio-padrão amostral entre cinco folds espaciais**, com o mesmo peso por fold. Esses desvios não são intervalos de confiança. As tabelas completas por divisão estão nos notebooks.

**Regressão — FRP em MW:**

| Modelo | R² | MAE (MW) | RMSE (MW) |
|---|---:|---:|---:|
| Média da FRP do treino | −0,001120 ± 0,001496 | 61,503819 ± 2,951083 | 151,839332 ± 17,511955 |
| Univariada própria / scikit-learn | 0,011190 ± 0,008941 | 60,288176 ± 3,003300 | 150,928709 ± 17,725127 |
| Multivariada própria / scikit-learn | 0,021070 ± 0,007485 | 60,265830 ± 3,488068 | 150,171216 ± 17,582647 |
| Ridge própria, lambda = 0,01 | 0,021096 ± 0,007646 | 60,184801 ± 3,439357 | 150,170564 ± 17,600487 |

O R² é baixo e a melhora sobre a referência constante é pequena. Lambda = 0,01 foi escolhido entre seis configurações pela própria validação espacial de 2024; seu resultado não é uma avaliação independente após a seleção. O ganho de RMSE sobre a multivariada é de aproximadamente 0,00065 MW, sem evidência de melhora relevante. As versões próprias e de biblioteca reproduzem as métricas exibidas; a maior diferença de coeficientes da multivariada/Ridge nos folds espaciais foi 4,921 × 10⁻⁷, com `alpha = n_treino × lambda` na referência Ridge.

**Classificação — classe 1: FRP ≥ Q75 do treino; métricas entre 0 e 1:**

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---:|---:|---:|---:|
| Maioria do treino | 0,749825 ± 0,014465 | 0 ± 0 | 0 ± 0 | 0 ± 0 |
| ID3 próprio / árvore scikit-learn | 0,749312 ± 0,014511 | 0,365120 ± 0,210052 | 0,012801 ± 0,011364 | 0,024509 ± 0,021351 |
| Logística própria / scikit-learn | 0,756409 ± 0,015766 | 0,677025 ± 0,065360 | 0,048038 ± 0,019260 | 0,089201 ± 0,034251 |

Precisão, recall e F1 se referem à classe 1. A acurácia próxima de 75% precisa ser lida junto à maioria: as árvores não a superam e recuperam cerca de **1,28%** dos positivos; a logística recupera **4,80%**, com limiar de probabilidade 0,5. As versões próprias e de biblioteca tiveram as mesmas métricas em cada fold espacial. A maior diferença entre probabilidades logísticas foi 1,063 × 10⁻⁶. A concordância numérica verifica a implementação, mas o desempenho ainda limita sua utilidade preditiva.

Os **cinco diagnósticos temporais** usam treino crescente e validação no bimestre seguinte, dentro de 2024, e ficam separados da avaliação espacial. Neles, a multivariada teve R² médio −0,036788; a logística teve acurácia média 0,734486, frente a 0,738189 da maioria. O lambda do Ridge foi selecionado usando todos os folds espaciais de 2024, que incluem períodos posteriores aos primeiros treinos temporais: esse diagnóstico do Ridge não constitui uma avaliação prospectiva independente da escolha do parâmetro.

A discretização da árvore perde informação e torna precipitação constante no fold espacial 0. Dependência entre detecções, fronteiras municipais, ausências e características do sensor também limitam a generalização. As faixas de desempenho ilustradas no PDF não foram usadas como metas de ajuste. **2025 não foi usado para ajustar, selecionar ou avaliar modelos nestes notebooks.**

**Etapa 5 — Rede Bayesiana (`M, R, S, H, F`):**
- **Estrutura do DAG:** $M \to R$, $M \to S$, $R \to S$, $S \to F$ e $H \to F$, induzindo $P(M,R,S,H,F) = P(M)\,P(H)\,P(R\mid M)\,P(S\mid M,R)\,P(F\mid S,H)$.
- **Economia de parâmetros:** reduz de **31 parâmetros livres** (conjunta completa de 5 variáveis binárias) para **12 parâmetros livres** (economia de **19 parâmetros**, ou $61{,}3\%$).
- **Inferência por enumeração vs. `pgmpy`:** maior diferença absoluta entre a implementação própria (`ENUMERATION-ASK`) e `VariableElimination` do `pgmpy` foi $\le 1{,}11 \times 10^{-16}$.
- **d-separação e colisor:** verificadas $M \perp_d F \mid S$ e $R \perp_d H \mid S$; demonstrado o colisor $S \to F \leftarrow H$, no qual observar $F=1$ abre o caminho entre $S$ e $H$ (*explaining away*: $P(S=1\mid F=1, H=0) = 0{,}655837$ contra $P(S=1\mid F=1, H=1) = 0{,}634217$ no fold 0).
- **Fidelidade da conjunta nos 5 folds espaciais:** Distância de Variação Total (TVD) média na validação de $0{,}0969 \pm 0{,}0577$ na Rede Bayesiana contra $0{,}1768 \pm 0{,}0441$ no Bernoulli Naive Bayes.

## Documentação

| Documento | Conteúdo |
|---|---|
| [Guia do dataset](docs/dataset.md) | Explicação do dataset, limpeza, atributos, alvos, ambiente e exemplos de uso nos notebooks |
| [Dicionário](data/dicionario.csv) | Tipos, unidades, origem e papel das 27 colunas |
| [Auditoria completa](docs/auditoria-dataset.json) | Cobertura, filtragem, ausências e cinco folds espaciais/temporais |
| [Fontes de 2024–2025](data/fontes_2024_2025.json) e [produtos](data/dataset/manifesto.json) | URLs, hashes e contagens do dataset integral |
| [Análise inicial dos dados](docs/analise-dados.md) | Estudo anterior de viabilidade com cinco meses, preservado como referência |
| [Fontes](docs/fontes.md) | Fontes oficiais e distinção entre consultado e ainda não verificado |
| [Uso de IA](docs/uso-de-ia.md) | Registro de prompts e escolhas metodológicas para o apêndice exigido |
| [Prompt de implementação das etapas 1 e 2](docs/prompts/implementacao-etapas-1-2.md) | Guia integral preservado como prompt de referência para a implementação assistida por IA |

## Dados prontos para os notebooks

- [Desenvolvimento de 2024](data/dataset/desenvolvimento_2024.csv.gz): usar para implementar, comparar e selecionar modelos.
- [Teste de 2025](data/dataset/teste_2025.csv.gz): reservar para a avaliação final, após encerrar escolhas em 2024.

**140.346 registros em 2024** e **43.131 em 2025**, em dois CSVs comprimidos que somam aproximadamente 8,89 MB. Os dois arquivos derivados estão no GitHub. Eles contêm todas as detecções válidas do recorte nos 24 meses, sem amostragem. Os originais maiores ficam fora do Git e podem ser obtidos pelo manifesto. Esta versão utiliza somente o INPE, inclusive para os atributos meteorológicos; não houve associação com estações externas.

## Ambiente e início rápido

Ambiente validado: **Python 3.12.14**. Na raiz do projeto, com `uv`:

```sh
uv venv --python 3.12.14
uv pip sync --python .venv/bin/python requirements.txt
.venv/bin/python -m ipykernel install --prefix .venv --name fia --display-name "FIA (Python 3.12)"
.venv/bin/jupyter lab
```

Usar o kernel **FIA (Python 3.12)**. Para ler e preparar uma divisão, sem treinar modelos:

```python
from utils import carregar_dados, iterar_folds, preparar_particao

dados = carregar_dados(2024)
fold, treino, validacao = next(iterar_folds(dados, "espacial"))
particao = preparar_particao(treino, validacao, "padronizada")
X_treino = particao.X_treino
y_reg_treino = particao.y_reg_treino
y_cls_treino = particao.y_cls_treino
```

A imputação, a padronização e o Q75 são calculados somente no treino da divisão. Para a árvore didática, usar `"discretizada"`; para valores sem padronização, `"numerica"`. O [guia](docs/dataset.md) explica os alvos, as duas estratégias de validação e os limites de interpretação.

### Executar os notebooks

Abrir primeiro `etapa1.ipynb` e depois `etapa2.ipynb`. Em cada um, selecionar o kernel **FIA (Python 3.12)** e usar **Reiniciar kernel e executar todas as células**. As saídas versionadas incluem tabelas, testes e figuras. Ambos usam somente os dados de desenvolvimento de 2024.

Para reexecutar pelo terminal, na raiz do projeto, salvando cópias em uma pasta ignorada pelo Git:

```sh
mkdir -p tmp/reexecucao
.venv/bin/jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.kernel_name=fia \
  --ExecutePreprocessor.timeout=600 \
  --output-dir=tmp/reexecucao etapa1.ipynb etapa2.ipynb
```

Na revisão, passaram as verificações sintéticas de ID3, XOR, categorias inéditas, gradientes por diferenças finitas, estabilidade da sigmoide, convergência e regularização. Também foram conferidos os cinco folds espaciais e os cinco temporais de cada notebook, além das quatro figuras. Os arquivos de implementação e suas saídas originais foram preservados.

## Reconstruir e verificar

```sh
.venv/bin/python scripts/baixar_fontes.py --baixar
.venv/bin/python scripts/preparar_dados.py
.venv/bin/python -m unittest discover -s tests -v
```

O primeiro comando baixa somente os originais ausentes e confere os hashes dos 24 meses. O segundo reconstrói as tabelas e a auditoria, sem ajustar modelos. Os testes verificam qualidade, separações espaciais/temporais e ausência de vazamento na preparação. Detalhes em [data/README.md](data/README.md).

As instruções de compilação do artigo serão adicionadas quando o relatório existir.

## Histórico e autoria

Autoria: **Matheus Garcia**. O histórico registra contribuições incrementais com `Signed-off-by`.
