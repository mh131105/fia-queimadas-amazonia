# Registro de uso de IA

Data: **31/10/2026**. Assistente utilizado nesta conversa: Codex. Este registro documenta as solicitações humanas, decisões metodológicas, o prompt de referência para implementação e a revisão dos resultados. As seções seguem a ordem das entregas; as descrições de fases anteriores são históricas.

A seção 13 da especificação do trabalho (p. 9) pede documentar os prompts no apêndice. Este arquivo e o [prompt integral das etapas 1 e 2](prompts/implementacao-etapas-1-2.md) fornecem o material para esse apêndice, a ser preparado com o relatório final.

## Solicitação original

```text
/goal # PAPEL

Você é um cientista de dados formado em harvard com pós graduação em engenharia florestal, com experiência em times que monitoram focos de queimada na região da amazonia.

# MISSÃO

Sua missão é desenvolver o que se pede no trabalho, vamos usar u conjunto de dados diferente do do link da especificação mas foi autorizdo pelo professor.

O resultado deve permitir entregar o repositório completo, com tudo o que se pede

# MATERIAL DE REFERÊNCIA

https://dados.gov.br/dados/conjuntos-dados/monitoramento-de-queimadas

Utilize esse material como referência para executar a tarefa, junto com o pdf anexado.

# INSTRUÇÕES

Execute a tarefa seguindo estas etapas:

1. Análise: Analise o conjunto de dados e idealize como ele pode ser usado no trabalho
2. Planejamento: leia atentamente todas as etapas do trabalho, crie um repositório no github para organizar a entra
3. Execução: Voce sepre deve executar uma etapa de cada vez
4. Validação: faça sua analise com base em dados, validação humana só na entrega final

# REGRAS

- Siga rigorosamente as instruções fornecidas.
- Priorize informações precisas, verificáveis e relevantes.
- Não invente informações, dados ou referências.
- Diferencie fatos confirmados de hipóteses e suposições.
- Utilize linguagem clara, objetiva e adequada ao público-alvo.
- Evite explicações redundantes e conteúdo fora do escopo.
- Quando for escrever os programas, sempre não faça um commit apenas no final, divida as subtarefas por commits, assinados em meu nome e com mensagens atomicas e em portugues.

# CRITÉRIOS DE DECISÃO

Quando houver múltiplas alternativas:

1. Priorize interpretar o que esta escrito na especificação do trabalho.
2. Considere o contexto do tema que estamos trabalhando
3. Avalie a veracidade dos fatos
4. Justifique escolhas relevantes com base nos critérios definidos.

# FALLBACK

Caso existam informações insuficientes, ambiguidades ou limitações:

- Utilize o contexto disponível antes de solicitar esclarecimentos.
- Faça suposições razoáveis apenas quando não comprometerem o resultado, identificando-as explicitamente.
- Solicite informações adicionais somente quando forem indispensáveis.
- Se uma etapa não puder ser concluída, explique a limitação e apresente o resultado parcial possível.

# TAREFA

Realizar as etapas um e dois, quero que me pergunte decisões críticas via human in the loop, vamos fazer isso juntos
```

Anexo: `2o-Trab-SimbML-Incerteza.pdf`. A formatação de espaços e marcadores foi normalizada nesta transcrição; o conteúdo da solicitação foi preservado.

## Orientações adicionais do responsável

```text
A gente pode buscar dados de outras fontes desde que sejam fontes relevantes e confiáveis
```

```text
vamos tentar usar a maioria do link que eu mandei ok? mas pode complementar. a gente onta um dataset para esse trabalho
```

Ao discutir percentil 75 e mediana, o responsável solicitou:

```text
No que isso implica? eu quero entender melhor a diferença
```

Foi explicado que Q75 destaca aproximadamente um quarto das observações de treino, enquanto a mediana divide aproximadamente em metades; empates alteram proporções; o limiar será congelado para o teste. Também se explicou o baseline de maioria e que o rótulo não representa gravidade ambiental.

## Respostas às decisões críticas

| Tema apresentado pelo assistente | Escolha humana |
|---|---|
| Potência de focos detectados vs. previsão de novas ocorrências | Potência dos focos detectados |
| Bioma Amazônia vs. estado do Amazonas vs. Amazônia Legal | Bioma Amazônia no Brasil |
| Dois anos com teste externo vs. um ano com divisão interna | 2024 desenvolvimento e 2025 teste, sensor AQUA_M-T |
| Percentil 75 vs. mediana, após explicação | Percentil 75: destacar o quarto de maior potência |

## Análise e planejamento: contribuição da IA e evidência independente

A IA auxiliou na leitura dos requisitos, busca de fontes oficiais, elaboração de um programa de auditoria descritiva, análise dos arquivos, documentação do contrato de dados e planejamento dos seis módulos. A análise numérica foi executada sobre arquivos oficiais com hashes registrados, não gerada por estimativa textual.

Nenhum algoritmo acadêmico foi implementado ou treinado nessa fase inicial. Naquele momento, não havia métricas preditivas, revisão técnica por especialista ou validação operacional para relatar. O relatório final deverá incluir apêndice compacto com referência a este arquivo, respeitando 6–8 páginas.

## Construção do dataset e preparação dos notebooks

Solicitação adicional do responsável em 31/10/2026:

```text
Por favor favor, contrói e audita esse dataset para que eu possa implementar os notebooks, cria um documento auxiliar explicando o dataset.
```

O trecho selecionado para contexto pedia construir e auditar o dataset completo de 2024–2025, preparar o tratamento de ausências, as divisões de validação e o ambiente reproduzível. Foi mantida a orientação de publicar no GitHub somente os materiais acadêmicos e os necessários à reprodução.

A IA auxiliou na aquisição dos 24 arquivos oficiais, filtragem Brasil/Amazônia/AQUA_M-T, preservação de ausências, rastreabilidade e preparação dos folds/transformações. Produziu funções auxiliares de leitura/preparação, testes, ambiente com versões fixadas, dicionário e guia explicativo. As decisões de fonte, anos, sensor, atributos e quartil superior foram preservadas.

O resultado contém 140.346 registros de desenvolvimento e 43.131 de teste. Foram executados 11 testes, conferência agregada com os produtos anuais do próprio INPE, reconstrução idêntica dos CSVs comprimidos e execução dos exemplos no kernel Jupyter. Na entrega do dataset ainda não haviam sido implementados os notebooks acadêmicos nem ajustados modelos.

## Prompt de referência para implementação assistida por IA

Após discutir as comparações entre implementações próprias e bibliotecas, o responsável solicitou:

```text
Ok cria um arquivo markdown contendo o passo a passo para codar os dois notebooks e os dois arquivos arvore.py e linear.py. Apenas isso por enquanto, o relatório será feito ao final. Apenas esses 4 arquivos devem ser alterados
```

O Codex produziu o guia agora preservado em [prompts/implementacao-etapas-1-2.md](prompts/implementacao-etapas-1-2.md). Ele é registrado como **prompt de referência para implementação com assistência de IA**: contém o contexto, as interfaces, a sequência de trabalho, as fórmulas, as verificações e os requisitos de comparação. O corpo original foi preservado integralmente; o novo cabeçalho explica sua procedência e o escopo histórico.

O guia foi inicialmente mantido como material interno. Sua publicação como registro acadêmico de uso de IA foi solicitada na revisão abaixo. As restrições originais de alterar somente quatro arquivos pertencem à rodada de implementação; a atualização documental foi autorizada posteriormente.

SHA-256 do corpo original, antes do cabeçalho de registro:

```text
221222a4d29a55e0b3d61183fe559ea7101705f10b6029cc81c2547e6f172396
```

O registro identifica o roteiro disponibilizado para esse uso; a conversa de implementação não foi transcrita aqui. O responsável deve acrescentar outros prompts e intervenções humanas efetivamente usados, caso existam, antes de concluir o apêndice final.

### Arquivos resultantes e rastreabilidade

| Entrega | Commit |
|---|---|
| `arvore.py`: entropia, ganho e ID3 | [`0380670`](https://github.com/mh131105/fia-queimadas-amazonia/commit/0380670a67576028cafd2020909f1d3e8af31986) |
| `etapa1.ipynb`: experimentos e comparação da árvore | [`c7e2404`](https://github.com/mh131105/fia-queimadas-amazonia/commit/c7e2404723b1e52d29492957e209ae241f042a4b) |
| `linear.py`: fórmula fechada, gradiente, L2 e logística | [`88f0f1e`](https://github.com/mh131105/fia-queimadas-amazonia/commit/88f0f1e460c5d56974111084ae323a3752275412) |
| `etapa2.ipynb`: verificações e comparações das regressões | [`a333fbb`](https://github.com/mh131105/fia-queimadas-amazonia/commit/a333fbb5bc577ba50e947067c12ecd749a3aba33) |

## Revisão dos resultados e atualização documental

Solicitação do responsável:

```text
Revisa os resultados, documenta o guia que voce criou como prompt para implementação com assistencia de IA, depois atualiza o readme
```

A revisão usou a versão `a333fbb5bc577ba50e947067c12ecd749a3aba33`, em Python 3.12.14 com as dependências fixadas do projeto. A IA auxiliou na inspeção do código, no confronto com as tarefas das seções 4.2 e 5.2 do PDF e na interpretação das métricas. A verificação executável incluiu:

- Os 11 testes existentes de integridade, partições e preparação do dataset.
- Execução completa dos dois notebooks com kernels novos, incluindo os testes sintéticos, os cinco folds espaciais e os cinco diagnósticos temporais de cada etapa.
- Conferência das saídas textuais e numéricas com as salvas no repositório e geração das quatro figuras.
- Verificação dos hashes dos notebooks originais, preservando os quatro arquivos de implementação.

As métricas foram reproduzidas. As implementações próprias concordam numericamente com as referências, mas a utilidade preditiva observada é limitada: o ID3 teve recall espacial médio 0,012801 e a logística, 0,048038; a multivariada teve R² médio 0,021070. Esses resultados e as referências de maioria/média estão no [README](../README.md), com os desvios entre folds e os limites de interpretação.

A revisão também explicitou que o lambda do Ridge foi selecionado com todos os folds espaciais de 2024. Seu diagnóstico temporal reutiliza essa escolha e não constitui uma avaliação prospectiva independente da seleção. Os notebooks não usam 2025 para ajuste, seleção ou avaliação de modelos. A inspeção dos dados na auditoria de qualidade é distinta dessa avaliação preditiva final, ainda pendente.

Esta rodada atualiza a documentação e o registro do prompt. O relatório e seu apêndice, as etapas 3–6 e a avaliação final continuam para entregas posteriores.


## Etapa 3 — Probabilidade (`etapa3.ipynb`)

Data: **01/11/2026**. Assistente: Claude. A Etapa 3 foi solicitada com o PDF da especificação e o repositório até as etapas 1 e 2.

```text
eu tenho um trabalho com a especificação que está no pdf que estou te enviando. foi dividido o trabalho em 6 partes, fiquei responsável pela terceira parte. o zip contêm o que já existe do trabalho até agora. a terceira parte seria essa, e gostaria que fizesse tudo que seja necessário relacionado a essa etapa somente:
"Etapa 3 — Entender as probabilidades do problema
Em etapa3.ipynb, podemos investigar perguntas como: “Entre os focos detectados, como a proporção de FRP elevada varia com períodos longos sem chuva?”. Precisamos construir uma distribuição conjunta de duas variáveis binárias, calcular probabilidades marginais e condicionais, aplicar a regra da cadeia com três variáveis e o Teorema de Bayes. Também devemos verificar se dois atributos são independentes quando conhecemos a classe. Os cortes das variáveis serão definidos no treino.
Exigência: seção 6, p. 6. "
```

Escolhas metodológicas registradas: F = FRP ≥ Q75 do treino; S = dias sem chuva ≥ mediana do treino (mesmo corte da representação `discretizada`); R = precipitação > mediana do treino (0 mm; desigualdade estrita porque a mediana é zero); intervalos por bootstrap de municípios; independência condicional avaliada por φ, informação mútua e qui-quadrado, priorizando a magnitude. 2025 não foi usado.


## Etapa 5 — Redes Bayesianas (`rede_bayesiana.py` e `etapa5.ipynb`)

Data: **06/11/2026**. Assistente: Antigravity (Gemini). A Etapa 5 foi solicitada com a especificação em PDF e o repositório contendo as Etapas 1 a 4.

```text
Oi, eu tenho um trabalho de IA para realizar,, a especificação está dentro do repositório, além das partes 1-4, fiquei responsável pela parte 5, me explique ela na íntegra, confira as partes anteriores, monte um pipeline operacional para fazer a parte 5 e a faça. Tudo isso verificando comigo ao final de cada etapa definida
```

Escolhas metodológicas e verificações registradas (validadas passo a passo via *human-in-the-loop*):
- **Variáveis binárias do DAG (5 nós):** mantidas as mesmas definições e cortes aprendidos exclusivamente no treino da Etapa 3: $M$ (`mes >= mediana`), $R$ (`precipitacao > 0` mm, desigualdade estrita), $S$ (`numero_dias_sem_chuva >= mediana`), $H$ (`hora_utc >= mediana`) e $F$ (`frp >= Q75`).
- **Arestas justificadas (5 arestas):** $M \to R$, $M \to S$, $R \to S$ (capturando a dependência entre precipitação instantânea, sazonalidade e período sem chuva observada na triagem da Etapa 3), $S \to F$ e $H \to F$.
- **Economia de parâmetros:** $12$ parâmetros livres na Rede Bayesiana ($1 + 1 + 2 + 4 + 4$) frente a $31$ na conjunta completa ($2^5 - 1$), economizando $19$ parâmetros ($61{,}3\%$).
- **Implementação e testes (`rede_bayesiana.py` e `tests/test_rede_bayesiana.py`):** estimação empírica de CPTs no treino, verificação exata de d-separação por inspeção de triplas (cadeias, bifurcações e colisores, inclusive por descendentes) e inferência exata por enumeração (`ENUMERATION-ASK`) implementada do zero e confrontada com `VariableElimination` do `pgmpy` (diferença numérica máxima $\le 1{,}11 \times 10^{-16}$).
- **Demonstração de colisor:** $S \to F \leftarrow H$, mostrando bloqueio quando $F$ não é observado ($S \perp_d H$) e abertura do caminho (*explaining away*) quando $F=1$ é observado ($S \not\perp_d H \mid F$). 2025 permaneceu reservado.

## Etapa 6 — Lógica Fuzzy (`fuzzy.py` e `etapa6.ipynb`)

Data: **06/11/2026**. Assistente: **DeepSeek**. A Etapa 6 foi solicitada com o PDF da especificação, o repositório até a Etapa 5 e os documentos auxiliares.

```text
Faça a Etapa 6 (Lógica Fuzzy, seção 9 do PDF) combinando saídas das Etapas 1, 2, 4 e 5
em uma decisão interpretável via Mamdani. Entradas linguísticas: confiança da árvore ID3
(Etapa 1), probabilidade da logística (Etapa 2), probabilidade do Bernoulli Naive Bayes
(Etapa 4) e probabilidade da Rede Bayesiana (Etapa 5). Saída: urgência em [0,100] com
funções triangulares. Seis regras SE-ENTÃO legíveis por especialista. Documente por que a
Etapa 3 não é usada como entrada. Compare com o threshold rígido de 0,5. Protocolo: cinco
folds espaciais de 2024; 2025 reservado.
```

Escolhas metodológicas registradas: entradas `conf_arvore`, `prob_logistica`, `prob_nb` e `risco_bayesiano`; saída `urgencia` em [0,100] com três termos triangulares; seis regras que combinam ao menos duas fontes; threshold fuzzy de 50, fixado antes da avaliação; `fuzzy.py` comparado a `scikit-fuzzy` (diferença máxima < 1e-6). O enunciado não é uniforme entre as seções 9.1, 9.2.5 e 10; optamos pela leitura ampla da Seção 9.1 e incluímos a Etapa 2. A Etapa 3 não entra como entrada porque não produz probabilidade de F nem decisão classificável: ela define as variáveis `M`, `R`, `S`, `H`, `F` e seus cortes em `utils.py`, reutilizados pelas etapas seguintes. 2025 não foi usado.
