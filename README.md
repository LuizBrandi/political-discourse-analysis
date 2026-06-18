# Análise de Coerência entre Pautas e Discursos Políticos

Este repositório reúne o código, os dados e a documentação de um projeto que compara **o que os partidos dizem que defendem** (suas pautas) com **o que seus parlamentares efetivamente falam** (seus discursos).

---

## 1. Visão Geral do Projeto

O projeto investiga o quanto a agenda declarada de um partido aparece, na prática, nos discursos de seus parlamentares. Para isso, parte de dois materiais: as pautas dos partidos (programas e manifestos) e os discursos proferidos na Câmara dos Deputados, sempre organizados por partido e por período.

O problema que ele ajuda a responder é simples de enunciar: *o discurso acompanha a pauta?* A análise é feita ao redor das eleições de 2022, comparando o período anterior e o posterior, para observar se o alinhamento entre pauta e discurso muda nesse intervalo.

De forma geral, o sistema lê os textos originais, organiza e resume o conteúdo em temas, e então mede a proximidade entre os temas das pautas e os temas dos discursos. O resultado é um conjunto de medidas de similaridade que indicam, partido a partido, quão próximos esses dois lados estão.

Os partidos analisados são **MDB, NOVO, PL, PSOL, PT e UNIÃO**. A leitura deste projeto não exige conhecimento técnico: cada etapa é explicada em alto nível tanto aqui quanto na pasta de documentação.

---

## 2. Como Executar o Projeto

O projeto é executado principalmente por meio de dois notebooks, um para cada lado da comparação. Ambos podem ser abertos no Jupyter, no VS Code ou em qualquer ambiente que rode notebooks, desde que as dependências já tenham sido instaladas (ver a seção **Configuração Inicial**).

As células de cada notebook devem ser executadas **de cima para baixo, na ordem em que aparecem**. Cada etapa possui uma célula de texto explicando o que faz, o que ela usa e o que produz. Como o passo final de cada notebook compara pautas com discursos, recomenda-se executar **os dois notebooks** antes de interpretar os resultados de similaridade.

### 2.1 Agenda Extraction

Notebook: [notebooks/agenda/agenda_extraction.ipynb](notebooks/agenda/agenda_extraction.ipynb)

Este notebook cuida do lado das **pautas dos partidos**. Ele parte dos documentos das pautas (em PDF, já incluídos no projeto), extrai o texto, prepara o conteúdo, identifica os principais temas de cada partido e gera as representações desses temas.

- **Como abrir:** abra o arquivo no Jupyter ou no VS Code e execute as células em sequência.
- **Ordem de execução:** de cima para baixo; a primeira célula define os parâmetros (partido, versão de preparação do texto e abordagem de temas).
- **Dependências:** os PDFs das pautas já acompanham o projeto, então não é necessário acesso à internet para as primeiras etapas (apenas para baixar os modelos de linguagem na primeira execução).
- **Resultado esperado:** ao final, são gerados os temas das pautas e suas representações e, na última etapa, a similaridade entre os temas das pautas e os temas dos discursos (que dependem do notebook de discursos).

### 2.2 Discourse Extraction

Notebook: [notebooks/discourse/discourse_extraction.ipynb](notebooks/discourse/discourse_extraction.ipynb)

Este notebook cuida do lado dos **discursos parlamentares**. A principal diferença em relação ao notebook da agenda é que ele **coleta os discursos a partir de uma fonte externa** (a Câmara dos Deputados), em vez de partir de documentos já existentes no projeto.

- **Como executar:** abra o notebook e execute as células em sequência. A primeira célula define o período de análise e os partidos.
- **Diferença em relação ao outro notebook:** ele baixa os discursos da internet e os organiza por partido e período (antes e depois da eleição), exigindo conexão de rede durante a coleta.
- **O que é gerado:** os discursos coletados e preparados, os temas dos discursos, suas representações e, ao final, a similaridade entre os temas dos discursos e os temas das pautas (que dependem do notebook da agenda).

> Como cada notebook calcula, no fim, a similaridade usando os temas produzidos pelo outro, o ideal é rodar os dois. O passo de similaridade só fica completo quando os temas dos dois lados já existem.

---

## 3. Estrutura de Pastas

```
political-discourse-analysis/
├── notebooks/        # Notebooks que executam e explicam o pipeline
│   ├── agenda/       # Fluxo das pautas dos partidos
│   └── discourse/    # Fluxo dos discursos parlamentares
├── src/              # Código-fonte com as funções usadas pelos notebooks
│   ├── agenda/       # Funções do fluxo das pautas
│   └── discourses/   # Funções do fluxo dos discursos
├── data/             # Dados de entrada e todos os resultados gerados
│   ├── party_agenda/ # Tudo relativo às pautas
│   └── discourses/   # Tudo relativo aos discursos
├── documentation/    # Documentação detalhada e o artigo (LaTeX/Overleaf)
├── reports/          # Relatórios do projeto em PDF
├── pyproject.toml    # Lista de dependências do projeto
└── README.md         # Este arquivo
```

- **notebooks/** — Ponto de entrada do projeto. Contém os notebooks que executam o processo de ponta a ponta e o documentam etapa por etapa. Use esta pasta para rodar os fluxos. Além dos dois notebooks principais, há notebooks auxiliares que geram os gráficos comparativos.
- **src/** — Reúne o código que dá suporte aos notebooks, separado entre o fluxo das pautas (`agenda`) e o dos discursos (`discourses`). Normalmente não é preciso editá-lo para usar o projeto; ele é chamado pelos notebooks.
- **data/** — Concentra tanto os materiais de entrada quanto os resultados produzidos em cada etapa. É aqui que ficam os textos originais, os dados preparados e todos os arquivos gerados (ver a seção seguinte).
- **documentation/** — Explicações detalhadas de cada etapa e os arquivos do artigo científico, incluindo os gráficos usados nele. Use esta pasta para entender o projeto em profundidade.
- **reports/** — Relatórios em PDF que descrevem a proposta e a versão de referência (baseline) do trabalho.

---

## 4. Onde os Resultados são Gerados

Todos os resultados ficam dentro de **`data/`**, separados entre o lado das pautas (`data/party_agenda/`) e o lado dos discursos (`data/discourses/`). Os arquivos são organizados por partido (e, no caso dos discursos, também por período). Os principais tipos de resultado são:

- **Textos extraídos**
  - Pautas em texto: `data/party_agenda/party/<PARTIDO>/txt/`
  - Discursos coletados (brutos): `data/discourses/raw/running_files/<PARTIDO>/`

- **Dados preparados (texto limpo e organizado)**
  - Pautas: `data/party_agenda/preprocessing/tokenization/`
  - Discursos: `data/discourses/preprocessing/running_files/<PARTIDO>/`

- **Representações semânticas (embeddings)**
  - Pautas: `data/party_agenda/embeddings/<PARTIDO>/`
  - Discursos: `data/discourses/embeddings/discourses/<PARTIDO>/`

- **Temas identificados**
  - Pautas: `data/party_agenda/topics/<abordagem>/<PARTIDO>/`
  - Discursos: `data/discourses/topics/<PARTIDO>/`

- **Resultado final — similaridade entre pauta e discurso**
  - A partir das pautas: `data/party_agenda/embeddings/<PARTIDO>/similaridade/topics/.../<periodo>/`
  - A partir dos discursos: `data/discourses/embeddings/discourses/<PARTIDO>/topics_similarity/`

- **Gráficos e tabelas**
  - Os gráficos comparativos são produzidos pelos notebooks auxiliares de gráficos e ficam em `documentation/overleaf/figures/`, sendo reaproveitados no artigo.

---

## 5. Fluxo Geral do Projeto

Em alto nível, o projeto segue sempre o mesmo caminho, tanto para as pautas quanto para os discursos:

1. **Entrada de dados** — As pautas chegam como documentos (PDF) já incluídos no projeto; os discursos são coletados de uma fonte externa.
2. **Processamento pelos notebooks** — O texto é limpo e organizado, resumido em temas, e cada tema ganha uma representação que captura o seu significado.
3. **Geração de resultados** — Os temas das pautas e dos discursos são comparados, medindo-se o quanto se aproximam, com os resultados separados por partido e por período.
4. **Armazenamento dos outputs** — Cada etapa salva seus arquivos em `data/`, e os gráficos comparativos finais ficam na pasta de documentação.

---

## 6. Configuração Inicial

**Pré-requisitos do ambiente**

- **Python 3.13** ou superior.
- **uv** como gerenciador de dependências (as dependências estão declaradas em `pyproject.toml`).
- Conexão com a internet na primeira execução (para baixar os modelos de linguagem) e durante a coleta dos discursos.

**1) Instalar as dependências**

```bash
uv sync
```

**2) Baixar os recursos de linguagem**

```bash
uv run python -m nltk.downloader stopwords
uv run python -m spacy download pt_core_news_lg
```

**3) Abrir e executar os notebooks**

Com o ambiente pronto, abra os notebooks da pasta `notebooks/` e execute as células em ordem, conforme descrito na seção **Como Executar o Projeto**.

**Problemas comuns**

- **Aviso sobre CUDA/placa de vídeo:** se aparecer uma mensagem sobre driver de GPU ou CUDA, ela pode ser ignorada — o processamento funciona normalmente na CPU, apenas mais devagar.
- **Demora na primeira execução:** na primeira vez, os modelos de linguagem são baixados automaticamente; isso pode levar alguns minutos.
- **Falha ao coletar discursos:** a etapa de coleta do notebook de discursos depende de acesso à internet e da disponibilidade da fonte externa; sem conexão, essa etapa não conclui.
- **Passo de similaridade incompleto:** a comparação final só funciona quando os temas das pautas e dos discursos já foram gerados; por isso, execute os dois notebooks antes de analisar a similaridade.
