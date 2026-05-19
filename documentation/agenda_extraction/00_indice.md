# Documentação do Pipeline: agenda_extraction.ipynb

Este conjunto de documentos descreve cada etapa do notebook `notebooks/agenda/agenda_extraction.ipynb`, que implementa o pipeline completo de análise de agenda política por partido.

## O que o pipeline faz

O pipeline transforma documentos PDF de agenda política partidária em resultados de similaridade entre os tópicos da agenda e os tópicos dos discursos parlamentares. O fluxo é totalmente sequencial: cada etapa depende dos artefatos gerados pela etapa anterior.

## Etapas

| # | Arquivo | Título |
|---|---------|--------|
| 1 | [01_configuracao_inicial.md](01_configuracao_inicial.md) | Configuração inicial e variáveis de controle |
| 2 | [02_pdf_para_txt.md](02_pdf_para_txt.md) | Conversão de PDF para TXT (layout-aware) |
| 3 | [03_preprocessamento.md](03_preprocessamento.md) | Pré-processamento e tokenização da agenda |
| 4 | [04_embeddings_chunks.md](04_embeddings_chunks.md) | Geração de embeddings semânticos por chunks |
| 5 | [05_embeddings_sem_stopwords.md](05_embeddings_sem_stopwords.md) | Geração de embeddings sem stopwords (tokens) |
| 6 | [06_modelagem_topicos.md](06_modelagem_topicos.md) | Modelagem de tópicos (LDA e BERTopic) |
| 7 | [07_embeddings_topicos_termos.md](07_embeddings_topicos_termos.md) | Embeddings dos termos dos tópicos da agenda |
| 8 | [08_similaridade_cosseno.md](08_similaridade_cosseno.md) | Cálculo de similaridade cosseno entre tópicos |

## Fluxo geral de dados

```
PDF (agenda partidária)
    │
    ▼ Etapa 2
TXT (texto extraído, layout preservado)
    │
    ├──▶ Etapa 4: Embeddings semânticos por chunks ──▶ CSV + NPY de embeddings
    │
    ▼ Etapa 3
CSV de tokens (pré-processados)
    │
    ├──▶ Etapa 5: Embeddings sem stopwords ──▶ CSV + NPY de embeddings
    │
    ▼ Etapa 6
Tópicos da agenda (LDA ou BERTopic) ──▶ CSV de termos por tópico
    │
    ▼ Etapa 7
Embeddings dos termos dos tópicos ──▶ CSV com vetores por tópico
    │
    ▼ Etapa 8
Similaridade cosseno com tópicos dos discursos ──▶ CSV de similaridade por período
```

## Dependências entre etapas

- **Etapa 3** depende da **Etapa 2** (precisa dos TXTs)
- **Etapa 4** depende da **Etapa 2** (usa os mesmos TXTs)
- **Etapa 5** depende da **Etapa 3** (usa os tokens gerados)
- **Etapa 6** depende da **Etapa 3** e, se BERTopic, também da **Etapa 4** ou **Etapa 5**
- **Etapa 7** depende da **Etapa 6** (usa os termos dos tópicos gerados)
- **Etapa 8** depende da **Etapa 7** e dos embeddings de tópicos dos discursos (pipeline separado)

## Modelos de tópicos suportados

| Modelo | Versões de preprocessamento | Usa embeddings |
|--------|----------------------------|----------------|
| LDA | v1 (com remoção de stopwords), v2 (com remoção de acentos) | Não |
| BERTopic | v1 apenas | Sim (Etapa 4 ou 5) |
