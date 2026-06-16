# Etapa 6 — Modelagem de tópicos (LDA e BERTopic)

## Propósito

Esta é a etapa central do pipeline de agenda. A partir dos documentos pré-processados (tokens ou embeddings), ela identifica automaticamente os **tópicos temáticos** presentes no manifesto de cada partido.

Um **tópico** é um grupo de palavras que tendem a aparecer juntas e que representam um tema latente no texto — por exemplo, "economia", "meio ambiente" ou "saúde pública". O resultado desta etapa permite saber quais são os grandes temas defendidos por cada partido em sua agenda.

O pipeline suporta dois algoritmos de modelagem de tópicos:

## LDA (Latent Dirichlet Allocation)

LDA é um modelo probabilístico clássico. Ele assume que cada documento é uma mistura de tópicos e cada tópico é uma distribuição de palavras. O LDA é alimentado pelos **tokens** produzidos na Etapa 3.

### Como o LDA é treinado

1. Os tokens de todos os documentos do partido são carregados via `load_party_tokens_dataframe`.
2. É feita uma **busca por coerência**: modelos com diferentes números de tópicos (de `topic_start` a `topic_limit`) são treinados e avaliados pela métrica `c_v`. O número de tópicos com maior coerência é selecionado.
3. Um modelo final é treinado com o número ideal de tópicos, com mais passagens (`final_passes`) e iterações (`final_iterations`) para maior qualidade.

### Parâmetros do LDA

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `topic_start` | `2` | Mínimo de tópicos a testar |
| `topic_limit` | `8` | Máximo de tópicos a testar |
| `topic_step` | `1` | Incremento entre números de tópicos testados |
| `search_passes` | `10` | Passagens na busca por coerência |
| `search_iterations` | `100` | Iterações na busca por coerência |
| `final_passes` | `30` | Passagens no treinamento final |
| `final_iterations` | `300` | Iterações no treinamento final |

## BERTopic

BERTopic é um modelo moderno baseado em embeddings e clustering. Em vez de modelar distribuições de palavras, ele agrupa documentos semanticamente similares e usa c-TF-IDF com representação KeyBERT para descrever cada grupo como um tópico.

### Como o BERTopic é treinado

1. Os embeddings dos chunks (Etapa 4 ou 5, dependendo de `use_embeddings_without_stopwords`) são carregados.
2. O pipeline interno do BERTopic é executado:
   - **UMAP**: reduz a dimensionalidade dos vetores (configurado adaptativamente ao tamanho do corpus).
   - **HDBSCAN**: agrupa os documentos em clusters temáticos (configurado adaptativamente).
   - **CountVectorizer**: extrai os termos mais representativos de cada cluster (ngrams de 1 a 3 palavras, com filtros `min_df` e `max_df` adaptativos).
   - **KeyBERTInspired**: refina a representação dos tópicos usando similaridade semântica com o modelo de embedding.
3. Se o pipeline falhar (ex: poucos documentos), um fallback UMAP com parâmetros mínimos é ativado automaticamente.

### Adaptação ao tamanho do corpus

Os parâmetros de UMAP e HDBSCAN são ajustados automaticamente com base no número de documentos:

| Tamanho do corpus | UMAP `n_neighbors` | UMAP `n_components` | HDBSCAN `min_cluster_size` |
|-------------------|--------------------|---------------------|---------------------------|
| ≤ 5 docs | `n_docs - 1` | 2 | 2 |
| 6–15 docs | até 5 | até 3 | 2 |
| > 15 docs | até 15 | até 10 | `max(5, n_docs * 5%)` |

## Dados de entrada

### Para LDA

O caminho de entrada do LDA depende de `preprocessing_version` definido na Etapa 1. O notebook foi executado tanto com `"v1"` quanto com `"v2"`, gerando dois conjuntos independentes de tópicos LDA — refletidos nos subdiretórios `topics/lda/v1/` e `topics/lda/v2/` dos dados de saída.

| Versão | Localização | Formato | Descrição |
|--------|-------------|---------|-----------|
| `v1` | `data/party_agenda/preprocessing/tokenization/tokens/{PARTIDO}/CSV/*.csv` | CSV | Tokens sem stopwords, lematizados (spaCy) |
| `v2` | `data/party_agenda/preprocessing/tokenization/tokensV2/{PARTIDO}/CSV/*.csv` | CSV | Tokens com stopwords, sem lematização, sem acentos |

### Para BERTopic

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/embeddings/{PARTIDO}/agenda_embeddings_*.csv` | CSV | Chunks com texto (quando `use_no_stop = False`) |
| `data/party_agenda/embeddings/{PARTIDO}/agenda_embeddings_*.npy` | NPY | Vetores de embeddings correspondentes |
| `data/party_agenda/embeddingsWithoutStopwords/{PARTIDO}/agenda_embeddings_*.csv` | CSV | Chunks de tokens (quando `use_no_stop = True`) |
| `data/party_agenda/embeddingsWithoutStopwords/{PARTIDO}/agenda_embeddings_*.npy` | NPY | Vetores correspondentes |

## Dados produzidos

### Para LDA

Salvos em `data/party_agenda/topics/lda/{version}/{PARTIDO}/`:

| Arquivo | Descrição |
|---------|-----------|
| `lda_model.model` | Modelo LDA serializado (Gensim) |
| `lda_dictionary.dict` | Dicionário de vocabulário (Gensim) |
| `lda_topicos_termos.csv` | Termos e pesos de cada tópico identificado |
| `lda_topN_docs_por_topico.csv` | Top-N documentos mais representativos por tópico |
| `lda_distribuicao_docs.csv` | Distribuição de probabilidade de todos os documentos sobre todos os tópicos |
| `lda_coherence_scores.csv` | Pontuação de coerência para cada número de tópicos testado |

### Para BERTopic

Salvos em `data/party_agenda/topics/bertopic/{PARTIDO}/`:

| Arquivo | Descrição |
|---------|-----------|
| `bertopic_model` | Modelo BERTopic serializado |
| `bertopic_topicos_termos.csv` | Termos e pesos de cada tópico identificado |
| `bertopic_topN_docs_por_topico.csv` | Top-N documentos mais representativos por tópico |
| `bertopic_distribuicao_docs.csv` | Distribuição de todos os documentos sobre todos os tópicos |

### Estrutura de `*_topicos_termos.csv`

| Coluna | Tipo | Exemplo |
|--------|------|---------|
| `topic` | inteiro | `0` |
| `terms` | string | `0.012*"brasil" + 0.008*"social" + ...` |

Cada linha é um tópico. Os termos são listados com seus pesos relativos dentro do tópico.

### Estrutura de `*_distribuicao_docs.csv`

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `doc_id` | inteiro | Índice do documento |
| `topic` | inteiro | ID do tópico |
| `probability` | float | Probabilidade do documento pertencer ao tópico |

## Relação com outras etapas

- Depende da **Etapa 3** (tokens para LDA) e das **Etapas 4 ou 5** (embeddings para BERTopic).
- Produz os arquivos de termos de tópicos (`*_topicos_termos.csv`) que a **Etapa 7** usa para gerar embeddings dos tópicos da agenda.
