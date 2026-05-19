# Etapa 4 — Geração de embeddings semânticos por chunks

## Propósito

Esta etapa transforma o texto bruto dos manifestos em **representações numéricas** (vetores de alta dimensão chamados embeddings) que capturam o significado semântico de cada trecho do documento.

O texto de cada manifesto é dividido em **chunks** — fragmentos menores e semanticamente coesos — e cada chunk recebe um vetor embedding gerado por um modelo de linguagem pré-treinado. Estes vetores são usados posteriormente pelo BERTopic para identificar grupos temáticos e pelo pipeline de similaridade para comparar temas da agenda com temas dos discursos.

## O que é executado

A função `generate_agenda_embeddings_from_txt` do módulo `src/agenda/embeddings.py` é chamada para cada arquivo TXT de cada partido. O processo:

1. Lê o TXT do manifesto.
2. Divide o texto em chunks usando similaridade semântica entre sentenças (fusão de sentenças similares e separação de sentenças muito distintas), respeitando os parâmetros `similarity_threshold`, `min_sentences_per_chunk` e `max_sentences_per_chunk`.
3. Codifica cada chunk com um modelo de sentence-transformers.
4. Salva os resultados em CSV e NPY.

## Parâmetros da função

| Parâmetro | Valor usado | Significado |
|-----------|-------------|-------------|
| `similarity_threshold` | `0.45` | Limiar de similaridade para agrupar sentenças num mesmo chunk |
| `min_sentences_per_chunk` | `1` | Mínimo de sentenças por chunk |
| `max_sentences_per_chunk` | `None` | Sem limite máximo |
| `batch_size` | `32` | Sentenças processadas por vez pelo modelo |
| `save_files` | `True` | Salva os arquivos CSV e NPY |

## Variável de controle

| Variável | Valor padrão | Efeito |
|----------|--------------|--------|
| `force_recompute_embeddings` | `False` | Se `True`, recalcula embeddings mesmo que os arquivos já existam |

## Dados de entrada

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/party/{PARTIDO}/txt/*.txt` | TXT | Textos extraídos dos PDFs na Etapa 2 |

## Dados produzidos

Para cada arquivo TXT de cada partido, dois arquivos são gerados:

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/embeddings/{PARTIDO}/agenda_embeddings_{nome}.csv` | CSV | Metadados de cada chunk com seu texto |
| `data/party_agenda/embeddings/{PARTIDO}/agenda_embeddings_{nome}.npy` | NPY (NumPy) | Matriz de embeddings — uma linha por chunk |

### Estrutura do CSV produzido

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `source_id` | string | Nome do arquivo TXT de origem |
| `chunk_id` | inteiro | Índice sequencial do chunk dentro do documento |
| `chunk_text` | string | Texto do chunk |
| `embedding` | lista JSON | Vetor de embedding (mesma informação do NPY, em formato legível) |

### Sobre o arquivo NPY

O arquivo `.npy` contém a matriz de embeddings em formato NumPy binário, com dimensões `(N, D)` onde:
- `N` = número de chunks
- `D` = dimensão dos vetores do modelo (varia por modelo, tipicamente 384 ou 768)

O NPY é a forma eficiente de carregar os vetores para operações matemáticas (como cálculo de similaridade cosseno).

## Relação com outras etapas

- **Etapa 6 (BERTopic)**: quando `use_embeddings_without_stopwords = False`, o BERTopic carrega os embeddings produzidos aqui como entrada para o treinamento. Os textos dos chunks também são usados como documentos.
- **Etapa 8 (Similaridade)**: **não** usa diretamente estes embeddings. A similaridade é calculada entre os vetores dos *tópicos* (Etapa 7), não entre os vetores dos chunks.
