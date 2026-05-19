# Etapa 5 — Geração de embeddings sem stopwords (a partir dos tokens)

## Propósito

Esta etapa é uma alternativa à Etapa 4 para geração de embeddings. Enquanto a Etapa 4 usa o texto bruto dos manifestos, esta etapa usa os **tokens já pré-processados** (sem stopwords) gerados na Etapa 3.

A ideia é que ao remover stopwords antes de gerar os embeddings, os vetores resultantes capturam mais precisamente os termos temáticos do documento, evitando que palavras de alta frequência e baixo valor semântico (como "de", "para", "que") influenciem a representação.

Os embeddings produzidos aqui são salvos na pasta `embeddingsWithoutStopwords/` e são utilizados pelo BERTopic quando `use_embeddings_without_stopwords = True`.

## O que é executado

Para cada partido e para cada arquivo TXT de tokens:

1. Lê o arquivo TXT linha a linha — cada linha é uma lista Python de tokens (ex: `['brasil', 'social', 'encontro']`).
2. Converte cada lista de tokens em uma string de texto juntando os tokens com espaço (`" ".join(tokens)`).
3. Codifica cada string com o modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
4. Salva os resultados em CSV e NPY.

## Modelo utilizado

| Modelo | Dimensão dos vetores | Idiomas |
|--------|----------------------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilíngue (inclui português) |

Este é um modelo menor e mais rápido do que o `BAAI/bge-m3` usado em outras etapas, adequado para gerar embeddings em grande volume.

## Dados de entrada

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/preprocessing/tokenization/tokens/{PARTIDO}/TXT/*.txt` | TXT | Arquivos de tokens gerados na Etapa 3 (versão v1) |

Cada linha do TXT contém uma lista de tokens no formato Python: `['token1', 'token2', ...]`

## Dados produzidos

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/embeddingsWithoutStopwords/{PARTIDO}/agenda_embeddings_{nome}.csv` | CSV | Metadados de cada chunk de tokens |
| `data/party_agenda/embeddingsWithoutStopwords/{PARTIDO}/agenda_embeddings_{nome}.npy` | NPY | Matriz de embeddings, uma linha por chunk |

### Estrutura do CSV produzido

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `source_id` | string | Nome do arquivo TXT de origem |
| `chunk_id` | inteiro | Índice sequencial do chunk (linha no TXT) |
| `chunk_text` | string | Tokens do chunk reunidos em uma string |
| `embedding` | lista JSON | Vetor de embedding em formato legível |

## Diferença em relação à Etapa 4

| Aspecto | Etapa 4 | Etapa 5 |
|---------|---------|---------|
| Texto de entrada | Texto bruto do manifesto | Tokens pré-processados (sem stopwords) |
| Chunking | Feito por similaridade entre sentenças | Cada linha de tokens é um chunk |
| Pasta de saída | `embeddings/` | `embeddingsWithoutStopwords/` |
| Modelo | Configurável (default: `BAAI/bge-m3` no BERTopic) | `paraphrase-multilingual-MiniLM-L12-v2` |

## Variável de controle

| Variável | Efeito |
|----------|--------|
| `use_embeddings_without_stopwords` | Se `True`, a Etapa 6 (BERTopic) carregará os embeddings desta pasta |

## Relação com outras etapas

- Depende da **Etapa 3** (tokens precisam ter sido gerados).
- É consumida pela **Etapa 6 (BERTopic)** quando `use_embeddings_without_stopwords = True`.
- Quando `use_embeddings_without_stopwords = False`, esta etapa pode ser ignorada — o BERTopic usará os embeddings da Etapa 4.
