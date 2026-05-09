# Fluxo da agenda politica

Este guia descreve o fluxo do modulo de agenda, do PDF ate os topicos, e explica os dados gerados em cada etapa.

## Visao geral do fluxo
1. PDF -> TXT (extracao do texto)
2. Pre-processamento e tokenizacao (V1 ou V2)
3. Embeddings por chunks semanticos
4. Modelagem de topicos (LDA)
5. Embeddings de termos de topicos + similaridade com discursos

Cada etapa depende da anterior. O notebook [notebooks/agenda/agenda_extraction.ipynb](../notebooks/agenda/agenda_extraction.ipynb) orquestra esse fluxo.

## 1) PDF -> TXT (entrada do fluxo)
Origem: [data/party_agenda/party](../data/party_agenda/party)

Estrutura:
- Cada partido tem:
  - pdf/ (arquivo original)
  - txt/ (texto extraido)

O notebook faz a extracao usando PyMuPDF (layout-aware) e grava:
- data/party_agenda/party/<PARTIDO>/txt/<arquivo>.txt

Para que serve:
- Transformar o PDF em texto puro para ser processado nas etapas seguintes.

## 2) Pre-processamento e tokenizacao (V1 ou V2)
Codigo principal: [src/agenda/pre_processing.py](../src/agenda/pre_processing.py)

Entradas:
- TXT dos partidos (data/party_agenda/party/<PARTIDO>/txt)

Variantes:
- V1: remove stopwords, mantem acentos
- V2: mantem stopwords, remove acentos

Saidas:
- V1: [data/party_agenda/preprocessing/tokenization/tokens](../data/party_agenda/preprocessing/tokenization/tokens)
- V2: [data/party_agenda/preprocessing/tokenization/tokensV2](../data/party_agenda/preprocessing/tokenization/tokensV2)

Estrutura por partido:
- TXT/ <arquivo>_tokens.txt
- CSV/ <arquivo>_preprocess.csv

Como ler:
- TXT: uma linha por chunk; cada linha e uma lista JSON de tokens.
- CSV: colunas relevantes:
  - chunk_id: id do chunk
  - chunk_text: texto bruto do chunk
  - preprocess_agenda: texto preprocessado
  - tokens: lista de tokens (JSON)

Para que serve:
- Preparar os textos para modelagem e garantir consistencia linguistica.

## 3) Embeddings por chunks semanticos
Codigo principal: [src/agenda/embeddings.py](../src/agenda/embeddings.py)
Saida: [data/party_agenda/embeddings](../data/party_agenda/embeddings)

Arquivos gerados por partido:
- agenda_embeddings_<source>_<timestamp>.csv
- agenda_embeddings_<source>_<timestamp>.npy

Como ler:
- CSV: metadados dos chunks (texto, ids, etc).
- NPY: matriz numpy com os embeddings (mesma ordem do CSV).

Para que serve:
- Representar cada chunk em um vetor numerico para buscas e similares.

## 4) Topicos da agenda (LDA)
Codigo principal: [src/agenda/topics.py](../src/agenda/topics.py)
Saida: [data/party_agenda/topics/lda](../data/party_agenda/topics/lda)

Estrutura:
- LDA + versao do preprocessing:
  - data/party_agenda/topics/lda/v1/<PARTIDO>/
  - data/party_agenda/topics/lda/v2/<PARTIDO>/

Arquivos gerados:
- lda_model.model: modelo treinado
- lda_dictionary.dict: dicionario do corpus
- lda_topicos_termos.csv: topicos e termos principais
- lda_distribuicao_docs.csv: distribuicao de topicos por documento
- lda_topN_docs_por_topico.csv: documentos mais representativos por topico
- lda_coherence_scores.csv: diagnostico de coerencia
- arquivos auxiliares do modelo (state, id2word, expElogbeta)

Como ler:
- lda_topicos_termos.csv: principal para entender os topicos.
- lda_topN_docs_por_topico.csv: exemplos reais de documentos por topico.
- lda_distribuicao_docs.csv: peso de cada topico por documento.

Para que serve:
- Resumir os textos da agenda em topicos interpretaveis.

## 5) Embeddings de termos de topicos + similaridade com discursos
Saida: [data/party_agenda/embeddings](../data/party_agenda/embeddings)

Arquivos gerados por partido:
- topicos_embeddings_termos.csv: embeddings de termos dos topicos LDA
- topicos_similaridade_discursos.csv: melhor match entre topicos da pauta e topicos dos discursos

Como ler:
- topicos_embeddings_termos.csv: coluna embedding contem vetores em JSON.
- topicos_similaridade_discursos.csv: mostra o topico da pauta e o topico de discurso mais similar (cosseno).

Para que serve:
- Comparar a agenda do partido com os topicos dos discursos.

## Relacao entre etapas
- PDF -> TXT alimenta o pre-processamento.
- Pre-processamento gera tokens usados na modelagem de topicos.
- Embeddings por chunks sao usados para analise semantica e inspecao.
- LDA usa os tokens para gerar topicos.
- Embeddings de topicos permitem similaridade entre agenda e discursos.

## Dica de execucao no notebook
- Defina `preprocessing_version` antes do pre-processamento.
- Defina `topic_model` antes de rodar a etapa de topicos.
- Verifique as pastas de saida para confirmar o fluxo correto (v1 vs v2).
