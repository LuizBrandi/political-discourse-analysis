# Etapa 8 — Cálculo de similaridade cosseno entre tópicos

## Propósito

Esta é a etapa final do pipeline e produz a **resposta à pergunta central da pesquisa**: os partidos brasileiros discursam em plenário sobre os mesmos temas que defendem em seus manifestos?

Para cada partido, cada tópico da agenda (manifesto) é comparado com cada tópico do discurso parlamentar (gerado por pipeline separado). A comparação é feita por **similaridade cosseno** entre os vetores de embedding dos tópicos. O par mais similar entre agenda e discurso é registrado como resultado.

Além disso, os resultados são separados por **período eleitoral**: antes e depois da eleição de 2022, permitindo analisar se o alinhamento entre agenda e discurso muda conforme o contexto político.

## O que é executado

Para cada combinação de (partido, técnica de tópicos, arquivo de discurso):

1. Carrega os embeddings dos tópicos da **agenda** (Etapa 7).
2. Carrega os embeddings dos tópicos dos **discursos** (pipeline de discursos, externo a este notebook).
3. Classifica o arquivo de discurso como `antesDaEleicao` ou `depoisDaEleicao` com base nas datas no nome do arquivo (cutoff: 30/10/2022).
4. Calcula a **matriz de similaridade cosseno** entre todos os pares de tópicos (agenda × discurso).
5. Para cada tópico da agenda, seleciona o tópico de discurso com maior similaridade.
6. Salva o resultado em CSV.

## Identificação do período eleitoral

O nome dos arquivos de discurso segue o padrão:
```
embeddings_topics_termos_..._ini_DDMMYYYY_fim_DDMMYYYY.csv
```

A lógica de classificação:
- `fim <= 30/10/2022` → `antesDaEleicao`
- `inicio > 30/10/2022` → `depoisDaEleicao`
- Períodos que cruzam a data de corte são ignorados.

## Dados de entrada

### Tópicos da agenda (gerados na Etapa 7)

| Localização | Formato | Condição |
|-------------|---------|----------|
| `data/party_agenda/embeddings/{PARTIDO}/embeddings_topics_termos_bertopic.csv` | CSV | BERTopic |
| `data/party_agenda/embeddings/{PARTIDO}/embeddings_topics_termos_lda_{version}.csv` | CSV | LDA |

### Tópicos dos discursos (pipeline externo)

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/discourses/embeddings/discourses/{PARTIDO}/embeddings_topics_termos_*ini_*_fim_*.csv` | CSV | Embeddings dos tópicos dos discursos parlamentares por partido e período |

> Os arquivos de discurso são gerados por um pipeline separado (não coberto por este notebook). Eles seguem a mesma estrutura dos arquivos de agenda: coluna `topic`, `terms` e `embedding`.

## Dados produzidos

Para cada combinação de (partido, técnica, período), um CSV de similaridade é salvo:

### Para BERTopic

| Localização | Formato |
|-------------|---------|
| `data/party_agenda/embeddings/{PARTIDO}/similaridade/topics/bertopic/{periodo}/similaridade_topics_discurso_topics_agenda.csv` | CSV |

### Para LDA

| Localização | Formato |
|-------------|---------|
| `data/party_agenda/embeddings/{PARTIDO}/similaridade/topics/lda/{version}/{periodo}/similaridade_topics_discurso_topics_agenda.csv` | CSV |

### Estrutura do CSV produzido

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `agenda_topic` | inteiro | ID do tópico da agenda |
| `agenda_terms` | string | Termos descritivos do tópico da agenda |
| `discourse_topic` | inteiro | ID do tópico de discurso mais similar |
| `discourse_terms` | string | Termos descritivos do tópico de discurso mais similar |
| `cosine_similarity` | float | Valor de similaridade cosseno entre os dois tópicos (0 a 1) |

### Exemplo de linha

```
agenda_topic: 0
agenda_terms: 0.656*"caminho brasil país" + 0.638*"pensar brasil" + ...
discourse_topic: 1
discourse_terms: 0.014*"mdb" + 0.008*"brasil" + 0.006*"grande" + ...
cosine_similarity: 0.6304
```

## Interpretação dos resultados

- **Similaridade alta (próxima de 1)**: o partido discursa sobre temas muito próximos aos que defende em seu manifesto.
- **Similaridade baixa (próxima de 0)**: há distância entre a agenda escrita e os temas abordados nos discursos.
- **Diferença antes/depois da eleição**: permite observar se os partidos mudam sua coerência programática após conquistar ou perder o poder.

## Relação com outras etapas

- Depende da **Etapa 7** (embeddings dos tópicos da agenda).
- Depende do **pipeline de discursos** (externo), que gera os embeddings dos tópicos dos discursos parlamentares com o mesmo modelo de embedding.
- Os CSVs produzidos aqui são consumidos pelos notebooks de visualização:
  - `notebooks/agenda/similarity_lda_graphic.ipynb` — gráficos para resultados LDA
  - `notebooks/agenda/similarity_bertopic_graphic.ipynb` — gráficos para resultados BERTopic
