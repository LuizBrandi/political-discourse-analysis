# Etapa 7 — Embeddings dos termos dos tópicos da agenda

## Propósito

Nesta etapa, cada tópico identificado na Etapa 6 é transformado em um **vetor numérico (embedding)** que representa seu significado semântico.

Para isso, os termos que descrevem cada tópico (ex: `"brasil + social + encontro + país"`) são transformados em uma string de texto e codificados por um modelo de linguagem. O resultado é um vetor por tópico que pode ser matematicamente comparado com vetores de tópicos dos discursos parlamentares — o que é exatamente o que a Etapa 8 faz.

Esta etapa é o **ponto de conexão** entre o pipeline de agenda e o pipeline de discursos: é aqui que os dois mundos se tornam comparáveis.

## O que é executado

Para cada partido e para cada modelo de tópicos (LDA ou BERTopic):

1. Lê o arquivo de termos dos tópicos gerado na Etapa 6 (`lda_topicos_termos.csv` ou `bertopic_topicos_termos.csv`).
2. Extrai os termos de cada tópico, limpando os pesos numéricos (ex: `0.012*"brasil"` → `"brasil"`).
3. Codifica o texto de termos de cada tópico usando o modelo `paraphrase-multilingual-MiniLM-L12-v2`.
4. Salva o resultado em um CSV com os embeddings incluídos.

### Limpeza dos termos

O formato de saída do LDA é:
```
0.012*"brasil" + 0.008*"social" + 0.008*"encontro" + ...
```

O formato de saída do BERTopic é:
```
0.656*"caminho brasil país" + 0.638*"pensar brasil" + ...
```

Em ambos os casos, a função `extract_terms` remove os pesos e extrai apenas as palavras entre aspas, produzindo uma string como `"brasil social encontro"`.

## Modelo utilizado

| Modelo | Dimensão dos vetores | Idiomas |
|--------|----------------------|---------|
| `paraphrase-multilingual-MiniLM-L12-v2` | 384 | Multilíngue (inclui português) |

O mesmo modelo é usado para os embeddings dos tópicos dos discursos (pipeline separado), garantindo que os vetores dos dois lados sejam comparáveis no mesmo espaço semântico.

## Dados de entrada

### Para LDA

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/topics/lda/{version}/{PARTIDO}/lda_topicos_termos.csv` | CSV | Termos e pesos dos tópicos LDA |

### Para BERTopic

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/topics/bertopic/{PARTIDO}/bertopic_topicos_termos.csv` | CSV | Termos e pesos dos tópicos BERTopic |

## Dados produzidos

Os arquivos são salvos na pasta de embeddings de cada partido:

### Para LDA

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/embeddings/{PARTIDO}/embeddings_topics_termos_lda_{version}.csv` | CSV | Embeddings dos tópicos LDA por partido |

### Para BERTopic

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/embeddings/{PARTIDO}/embeddings_topics_termos_bertopic.csv` | CSV | Embeddings dos tópicos BERTopic por partido |

### Estrutura do CSV produzido

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `topic` | inteiro | ID do tópico (mesmo ID do arquivo de termos) |
| `terms` | string | String original de termos e pesos do tópico |
| `terms_clean` | string | Termos extraídos sem pesos (usados para gerar o embedding) |
| `embedding` | JSON string | Vetor numérico de 384 dimensões serializado como JSON |

### Exemplo de linha

```
topic: 0
terms: 0.012*"brasil" + 0.008*"social" + 0.008*"encontro" + 0.007*"país"
terms_clean: brasil social encontro país
embedding: [0.021, -0.134, 0.087, ...]  (384 valores)
```

## Relação com outras etapas

- Depende da **Etapa 6** (os arquivos `*_topicos_termos.csv` precisam existir).
- É consumida diretamente pela **Etapa 8** (cálculo de similaridade cosseno), que compara os vetores de tópicos da agenda com os vetores de tópicos dos discursos.
- O mesmo arquivo de modelo e mesmo tipo de embedding são usados no pipeline de discursos para garantir que os espaços vetoriais sejam comparáveis.
