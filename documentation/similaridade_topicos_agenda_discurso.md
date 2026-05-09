# Similaridade entre topicos da pauta e topicos do discurso

Este documento descreve, de forma detalhada, como o arquivo de similaridade de topicos e gerado a partir do notebook de agenda. O processo compara os topicos extraidos da pauta partidaria com os topicos extraidos dos discursos, usando embeddings e similaridade cosseno.

## Visao geral do fluxo

1. Geracao de topicos da pauta (LDA ou BERTopic).
2. Geracao de embeddings para os termos de cada topico da pauta.
3. Geracao de embeddings para os termos de cada topico dos discursos.
4. Calculo de similaridade cosseno entre os embeddings dos topicos de pauta e de discurso.
5. Selecao, para cada topico da pauta, do topico de discurso mais parecido.
6. Gravacao do arquivo CSV de similaridade.

## Onde o calculo acontece

O calculo e executado no notebook de agenda, na secao:
"Calcula similaridade cosseno entre topicos da pauta e topicos dos discursos".

O CSV final fica em um caminho do tipo:

```
/data/party_agenda/embeddings/<PARTIDO>/similaridade/topics/<tecnica>/<versao>/similaridade_topics_discurso_topics_agenda.csv
```

Exemplo real (UNIAO, LDA v1):

```
political-discourse-analysis/data/party_agenda/embeddings/UNIAO/similaridade/topics/lda/v1/similaridade_topics_discurso_topics_agenda.csv
```

## Entradas usadas no calculo

O codigo usa dois arquivos principais (um da pauta e um dos discursos), ambos contendo embeddings de topicos:

### 1) Embeddings dos topicos da pauta

- Pasta base:
  - `data/party_agenda/embeddings/<PARTIDO>/`
- Arquivo esperado (LDA):
  - `embeddings_topics_termos_lda_<versao>.csv`
- Colunas importantes:
  - `topic`: id numerico do topico da pauta.
  - `terms`: lista de termos do topico (com pesos).
  - `embedding`: vetor do embedding do topico (string JSON com lista de floats).

### 2) Embeddings dos topicos dos discursos

- Pasta base:
  - `data/discourses/embeddings/discourses/<PARTIDO>/`
- Arquivo esperado (LDA):
  - `embeddings_topics_termos_lda_<versao>.csv`
- Colunas importantes:
  - `topic`: id numerico do topico do discurso.
  - `terms`: lista de termos do topico (com pesos).
  - `embedding`: vetor do embedding do topico (string JSON com lista de floats).

### Observacao sobre tecnica e versao

O codigo identifica automaticamente a tecnica e a versao com base no nome do arquivo da pauta:

- Se tiver `bertopic` no nome, tecnica = bertopic.
- Se tiver `lda_vX`, tecnica = lda e versao = vX.
- Se tiver apenas `lda`, tecnica = lda e versao indefinida.

Depois disso, ele procura o arquivo correspondente dos discursos seguindo a mesma tecnica e versao.

## Pre-processamentos importantes

Antes do calculo, ha duas etapas auxiliares:

1. Normalizacao do nome do partido
   - Remove acentos e caracteres especiais.
   - Converte para maiusculas.
   - Ajuda a localizar a pasta correta dos discursos.

2. Conversao dos embeddings
   - A coluna `embedding` e uma string JSON.
   - O codigo faz `json.loads` para transformar em lista de floats.
   - Em seguida, cria um array NumPy para o calculo.

## Calculo da similaridade

Para cada partido e para cada arquivo de topicos da pauta:

1. Le os embeddings dos topicos da pauta e dos discursos.
2. Calcula a matriz de similaridade cosseno usando `sentence_transformers.util.cos_sim`.
   - Linhas: topicos da pauta.
   - Colunas: topicos dos discursos.
3. Para cada topico da pauta, seleciona:
   - O topico de discurso com maior similaridade.
   - O valor dessa similaridade maxima.

O resultado e um registro por topico da pauta.

## Estrutura do CSV gerado

O CSV final tem as seguintes colunas:

- `agenda_topic`
  - ID do topico da pauta (numero inteiro).

- `agenda_terms`
  - Termos do topico da pauta, com pesos, no formato da modelagem (LDA ou BERTopic).

- `discourse_topic`
  - ID do topico de discurso mais similar ao topico da pauta.

- `discourse_terms`
  - Termos do topico de discurso mais similar, com pesos.

- `cosine_similarity`
  - Similaridade cosseno entre os embeddings dos dois topicos.
  - Varia de -1 a 1 (em pratica, geralmente de 0 a 1 se os embeddings sao normalizados).

## O que cada linha representa

Cada linha do CSV representa:

"Para o topico X da pauta, o topico de discurso mais semelhante foi o topico Y, com similaridade Z."

Ou seja:

- Existe 1 linha por topico da pauta.
- Nao ha repeticao de topicos da pauta.
- Um mesmo topico de discurso pode aparecer em varias linhas.

## Exemplo didatico

Imagine:

- Topicos da pauta (LDA):
  - Topico 0: "saude", "hospital", "medico"
  - Topico 1: "educacao", "escola", "professor"

- Topicos dos discursos (LDA):
  - Topico 3: "saude", "vacina", "hospital"
  - Topico 7: "educacao", "aluno", "professor"

A matriz de similaridade poderia ser algo assim:

```
            Discurso 3   Discurso 7
Pauta 0         0.88        0.22
Pauta 1         0.31        0.91
```

O CSV resultante seria:

```
agenda_topic,agenda_terms,discourse_topic,discourse_terms,cosine_similarity
0,"...saude...hospital...medico...",3,"...saude...vacina...hospital...",0.88
1,"...educacao...escola...professor...",7,"...educacao...aluno...professor...",0.91
```

Interpretacao:

- O topico 0 da pauta (saude) e mais parecido com o topico 3 dos discursos.
- O topico 1 da pauta (educacao) e mais parecido com o topico 7 dos discursos.

## Pontos de atencao e boas praticas

- Se um partido nao tiver embeddings de discursos, ele e ignorado.
- Se um arquivo de embeddings estiver vazio, ele e ignorado.
- O codigo nao tenta alinhar topicos pelo numero de topico, apenas por similaridade.
- A qualidade do matching depende diretamente dos embeddings e da modelagem de topicos.

## Resumo rapido

- Entrada: embeddings de topicos da pauta e dos discursos.
- Metodo: similaridade cosseno entre vetores.
- Saida: um CSV com o melhor par para cada topico da pauta.
- Uso principal: comparar o foco tematico da pauta partidaria com o foco tematico dos discursos.
