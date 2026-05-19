# Etapa 1 — Configuração inicial e variáveis de controle

## Propósito

Esta é a célula de entrada do notebook. Ela localiza a raiz do projeto, configura o `sys.path` para que os módulos de `src/` sejam importáveis, e define as variáveis de controle que determinam o comportamento de **todas as etapas seguintes**.

Alterar qualquer variável desta célula muda o caminho percorrido pelo pipeline inteiro.

## O que é executado

1. Localização automática da raiz do projeto (sobe pelos diretórios pais até encontrar a pasta `src/`).
2. Adição de `src/` ao `sys.path` para importação dos módulos internos.
3. Importação das funções principais dos módulos de agenda.
4. Definição das variáveis de controle listadas abaixo.

## Variáveis de controle

| Variável | Valores possíveis | Significado |
|----------|-------------------|-------------|
| `elemento_teste` | `"UNIAO"`, `"MDB"`, ..., `None` | Se informado, processa apenas esse partido. `None` processa todos. |
| `preprocessing_version` | `"v1"`, `"v2"` | Define a versão do pré-processamento de texto. |
| `topic_model` | `"lda"`, `"bertopic"` | Define o algoritmo de modelagem de tópicos. |
| `use_embeddings_without_stopwords` | `True`, `False` | Se `True`, o BERTopic usará os embeddings gerados a partir dos tokens sem stopwords (Etapa 5). Se `False`, usará os embeddings gerados diretamente dos chunks de texto (Etapa 4). |

### Detalhe sobre `preprocessing_version`

- **v1**: aplica remoção de stopwords, **não** remove acentos. Os tokens ficam em `data/party_agenda/preprocessing/tokenization/tokens/`.
- **v2**: **não** remove stopwords, remove acentos. Os tokens ficam em `data/party_agenda/preprocessing/tokenization/tokensV2/`.

> Quando `topic_model == "bertopic"`, a variável `preprocessing_version` é forçada para `"v1"` automaticamente, pois o BERTopic não usa as versões de tokenização da mesma forma que o LDA.

## Caminhos definidos

| Variável | Caminho |
|----------|---------|
| `pasta_agenda_politica` | `data/party_agenda/party/` |
| `pasta_saida_tokens` (v1) | `data/party_agenda/preprocessing/tokenization/tokens/` |
| `pasta_saida_tokens` (v2) | `data/party_agenda/preprocessing/tokenization/tokensV2/` |

## Dados de entrada

Nenhum dado de entrada é lido nesta etapa. Apenas configurações são definidas.

## Dados produzidos

Nenhum arquivo é gerado. O resultado desta etapa são variáveis em memória que guiam todas as etapas seguintes.

## Relação com outras etapas

Todas as etapas dependem desta. As variáveis `preprocessing_version`, `topic_model` e `use_embeddings_without_stopwords` definem quais caminhos de arquivo serão lidos e escritos nas etapas 3 a 8.
