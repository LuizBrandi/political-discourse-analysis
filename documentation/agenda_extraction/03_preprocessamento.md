# Etapa 3 — Pré-processamento e tokenização da agenda

## Propósito

Antes de aplicar modelos de tópicos (LDA), o texto bruto dos manifestos precisa ser normalizado e reduzido às suas unidades de significado. Esta etapa transforma os arquivos TXT da agenda em listas de **tokens** — palavras limpas, normalizadas e (opcionalmente) sem stopwords.

O resultado desta etapa é o dado de entrada direto para a **Etapa 6 (LDA)**. O BERTopic não usa estes tokens para treinar, mas a estrutura de pastas gerada aqui é usada para identificar os partidos disponíveis.

## O que é executado

A função `processar_todos_elementos` do módulo `src/agenda/pre_processing.py` é chamada com os parâmetros:

- `pasta_agenda_politica`: pasta onde estão os TXTs da agenda por partido.
- `pasta_saida_tokens`: pasta de destino dos tokens gerados.
- `elemento_teste`: se informado, processa apenas esse partido.
- `remove_stopwords`: se `True` (versão v1), remove palavras sem valor semântico (artigos, preposições etc.).
- `remove_accents`: se `True` (versão v2), normaliza acentuação.

## Versões de pré-processamento

| Versão | `remove_stopwords` | `remove_accents` | Pasta de saída |
|--------|--------------------|------------------|----------------|
| v1 | `True` | `False` | `tokens/` |
| v2 | `False` | `True` | `tokensV2/` |

A **versão v1** produz tokens mais "limpos" (sem stopwords), mais adequados para LDA, que se beneficia de vocabulários enxutos.

A **versão v2** mantém as stopwords mas remove acentos, útil para experimentos onde a frequência bruta de palavras importa mais.

## Dados de entrada

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/party/{PARTIDO}/txt/*.txt` | TXT | Textos extraídos dos PDFs na Etapa 2 |

## Dados produzidos

Para cada partido, dois tipos de arquivo são gerados em subpastas:

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `{pasta_tokens}/{PARTIDO}/TXT/*.txt` | TXT | Uma lista de tokens por linha, em formato de lista Python |
| `{pasta_tokens}/{PARTIDO}/CSV/*.csv` | CSV | Mesmo conteúdo em formato tabular com coluna `tokens` |

### Estrutura do CSV produzido

Cada linha do CSV corresponde a um **chunk de texto** (parágrafo ou seção) do manifesto:

| Coluna | Tipo | Descrição |
|--------|------|-----------|
| `tokens` | string | Lista de tokens em formato `['palavra1', 'palavra2', ...]` |
| `preprocess_agenda` | string | Metadado da versão de pré-processamento aplicada |

### Exemplo de conteúdo

```
tokens
"['brasil', 'social', 'encontro', 'país', 'público', 'nacional']"
"['responsabilidade', 'fiscal', 'desenvolvimento', 'sustentável']"
```

## Variável de controle relevante

| Variável | Efeito |
|----------|--------|
| `preprocessing_version` | Define qual conjunto de pastas é usado (tokens/ ou tokensV2/) |
| `elemento_teste` | Limita o processamento a um único partido |

## Relação com outras etapas

- Produz os dados que a **Etapa 5** (embeddings sem stopwords) e a **Etapa 6** (modelagem de tópicos) consomem.
- A **Etapa 6** (LDA) lê os CSVs produzidos aqui diretamente via `load_party_tokens_dataframe`.
- A **Etapa 5** lê os TXTs produzidos aqui para gerar embeddings a partir dos tokens.
