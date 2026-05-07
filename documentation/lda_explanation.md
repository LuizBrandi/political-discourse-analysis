# LDA - Exemplos rapidos

Este arquivo mostra exemplos simples de como ler os resultados do LDA e interpretar os CSVs.

## Exemplo 1: Ver topicos e termos principais
Arquivo: lda_topicos_termos.csv

Colunas esperadas:
- topic: id do topico (inteiro)
- terms: string com termos e pesos (formato do Gensim)

Exemplo de linha:
- topic: 0
- terms: "0.012*\"economia\" + 0.010*\"emprego\" + 0.009*\"renda\""

Como ler:
- As palavras com maior peso representam o tema central do topico.

## Exemplo 2: Ver distribuicao de topicos por documento
Arquivo: lda_distribuicao_docs.csv

Colunas esperadas:
- doc_id: indice do documento
- topic: id do topico
- probability: peso do topico naquele documento

Exemplo de interpretacao:
- Documento 12 tem 0.62 no topico 3 -> o topico 3 e dominante.

## Exemplo 3: Encontrar documentos mais representativos
Arquivo: lda_topN_docs_por_topico.csv

Colunas esperadas:
- doc_id
- topic
- probability
- source_file
- preprocess_agenda

Como usar:
- Para cada topico, olhe os maiores valores de probability.
- Abra o source_file correspondente para ver o texto original.

## Exemplo 4: Escolha de numero de topicos
Arquivo: lda_coherence_scores.csv

Colunas esperadas:
- num_topics
- coherence_c_v

Como ler:
- Quanto maior a coerencia, melhor tende a ser a separacao dos topicos.
- Use como guia, nao como regra fixa.

## Dicas praticas
- Comece sempre por lda_topicos_termos.csv.
- Em seguida, use lda_topN_docs_por_topico.csv para achar exemplos reais.
- Use lda_distribuicao_docs.csv para analises quantitativas.
