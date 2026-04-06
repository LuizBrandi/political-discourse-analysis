# Political Discourse Analysis

This project aims to analyze how coherent Brazilian parliamentary speeches are with each politician’s party program using NLP and text mining techniques.

The planned pipeline will:

- collect speeches from the Brazilian Chamber of Deputies API and party programs from official party sources;
- preprocess and segment texts by thematic units;
- represent text segments with semantic embeddings;
- compute coherence scores using cosine similarity;
- classify speech segments as coherent or incoherent with party guidelines.

The goal is to provide interpretable evidence of alignment (or misalignment) between political discourse and party agendas, supporting transparency, accountability, and future research in political communication.

## Pre-processing da agenda política

O script `pre_processing.py` executa as etapas de:

- conversão para minúsculas;
- remoção de stopwords (NLTK + lista adicional);
- lematização com spaCy (`pt_core_news_lg`);
- tokenização por arquivo `.txt`.

Saída:

- `data/preprocessing/tokenization/tokens/<Elemento>/<arquivo>_tokens.txt`

Exemplos de execução:

- Processar todos os Elementos:

	`python pre_processing.py`

- Processar apenas um Elemento (modo teste), ex.: UNIAO:

	`python pre_processing.py --elemento UNIAO`
