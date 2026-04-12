# Political Discourse Analysis

Project to analyze the coherence between Brazilian parliamentary speeches and party agendas using NLP, topic modeling, and semantic embeddings.

## Baseline (summary)

This baseline was defined as a minimum reproducible reference for comparing party agendas and political discourse, with initial validation on the **UNIÃO** party.

### Objective

Measure semantic alignment between:

- topics extracted from party agendas;
- topics and chunks from parliamentary speeches.

### Methodological approach

1. **Data ingestion**
	- Speeches collected from the Chamber of Deputies (2022 time window: 3 months before and 3 months after the electoral period).
	- Party agendas gathered from official sources (e.g., MDB, Novo, PL, PSOL, PT, and UNIÃO), with text extraction from PDFs.

2. **Text preprocessing**
	- Structural marker cleaning;
	- stopword removal;
	- text normalization;
	- lemmatization;
	- tokenization.

3. **Topic modeling (LDA)**
	- Identification of latent themes in speeches and agendas;
	- topic count selection based on coherence.

4. **Semantic embeddings**
	- Text segmentation into chunks;
	- embeddings with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.

5. **Coherence inference**
	- Cosine similarity between agenda topics and speech topics;
	- comparison between chunks of the most similar topics.

### Preliminary results (baseline)

- The baseline produced a functional pipeline for similarity calculation between agendas and speeches.
- Initial results showed alignment variation across topic pairs.
- Observed limitation: low thematic granularity of the agenda in some scenarios (few relevant topics), reducing comparison sensitivity.

## Environment

- Python **>= 3.13**
- Dependency manager: **uv**
- Dependencies declared in `pyproject.toml`

## How to run

### 1) Install dependencies

```bash
uv sync
```

### 2) Download required language resources

```bash
uv run python -m nltk.downloader stopwords
uv run python -m spacy download pt_core_news_lg
```

### 3) Run the full baseline

```bash
uv run python src/run_baseline.py \
  --agenda-party UNIAO \
  --agenda-txt data/party_agenda/party/UNIAO/txt/Manifesto_Uniao_BRASIL.layout_aware.txt \
  --discourse-party UNIAO \
  --discourse-csv data/discourses/political_discourses_ini_31102022_fim_02012023.csv
```

> Note: the speeches CSV must contain at least the columns `partido`, `preprocess_disc`, and `tokens`.

## Main outputs

- **Agenda topics**: `data/party_agenda/topics/<PARTIDO>/`
- **Agenda embeddings**: `data/party_agenda/embeddings/<PARTIDO>/`
- **Speech topics**: `data/discourses/lda_files/<PARTIDO>/`
- **Speech embeddings**: `data/discourses/embeddings/discourses/<PARTIDO>/`
- **Agenda vs. speeches similarity**:
  - `data/discourses/embeddings/discourses/<PARTIDO>/topicos_similaridade_pauta.csv`
  - `data/party_agenda/embeddings/<PARTIDO>/topicos_similaridade_discursos.csv`

## Baseline reference

Source document: `reports/02_baseline_political_discourse_analysis_guilherme_moura_luiz_brandi.pdf`
