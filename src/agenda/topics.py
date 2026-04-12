from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import gensim
import gensim.corpora as corpora
import pandas as pd
from gensim.models import CoherenceModel, LdaModel


def _parse_tokens(value: Any) -> list[str]:
	if isinstance(value, list):
		return [str(token) for token in value if str(token).strip()]

	if pd.isna(value):
		return []

	if isinstance(value, str):
		value = value.strip()
		if not value:
			return []

		try:
			parsed = ast.literal_eval(value)
			if isinstance(parsed, list):
				return [str(token) for token in parsed if str(token).strip()]
		except (ValueError, SyntaxError):
			pass

		return [token for token in value.split() if token.strip()]

	return []


def load_party_tokens_dataframe(
	pasta_tokens_base: str | Path,
	partido: str,
) -> pd.DataFrame:
	pasta_tokens_base = Path(pasta_tokens_base)
	pasta_csv = pasta_tokens_base / partido / "CSV"

	if not pasta_csv.exists():
		raise FileNotFoundError(f"Pasta de CSV não encontrada: {pasta_csv}")

	csv_files = sorted(pasta_csv.glob("*.csv"))
	if not csv_files:
		raise FileNotFoundError(f"Nenhum CSV encontrado em: {pasta_csv}")

	dataframes: list[pd.DataFrame] = []
	for csv_path in csv_files:
		df = pd.read_csv(csv_path)
		if "tokens" not in df.columns:
			raise ValueError(f"Coluna 'tokens' não encontrada em: {csv_path}")

		if "preprocess_agenda" not in df.columns:
			df["preprocess_agenda"] = ""

		df["source_file"] = csv_path.name
		dataframes.append(df[["source_file", "preprocess_agenda", "tokens"]].copy())

	resultado = pd.concat(dataframes, ignore_index=True)
	resultado["tokens"] = resultado["tokens"].apply(_parse_tokens)
	resultado = resultado[resultado["tokens"].map(len) > 0].reset_index(drop=True)

	if resultado.empty:
		raise ValueError(
			f"Nenhum documento com tokens válidos foi encontrado para o partido '{partido}'."
		)

	return resultado


def compute_coherence_values(
	dictionary: corpora.Dictionary,
	corpus: list[list[tuple[int, int]]],
	texts: list[list[str]],
	start: int = 2,
	limit: int = 13,
	step: int = 1,
	processes: int = 1,
	lda_passes: int = 10,
	lda_iterations: int = 100,
) -> tuple[list[LdaModel], list[float], list[int]]:
	print("...........................................")
	print("... Executando compute_coherence_values ...")

	coherence_values: list[float] = []
	model_list: list[LdaModel] = []
	topic_sizes = list(range(start, limit, step))

	total_models = len(topic_sizes)
	for idx, num_topics in enumerate(topic_sizes, start=1):
		t0 = time.perf_counter()
		print(f"... [{idx}/{total_models}] Testando {num_topics} tópicos ...")

		model = gensim.models.LdaModel(
			corpus=corpus,
			id2word=dictionary,
			num_topics=num_topics,
			random_state=42,
			chunksize=100,
			passes=lda_passes,
			iterations=lda_iterations,
			alpha="auto",
			per_word_topics=True,
		)
		model_list.append(model)

		for topic_id, topic_terms in model.print_topics(num_words=10):
			print(f"    Tópico {topic_id}: {topic_terms}")

		coherencemodel = CoherenceModel(
			model=model,
			texts=texts,
			dictionary=dictionary,
			coherence="c_v",
			processes=processes,
		)
		coherence_values.append(float(coherencemodel.get_coherence()))

		elapsed = time.perf_counter() - t0
		print(f"... [{idx}/{total_models}] concluído em {elapsed:.1f}s ...")

	return model_list, coherence_values, topic_sizes


def LDA_train(
	num_topics: int,
	dictionary: corpora.Dictionary,
	corpus: list[list[tuple[int, int]]],
	passes: int = 30,
	iterations: int = 300,
) -> LdaModel:
	print("... Função LDA_train iniciada! ...")
	return LdaModel(
		corpus=corpus,
		id2word=dictionary,
		num_topics=num_topics,
		random_state=42,
		passes=passes,
		iterations=iterations,
		alpha="auto",
		per_word_topics=True,
	)


def topics_main(
	dataframe: pd.DataFrame,
	partido: str,
	top_n: int = 5,
	topic_start: int = 2,
	topic_limit: int = 16,
	topic_step: int = 1,
	coherence_processes: int = 1,
	search_passes: int = 10,
	search_iterations: int = 100,
	final_passes: int = 30,
	final_iterations: int = 300,
	output_base_dir: str | Path = "data/party_agenda/topics",
) -> dict[str, Any]:
	print("....................................")
	print("... Função topics_main iniciada! ...")

	if dataframe.empty:
		raise ValueError("O dataframe de entrada está vazio.")

	partido_label = str(partido).strip()
	if not partido_label:
		raise ValueError("O parâmetro 'partido' deve ser informado.")

	working_df = dataframe.reset_index(drop=True).copy()
	if "tokens" not in working_df.columns:
		raise ValueError("A coluna 'tokens' não existe no dataframe.")

	if "preprocess_agenda" not in working_df.columns:
		working_df["preprocess_agenda"] = ""

	if "source_file" not in working_df.columns:
		working_df["source_file"] = "desconhecido"

	working_df["tokens"] = working_df["tokens"].apply(_parse_tokens)
	working_df = working_df[working_df["tokens"].map(len) > 0].reset_index(drop=True)

	if working_df.empty:
		raise ValueError("Nenhum documento com tokens válidos após o parse da coluna 'tokens'.")

	print(f"... Partido: {partido_label} ...")
	print(f"... Total de documentos considerados: {len(working_df)} ...")

	texts = working_df["tokens"].tolist()
	id2word = corpora.Dictionary(texts)
	corpus = [id2word.doc2bow(text) for text in texts]

	print("... Identificando valor mais adequado de tópicos ...")

	max_topics_by_docs = max(1, min(15, len(texts)))
	candidate_topics = [
		n for n in range(topic_start, topic_limit, topic_step) if n <= max_topics_by_docs
	]

	coherence_values: list[float] = []
	tested_topic_sizes: list[int] = []

	if len(texts) < 2 or not candidate_topics:
		selected_topic_num = 1
		current_top_coherence = None
		print("... Poucos documentos para busca por coerência; usando 1 tópico ...")
	else:
		_, coherence_values, tested_topic_sizes = compute_coherence_values(
			dictionary=id2word,
			corpus=corpus,
			texts=texts,
			start=min(candidate_topics),
			limit=max(candidate_topics) + topic_step,
			step=topic_step,
			processes=coherence_processes,
			lda_passes=search_passes,
			lda_iterations=search_iterations,
		)

		selected_topic_num = tested_topic_sizes[0]
		current_top_coherence = coherence_values[0]
		for topic_size, coherence in zip(tested_topic_sizes, coherence_values):
			if coherence > current_top_coherence:
				current_top_coherence = coherence
				selected_topic_num = topic_size

		print(
			"... A maior coerência identificada foi:",
			round(float(current_top_coherence), 4),
			"quando usando",
			selected_topic_num,
			"tópicos! ...",
		)

	print("... Treinando o modelo ...")
	lda_model = LDA_train(
		num_topics=selected_topic_num,
		dictionary=id2word,
		corpus=corpus,
		passes=final_passes,
		iterations=final_iterations,
	)

	topic_terms_rows: list[dict[str, Any]] = []
	for idx, topic in lda_model.print_topics(num_words=10):
		print(f"Tópico {idx}: {topic}")
		topic_terms_rows.append({"topic": idx, "terms": topic})
	topic_terms_df = pd.DataFrame(topic_terms_rows)

	print("... Obtendo distribuição de tópicos para todos os documentos ...")
	all_doc_topics: list[dict[str, Any]] = []
	for i, doc_bow in enumerate(corpus):
		doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
		for topic_id, prob in doc_topics:
			all_doc_topics.append(
				{
					"doc_id": i,
					"topic": int(topic_id),
					"probability": float(prob),
				}
			)

	df_doc_topics = pd.DataFrame(all_doc_topics)

	top_docs_per_topic_n = pd.DataFrame(columns=["doc_id", "topic", "probability"])
	for topic in range(int(df_doc_topics["topic"].max()) + 1):
		tmp_df = df_doc_topics[df_doc_topics["topic"] == topic]
		top_docs_per_topic_n = pd.concat(
			[top_docs_per_topic_n, tmp_df.sort_values("probability", ascending=False).head(top_n)]
		)

	top_docs_per_topic_n = top_docs_per_topic_n.merge(
		working_df[["source_file", "preprocess_agenda"]],
		left_on="doc_id",
		right_index=True,
		how="left",
	).reset_index(drop=True)

	output_dir = Path(output_base_dir) / partido_label
	output_dir.mkdir(parents=True, exist_ok=True)

	lda_model.save(str(output_dir / "lda_model.model"))
	id2word.save(str(output_dir / "lda_dictionary.dict"))
	top_docs_per_topic_n.to_csv(
		output_dir / "lda_topN_docs_por_topico.csv",
		index=False,
		encoding="utf-8",
	)
	df_doc_topics.to_csv(
		output_dir / "lda_distribuicao_docs.csv",
		index=False,
		encoding="utf-8",
	)
	topic_terms_df.to_csv(
		output_dir / "lda_topicos_termos.csv",
		index=False,
		encoding="utf-8",
	)

	if coherence_values:
		pd.DataFrame(
			{
				"num_topics": tested_topic_sizes,
				"coherence_c_v": coherence_values,
			}
		).to_csv(output_dir / "lda_coherence_scores.csv", index=False, encoding="utf-8")

	print(f"... Artefatos salvos em: {output_dir} ...")
	print("... Função topics_main encerrada! ...")
	print(".....................................")

	return {
		"selected_topic_num": selected_topic_num,
		"top_docs_per_topic_n": top_docs_per_topic_n,
		"doc_topics": df_doc_topics,
		"topic_terms": topic_terms_df,
		"coherence_scores": pd.DataFrame(
			{
				"num_topics": tested_topic_sizes,
				"coherence_c_v": coherence_values,
			}
		),
		"output_dir": str(output_dir),
	}