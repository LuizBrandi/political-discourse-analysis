import os
import re
import importlib
from datetime import datetime
from typing import Iterable
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _load_sentence_transformer(model_name: str):
    try:
        st_module = importlib.import_module("sentence_transformers")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pacote 'sentence-transformers' não encontrado. "
            "Instale com: pip install sentence-transformers"
        ) from exc

    SentenceTransformer = getattr(st_module, "SentenceTransformer")
    return SentenceTransformer(model_name)


def _split_sentences(text: str) -> list[str]:
    """Quebra um texto em sentenças usando pontuação final básica."""
    clean_text = re.sub(r"\s+", " ", str(text)).strip()
    if not clean_text:
        return []

    # Mantém delimitadores de sentença em português (. ! ?)
    sentences = re.split(r"(?<=[\.!?])\s+", clean_text)
    return [s.strip() for s in sentences if s and s.strip()]


def segment_text_semantic(
    text: str,
    model: Any,
    similarity_threshold: float = 0.45,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int | None = None,
) -> list[str]:
    """
    Segmenta um texto em chunks semânticos.

    A estratégia compara a próxima sentença com o centróide do chunk atual.
    Se a similaridade cair abaixo do threshold, inicia um novo chunk.
    """
    sentences = _split_sentences(text)

    if not sentences:
        return []
    if len(sentences) == 1:
        return sentences

    sent_embeddings = model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)

    try:
        st_module = importlib.import_module("sentence_transformers")
        util = getattr(st_module, "util")
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pacote 'sentence-transformers' não encontrado. "
            "Instale com: pip install sentence-transformers"
        ) from exc

    chunks: list[list[str]] = [[sentences[0]]]
    chunk_embs: list[list[np.ndarray]] = [[sent_embeddings[0]]]

    for i in range(1, len(sentences)):
        sentence = sentences[i]
        emb = sent_embeddings[i]

        current_chunk_sentences = chunks[-1]
        current_chunk_embs = chunk_embs[-1]

        chunk_centroid = np.mean(np.vstack(current_chunk_embs), axis=0)
        sim = float(util.cos_sim(emb, chunk_centroid).item())

        must_keep_due_min = len(current_chunk_sentences) < min_sentences_per_chunk
        must_split_due_max = max_sentences_per_chunk is not None and len(current_chunk_sentences) >= max_sentences_per_chunk

        if must_split_due_max:
            chunks.append([sentence])
            chunk_embs.append([emb])
        elif sim < similarity_threshold and not must_keep_due_min:
            chunks.append([sentence])
            chunk_embs.append([emb])
        else:
            current_chunk_sentences.append(sentence)
            current_chunk_embs.append(emb)

    return [" ".join(chunk).strip() for chunk in chunks if chunk]


def _normalize_party_filter(party: str | Iterable[str] | None) -> set[str] | None:
    if party is None:
        return None
    if isinstance(party, str):
        return {party.strip().upper()}
    return {str(p).strip().upper() for p in party}


def _extract_period_from_discourse_filename(source_csv_name: str | None) -> tuple[str | None, str | None]:
    """
    Extrai o período (ini/fim) de nomes no formato:
    political_discourses_ini_02072022_fim_29102022.csv
    """
    if not source_csv_name:
        return None, None

    file_name = os.path.basename(str(source_csv_name))
    match = re.search(r"ini_(\d{8})_fim_(\d{8})", file_name)
    if not match:
        return None, None

    dt_ini, dt_fim = match.group(1), match.group(2)
    return dt_ini, dt_fim


def generate_discourse_embeddings(
    dataframe: pd.DataFrame,
    party: str | Iterable[str] | None = None,
    source_csv_name: str | None = None,
    text_col: str = "preprocess_disc",
    party_col: str = "partido",
    similarity_threshold: float = 0.35,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    output_dir: str = "data/running_files/embeddings",
    save_files: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    """
    Gera embeddings dos discursos após segmentação semântica em chunks.

    Args:
        dataframe: DataFrame com os discursos.
        party: Partido (str), lista de partidos ou None para todos.
        source_csv_name: Nome/caminho do CSV de origem dos discursos para
            extrair período (ini/fim) e usar no nome dos arquivos de saída.
        text_col: Nome da coluna que contém o texto para segmentação.
        party_col: Nome da coluna de partido.
        similarity_threshold: Limiar de similaridade para quebra de chunk.
        min_sentences_per_chunk: Mínimo de sentenças por chunk.
        max_sentences_per_chunk: Máximo de sentenças por chunk.
        model_name: Modelo SentenceTransformer.
        batch_size: Batch para geração dos embeddings.
        output_dir: Pasta para salvar arquivos.
        save_files: Se True, salva CSV + NPY.

    Returns:
        embeddings_df: DataFrame com metadados e chunks.
        embeddings_matrix: Matriz numpy (n_chunks, embedding_dim).
        base_name: Nome base dos arquivos gerados (ou None se save_files=False).
    """
    if text_col not in dataframe.columns:
        raise ValueError(f"Coluna '{text_col}' não encontrada no DataFrame.")
    if party_col not in dataframe.columns:
        raise ValueError(f"Coluna '{party_col}' não encontrada no DataFrame.")

    print("...................................................")
    print("... Função generate_discourse_embeddings iniciada ...")

    model = _load_sentence_transformer(model_name)

    party_filter = _normalize_party_filter(party)
    df = dataframe.copy()

    if party_filter is not None:
        normalized_party = df[party_col].astype(str).str.upper().str.strip()
        df = df[normalized_party.isin(party_filter)].copy()

    df = df[df[text_col].notna()].copy()

    if df.empty:
        print("... Nenhum discurso encontrado para os filtros informados ...")
        empty = pd.DataFrame(columns=["source_index", "chunk_id", "chunk_text", "partido"])
        return empty, np.empty((0, 0)), None

    records = []
    for idx, row in df.iterrows():
        chunks = segment_text_semantic(
            text=row[text_col],
            model=model,
            similarity_threshold=similarity_threshold,
            min_sentences_per_chunk=min_sentences_per_chunk,
            max_sentences_per_chunk=max_sentences_per_chunk,
        )

        for chunk_id, chunk_text in enumerate(chunks):
            item = row.to_dict()
            item.update(
                {
                    "source_index": idx,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                }
            )
            records.append(item)

    embeddings_df = pd.DataFrame(records)

    if embeddings_df.empty:
        print("... Nenhum chunk foi gerado ...")
        return embeddings_df, np.empty((0, 0)), None

    print(f"... Total de chunks gerados: {len(embeddings_df)} ...")
    embeddings_matrix = model.encode(
        embeddings_df["chunk_text"].tolist(),
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

    embeddings_df["embedding"] = [vec.tolist() for vec in embeddings_matrix]

    base_name = None
    if save_files:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        party_name = "ALL" if party_filter is None else "_".join(sorted(party_filter))
        dt_ini, dt_fim = _extract_period_from_discourse_filename(source_csv_name)

        if dt_ini and dt_fim:
            period_part = f"ini_{dt_ini}_fim_{dt_fim}"
        else:
            now = datetime.now().strftime("%Y%m%d_%H%M")
            period_part = f"periodo_desconhecido_{now}"

        base_name = f"discourses_embeddings_{party_name}_{period_part}"

        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        npy_path = os.path.join(output_dir, f"{base_name}.npy")

        embeddings_df.to_csv(csv_path, index=False)
        np.save(npy_path, embeddings_matrix)

        print(f"... Arquivo CSV salvo em: {csv_path} ...")
        print(f"... Arquivo NPY salvo em: {npy_path} ...")

    print("... Função generate_discourse_embeddings encerrada ...")
    print("....................................................")

    return embeddings_df, embeddings_matrix, base_name
