import os
import re
import importlib
from datetime import datetime
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

    sentences = re.split(r"(?<=[\.!?])\s+", clean_text)
    return [s.strip() for s in sentences if s and s.strip()]


def _read_text_file(file_path: str, encoding: str = "utf-8") -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    with open(file_path, "r", encoding=encoding) as file:
        return file.read()


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
        must_split_due_max = (
            max_sentences_per_chunk is not None and len(current_chunk_sentences) >= max_sentences_per_chunk
        )

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


def generate_text_embeddings(
    text: str,
    source_id: str = "text_input",
    similarity_threshold: float = 0.45,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    output_dir: str = "data/running_files",
    save_files: bool = True,
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    """
    Gera embeddings para um texto após segmentação semântica em chunks.

    Returns:
        embeddings_df: DataFrame com metadados e chunks.
        embeddings_matrix: Matriz numpy (n_chunks, embedding_dim).
        base_name: Nome base dos arquivos gerados (ou None se save_files=False).
    """
    print("............................................")
    print("... Função generate_text_embeddings iniciada ...")

    if not str(text).strip():
        print("... Texto vazio recebido ...")
        empty = pd.DataFrame(columns=["source_id", "chunk_id", "chunk_text"])
        return empty, np.empty((0, 0)), None

    model = _load_sentence_transformer(model_name)

    chunks = segment_text_semantic(
        text=text,
        model=model,
        similarity_threshold=similarity_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
    )

    if not chunks:
        print("... Nenhum chunk foi gerado ...")
        empty = pd.DataFrame(columns=["source_id", "chunk_id", "chunk_text"])
        return empty, np.empty((0, 0)), None

    embeddings_df = pd.DataFrame(
        {
            "source_id": [source_id] * len(chunks),
            "chunk_id": list(range(len(chunks))),
            "chunk_text": chunks,
        }
    )

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
        os.makedirs(output_dir, exist_ok=True)

        now = datetime.now().strftime("%Y%m%d_%H%M")
        safe_source_id = re.sub(r"[^\w\-.]", "_", source_id)
        base_name = f"agenda_embeddings_{safe_source_id}_{now}"

        csv_path = os.path.join(output_dir, f"{base_name}.csv")
        npy_path = os.path.join(output_dir, f"{base_name}.npy")

        embeddings_df.to_csv(csv_path, index=False)
        np.save(npy_path, embeddings_matrix)

        print(f"... Arquivo CSV salvo em: {csv_path} ...")
        print(f"... Arquivo NPY salvo em: {npy_path} ...")

    print("... Função generate_text_embeddings encerrada ...")
    print("..............................................")

    return embeddings_df, embeddings_matrix, base_name


def generate_agenda_embeddings_from_txt(
    txt_path: str,
    similarity_threshold: float = 0.45,
    min_sentences_per_chunk: int = 1,
    max_sentences_per_chunk: int | None = None,
    model_name: str = DEFAULT_MODEL_NAME,
    batch_size: int = 32,
    output_dir: str = "data/running_files",
    save_files: bool = True,
    encoding: str = "utf-8",
) -> tuple[pd.DataFrame, np.ndarray, str | None]:
    """
    Lê um arquivo .txt de agenda política e gera embeddings por chunk semântico.

    Args:
        txt_path: Caminho para o arquivo .txt.
        similarity_threshold: Limiar de similaridade para quebra de chunk.
        min_sentences_per_chunk: Mínimo de sentenças por chunk.
        max_sentences_per_chunk: Máximo de sentenças por chunk.
        model_name: Modelo SentenceTransformer.
        batch_size: Batch para geração dos embeddings.
        output_dir: Pasta para salvar arquivos.
        save_files: Se True, salva CSV + NPY.
        encoding: Codificação do arquivo texto.

    Returns:
        embeddings_df: DataFrame com metadados e chunks.
        embeddings_matrix: Matriz numpy (n_chunks, embedding_dim).
        base_name: Nome base dos arquivos gerados (ou None se save_files=False).
    """
    text = _read_text_file(txt_path, encoding=encoding)
    source_id = os.path.basename(txt_path)

    return generate_text_embeddings(
        text=text,
        source_id=source_id,
        similarity_threshold=similarity_threshold,
        min_sentences_per_chunk=min_sentences_per_chunk,
        max_sentences_per_chunk=max_sentences_per_chunk,
        model_name=model_name,
        batch_size=batch_size,
        output_dir=output_dir,
        save_files=save_files,
    )