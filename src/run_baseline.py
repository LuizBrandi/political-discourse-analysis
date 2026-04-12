from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").exists():
            return candidate
    raise RuntimeError("Nao foi possivel localizar a raiz do projeto (pasta com ./src).")


def normalize_name(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).upper().strip()


def resolve_party_label(df: pd.DataFrame, party: str, column: str) -> str:
    target = normalize_name(party)
    candidates = df[column].dropna().astype(str).unique().tolist()
    for candidate in candidates:
        if normalize_name(candidate) == target:
            return candidate
    raise ValueError(f"Partido '{party}' nao encontrado na coluna '{column}'.")


def extract_terms(text: str) -> str:
    terms = re.findall(r"\"([^\"]+)\"", str(text))
    if terms:
        return " ".join(terms)
    cleaned = re.sub(r"\d+\.\d+\*", " ", str(text))
    return re.sub(r"\s+", " ", cleaned.replace("+", " ")).strip()


def build_topic_embeddings(terms_csv: Path, output_csv: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(terms_csv)
    df["terms_clean"] = df["terms"].apply(extract_terms)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["terms_clean"].tolist(), normalize_embeddings=True)
    df["embedding"] = [json.dumps(vec.tolist(), ensure_ascii=False) for vec in embeddings]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8")
    return df


def compute_similarity(
    agenda_embeddings_csv: Path,
    discourse_embeddings_csv: Path,
    output_csv: Path,
) -> pd.DataFrame:
    agenda_df = pd.read_csv(agenda_embeddings_csv)
    discourse_df = pd.read_csv(discourse_embeddings_csv)

    agenda_vecs = np.array([json.loads(v) for v in agenda_df["embedding"]])
    discourse_vecs = np.array([json.loads(v) for v in discourse_df["embedding"]])

    sim_matrix = util.cos_sim(agenda_vecs, discourse_vecs).cpu().numpy()

    rows: list[dict[str, object]] = []
    for i in range(sim_matrix.shape[0]):
        for j in range(sim_matrix.shape[1]):
            rows.append(
                {
                    "agenda_topic": int(agenda_df.loc[i, "topic"]),
                    "agenda_terms": agenda_df.loc[i, "terms"],
                    "discourse_topic": int(discourse_df.loc[j, "topic"]),
                    "discourse_terms": discourse_df.loc[j, "terms"],
                    "cosine_similarity": float(sim_matrix[i, j]),
                }
            )

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8")
    return result_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa os fluxos de agenda e discursos.")
    parser.add_argument(
        "--agenda-party",
        default="UNIAO",
        help="Partido da agenda (ex.: UNIAO).",
    )
    parser.add_argument(
        "--agenda-txt",
        default="data/party_agenda/party/UNIAO/txt/Manifesto_Uniao_BRASIL.layout_aware.txt",
        help="Caminho do .txt da agenda.",
    )
    parser.add_argument(
        "--discourse-party",
        default="UNIAO",
        help="Partido dos discursos (ex.: UNIAO).",
    )
    parser.add_argument(
        "--discourse-csv",
        default="data/discourses/political_discourses_ini_31102022_fim_02012023.csv",
        help="CSV com discursos preprocessados.",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Modelo SentenceTransformer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root(Path.cwd())

    import sys

    src_path = (project_root / "src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from agenda.pre_processing import processar_todos_elementos
    from agenda.embeddings import generate_agenda_embeddings_from_txt
    from agenda.topics import load_party_tokens_dataframe, topics_main as agenda_topics_main
    from discourses._04_topics import topics_main as discourse_topics_main
    from discourses._05_embeddings import generate_discourse_embeddings

    agenda_party = args.agenda_party
    discourse_party = args.discourse_party

    # Agenda: preprocessamento
    agenda_party_dir = project_root / "data" / "party_agenda" / "party"
    agenda_tokens_dir = project_root / "data" / "party_agenda" / "preprocessing" / "tokenization" / "tokens"
    agenda_tokens_dir.mkdir(parents=True, exist_ok=True)
    processar_todos_elementos(
        pasta_agenda_politica=agenda_party_dir,
        pasta_saida_tokens=agenda_tokens_dir,
        elemento_teste=agenda_party,
    )

    # Agenda: embeddings
    agenda_txt_path = project_root / args.agenda_txt
    agenda_embeddings_dir = project_root / "data" / "party_agenda" / "embeddings" / agenda_party
    agenda_embeddings_dir.mkdir(parents=True, exist_ok=True)
    generate_agenda_embeddings_from_txt(
        txt_path=str(agenda_txt_path),
        output_dir=str(agenda_embeddings_dir),
        save_files=True,
    )

    # Agenda: topicos
    agenda_df = load_party_tokens_dataframe(agenda_tokens_dir, agenda_party)
    agenda_topics_main(
        dataframe=agenda_df,
        partido=agenda_party,
        output_base_dir=project_root / "data" / "party_agenda" / "topics",
    )

    # Discursos: embeddings
    discourse_csv_path = project_root / args.discourse_csv
    df_discourse = pd.read_csv(discourse_csv_path)
    discourse_party_label = resolve_party_label(df_discourse, discourse_party, "partido")
    generate_discourse_embeddings(
        dataframe=df_discourse,
        party=discourse_party_label,
        text_col="preprocess_disc",
        save_files=True,
        source_csv_name=str(discourse_csv_path.name),
    )

    # Discursos: topicos
    discourse_topics_main(
        df_discourse,
        discourse_party_label,
        output_base_dir=project_root / "data" / "discourses" / "lda_files",
    )

    # Embeddings dos topicos (agenda)
    agenda_terms_csv = project_root / "data" / "party_agenda" / "topics" / agenda_party / "lda_topicos_termos.csv"
    agenda_topic_embeddings_csv = agenda_embeddings_dir / "topicos_embeddings_termos.csv"
    build_topic_embeddings(agenda_terms_csv, agenda_topic_embeddings_csv, args.model)

    # Embeddings dos topicos (discursos)
    discourse_terms_csv = project_root / "data" / "discourses" / "lda_files" / discourse_party_label / "lda_topicos_termos.csv"
    discourse_embeddings_dir = project_root / "data" / "discourses" / "embeddings" / "discourses" / discourse_party_label
    discourse_embeddings_dir.mkdir(parents=True, exist_ok=True)
    discourse_topic_embeddings_csv = discourse_embeddings_dir / "topicos_embeddings_termos.csv"
    build_topic_embeddings(discourse_terms_csv, discourse_topic_embeddings_csv, args.model)

    # Similaridade (tudo contra tudo)
    similarity_out = discourse_embeddings_dir / "topicos_similaridade_pauta.csv"
    compute_similarity(agenda_topic_embeddings_csv, discourse_topic_embeddings_csv, similarity_out)

    agenda_similarity_out = agenda_embeddings_dir / "topicos_similaridade_discursos.csv"
    compute_similarity(agenda_topic_embeddings_csv, discourse_topic_embeddings_csv, agenda_similarity_out)

    print("Fluxo concluido.")


if __name__ == "__main__":
    main()
