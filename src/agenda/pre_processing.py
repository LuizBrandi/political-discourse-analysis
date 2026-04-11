from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import spacy
from nltk.corpus import stopwords


STOPWORDS_ADICIONAIS = [
    "senhor",
    "senhora",
    "senhores",
    "senhoras",
    "sr",
    "sra",
    "srs",
    "sras",
    "presidente",
    "presidenta",
    "deputado",
    "deputada",
    "deputados",
    "deputadas",
    "de",
    "do",
    "da",
    "e",
    "o",
    "a",
    "os",
    "as",
    "um",
    "uma",
    "nossos",
]


def carregar_stopwords() -> set[str]:
    try:
        stopwords_pt = set(stopwords.words("portuguese"))
    except LookupError as exc:
        raise RuntimeError(
            "Corpus stopwords do NLTK não encontrado. "
            "Execute: python -m nltk.downloader stopwords"
        ) from exc

    stopwords_pt.update(STOPWORDS_ADICIONAIS)
    return stopwords_pt


def carregar_modelo_spacy() -> spacy.language.Language:
    try:
        return spacy.load("pt_core_news_lg")
    except OSError as exc:
        raise RuntimeError(
            "Modelo spaCy 'pt_core_news_lg' não encontrado. "
            "Execute: python -m spacy download pt_core_news_lg"
        ) from exc


def preprocess_text(
    texto: str,
    nlp: spacy.language.Language,
    stopwords_pt: set[str],
) -> tuple[str, list[str]]:
    texto = str(texto)

    texto = texto.lower()
    texto = re.sub(r"[^a-zA-ZÀ-ÿ0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()

    termos_sem_stopwords = [token for token in texto.split() if token not in stopwords_pt]
    texto_sem_stopwords = " ".join(termos_sem_stopwords)

    doc = nlp(texto_sem_stopwords)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if (token.is_alpha or token.is_digit) and token.lemma_.strip() and token.lemma_.lower() not in stopwords_pt
    ]
    preprocess_agenda = " ".join(tokens)
    return preprocess_agenda, tokens


def processar_arquivo_txt(
    arquivo_txt: Path,
    pasta_saida_txt: Path,
    pasta_saida_csv: Path,
    nlp: spacy.language.Language,
    stopwords_pt: set[str],
) -> Path:
    texto = arquivo_txt.read_text(encoding="utf-8", errors="ignore")
    preprocess_agenda, tokens = preprocess_text(texto=texto, nlp=nlp, stopwords_pt=stopwords_pt)

    nome_saida = f"{arquivo_txt.stem}_tokens.txt"
    arquivo_saida = pasta_saida_txt / nome_saida
    arquivo_saida.write_text("\n".join(tokens), encoding="utf-8")

    nome_csv = f"{arquivo_txt.stem}_preprocess.csv"
    arquivo_csv = pasta_saida_csv / nome_csv
    with arquivo_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["preprocess_agenda", "tokens"])
        writer.writeheader()
        writer.writerow(
            {
                "preprocess_agenda": preprocess_agenda,
                "tokens": json.dumps(tokens, ensure_ascii=False),
            }
        )

    return arquivo_saida


def processar_elemento(
    pasta_elemento: Path,
    pasta_saida_tokens: Path,
    nlp: spacy.language.Language,
    stopwords_pt: set[str],
) -> int:
    pasta_txt = pasta_elemento / "txt"
    if not pasta_txt.exists():
        print(f"[AVISO] Pasta não encontrada: {pasta_txt}")
        return 0

    arquivos_txt = sorted(pasta_txt.glob("*.txt"))
    if not arquivos_txt:
        print(f"[AVISO] Nenhum arquivo .txt em {pasta_txt}")
        return 0

    pasta_saida_elemento = pasta_saida_tokens / pasta_elemento.name
    pasta_saida_elemento.mkdir(parents=True, exist_ok=True)
    pasta_saida_txt = pasta_saida_elemento / "TXT"
    pasta_saida_csv = pasta_saida_elemento / "CSV"
    pasta_saida_txt.mkdir(parents=True, exist_ok=True)
    pasta_saida_csv.mkdir(parents=True, exist_ok=True)

    print(f"\n[Elemento] {pasta_elemento.name}")
    processados = 0
    for arquivo_txt in arquivos_txt:
        arquivo_saida = processar_arquivo_txt(
            arquivo_txt=arquivo_txt,
            pasta_saida_txt=pasta_saida_txt,
            pasta_saida_csv=pasta_saida_csv,
            nlp=nlp,
            stopwords_pt=stopwords_pt,
        )
        processados += 1
        print(f"  - {arquivo_txt.name} -> {arquivo_saida.name}")

    return processados


def processar_todos_elementos(
    pasta_agenda_politica: Path,
    pasta_saida_tokens: Path,
    elemento_teste: str | None,
) -> None:
    stopwords_pt = carregar_stopwords()
    nlp = carregar_modelo_spacy()

    if elemento_teste:
        pasta_elemento = pasta_agenda_politica / elemento_teste
        if not pasta_elemento.exists() or not pasta_elemento.is_dir():
            raise ValueError(
                f"Elemento '{elemento_teste}' não encontrado em {pasta_agenda_politica}."
            )
        total = processar_elemento(
            pasta_elemento=pasta_elemento,
            pasta_saida_tokens=pasta_saida_tokens,
            nlp=nlp,
            stopwords_pt=stopwords_pt,
        )
        print(f"\nConcluído: {total} arquivo(s) processado(s) para o elemento {elemento_teste}.")
        return

    pastas_elementos = sorted([p for p in pasta_agenda_politica.iterdir() if p.is_dir()])
    total_arquivos = 0
    for pasta_elemento in pastas_elementos:
        total_arquivos += processar_elemento(
            pasta_elemento=pasta_elemento,
            pasta_saida_tokens=pasta_saida_tokens,
            nlp=nlp,
            stopwords_pt=stopwords_pt,
        )

    print(f"\nConcluído: {total_arquivos} arquivo(s) processado(s) no total.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pré-processa documentos .txt da agenda política e gera arquivos de tokens "
            "em data/preprocessing/tokenization/tokens/<Elemento>."
        )
    )
    parser.add_argument(
        "--elemento",
        type=str,
        default=None,
        help=(
            "Nome da pasta em data/political agenda para processar apenas um Elemento "
            "(ex.: UNIAO)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    raiz_projeto = Path(__file__).resolve().parent
    pasta_agenda_politica = raiz_projeto / "data" / "political agenda"
    pasta_saida_tokens = (
        raiz_projeto / "data" / "preprocessing" / "tokenization" / "tokens"
    )
    pasta_saida_tokens.mkdir(parents=True, exist_ok=True)

    processar_todos_elementos(
        pasta_agenda_politica=pasta_agenda_politica,
        pasta_saida_tokens=pasta_saida_tokens,
        elemento_teste=args.elemento,
    )


if __name__ == "__main__":
    main()