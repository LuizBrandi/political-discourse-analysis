
from src.discourses._01_list_extract import list_extract
from src.discourses._02_discourses_extract import discourses_extract
from src.discourses._03_discourse_preprocessing import preprocessing
from src.discourses._04_topics import topics_main
from src.discourses._05_embeddings import generate_discourse_embeddings
# from _05_llm_analysis import llm_analysis

import pandas as pd
from gensim.models import LdaModel

import os
import shutil

SOURCE_CSV_NAME = "political_discourses_ini_02072022_fim_29102022.csv"

########################################################

# >>> TLDR: <<<

# ETAPA 1, 2 e 3
df, name = preprocessing ( *discourses_extract ( *list_extract("09/05/2025", "09/09/2025") ) )

# # ETAPA 4 para o partido selecionado
topics_main (df, "UNIAO")

# ETAPA 5 para um partido específico
embeddings_df, embeddings_matrix, base_name = generate_discourse_embeddings(df, party="UNIÃO", source_csv_name=SOURCE_CSV_NAME)

