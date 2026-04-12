
import pandas as pd
import ast
import matplotlib.pyplot as plt
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from gensim.models import LdaModel
import os
import time
from pathlib import Path

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

############################### ###############################

############################### ###############################

# AUX FUNCTIONS:

# Função para calcular coerência para diferentes números de tópicos:
def compute_coherence_values(
    dictionary,
    corpus,
    texts,
    start=2,
    limit=13,
    step=2,
    processes=1,
    lda_passes=10,
    lda_iterations=100,
):
    print("...........................................")
    print("... Executando compute_coherence_values ...")
    
    coherence_values = []
    model_list = []
    total_models = len(range(start, limit, step))
    for idx, num_topics in enumerate(range(start, limit, step), start=1):
        t0 = time.perf_counter()
        print(f"... [{idx}/{total_models}] Testando {num_topics} tópicos ...")
        model = gensim.models.LdaModel(corpus=corpus,
                                       id2word=dictionary,
                                       num_topics=num_topics,
                                       random_state=42,
                                       chunksize=100,
                                       passes=lda_passes,
                                       iterations=lda_iterations,
                                       alpha='auto',
                                       per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model,
                                        texts=texts,
                                        dictionary=dictionary,
                                        coherence='c_v',
                                        processes=processes)
        coherence_values.append(coherencemodel.get_coherence())

        elapsed = time.perf_counter() - t0
        print(f"... [{idx}/{total_models}] concluído em {elapsed:.1f}s ...")
    return model_list, coherence_values

###############################

## Treina o modelo LDA ##
def LDA_train(NUM_TOPICS, dictionary, corpus, passes=30, iterations=300):
    print("... Função LDA_train iniciada! ...")
    return LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=NUM_TOPICS, # número de tópicos
        random_state=42,
        passes=passes,
        iterations=iterations,
        alpha='auto',
        per_word_topics=True
    )
    
############################### ###############################

############################### ###############################

# MAIN FUNCTION:

def topics_main(
    dataframe,
    partido=None,
    TOP_N=5,
    topic_start=2,
    topic_limit=16,
    topic_step=1,
    coherence_processes=1,
    search_passes=10,
    search_iterations=100,
    final_passes=30,
    final_iterations=300,
    output_base_dir="data/discourses/lda_files",
):
    print("....................................")
    print("... Função topics_main iniciada! ...")

    dataframe = dataframe.reset_index(drop=True).copy()

    # Filtro opcional por partido
    if partido is not None and str(partido).strip() != "":
        if "partido" not in dataframe.columns:
            raise ValueError("A coluna 'partido' não existe no dataframe para aplicar o filtro.")

        dataframe = dataframe[dataframe["partido"] == partido].reset_index(drop=True)
        if dataframe.empty:
            raise ValueError(f"Nenhum discurso encontrado para o partido '{partido}'.")

        partido_label = str(partido)
    else:
        partido_label = "TODOS"

    if dataframe.empty:
        raise ValueError("O dataframe de entrada está vazio após aplicar filtros.")

    print("... Partido: " + partido_label + " ...")
    print(f"... Total de discursos considerados: {len(dataframe)} ...")
    
    ### ### ###
    
    # SETUP #
    
    # Caso tokens esteja como string tipo "['a','b','c']"
    if isinstance(dataframe['tokens'].iloc[0], str):
        dataframe['tokens'] = dataframe['tokens'].apply(ast.literal_eval)

    # Criar o dicionário e o corpus a partir do dataframe fornecido:
    print("... Criando o dicionário e o corpus a partir do dataframe fornecido ...")
    texts = dataframe['tokens'].tolist()
    id2word = corpora.Dictionary(texts)
    corpus = [id2word.doc2bow(text) for text in texts]
    
    ### ### ###
    
    # NÚMERO DE TÓPICOS #
    print("... Identificando valor mais adequado de tópicos ...")
    # Roda compute_coherence_values para diferentes números de tópicos
    model_list, coherence_values = compute_coherence_values(dictionary=id2word,
                                                               corpus=corpus, 
                                                               texts=texts, 
                                                               start=topic_start,
                                                               limit=topic_limit,
                                                               step=topic_step,
                                                               processes=coherence_processes,
                                                               lda_passes=search_passes,
                                                               lda_iterations=search_iterations)

    x = range(topic_start, topic_limit, topic_step)

    # Visualizar os resultados
    plt.plot(x, coherence_values)
    plt.xlabel("Número de Tópicos")
    plt.ylabel("Coerência (C_v)")
    plt.title("Escolha do número ideal de tópicos")
    plt.show()


    # Ver o melhor valor
    selected_topic_num = 2
    current_top_coherence = 0
    for m, cv in zip(x, coherence_values):
        #print("Num Tópicos =", m, "Coerência =", round(cv, 4)) # <<<<<<<<<<<<<<<<
        if cv > current_top_coherence:
            current_top_coherence = cv
            selected_topic_num = m

    print("... A maior coerência identificada foi:",
          round(current_top_coherence,4),
          "quando usando",
          selected_topic_num,
          "tópicos! ...")

    ### ### ###

    print("... Treinando o modelo ...")
    lda_model = LDA_train(
        selected_topic_num,
        id2word,
        corpus,
        passes=final_passes,
        iterations=final_iterations,
    )
    for idx, topic in lda_model.print_topics(num_words=10):
        print(f"Tópico {idx}: {topic}")
        
    ### ### ###

    ## Documentos mais relevantes por tópico ##

    # Obter distribuição de tópicos para todos os docs
    print("... Obtendo distribuição de tópicos para todos os documentos ...")
    all_doc_topics = []
    for i, doc_bow in enumerate(corpus):
        doc_topics = lda_model.get_document_topics(doc_bow, minimum_probability=0)
        for topic_id, prob in doc_topics:
            all_doc_topics.append({
                "doc_id": i,
                "topic": topic_id,
                "probability": prob
            })

    df_doc_topics = pd.DataFrame(all_doc_topics)
    #print("--df_doc_topics")
    #print(df_doc_topics)

    ### ### ###

    top_docs_per_topic_n = pd.DataFrame(columns=['doc_id','topic','probability'])
    for topic in range(df_doc_topics["topic"].max() + 1):
        tmp_df = df_doc_topics[df_doc_topics["topic"] == topic]
        top_docs_per_topic_n = pd.concat( [top_docs_per_topic_n , tmp_df.sort_values("probability", ascending=False).head(TOP_N)] )    
    top_docs_per_topic_n = top_docs_per_topic_n.merge(dataframe[["preprocess_disc"]], left_on="doc_id", right_index=True, how="left")
    
    '''
    top_docs_per_topic_n = (
        df_doc_topics.sort_values(["topic", "probability"], ascending=[True, False])
        .groupby("topic")
        .head(TOP_N)
        .merge(dataframe[["preprocess_disc"]], left_on="doc_id", right_index=True)
    )
    '''
    #print("--top_docs_per_topic_n")
    #print(top_docs_per_topic_n)

    ### ### ###
    
    # SALVANDO CONTEÚDO #
    
    # Criando o diretório para os arquivos
    directory_path = Path(output_base_dir) / partido_label
    try:
        directory_path.mkdir(parents=True, exist_ok=True)
        print(f"Diretório '{directory_path}' pronto para uso!")
    except PermissionError:
        print(f"Negado: não foi possível criar: '{directory_path}'!")
        raise
    except Exception as e:
        print(f"ERRO!!!: {e}")
        raise
        
    # Salvar o modelo LDA treinado <<<<<
    lda_model.save(str(directory_path / "lda_model.model"))
    id2word.save(str(directory_path / "lda_dictionary.dict"))
    # Salvar documentos mais relevantes por tópico <<<<<
    top_docs_per_topic_n.to_csv(directory_path / "lda_topN_docs_por_topico.csv", index=False, encoding="utf-8")
    # Salvar distribuição de tópicos de todos os documentos
    df_doc_topics.to_csv(directory_path / "lda_distribuicao_docs.csv", index=False, encoding="utf-8")
    
    print("... Função topics_main encerrada! ...")
    print(".....................................")
    
    return top_docs_per_topic_n