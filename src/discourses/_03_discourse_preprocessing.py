
import pandas as pd
import spacy
import re
import nltk

try:
    from nltk.corpus import stopwords
    stopwords.words('portuguese')
except LookupError:
    nltk.download('stopwords')

try:
    nlp = spacy.load("pt_core_news_lg")
except OSError:
    import spacy.cli
    spacy.cli.download("pt_core_news_lg")
    nlp = spacy.load("pt_core_news_lg")

#FILE_NAME = "discursosRAW_138 discursos_ini 30062025_fim 1072025_consulta 20250913 1332.csv"
#df = pd.read_csv(FILE_NAME)

###############################

stopwords_pt = stopwords.words("portuguese")
stopwords_adicionais = ["senhor", "senhora", "senhores", "senhoras", "sr", "sra", "srs", "sras",
                        "presidente", "presidenta", "deputado","deputada","deputados","deputadas",
                        "de","do","da", "aqui","ali", "este","esse","nesse","neste","isso","aquilo","aquele","em","no","na",
                        "ir", "estar", "fazer", "dizer", "falar", "querer", "precisar", "obrigado", "agradeço", "porque", "ter", "poder",
                        "n"]
stopwords_pt = stopwords_pt + stopwords_adicionais


### INSTALAR VIA >>> python -m spacy download pt_core_news_lg ###
nlp = spacy.load("pt_core_news_lg")

###############################

def preprocess_steps (tmp_str):
    #print("... Função preprocess_steps iniciada! ...")
    
    #print(tmp_str)
    #print('\n')
    tmp_str = str(tmp_str)
    
    # Remover a identificação do atual falante ex: "O Sr. RICARDO AYRES (REPUBLICANOS-TO) pronuncia o seguinte discurso"
    tmp_str = re.sub(r'(?<=[Ss][Rr].\s)[A-ZÀ-Ÿ\s]*(?=\()', '', tmp_str)
    tmp_str = re.sub(r'(?<=[Ss][Rr][Aa].\s)[A-ZÀ-Ÿ\s]*(?=\()', '', tmp_str)
    
    # Remover nota presente em algumas notas
    tmp_str = re.sub(r'DISCURSO NA ÍNTEGRA ENCAMINHADO.*\(.*\)\.', '', tmp_str)
    
    # Remover trechos entre ()
    tmp_str = re.sub(r'\(.*\)', '', tmp_str)    
    
    # Identificação de entidades nomeadas compostas
    doc = nlp(tmp_str)
    new_tokens = []
    i = 0
    while i < len(doc):
        token = doc[i]
        if token.ent_iob_ == "B":  # início de entidade
            ent = [token.text]
            j = i + 1
            while j < len(doc) and doc[j].ent_iob_ == "I":
                ent.append(doc[j].text)
                j += 1
            new_tokens.append("_".join(ent))
            i = j
        else:
            new_tokens.append(token.text)
            i += 1
    tmp_str = " ".join(new_tokens)
    
    # Substituir qualquer "_" que não esteja entre duas r"\w".
    tmp_str = re.sub(r'(?<!\w)_(?!\w)', '', tmp_str)
    
    # Remover caracteres específicos
    #print("... Remoção de caracteres específicos ...")
    tmp_str = re.sub(r'[^a-zA-ZÀ-Ÿà-ÿ0-9\s_]', '', tmp_str)

    # Converter tudo para minúsculas
    #print("... Conversão para minúsculas ...")
    tmp_str = tmp_str.lower()

    # Remover stopwords
    #print("... 1ª Remoção de stopwords ...")
    tmp_str = ' '.join([word for word in tmp_str.split() if word not in stopwords_pt])

    # Lematização
    #print("... Lematização ...")
    
    doc = nlp(tmp_str)
    #lemmas = [token.lemma_.lower() for token in doc if token.is_alpha or token.is_digit]
    lemmas = [token.lemma_.lower() for token in doc if token.is_alpha or token.is_digit or "_" in token.text]
    tmp_str = ' '.join([word for word in lemmas])
    #'''
    
    # Remover stopwords 2
    #print("... 2ª Remoção de stopwords ...")
    tmp_str = ' '.join([word for word in tmp_str.split() if word not in stopwords_pt])
    
    # Substituir qualquer "_" que não esteja entre duas r"\w" 2.
    tmp_str = re.sub(r'(?<!\w)_(?!\w)', '', tmp_str)
    
    #print("... Função preprocess_steps encerrada! ...")
    return (tmp_str)

###############################

def preprocessing (dataframe, FILE_NAME):
    print("......................................")
    print("... Função preprocessing iniciada! ...")

    dataframe['preprocess_disc'] = dataframe['raw_disc'].apply(preprocess_steps)
    dataframe["tokens"] = dataframe["preprocess_disc"].apply(lambda x: x.split())

    ###############################

    num_discs = re.search(r"\d*(?= discursos)" , FILE_NAME).group()
    dt_ini = re.search(r"(?<=ini )\d*" , FILE_NAME).group()
    dt_fim = re.search(r"(?<=fim )\d*" , FILE_NAME).group()
    extract_date = re.search(r"(?<=consulta )[\d\W]*" , FILE_NAME).group()

    file_name_03 = "discursos03PREPROCESS_{} discursos_ini {}_fim {}_consulta {}csv".format(num_discs,
                                                                              dt_ini,
                                                                              dt_fim,
                                                                              extract_date)

    running_file_name = "political_discourses_ini_{}_fim_{}.csv".format(dt_ini, dt_fim)

    print("... Salvando as informações! ...")
    dataframe.to_csv("data/backup/"+file_name_03,index=False)
    dataframe.to_csv("data/running_files/"+running_file_name,index=False)
    
    print("... Função preprocessing encerrada! ...")
    print(".......................................")
    
    return dataframe, file_name_03