# PRE PROCESSING OF POLICITAL AGENDA


##### STEPS #####

"""
    1) Converter tudo para minúsculas
    2) Remover stopwords
        vírgulas, pontos, "Nossos",
        "e", "o", 
        
        2.1) 
            a) Stopwords python library
            from nltk.corpus import stopwords
            stopwords.words('portuguese')
            b) stopwords_iso_pt.txt
            https://github.com/stopwords-iso/stopwords-pt
            
            # OBS: Testar com stopwords e sem stopwords
            
        
    3) Lematização
        Converter palavras em uma forma canonica
        ou em seu radical
    4) tokenizar com spacy e testar com outras tecnicas
    

"""