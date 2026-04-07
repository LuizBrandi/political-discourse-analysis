
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re

#FILE_NAME = "running_files/political_discourses.csv"
#df = pd.read_csv(FILE_NAME)

###############################

# Cabeçalhos para simular um navegador real
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Referer": "https://www.camara.leg.br/",
    "Connection": "keep-alive"
}

# Criar sessão
s = requests.Session()
s.headers.update(headers)

###############################

def discourses_extract (dataframe, FILE_NAME):
    print("...........................................")
    print("... Função discourses_extract iniciada! ...")
    
    dataframe["raw_disc"] = None
    
    counter = 1
    for idx, tmp_url in dataframe["link_discurso"].items():
        print("... Extaíndo discurso da", str(counter) + "ª URL:", tmp_url, "...")
        counter += 1
        
        try:
            #print("... Acessando página ...")
            main_taq_page = s.get(tmp_url)
            #print("... Status:", main_taq_page.status_code, "...")
            main_taq_page.encoding = "utf-8"
            
            tmp_disc = BeautifulSoup(main_taq_page.text, features="html.parser").find('p').get_text(separator=" ", strip=True)
            
            # lidando com algumas palavras que foram extraídas fragmentadas: "Senhor" > "S enhor"
            # # "(?<=[^À-Ÿ\w\.])[b-df-hj-np-tv-z] \w*"   >>>>>   "(?<=[^À-Ÿà-ÿ\w\.])[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z] \w*"
            tmp_disc = re.sub(r"(?<=[^À-Ÿà-ÿ\w\.])[B-DF-HJ-NP-TV-Zb-df-hj-np-tv-z] \w*", lambda m: m.group(0).replace(" ", ""), tmp_disc)
            
            dataframe.at[idx, "raw_disc"] = tmp_disc
            
        except:
            dataframe.at[idx, "raw_disc"] = None
            print("... Bloqueado ou erro!!! ...")
    
    ###############################
    
    num_discs = re.search(r"\d*(?= discursos)" , FILE_NAME).group()
    dt_ini = re.search(r"(?<=ini )\d*" , FILE_NAME).group()
    dt_fim = re.search(r"(?<=fim )\d*" , FILE_NAME).group()
    extract_date = re.search(r"(?<=consulta )[\d\W]*" , FILE_NAME).group()
    
    
    
    file_name_02 = "discursos02RAW_{} discursos_ini {}_fim {}_consulta {}csv".format(num_discs,
                                                                              dt_ini,
                                                                              dt_fim,
                                                                              extract_date)
    
    running_file_name = "political_discourses_ini_{}_fim_{}.csv".format(dt_ini, dt_fim)

    print("... Salvando as informações! ...")
    dataframe.to_csv("data/backup/"+file_name_02,index=False)
    dataframe.to_csv("data/running_files/"+running_file_name,index=False)
    
    print("... Função discourses_extract encerrada! ...")
    print("............................................")
    
    return dataframe, file_name_02