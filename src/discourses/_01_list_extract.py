
import requests
from bs4 import BeautifulSoup
import pandas as pd
import html
from datetime import datetime
import re
import math
from urllib.parse import urljoin, urlencode, urlparse, parse_qs
import os

###############################################

### SETUP ###
# Número de elementos retornados por página
PAGE_SIZE = 550 # tamanhos superiores a 1350 tendem a dar problema

base_url = "https://www.camara.leg.br/internet/sitaqweb/resultadoPesquisaDiscursos.asp?txIndexacao=&CurrentPage={}&BasePesq=plenario&txOrador=&txPartido=&dtInicio={}&dtFim={}&txUF=&txSessao=&listaTipoSessao=&listaTipoInterv=&inFalaPres=&listaTipoFala=&listaFaseSessao=&txAparteante=&listaEtapa=&CampoOrdenacao=dtSessao&TipoOrdenacao=DESC&PageSize={}&txTexto=&txSumario="

# CRAWLER SETUP:
# Cabeçalhos para simular um navegador real
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/128.0.0.0 Safari/537.36",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Referer": "https://www.camara.leg.br/",
    "Connection": "keep-alive"
}

BASIC_URL = "https://www.camara.leg.br/internet/sitaqweb/"

###############################################

def list_extract (DATA_INI, DATA_FIM, PAGE_SIZE = PAGE_SIZE):
    print(".....................................")
    print("... Função list_extract iniciada! ...")
    
    ### FUNCTION SETUP ###
    # Período da consulta
    #DATA_INI = "30/06/2025" # "09/07/2025"
    #DATA_FIM = "1/07/2025" # "17/08/2025"
    
    main_url = base_url.format(1, DATA_INI, DATA_FIM, PAGE_SIZE)
    print("... URL base: ", main_url, "...")
    
    # Criar sessão
    s = requests.Session()
    s.headers.update(headers)
    
    # Momento da consulta
    AGORA = datetime.now().strftime('%Y%m%d %H%M')
    
    # Acessa o link desejado
    print("... Acessando página! ...")
    main_taq_page = s.get(main_url)
    main_taq_page.encoding = "utf-8"
    
    print("... Status:", main_taq_page.status_code, "...")
    
    if main_taq_page.status_code == 200:
        print("... Sucesso!!! ...")
        #print(main_taq_page.text)
    else:
        print("... Bloqueado ou erro!!! ...")
    
    ###############################################
    
    TOTAL_DISC = re.search(r"(?<=\"TotalRecords\" value=\")\d*" , main_taq_page.text)
    
    print("... O número total de discursos encontrados é:", TOTAL_DISC.group(), "...")
    
    total_pages = math.ceil( int(TOTAL_DISC.group()) / PAGE_SIZE )
    
    print("... O número total de páginas é:", total_pages, "...")
    
    ###############################################
    
    # tbody é a parte da página que contém as informações sobre TODOS os discursos
    # tr são as sub-caixas que contém as infos sobre CADA discurso (de dois em dois tr)
    
    # Informações sobre o discurso
    elems_taq = BeautifulSoup(main_taq_page.text, features="html.parser").find('tbody')
    elems_taq = [tr for tr in elems_taq.find_all("tr") if not tr.get("id")]
    
    # Sumário do discurso
    elems_sum = BeautifulSoup(main_taq_page.text, features="html.parser").find('tbody')
    elems_sum = [tr for tr in elems_sum.find_all("tr") if tr.get("id")]
    
    # PARA OS CASOS EM QUE A PESQUISA RETORNE MAIS DE UMA PÁGINA
    if total_pages > 1:
        print("... Acessando páginas adicionais! ...")
    
        for page_num in range(1,total_pages):
            current_page = page_num + 1
            print("... Adicionando os discursos da página: ...", current_page)
                    
            tmp_url = base_url.format(current_page, DATA_INI, DATA_FIM, PAGE_SIZE)
            tmp_taq_page = s.get(tmp_url)
            tmp_taq_page.encoding = "utf-8"
            
            tmp_elems_taq = BeautifulSoup(tmp_taq_page.text, features="html.parser").find('tbody')
            tmp_elems_taq = [tr for tr in tmp_elems_taq.find_all("tr") if not tr.get("id")]
            elems_taq = elems_taq + tmp_elems_taq
            
            tmp_elems_sum = BeautifulSoup(tmp_taq_page.text, features="html.parser").find('tbody')
            tmp_elems_sum = [tr for tr in tmp_elems_sum.find_all("tr") if tr.get("id")]
            elems_sum = elems_sum + tmp_elems_sum
    
    ###############################################
    
    print("... Formatando dados extraídos! ...")
    dados = []
    for tr, tr_sum in zip(elems_taq, elems_sum):
        tds = tr.find_all('td')
        if not tds:  # ignora se não tiver colunas
            continue
        
        # Extrai o link para o discurso
        link_tag = tds[3].find('a')
        link = None
        if link_tag and link_tag.get("href"):
            raw_href = html.unescape(link_tag["href"])  # garante que "&amp;" virem "&"
            parsed = urlparse(raw_href)
    
            # quebra query em dict (cada valor vem como lista)
            query_dict = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    
            # remonta URL absoluta com encoding correto
            link = urljoin(BASIC_URL, f"{parsed.path}?{urlencode(query_dict)}")
    
        # Nome e partido (ex: "Chico Alencar, PSOL-RJ")
        nome_partido = tds[5].get_text(strip=True)
        uf_partido = None
        partido = None
        if "," in nome_partido:
            nome, partido_raw = [p.strip() for p in nome_partido.split(",", 1)]
    
            # separa partido de UF (ex: "PT-RJ" -> PT / RJ)
            if "-" in partido_raw:
                partido, uf_partido = partido_raw.split("-", 1)
                partido, uf_partido = partido.strip(), uf_partido.strip()
            else:
                partido = partido_raw.strip()
        else:
            nome = nome_partido
        
        # Sumário
        tmp_soup = BeautifulSoup(str( tr_sum ), "html.parser")
        tmp_suma = tmp_soup.find("td", {"class": "Sumario"}).get_text(strip=True)
    
        item = {
            "parlamentar": nome,                          # Chico Alencar
            "partido": partido,                           # PT
            "uf_partido": uf_partido,                     # RJ
            "sessao": tds[1].get_text(strip=True),        # 150.2025
            "fase": tds[2].get_text(strip=True),          # ORDEM DO DIA
            "data": tds[0].get_text(strip=True),          # 14/08/2025
            "hora": tds[6].get_text(strip=True),          # 11h04
            "publicacao": tds[7].get_text(strip=True),    # DCD 15/08/2025
            "link_discurso": link,                        # URL completa e encodada
            "sumario": tmp_suma,                          # Sumário
        }
        dados.append(item)
    
    ###############################################
    
    df = pd.DataFrame(dados)
    #df
    
    file_name_01 = "discursos01LISTA_{} discursos_ini {}_fim {}_pagesize {}_consulta {}.csv".format(TOTAL_DISC.group(),
                                                                              re.sub("/","",DATA_INI),
                                                                              re.sub("/","",DATA_FIM),
                                                                              PAGE_SIZE,
                                                                              AGORA)
    print("... Salvando as informações! ...")
    if not os.path.exists("data/backup"):
        os.makedirs("data/backup")
    if not os.path.exists("data/running_files"):
        os.makedirs("data/running_files")
    running_file_name = "political_discourses_ini_{}_fim_{}.csv".format(
        re.sub("/","",DATA_INI),
        re.sub("/","",DATA_FIM)
    )
    df.to_csv("data/backup/"+file_name_01,index=False)
    df.to_csv("data/running_files/"+running_file_name,index=False)
        
    print("... Função list_extract encerrada! ...")
    print("......................................")
    
    return df, file_name_01