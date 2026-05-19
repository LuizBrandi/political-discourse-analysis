# Etapa 2 — Conversão de PDF para TXT (layout-aware)

## Propósito

Os manifestos e documentos de agenda política dos partidos são originalmente disponibilizados em PDF. Esta etapa converte cada PDF em um arquivo de texto simples (`.txt`), preservando ao máximo a ordem de leitura correta do documento — incluindo documentos com layout em múltiplas colunas.

O processo é chamado de **layout-aware** porque, ao invés de extrair o texto na ordem em que os blocos aparecem internamente no PDF (que pode misturar colunas), a etapa detecta se a página tem múltiplas colunas e ordena os blocos da esquerda para a direita, de cima para baixo, dentro de cada coluna.

## O que é executado

1. Varre recursivamente a pasta `data/party_agenda/party/` em busca de subpastas `pdf/` dentro de cada partido.
2. Para cada PDF encontrado, verifica se já existe um TXT correspondente na pasta `txt/` do mesmo partido.
3. Se o TXT não existe (ou se `force_rebuild_txt = True`), extrai o texto do PDF e salva o TXT.

### Como a extração funciona

- Para cada página do PDF, todos os blocos de texto são recuperados com suas coordenadas `(x, y)`.
- A função `_guess_column_split_x` analisa as posições horizontais dos blocos para detectar se a página tem duas colunas. Se o maior intervalo horizontal entre blocos for maior que 15% da largura da página, a página é tratada como bicoluna.
- Blocos da coluna esquerda são ordenados por `(y, x)` e concatenados; depois vêm os blocos da coluna direita, também ordenados por `(y, x)`.
- O texto final é limpo: quebras de linha múltiplas são reduzidas, hifenizações no final de linha são desfeitas, espaços extras são removidos.

## Variável de controle

| Variável | Valor padrão | Significado |
|----------|--------------|-------------|
| `force_rebuild_txt` | `False` | Se `True`, reconverte todos os PDFs mesmo que o TXT já exista. |

## Dados de entrada

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/party/{PARTIDO}/pdf/*.pdf` | PDF | Documentos de agenda política por partido |

A estrutura esperada de pastas:
```
data/party_agenda/party/
├── MDB/
│   ├── pdf/
│   │   └── Manifesto_MDB.pdf
│   └── txt/          ← criada automaticamente
├── UNIAO/
│   ├── pdf/
│   │   └── Manifesto_Uniao_BRASIL.pdf
│   └── txt/
...
```

## Dados produzidos

| Localização | Formato | Descrição |
|-------------|---------|-----------|
| `data/party_agenda/party/{PARTIDO}/txt/{nome}.txt` | TXT (UTF-8) | Texto extraído do PDF com ordem de leitura preservada |

O nome do arquivo TXT é idêntico ao nome do PDF, apenas com a extensão trocada para `.txt`.

## Relação com outras etapas

- **Etapa 3** (Pré-processamento) lê os TXTs gerados aqui para tokenizar o texto.
- **Etapa 4** (Embeddings por chunks) também lê os TXTs gerados aqui para criar os embeddings semânticos.

Sem os TXTs produzidos por esta etapa, as etapas 3 e 4 não têm dados de entrada.
