
# # Previsão de Vendas de Smartphones 

# Importando blibliotecas do projeto 

import re
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    balanced_accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, confusion_matrix,
)
from xgboost import XGBRegressor, XGBClassifier

BASE = r"C:\Users\kelvim\OneDrive\Desktop\planilhas atualizadas tcc" 


#Carregando o dataset para o python e consolidando planilhas 

# Arquivos reais: 2022="terminais", 2023="terminais 2023",
# 2024="DADOS TERMINAIS 2024", 2025="TERMINAIS " .


sheets = {
    2022: (f"{BASE}\\2022.xlsx", "terminais"),
    2023: (f"{BASE}\\2023.xlsx", "terminais 2023"),
    2024: (f"{BASE}\\2024.xlsx", "DADOS TERMINAIS 2024"),
    2025: (f"{BASE}\\2025.xlsx", "TERMINAIS "),
}

#  Dados pessoais de clientes, devo remover (LGPD)
colunas_pii = ["Nome Cliente", "CPF/CNPJ", "Endereço", "E-mail", "Chave de Acesso",
               "Nº de Acesso", "Nº Comprovante Fiscal"]

# Unificando nomes de colunas, para nao ter problema com duplicatas depois 

mapa_colunas = {
    "Categoria Tipo": "CATEGORIA", "CATEGORIA": "CATEGORIA",
    "Subcategoria": "SUBCATEGORIA_GERENCIAL", "SUBCATEGORIA_GERENCIAL": "SUBCATEGORIA_GERENCIAL",
}

colunas_finais = ["Filial", "UF", "CATEGORIA", "SUBCATEGORIA_GERENCIAL", "Status", "Nº Venda",
                   "Marca", "Modelo Comercial", "Modelo DPGC", "SKU", "Data de Emissão da Nota",
                   "Cor", "Serial", "Filial de Compra", "Fornecedor", "Promoção",
                   "Data da compra", "Data da venda", "Hora da venda", "Qtde.",
                   "Valor de Custo", "Valor de Custo Líquido", "Valor Produto",
                   "Valor de Venda", "Condição Pagamento"]

dfs = []
for ano, (caminho, aba) in sheets.items():
    df_ano = pd.read_excel(caminho, sheet_name=aba)
    df_ano = df_ano.drop(columns=[c for c in colunas_pii if c in df_ano.columns], errors="ignore")
    df_ano = df_ano.rename(columns=mapa_colunas)
    colunas_presentes = [c for c in colunas_finais if c in df_ano.columns]
    df_ano = df_ano[colunas_presentes].copy()
    df_ano["ANO_PLANILHA"] = ano
    dfs.append(df_ano)
    print(f"{ano}: {df_ano.shape[0]} linhas, {df_ano.shape[1]} colunas")

df = pd.concat(dfs, ignore_index=True)
print("\nTotal consolidado:", df.shape)

#2. Limpeza dos dados

# 2.1 Padronizar nomes de colunas


df.columns = (
    df.columns.str.strip()
    .str.normalize("NFKD")
    .str.encode("ascii", errors="ignore")
    .str.decode("utf-8")
    .str.upper()
    .str.replace(" ", "_")
)
print(df.columns.tolist())

# 2.2 Conversão de tipos


df["DATA_DA_VENDA"] = pd.to_datetime(df["DATA_DA_VENDA"], errors="coerce")
df["DATA_DE_EMISSAO_DA_NOTA"] = pd.to_datetime(df["DATA_DE_EMISSAO_DA_NOTA"], errors="coerce")
df["DATA_DA_COMPRA"] = pd.to_datetime(df["DATA_DA_COMPRA"], errors="coerce")

colunas_numericas = ["QTDE.", "VALOR_DE_CUSTO", "VALOR_DE_CUSTO_LIQUIDO",
                      "VALOR_PRODUTO", "VALOR_DE_VENDA"]
for col in colunas_numericas:
    if df[col].dtype == object:
        df[col] = (df[col].astype(str)
                   .str.replace(".", "", regex=False)
                   .str.replace(",", ".", regex=False))
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 2.3 Remover nulos e inválidos e duplicatas 
## Tem alguns valores nulos, por serem uma quantidade pequena, irei descartar os valores


n_antes = len(df)

df = df.dropna(subset=["DATA_DA_VENDA", "VALOR_DE_VENDA", "QTDE."])
print(f"Removidas por nulo em data/valor/qtde: {n_antes - len(df)}")

print("\nDistribuição de STATUS antes do filtro de qtde/valor:")
print(df["STATUS"].value_counts())

n_antes = len(df)
df = df[df["QTDE."] > 0]
df = df[df["VALOR_DE_VENDA"] > 0]
print(f"Removidas por qtde/valor <= 0: {n_antes - len(df)}")

n_antes = len(df)
df = df.drop_duplicates()
print(f"Removidas por duplicata exata: {n_antes - len(df)}")

print("\nNulos restantes por coluna:")
print(df.isnull().sum().sort_values(ascending=False))
print("\nLinhas após limpeza básica:", len(df))

#2.4 Cor não deve entrar em nenhuma etapa da modelagem, ira atrapalhar remover  a coluna


df = df.drop(columns=["COR"], errors="ignore")



# #  3. Apos limpeza comecar a tratar os dados, sera utilizado somente smartphones, invalidar demais dados 

# 3.1 Isolar smartphones (a base "terminais" também traz Box/Modem/FWT/Bem-Estar)


n_antes = len(df)
cat_normalizada = df["CATEGORIA"].astype(str).str.upper().str.strip()
cat_vazia = df["CATEGORIA"].isna()
marca_smartphone = df["MARCA"].astype(str).str.upper().str.strip().isin(["APPLE", "SAMSUNG", "MOTOROLA"])
print(f"Recuperadas com categoria vazia + marca Apple/Samsung/Motorola: {(cat_vazia & marca_smartphone).sum()}")
df = df[(cat_normalizada == "SMARTPHONE") | (cat_vazia & marca_smartphone)].copy()
print(f"Removidas por categoria != SMARTPHONE: {n_antes - len(df)}")
print("Linhas após filtrar só smartphones:", len(df))
print("Modelos únicos:", df["MODELO_COMERCIAL"].nunique())

# 3.2 Utilizar somente  às marcas trabalhadas atualmente (Apple, Samsung, Motorola)
# Outras marcas ficam sinalizadas (MARCA_ATIVA=False) e não descartadas do bruto, 
#fazer isso porque atualmente a empresa trabalha  apenas com as tres
# demais marcas não entram na base de treino/previsão pois foram descontinuadas .


df["MARCA"] = df["MARCA"].astype(str).str.upper().str.strip()
df["MARCA_ATIVA"] = df["MARCA"].isin(["APPLE", "SAMSUNG", "MOTOROLA"])

print("Distribuição de marcas (histórico completo):")
print(df["MARCA"].value_counts())
print(f"\nLinhas com marca ativa (Apple/Samsung/Motorola): {df['MARCA_ATIVA'].sum()} de {len(df)}")

df_hist = df.copy()         
df = df[df["MARCA_ATIVA"]].copy()   
print("Linhas após restringir a Apple/Samsung/Motorola:", len(df))

# 4. Tabela de datas de lançamento 
# adicionar tabela com datas de lnaçamento para validar tempo 
# # Lista com uma linha por modelo, montada a partir dos comunicados oficiais de
# lançamento da Apple, Samsung e Motorola e conferida manualmente.



dados_lancamento = pd.DataFrame([
    {"Marca": "Apple", "Modelo": "iPhone 18", "Ano": 2026, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 18 Pro", "Ano": 2026, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 18 Max", "Ano": 2026, "Mes": "Setembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A57", "Ano": 2026, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A37", "Ano": 2026, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy S26", "Ano": 2026, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S26+", "Ano": 2026, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S26 Ultra", "Ano": 2026, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A17", "Ano": 2026, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Edge 70 Pro", "Ano": 2026, "Mes": "Maio"},
    {"Marca": "Motorola", "Modelo": "Edge 70 Ultra", "Ano": 2026, "Mes": "Maio"},
    {"Marca": "Motorola", "Modelo": "Motorola Signature", "Ano": 2026, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G17", "Ano": 2026, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G67", "Ano": 2026, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G77", "Ano": 2026, "Mes": "Janeiro"},
    {"Marca": "Apple", "Modelo": "iPhone 17", "Ano": 2025, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 17 Pro", "Ano": 2025, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 17 Max", "Ano": 2025, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 16e", "Ano": 2025, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A56", "Ano": 2025, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A36", "Ano": 2025, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy S25", "Ano": 2025, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S25+", "Ano": 2025, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S25 Ultra", "Ano": 2025, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A26", "Ano": 2025, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A16", "Ano": 2025, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G86 5G", "Ano": 2025, "Mes": "Julho"},
    {"Marca": "Motorola", "Modelo": "Edge 60 Pro", "Ano": 2025, "Mes": "Maio"},
    {"Marca": "Motorola", "Modelo": "Edge 60 Ultra", "Ano": 2025, "Mes": "Maio"},
    {"Marca": "Motorola", "Modelo": "Moto G56 5G", "Ano": 2025, "Mes": "Maio"},
    {"Marca": "Apple", "Modelo": "iPhone 16", "Ano": 2024, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 16 Pro", "Ano": 2024, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 16 Max", "Ano": 2024, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto G85", "Ano": 2024, "Mes": "Junho"},
    {"Marca": "Motorola", "Modelo": "Edge 50 Pro", "Ano": 2024, "Mes": "Abril"},
    {"Marca": "Motorola", "Modelo": "Edge 50 Fusion", "Ano": 2024, "Mes": "Abril"},
    {"Marca": "Motorola", "Modelo": "Edge 50 Ultra", "Ano": 2024, "Mes": "Abril"},
    {"Marca": "Samsung", "Modelo": "Galaxy A55", "Ano": 2024, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A35", "Ano": 2024, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy S24", "Ano": 2024, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S24+", "Ano": 2024, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S24 Ultra", "Ano": 2024, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A15", "Ano": 2024, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G24", "Ano": 2024, "Mes": "Janeiro"},
    {"Marca": "Apple", "Modelo": "iPhone 15", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 15 Pro", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 15 Max", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto G84", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Razr 40 Ultra", "Ano": 2023, "Mes": "Junho"},
    {"Marca": "Motorola", "Modelo": "Edge 40", "Ano": 2023, "Mes": "Maio"},
    {"Marca": "Samsung", "Modelo": "Galaxy A54", "Ano": 2023, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A34", "Ano": 2023, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy S23", "Ano": 2023, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S23+", "Ano": 2023, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S23 Ultra", "Ano": 2023, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A14", "Ano": 2023, "Mes": "Janeiro"},
    {"Marca": "Apple", "Modelo": "iPhone 14", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 14 Pro", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 14 Max", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto G62", "Ano": 2022, "Mes": "Junho"},
    {"Marca": "Motorola", "Modelo": "Moto G82", "Ano": 2022, "Mes": "Maio"},
    {"Marca": "Motorola", "Modelo": "Moto G52", "Ano": 2022, "Mes": "Abril"},
    {"Marca": "Apple", "Modelo": "iPhone SE (3ª Ger)", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A53", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A33", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Motorola", "Modelo": "Moto G22", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy S22", "Ano": 2022, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S22+", "Ano": 2022, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S22 Ultra", "Ano": 2022, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Edge 30 Pro", "Ano": 2022, "Mes": "Fevereiro"},
    {"Marca": "Apple", "Modelo": "iPhone 13", "Ano": 2021, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 13 Pro", "Ano": 2021, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 13 Max", "Ano": 2021, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto G60", "Ano": 2021, "Mes": "Abril"},
    {"Marca": "Samsung", "Modelo": "Galaxy A52", "Ano": 2021, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A52s", "Ano": 2021, "Mes": "Março"},
    {"Marca": "Motorola", "Modelo": "Moto G100", "Ano": 2021, "Mes": "Março"},
    {"Marca": "Motorola", "Modelo": "Moto G30", "Ano": 2021, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy S21", "Ano": 2021, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S21+", "Ano": 2021, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S21 Ultra", "Ano": 2021, "Mes": "Janeiro"},
    {"Marca": "Apple", "Modelo": "iPhone 12", "Ano": 2020, "Mes": "Outubro"},
    {"Marca": "Apple", "Modelo": "iPhone 12 Pro", "Ano": 2020, "Mes": "Outubro"},
    {"Marca": "Apple", "Modelo": "iPhone 12 Max", "Ano": 2020, "Mes": "Outubro"},
    {"Marca": "Motorola", "Modelo": "Moto G9 Plus", "Ano": 2020, "Mes": "Setembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy Note 20", "Ano": 2020, "Mes": "Agosto"},
    {"Marca": "Samsung", "Modelo": "Galaxy Note 20 Ultra", "Ano": 2020, "Mes": "Agosto"},
    {"Marca": "Motorola", "Modelo": "Moto G9 Play", "Ano": 2020, "Mes": "Agosto"},
    {"Marca": "Apple", "Modelo": "iPhone SE (2ª Ger)", "Ano": 2020, "Mes": "Abril"},
    {"Marca": "Samsung", "Modelo": "Galaxy S20", "Ano": 2020, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S20+", "Ano": 2020, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S20 Ultra", "Ano": 2020, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G8 Power", "Ano": 2020, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A51", "Ano": 2020, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A71", "Ano": 2020, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G8 Play", "Ano": 2019, "Mes": "Outubro"},
    {"Marca": "Motorola", "Modelo": "Moto G8 Plus", "Ano": 2019, "Mes": "Outubro"},
    {"Marca": "Apple", "Modelo": "iPhone 11", "Ano": 2019, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 11 Pro", "Ano": 2019, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 11 Max", "Ano": 2019, "Mes": "Setembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S10", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S10e", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S10+", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A50", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A30", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A10", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G7", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G7 Plus", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G7 Power", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G7 Play", "Ano": 2019, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G42", "Ano": 2022, "Mes": "Junho"},
    {"Marca": "Motorola", "Modelo": "Moto G53", "Ano": 2023, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto E13", "Ano": 2023, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto G54", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto G35", "Ano": 2024, "Mes": "Outubro"},
    {"Marca": "Motorola", "Modelo": "Moto G75", "Ano": 2024, "Mes": "Outubro"},
    {"Marca": "Motorola", "Modelo": "Moto G55", "Ano": 2024, "Mes": "Outubro"},
    {"Marca": "Motorola", "Modelo": "Moto G34", "Ano": 2024, "Mes": "Janeiro"},
    {"Marca": "Motorola", "Modelo": "Moto G15", "Ano": 2024, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Moto G31", "Ano": 2021, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Moto G71", "Ano": 2021, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Moto G200", "Ano": 2021, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Moto E22", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto E20", "Ano": 2021, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto E40", "Ano": 2021, "Mes": "Outubro"},
    {"Marca": "Motorola", "Modelo": "Moto E7 Power", "Ano": 2021, "Mes": "Fevereiro"},
    {"Marca": "Motorola", "Modelo": "Moto E7", "Ano": 2020, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Moto E7 Plus", "Ano": 2020, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Moto G9 Power", "Ano": 2020, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Moto G 5G", "Ano": 2020, "Mes": "Dezembro"},
    {"Marca": "Motorola", "Modelo": "Edge 20", "Ano": 2021, "Mes": "Agosto"},
    {"Marca": "Motorola", "Modelo": "Edge 20 Pro", "Ano": 2021, "Mes": "Agosto"},
    {"Marca": "Motorola", "Modelo": "Edge 20 Lite", "Ano": 2021, "Mes": "Agosto"},
    {"Marca": "Motorola", "Modelo": "Edge 30 Neo", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Edge 30 Fusion", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Edge 30 Ultra", "Ano": 2022, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Edge 40 Neo", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Edge 50 Neo", "Ano": 2024, "Mes": "Setembro"},
    {"Marca": "Motorola", "Modelo": "Edge 60 Fusion", "Ano": 2025, "Mes": "Abril"},
    {"Marca": "Samsung", "Modelo": "Galaxy A13", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A32", "Ano": 2021, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A22", "Ano": 2021, "Mes": "Junho"},
    {"Marca": "Samsung", "Modelo": "Galaxy A12", "Ano": 2020, "Mes": "Novembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A23", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A73", "Ano": 2022, "Mes": "Março"},
    {"Marca": "Samsung", "Modelo": "Galaxy A25", "Ano": 2023, "Mes": "Dezembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A03", "Ano": 2021, "Mes": "Novembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A03 Core", "Ano": 2021, "Mes": "Novembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A03s", "Ano": 2021, "Mes": "Agosto"},
    {"Marca": "Samsung", "Modelo": "Galaxy A02", "Ano": 2021, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A02s", "Ano": 2020, "Mes": "Novembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A04e", "Ano": 2022, "Mes": "Outubro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A05", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A06", "Ano": 2025, "Mes": "Fevereiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy A10s", "Ano": 2019, "Mes": "Agosto"},
    {"Marca": "Samsung", "Modelo": "Galaxy A21s", "Ano": 2020, "Mes": "Maio"},
    {"Marca": "Samsung", "Modelo": "Galaxy S20 FE", "Ano": 2020, "Mes": "Setembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S21 FE", "Ano": 2022, "Mes": "Janeiro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S23 FE", "Ano": 2023, "Mes": "Outubro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S24 FE", "Ano": 2024, "Mes": "Outubro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S25 FE", "Ano": 2025, "Mes": "Setembro"},
    {"Marca": "Samsung", "Modelo": "Galaxy S25 Edge", "Ano": 2025, "Mes": "Maio"},
    {"Marca": "Samsung", "Modelo": "Galaxy Z Flip 3", "Ano": 2021, "Mes": "Agosto"},
    {"Marca": "Samsung", "Modelo": "Galaxy Z Flip 4", "Ano": 2022, "Mes": "Agosto"},
    {"Marca": "Samsung", "Modelo": "Galaxy Z Flip 5", "Ano": 2023, "Mes": "Agosto"},
    {"Marca": "Samsung", "Modelo": "Galaxy Z Flip 6", "Ano": 2024, "Mes": "Julho"},
    {"Marca": "Samsung", "Modelo": "Galaxy Z Flip 7", "Ano": 2025, "Mes": "Julho"},
    {"Marca": "Samsung", "Modelo": "Galaxy Z Fold 7", "Ano": 2025, "Mes": "Julho"},
    {"Marca": "Apple", "Modelo": "iPhone 12 Mini", "Ano": 2020, "Mes": "Novembro"},
    {"Marca": "Apple", "Modelo": "iPhone 13 Mini", "Ano": 2021, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone 14 Plus", "Ano": 2022, "Mes": "Outubro"},
    {"Marca": "Apple", "Modelo": "iPhone 15 Plus", "Ano": 2023, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone Air", "Ano": 2025, "Mes": "Setembro"},
    {"Marca": "Apple", "Modelo": "iPhone XR", "Ano": 2018, "Mes": "Outubro"},
], columns=["Marca", "Modelo", "Ano", "Mes"])

meses_pt = {
    "Janeiro": 1, "Fevereiro": 2, "Março": 3, "Abril": 4, "Maio": 5, "Junho": 6,
    "Julho": 7, "Agosto": 8, "Setembro": 9, "Outubro": 10, "Novembro": 11, "Dezembro": 12
}
dados_lancamento["Mes_Num"] = dados_lancamento["Mes"].map(meses_pt)
dados_lancamento["DATA_LANCAMENTO"] = pd.to_datetime(
    dict(year=dados_lancamento["Ano"], month=dados_lancamento["Mes_Num"], day=1)
)
dados_lancamento_final = dados_lancamento[["Marca", "Modelo", "DATA_LANCAMENTO"]]
dados_lancamento_final.to_excel(f"{BASE}\\datas_lancamento_exploded.xlsx", index=False)
print(f"{len(dados_lancamento_final)} modelos na tabela de lançamento curada")

# 5. Fuzzy matching entre nome de venda e tabela de lançamento
# remover marca ,armazenamento, cor, 4G/5G e códigos entre parênteses não atrapalhar a comparação 


def canon(nome):
    
    n = str(nome).upper()
    n = re.sub(r"\([^)]*\)", " ", n)
    n = re.sub(r"\d+\s*(GB|TB)\b", " ", n)
    n = re.sub(r"\s*-\s*[A-ZÀ-Ú ]+$", " ", n)
    n = re.sub(r"\b(SAMSUNG|APPLE|MOTOROLA|XIAOMI)\b", " ", n)
    n = re.sub(r"\b[45]G\b", " ", n)
    n = re.sub(r"[^A-Z0-9+ ]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    n = re.sub(r"^IPHONE (\d+) MAX$", r"IPHONE \1 PRO MAX", n)   
    return ALIAS.get(n, n)

# nomes diferentes que estava esquecendo de validar 
ALIAS = {"GALAXY S21 PLUS": "GALAXY S21+", "GALAXY S20+ BTS": "GALAXY S20+", "GALAXY A176B": "GALAXY A17"}

def canon_lancamento(nome):
    nome = str(nome).replace("(2ª Ger)", "2 GER").replace("(3ª Ger)", "")
    return canon(nome)

tabela = dados_lancamento_final.assign(CANON=dados_lancamento_final["Modelo"].map(canon_lancamento))
repetidos = tabela[tabela["CANON"].duplicated(keep=False)]
if len(repetidos):
    raise ValueError(f"Nomes repetidos na tabela de lançamento: {repetidos.to_string()}")
nome_oficial = dict(zip(tabela["CANON"], tabela["Modelo"]))
data_oficial = dict(zip(tabela["CANON"], tabela["DATA_LANCAMENTO"]))


df["CANON"] = df["MODELO_COMERCIAL"].map(canon)
df["Modelo_Lancamento_Sugerido"] = df["CANON"].map(nome_oficial)
df["DATA_LANCAMENTO"] = df["CANON"].map(data_oficial)

df_correspondencias = (df.groupby("MODELO_COMERCIAL")
                         .agg(Nome_Aparelho=("CANON", "first"),
                              Modelo_Lancamento=("Modelo_Lancamento_Sugerido", "first"),
                              Unidades=("QTDE.", "sum"))
                         .reset_index())
df_correspondencias.to_csv(f"{BASE}\\correspondencias_nomes.csv", index=False, sep=";", encoding="utf-8-sig")
sem_data = df_correspondencias[df_correspondencias["Modelo_Lancamento"].isna()]
print(f"Nomes de venda: {len(df_correspondencias)} | sem data de lançamento na tabela: {len(sem_data)} "
      f"({sem_data['Unidades'].sum():.0f} unidades) -> usam a data da primeira venda")
print(sem_data.to_string())

fallback = df.groupby("MODELO_COMERCIAL")["DATA_DA_VENDA"].min().rename("DATA_LANCAMENTO_FALLBACK")
df = df.merge(fallback, on="MODELO_COMERCIAL", how="left")
df["LANCAMENTO_APROXIMADO"] = df["DATA_LANCAMENTO"].isna()
df["DATA_LANCAMENTO"] = df["DATA_LANCAMENTO"].fillna(df["DATA_LANCAMENTO_FALLBACK"])
df = df.drop(columns=["DATA_LANCAMENTO_FALLBACK"])

pct_aprox = df["LANCAMENTO_APROXIMADO"].mean() * 100
print(f"Registros usando fallback por primeira venda: {pct_aprox:.1f}%")
print(f"Modelos usando fallback: {df.loc[df['LANCAMENTO_APROXIMADO'], 'MODELO_COMERCIAL'].nunique()}")

# # Junta as variações de cor e armazenamento de um mesmo celular em um nome só 
#  para que as vendas de cada aparelho não fiquem divididas.

n_bruto = df["MODELO_COMERCIAL"].nunique()
df["MODELO_COMERCIAL_ORIGINAL"] = df["MODELO_COMERCIAL"]
df["MODELO_COMERCIAL"] = df["Modelo_Lancamento_Sugerido"].fillna(df["CANON"].str.title())
n_canonico = df["MODELO_COMERCIAL"].nunique()
print(f"Modelos únicos: {n_bruto} (string bruta) -> {n_canonico} (nome canônico consolidado)")


# atribuindo valores ao ciclo de vida 


df["ANO"] = df["DATA_DA_VENDA"].dt.year
df["MES"] = df["DATA_DA_VENDA"].dt.month
df["DIA_SEMANA"] = df["DATA_DA_VENDA"].dt.dayofweek
df["TRIMESTRE"] = df["DATA_DA_VENDA"].dt.quarter

df["DIAS_DESDE_LANCAMENTO"] = (df["DATA_DA_VENDA"] - df["DATA_LANCAMENTO"]).dt.days
df["FASE_CICLO"] = pd.cut(
    df["DIAS_DESDE_LANCAMENTO"],
    bins=[-1, 90, 270, 365, np.inf],
    labels=["Lancamento", "Maturidade", "Fim_de_ciclo", "Legado"]
)
print(df["FASE_CICLO"].value_counts())



fase_counts = df["FASE_CICLO"].value_counts()

plt.figure(figsize=(5, 5))
plt.pie(fase_counts.values, labels=fase_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Distribuição das vendas por fase do ciclo de vida")
plt.tight_layout()
plt.savefig(f"{BASE}\\distribuicao_fase_ciclo.png", dpi=300, bbox_inches="tight")
plt.show()

# base mensal agregada para previsão, regressão linear 


base_mensal = (
    df.groupby(["MODELO_COMERCIAL", "UF", pd.Grouper(key="DATA_DA_VENDA", freq="ME")])
    .agg(
        QTDE_VENDIDA=("QTDE.", "sum"),
        VALOR_TOTAL=("VALOR_DE_VENDA", "sum"),
        TICKET_MEDIO=("VALOR_DE_VENDA", "mean"),
        DIAS_DESDE_LANCAMENTO=("DIAS_DESDE_LANCAMENTO", "mean"),
    )
    .reset_index()
)

base_mensal = base_mensal.sort_values(["MODELO_COMERCIAL", "UF", "DATA_DA_VENDA"])
for lag in [1, 2, 3]:
    base_mensal[f"QTDE_LAG_{lag}"] = (
        base_mensal.groupby(["MODELO_COMERCIAL", "UF"])["QTDE_VENDIDA"].shift(lag)
    )

# Adicionando as datas  comemorativas o mês

base_mensal["MES"] = base_mensal["DATA_DA_VENDA"].dt.month
base_mensal["IS_BLACK_FRIDAY"] = (base_mensal["MES"] == 11).astype(int)   # Novembro
base_mensal["IS_NATAL"] = (base_mensal["MES"] == 12).astype(int)          # Dezembro
base_mensal["IS_DIA_DAS_MAES"] = (base_mensal["MES"] == 5).astype(int)    # Maio
base_mensal["IS_DIA_DOS_PAIS"] = (base_mensal["MES"] == 8).astype(int)    # Agosto

# Fase do ciclo de vida na base mensal, para modelos de regressão e de classificação.
base_mensal["FASE_CICLO"] = pd.cut(
    base_mensal["DIAS_DESDE_LANCAMENTO"],
    bins=[-1, 90, 270, 365, np.inf],
    labels=["Lancamento", "Maturidade", "Fim_de_ciclo", "Legado"]
)
fase_dummies = pd.get_dummies(base_mensal["FASE_CICLO"], prefix="FASE", dtype=int)
base_mensal = pd.concat([base_mensal, fase_dummies], axis=1)

base_mensal = base_mensal.dropna()
print("Base mensal agregada:", base_mensal.shape)



base_mensal = base_mensal.sort_values("DATA_DA_VENDA")
corte = base_mensal["DATA_DA_VENDA"].quantile(0.8)

treino = base_mensal[base_mensal["DATA_DA_VENDA"] <= corte]
teste = base_mensal[base_mensal["DATA_DA_VENDA"] > corte]

features = (["DIAS_DESDE_LANCAMENTO", "QTDE_LAG_1", "QTDE_LAG_2", "QTDE_LAG_3",
             "MES", "IS_BLACK_FRIDAY", "IS_NATAL", "IS_DIA_DAS_MAES", "IS_DIA_DOS_PAIS"]
            + list(fase_dummies.columns))
X_treino = treino[features].astype("float64")
X_teste = teste[features].astype("float64")
y_treino, y_teste = treino["QTDE_VENDIDA"], teste["QTDE_VENDIDA"]
print(f"Treino: {len(treino)} linhas até {corte.date()} | Teste: {len(teste)} linhas depois disso")

#   Modelos de regressão (volume de vendas)


def smape(y_teste, y_pred):
    y_teste = np.asarray(y_teste, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_teste) + np.abs(y_pred)) / 2
    diff = np.abs(y_teste - y_pred)
    return float(np.mean(np.where(denom == 0, 0, diff / denom)) * 100)

def avaliar_regressao(nome_modelo, y_teste, y_pred):
    return pd.DataFrame({
        "MAE": [mean_absolute_error(y_teste, y_pred)],
        "RMSE": [np.sqrt(mean_squared_error(y_teste, y_pred))],
        "R2": [r2_score(y_teste, y_pred)],
        "SMAPE_%": [smape(y_teste, y_pred)],
    }, index=[nome_modelo])

modelos = {
    "Regressao Linear": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42),
}

resultados_reg = []
for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)
    pred = modelo.predict(X_teste)
    resultados_reg.append(avaliar_regressao(nome, y_teste, pred))

resultados_reg = pd.concat(resultados_reg)
print(resultados_reg)

# Validação cruzada apropriada para série temporal


tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(
    XGBRegressor(n_estimators=300, random_state=42),
    X_treino, y_treino, cv=tscv, scoring="neg_mean_absolute_error"
)
print("MAE médio (TimeSeriesSplit, 5 folds):", -scores.mean())

# Ajuste  do XGBoost (GridSearchCV + TimeSeriesSplit)
#
# O XGBoost "de fábrica" (seção 8) ficou atrás do Random Forest. Antes de descartar,
# testar várias combinações de hiperparâmetros para ver se ele melhora com ajuste fino.


parametros_xgb = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.3, 0.1, 0.05],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

gs_xgb = GridSearchCV(
    XGBRegressor(n_estimators=300, random_state=42),
    param_grid=parametros_xgb,
    scoring="neg_mean_absolute_error",
    cv=TimeSeriesSplit(n_splits=5),
    n_jobs=1,  
)
gs_xgb.fit(X_treino, y_treino)
print("Melhores parâmetros encontrados:", gs_xgb.best_params_)
print("MAE médio na validação cruzada (melhor combinação):", -gs_xgb.best_score_)

xgb_ajustado = gs_xgb.best_estimator_
pred_ajustado = xgb_ajustado.predict(X_teste)
resultado_xgb_ajustado = avaliar_regressao("XGBoost (ajustado)", y_teste, pred_ajustado)

print("\nComparação no conjunto de teste:")
print(pd.concat([resultados_reg, resultado_xgb_ajustado]))

# Seleção  do melhor modelo
#
# Escolher o modelo com menor MAE no conjunto de teste entre os quatro candidatos
# (Regressão Linear, Random Forest, XGBoost e XGBoost ajustado). Esse modelo é o usado
# no gráfico de importância das variáveis, nas previsões exportadas e no arquivo `.pkl`.


candidatos = {
    "Regressao Linear": modelos["Regressao Linear"],
    "Random Forest": modelos["Random Forest"],
    "XGBoost": modelos["XGBoost"],
    "XGBoost (ajustado)": xgb_ajustado,
}
comparacao_final = pd.concat([resultados_reg, resultado_xgb_ajustado])
nome_melhor_modelo = comparacao_final["MAE"].idxmin()
melhor_modelo = candidatos[nome_melhor_modelo]
print(f"Melhor modelo (menor MAE no teste): {nome_melhor_modelo}")
print(comparacao_final)

#10. Modelo  de classificação fim de ciclo de vida


base_mensal["MEDIA_3M"] = base_mensal[["QTDE_LAG_1", "QTDE_LAG_2", "QTDE_LAG_3"]].mean(axis=1)
base_mensal["DECLINIO"] = (base_mensal["QTDE_VENDIDA"] < base_mensal["MEDIA_3M"] * 0.7).astype(int)

print("Proporção de declínio:")
print(base_mensal["DECLINIO"].value_counts(normalize=True))

# 
declinio_counts = base_mensal["DECLINIO"].value_counts().rename({0: "Sem declínio", 1: "Declínio"})

plt.figure(figsize=(5, 5))
plt.pie(declinio_counts.values, labels=declinio_counts.index, autopct="%1.1f%%", startangle=90)
plt.title("Distribuição da variável DECLINIO")
plt.tight_layout()
plt.savefig(f"{BASE}\\distribuicao_declinio.png", dpi=300, bbox_inches="tight")
plt.show()


X = base_mensal[features].astype("float64")
y = base_mensal["DECLINIO"]
X_treino_c, y_treino_c = X.loc[treino.index], y.loc[treino.index]
X_teste_c, y_teste_c = X.loc[teste.index], y.loc[teste.index]

qtd_classe_0 = (y_treino_c == 0).sum()
qtd_classe_1 = (y_treino_c == 1).sum()

classificadores = {
    "Regressao Logistica": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                             n_jobs=-1, random_state=42),
    "XGBoost": XGBClassifier(scale_pos_weight=qtd_classe_0 / max(qtd_classe_1, 1),
                              n_estimators=300, random_state=42, eval_metric="logloss"),
}


# precisão, revocação, F1 e Kappa, além da matriz de confusão.


def avaliar_classificacao(nome_modelo, y_teste, y_pred):
    resultados = pd.DataFrame({
        "Acuracia_Balanceada": [balanced_accuracy_score(y_teste, y_pred)],
        "Precisao": [precision_score(y_teste, y_pred, pos_label=1, zero_division=0)],
        "Revocacao": [recall_score(y_teste, y_pred, pos_label=1, zero_division=0)],
        "F1": [f1_score(y_teste, y_pred, pos_label=1, zero_division=0)],
        "Kappa": [cohen_kappa_score(y_teste, y_pred)],
    }, index=[nome_modelo]).round(3)

    matriz_confusao = pd.DataFrame(
        confusion_matrix(y_teste, y_pred),
        index=["Real: Sem declinio", "Real: Declinio"],
        columns=["Previsto: Sem declinio", "Previsto: Declinio"],
    )
    return resultados, matriz_confusao

resultados_clf = []
for nome, clf in classificadores.items():
    clf.fit(X_treino_c, y_treino_c)
    pred = clf.predict(X_teste_c)
    linha, matriz_confusao = avaliar_classificacao(nome, y_teste_c, pred)
    resultados_clf.append(linha)
    print(f"\n--- {nome} ---")
    print(matriz_confusao)

resultados_clf = pd.concat(resultados_clf).sort_values("F1", ascending=False)
print("\nComparação dos modelos de classificação (fim de ciclo):")
print(resultados_clf)

#11. Importância de atributos (melhor modelo)


if hasattr(melhor_modelo, "feature_importances_"):
    valores_importancia = melhor_modelo.feature_importances_
else:
   
    valores_importancia = np.abs(melhor_modelo.coef_)

importancias = pd.DataFrame({
    "feature": features,
    "importance": valores_importancia
}).sort_values(by="importance", ascending=False)

plt.figure(figsize=(7, 6))
sns.barplot(data=importancias, x="importance", y="feature", palette="magma")
plt.xlabel("Importância")
plt.ylabel("Variável")
plt.title(f"Importância dos atributos — {nome_melhor_modelo}")
plt.tight_layout()
plt.savefig(f"{BASE}\\importancia_variaveis.png", dpi=300, bbox_inches="tight")
plt.show()

#  Salvar modelo e exportar previsões


with open(f"{BASE}\\modelo_previsao_vendas.pkl", "wb") as f:
    pickle.dump(melhor_modelo, f)

df_previsoes = teste[["MODELO_COMERCIAL", "UF", "DATA_DA_VENDA"]].copy()
df_previsoes["QTDE_REAL"] = y_teste.values
df_previsoes["QTDE_PREVISTA"] = melhor_modelo.predict(X_teste)
df_previsoes.to_csv(f"{BASE}\\previsoes_vendas.csv", index=False, sep=";", decimal=",")

print(f"Modelo salvo em modelo_previsao_vendas.pkl ({nome_melhor_modelo})")
print("Previsões exportadas para previsoes_vendas.csv")


# Previsto x real  vai mostrar
#  o mais perto o melhor modelo chega da diagonal ideal (previsão = real) no teste. 
limite = max(df_previsoes["QTDE_REAL"].max(), df_previsoes["QTDE_PREVISTA"].max())

plt.figure(figsize=(6, 6))
plt.scatter(df_previsoes["QTDE_REAL"], df_previsoes["QTDE_PREVISTA"], alpha=0.4, s=20)
plt.plot([0, limite], [0, limite], "--", color="gray", label="Previsão = Real")
plt.xlabel("Quantidade real")
plt.ylabel("Quantidade prevista")
plt.title(f"Previsto x Real — {nome_melhor_modelo}")
plt.legend()
plt.tight_layout()
plt.savefig(f"{BASE}\\previsto_vs_real.png", dpi=300, bbox_inches="tight")
plt.show()

# Comparação da FASE_CICLO com o ciclo de vida usado pela empresa (ESTOQUE.xlsx)

estoque = pd.read_excel(f"{BASE}\\ESTOQUE.xlsx", sheet_name="ESTOQUE TERMINAIS")


canon_para_modelo = df.drop_duplicates("CANON").set_index("CANON")["MODELO_COMERCIAL"]
correspondencia_estoque = {n: canon_para_modelo.get(canon(n)) for n in estoque["NOME_COMERCIAL"].dropna().unique()}
correspondencia_estoque = {k: v for k, v in correspondencia_estoque.items() if v is not None}

estoque["MODELO_CANONICO_VENDA"] = estoque["NOME_COMERCIAL"].map(correspondencia_estoque)
print(f"Modelos do estoque casados com nome canônico de venda: "
      f"{estoque['MODELO_CANONICO_VENDA'].notna().sum()} de {len(estoque)} linhas "
      f"({estoque.loc[estoque['MODELO_CANONICO_VENDA'].notna(), 'NOME_COMERCIAL'].nunique()} modelos únicos).")


ciclo_vida_real = (
    estoque.dropna(subset=["MODELO_CANONICO_VENDA"])
    .groupby("MODELO_CANONICO_VENDA")["CICLO_VIDA"]
    .agg(lambda s: s.mode().iat[0])
    .rename("CICLO_VIDA_REAL")
)

venda_mais_recente = df.sort_values("DATA_DA_VENDA").groupby("MODELO_COMERCIAL").tail(1)
fase_calculada = venda_mais_recente.set_index("MODELO_COMERCIAL")["FASE_CICLO"]

comparacao_ciclo = pd.concat([fase_calculada, ciclo_vida_real], axis=1, join="inner").dropna()
print(f"Modelos comparáveis (venda + estoque): {len(comparacao_ciclo)}")

tabela_cruzada = pd.crosstab(comparacao_ciclo["FASE_CICLO"], comparacao_ciclo["CICLO_VIDA_REAL"])
print(tabela_cruzada)

plt.figure(figsize=(7, 5))
sns.heatmap(tabela_cruzada, annot=True, fmt="d", cmap="magma")
plt.xlabel("CICLO_VIDA real (ESTOQUE.xlsx)")
plt.ylabel("FASE_CICLO calculada (pipeline)")
plt.title("FASE_CICLO calculada x CICLO_VIDA real da empresa")
plt.tight_layout()
plt.savefig(f"{BASE}\\fase_ciclo_vs_estoque.png", dpi=300, bbox_inches="tight")
plt.show()
