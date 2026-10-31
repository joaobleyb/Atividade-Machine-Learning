"""Etapas 1 e 3: carregar o CSV e fazer a limpeza básica."""

import urllib.request

import pandas as pd

from src.config import ARQUIVO_CSV, COLUNAS_INUTEIS, PASTA_DADOS, URL_DATASET
from src.tela import imprimir_titulo


def carregar_dados():
    """Lê o CSV de data/. Se ele não existir, baixa da internet primeiro."""
    if not ARQUIVO_CSV.exists():
        print(f"Arquivo local não encontrado. Baixando de:\n  {URL_DATASET}")
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_DATASET, ARQUIVO_CSV)

    return pd.read_csv(ARQUIVO_CSV)


def limpar_dados(df):
    """Corrige o tipo da Pclass, remove duplicados e apaga as colunas inúteis.

    Aqui só entram regras fixas, que não "aprendem" nada com os dados.
    Por isso podem rodar antes da divisão treino/teste sem vazar informação.

    Os valores ausentes de Age e Embarked NÃO são preenchidos aqui.
    Isso fica para o Pipeline (src/modelos.py), que usa só os dados de treino.
    """
    imprimir_titulo("TRATAMENTO DOS DADOS")
    df = df.copy()  # trabalha numa cópia para não alterar a tabela original

    # 1) Pclass vem como número (1, 2, 3), mas é uma categoria:
    #    a 3ª classe não vale "três vezes" a 1ª. Então vira texto.
    df["Pclass"] = df["Pclass"].astype(str)
    print("Pclass convertida de número para categoria.")

    # 2) Remove linhas repetidas
    linhas_antes = len(df)
    df = df.drop_duplicates()
    print(f"Linhas duplicadas removidas: {linhas_antes - len(df)}")

    # 3) Remove as colunas que não ajudam a prever
    df = df.drop(columns=COLUNAS_INUTEIS)
    print(f"Colunas removidas: {', '.join(COLUNAS_INUTEIS)}")

    # 4) Mostra o que ainda falta preencher (o Pipeline vai cuidar disso)
    print("Ausentes que serão preenchidos no Pipeline (só com dados de treino):")
    print(f"  Age: {df['Age'].isna().sum()} ausentes -> mediana")
    print(f"  Embarked: {df['Embarked'].isna().sum()} ausentes -> valor mais comum")

    print(f"Formato final após a limpeza: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df
