"""Carregamento e limpeza básica dos dados."""

import urllib.request

import pandas as pd

from src.config import ARQUIVO_CSV, COLUNAS_INUTEIS, PASTA_DADOS, URL_DATASET


def carregar_dados() -> pd.DataFrame:
    """Lê o CSV da pasta data/. Se o arquivo não existir, baixa da internet e salva uma cópia."""
    if not ARQUIVO_CSV.exists():
        print(f"Arquivo local não encontrado. Baixando de:\n  {URL_DATASET}")
        PASTA_DADOS.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL_DATASET, ARQUIVO_CSV)

    return pd.read_csv(ARQUIVO_CSV)


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Corrige tipos, remove duplicados e colunas inúteis.

    Essas etapas são regras fixas (não "aprendem" nada com os dados),
    então podem ser feitas antes da divisão treino/teste sem vazar informação.

    Os valores ausentes de Age e Embarked NÃO são preenchidos aqui: isso é feito
    dentro do Pipeline (src/modelos.py), usando só os dados de treino.
    """
    df = df.copy()
    print("\n" + "=" * 60)
    print("TRATAMENTO DOS DADOS")
    print("=" * 60)

    # 1) Pclass vem como número (1, 2, 3), mas na verdade é uma categoria:
    #    a 3ª classe não é "três vezes" a 1ª. Por isso convertemos para texto,
    #    e assim ela passa pelo OneHotEncoder em vez do StandardScaler.
    df["Pclass"] = df["Pclass"].astype(str)
    print("Pclass convertida de número para categoria.")

    # 2) Remove linhas totalmente duplicadas (cadastro repetido).
    antes = len(df)
    df = df.drop_duplicates()
    print(f"Linhas duplicadas removidas: {antes - len(df)}")

    # 3) Remove as colunas que não ajudam a prever.
    #    A Cabin cai aqui também, então seus valores ausentes deixam de ser problema.
    df = df.drop(columns=COLUNAS_INUTEIS)
    print(f"Colunas removidas: {', '.join(COLUNAS_INUTEIS)}")

    # 4) Mostra o que ainda falta preencher (será tratado no Pipeline)
    faltando = df.isna().sum()
    faltando = faltando[faltando > 0]
    print("Ausentes que serão preenchidos no Pipeline (só com dados de treino):")
    for coluna, qtd in faltando.items():
        estrategia = "mediana" if pd.api.types.is_numeric_dtype(df[coluna]) else "valor mais comum"
        print(f"  {coluna}: {qtd} ausentes -> {estrategia}")

    print(f"Formato final após a limpeza: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df
