"""Pré-processamento (Pipeline + ColumnTransformer) e definição dos modelos."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RANDOM_STATE


def criar_preprocessador(X: pd.DataFrame) -> ColumnTransformer:
    """Monta o tratamento das colunas.

    - Numéricas: preenche ausentes com a mediana e padroniza (StandardScaler).
    - Categóricas: preenche ausentes com o valor mais comum e aplica OneHotEncoder.

    Como isso fica dentro de um Pipeline, a média, o desvio padrão e as categorias
    são calculados SOMENTE com os dados de treino, evitando vazamento de dados.
    """
    colunas_numericas = X.select_dtypes(include="number").columns.tolist()
    colunas_categoricas = X.select_dtypes(exclude="number").columns.tolist()

    tratamento_numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("padronizador", StandardScaler()),
    ])

    tratamento_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore" evita erro se aparecer uma categoria nova no teste
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", tratamento_numerico, colunas_numericas),
        ("cat", tratamento_categorico, colunas_categoricas),
    ])


def criar_modelos(X: pd.DataFrame) -> dict[str, Pipeline]:
    """Retorna um dicionário com os modelos, cada um já com o pré-processamento acoplado."""
    algoritmos = {
        "Regressão Logística": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    }

    return {
        nome: Pipeline([
            ("preprocessamento", criar_preprocessador(X)),
            ("modelo", algoritmo),
        ])
        for nome, algoritmo in algoritmos.items()
    }
