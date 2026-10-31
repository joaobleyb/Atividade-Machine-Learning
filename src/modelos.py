"""Etapa 5: montar os dois modelos, cada um com o seu pré-processamento.

Cada modelo é um Pipeline, uma "linha de montagem" com duas estações:
  1. pré-processamento: preenche os ausentes e transforma as colunas em números
  2. modelo: o algoritmo que aprende a prever quem sobreviveu
"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import COLUNAS_CATEGORICAS, COLUNAS_NUMERICAS, RANDOM_STATE


def criar_preprocessador():
    """Diz o que fazer com cada tipo de coluna.

    Numéricas (idade, parentes, valor da passagem):
      - ausentes viram a MEDIANA da coluna
      - depois são padronizadas (StandardScaler), para ficarem na mesma escala

    Categóricas (classe, sexo, porto):
      - ausentes viram o valor MAIS COMUM da coluna
      - depois cada categoria vira uma coluna de 0 e 1 (OneHotEncoder)
        exemplo: Sex vira "Sex_female" e "Sex_male"

    A mediana, a média, o desvio padrão e as categorias são aprendidos
    no .fit(), que só recebe os dados de treino. Assim o teste fica de fora
    e não há vazamento de dados.
    """
    tratamento_numerico = Pipeline([
        ("imputador", SimpleImputer(strategy="median")),
        ("padronizador", StandardScaler()),
    ])

    tratamento_categorico = Pipeline([
        ("imputador", SimpleImputer(strategy="most_frequent")),
        # handle_unknown="ignore": se aparecer uma categoria nova no teste, não dá erro
        ("codificador", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        [
            ("num", tratamento_numerico, COLUNAS_NUMERICAS),
            ("cat", tratamento_categorico, COLUNAS_CATEGORICAS),
        ],
        # Mantém os nomes simples ("Age", "Sex_male") sem o prefixo "num__"/"cat__"
        verbose_feature_names_out=False,
    )


def criar_modelos():
    """Cria os dois modelos que vamos comparar."""
    # Regressão Logística: modelo simples, uma "conta ponderada" das colunas
    regressao_logistica = Pipeline([
        ("preprocessamento", criar_preprocessador()),
        ("modelo", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])

    # Random Forest: 300 árvores de decisão que votam juntas
    random_forest = Pipeline([
        ("preprocessamento", criar_preprocessador()),
        ("modelo", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)),
    ])

    return {
        "Regressão Logística": regressao_logistica,
        "Random Forest": random_forest,
    }
