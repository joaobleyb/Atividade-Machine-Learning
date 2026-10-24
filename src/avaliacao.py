"""Avaliação dos modelos no conjunto de teste."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import PASTA_SAIDAS
from src.traducao import traduzir_feature


def avaliar_modelo(nome: str, modelo: Pipeline, X_teste: pd.DataFrame, y_teste: pd.Series) -> dict:
    """Calcula as métricas, mostra a matriz de confusão e salva a imagem dela."""
    y_pred = modelo.predict(X_teste)

    metricas = {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_teste, y_pred),
        "Precisão": precision_score(y_teste, y_pred),
        "Recall": recall_score(y_teste, y_pred),
        "F1": f1_score(y_teste, y_pred),
    }

    matriz = confusion_matrix(y_teste, y_pred)
    print(f"\n--- {nome} ---")
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    print(pd.DataFrame(matriz, index=["Real: Não sobreviveu", "Real: Sobreviveu"],
                       columns=["Prev: Não sobreviveu", "Prev: Sobreviveu"]).to_string())

    # Salva a matriz de confusão como imagem
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ConfusionMatrixDisplay(matriz, display_labels=["Não sobreviveu", "Sobreviveu"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusão - {nome}")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    nome_arquivo = nome.lower().replace(" ", "_").replace("ã", "a").replace("í", "i")
    fig.savefig(PASTA_SAIDAS / f"matriz_confusao_{nome_arquivo}.png", dpi=100)
    plt.close(fig)

    return metricas


def importancia_features(modelo: Pipeline, top: int = 15) -> pd.Series:
    """Pega as features mais importantes do Random Forest e salva um gráfico."""
    nomes = modelo.named_steps["preprocessamento"].get_feature_names_out()
    importancias = pd.Series(modelo.named_steps["modelo"].feature_importances_, index=nomes)
    # Tira os prefixos "num__" e "cat__" e traduz para português
    importancias.index = [traduzir_feature(n.split("__", 1)[1]) for n in importancias.index]
    importancias = importancias.sort_values(ascending=False).head(top)

    fig, ax = plt.subplots(figsize=(8, 6))
    importancias.sort_values().plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_title(f"As {top} variáveis mais importantes - Random Forest")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / "importancia_features_random_forest.png", dpi=100)
    plt.close(fig)

    return importancias
