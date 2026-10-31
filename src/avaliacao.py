"""Etapa 6: medir como cada modelo foi no conjunto de teste."""

import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janela
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

from src.config import PASTA_SAIDAS
from src.traducao import traduzir_feature

# Nome do arquivo de imagem da matriz de confusão de cada modelo
ARQUIVOS_MATRIZ = {
    "Regressão Logística": "matriz_confusao_regressao_logistica.png",
    "Random Forest": "matriz_confusao_random_forest.png",
}


def avaliar_modelo(nome, modelo, X_teste, y_teste):
    """Calcula as métricas do modelo e salva a matriz de confusão."""
    # O modelo "chuta" quem sobreviveu no teste...
    y_previsto = modelo.predict(X_teste)

    # ...e comparamos com a resposta certa (y_teste)
    metricas = {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_teste, y_previsto),   # % de acertos no total
        "Precisão": precision_score(y_teste, y_previsto),  # dos que ele disse "sobreviveu", quantos sobreviveram
        "Recall": recall_score(y_teste, y_previsto),       # dos que sobreviveram, quantos ele achou
        "F1": f1_score(y_teste, y_previsto),               # equilíbrio entre precisão e recall
    }

    # Matriz de confusão: onde o modelo acertou e onde errou
    matriz = confusion_matrix(y_teste, y_previsto)
    print(f"\n--- {nome} ---")
    print("Matriz de confusão (linhas = real, colunas = previsto):")
    tabela = pd.DataFrame(
        matriz,
        index=["Real: Não sobreviveu", "Real: Sobreviveu"],
        columns=["Prev: Não sobreviveu", "Prev: Sobreviveu"],
    )
    print(tabela.to_string())

    salvar_matriz_confusao(nome, matriz)
    return metricas


def salvar_matriz_confusao(nome, matriz):
    """Salva a matriz de confusão como imagem em outputs/."""
    fig, ax = plt.subplots(figsize=(5.5, 4))
    grafico = ConfusionMatrixDisplay(matriz, display_labels=["Não sobreviveu", "Sobreviveu"])
    grafico.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Matriz de confusão - {nome}")
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / ARQUIVOS_MATRIZ[nome], dpi=100)
    plt.close(fig)


def importancia_features(modelo, top=15):
    """Mostra quais colunas mais pesaram na decisão do Random Forest e salva um gráfico."""
    # Nomes das colunas depois do pré-processamento (ex.: "Age", "Sex_male")
    nomes = modelo.named_steps["preprocessamento"].get_feature_names_out()
    # Quanto cada coluna ajudou nas decisões das árvores
    valores = modelo.named_steps["modelo"].feature_importances_

    importancias = pd.Series(valores, index=[traduzir_feature(n) for n in nomes])
    importancias = importancias.sort_values(ascending=False).head(top)

    fig, ax = plt.subplots(figsize=(8, 6))
    importancias.sort_values().plot(kind="barh", color="#4C72B0", ax=ax)
    ax.set_title(f"As {top} variáveis mais importantes - Random Forest")
    ax.set_xlabel("Importância")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / "importancia_features_random_forest.png", dpi=100)
    plt.close(fig)

    return importancias
