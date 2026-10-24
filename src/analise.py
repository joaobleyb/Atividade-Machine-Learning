"""Análise exploratória: informações gerais e gráficos salvos em outputs/."""

import matplotlib

matplotlib.use("Agg")  # gera os gráficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd

from src.config import COLUNA_ALVO, PASTA_SAIDAS
from src.traducao import traduzir_coluna, traduzir_valor

AZUL = "#4C72B0"
LARANJA = "#DD8452"


def analise_exploratoria(df: pd.DataFrame) -> None:
    """Mostra um resumo dos dados brutos no terminal e salva alguns gráficos."""
    PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ANÁLISE EXPLORATÓRIA")
    print("=" * 60)

    print(f"\nFormato: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print("\nTipos das colunas:")
    print(df.dtypes.to_string())

    print("\nValores ausentes por coluna:")
    nulos = df.isna().sum()
    nulos = nulos[nulos > 0]
    if nulos.empty:
        print("Nenhum valor ausente.")
    else:
        print(pd.DataFrame({
            "ausentes": nulos,
            "%": (nulos / len(df) * 100).round(1),
        }).to_string())

    print(f"\nLinhas duplicadas: {df.duplicated().sum()}")

    print(f"\nDistribuição da variável alvo ({COLUNA_ALVO}):")
    contagem = df[COLUNA_ALVO].value_counts()
    porcentagem = df[COLUNA_ALVO].value_counts(normalize=True) * 100
    tabela = pd.DataFrame({"quantidade": contagem, "%": porcentagem.round(1)})
    tabela.index = tabela.index.map(lambda v: traduzir_valor(COLUNA_ALVO, v))
    print(tabela.to_string())

    print("\nTaxa de sobrevivência por sexo (%):")
    print((df.groupby("Sex")[COLUNA_ALVO].mean() * 100).round(1).to_string())
    print("\nTaxa de sobrevivência por classe (%):")
    print((df.groupby("Pclass")[COLUNA_ALVO].mean() * 100).round(1).to_string())

    _graficos(df)
    print(f"\nGráficos salvos em: {PASTA_SAIDAS}")


def _grafico_taxa(df: pd.DataFrame, coluna: str, titulo: str, arquivo: str, cor: str) -> None:
    """Gráfico de barras com a % de sobreviventes em cada categoria de uma coluna."""
    taxa = df.groupby(coluna)[COLUNA_ALVO].mean() * 100
    taxa.index = taxa.index.map(lambda v: traduzir_valor(coluna, v))

    fig, ax = plt.subplots(figsize=(6, 4))
    taxa.plot(kind="bar", color=cor, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel("% de passageiros que sobreviveram")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)
    # Escreve o valor em cima de cada barra
    for i, valor in enumerate(taxa):
        ax.text(i, valor + 2, f"{valor:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / arquivo, dpi=100)
    plt.close(fig)


def _graficos(df: pd.DataFrame) -> None:
    """Cria e salva os gráficos básicos."""
    # 1) Quantos passageiros sobreviveram e quantos não
    fig, ax = plt.subplots(figsize=(6, 4))
    contagem = df[COLUNA_ALVO].value_counts().sort_index()
    contagem.index = contagem.index.map(lambda v: traduzir_valor(COLUNA_ALVO, v))
    contagem.plot(kind="bar", color=[AZUL, LARANJA], ax=ax)
    ax.set_title("Distribuição de sobreviventes")
    ax.set_xlabel("")
    ax.set_ylabel("Quantidade de passageiros")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / "01_distribuicao_sobreviventes.png", dpi=100)
    plt.close(fig)

    # 2) Histogramas de idade e valor da passagem, separados por sobrevivência
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, coluna in zip(axes, ["Age", "Fare"]):
        for valor, cor, rotulo in [(0, AZUL, "Não sobreviveu"), (1, LARANJA, "Sobreviveu")]:
            df.loc[df[COLUNA_ALVO] == valor, coluna].dropna().plot(
                kind="hist", bins=30, alpha=0.6, color=cor, label=rotulo, ax=ax
            )
        ax.set_title(traduzir_coluna(coluna))
        ax.set_ylabel("Quantidade")
        ax.legend()
    fig.suptitle("Variáveis numéricas por sobrevivência")
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / "02_histogramas_numericas.png", dpi=100)
    plt.close(fig)

    # 3) e 4) Taxa de sobrevivência por sexo e por classe
    _grafico_taxa(df, "Sex", "Taxa de sobrevivência por sexo", "03_sobrevivencia_por_sexo.png", LARANJA)
    _grafico_taxa(df, "Pclass", "Taxa de sobrevivência por classe", "04_sobrevivencia_por_classe.png", "#55A868")
