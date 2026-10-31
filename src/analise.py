"""Etapa 2: análise exploratória. Resume os dados no terminal e salva gráficos em outputs/."""

import matplotlib

matplotlib.use("Agg")  # salva os gráficos em arquivo, sem abrir janela
import matplotlib.pyplot as plt
import pandas as pd

from src.config import COLUNA_ALVO, PASTA_SAIDAS
from src.tela import imprimir_titulo
from src.traducao import traduzir_coluna, traduzir_indice

AZUL = "#4C72B0"
LARANJA = "#DD8452"
VERDE = "#55A868"


def analise_exploratoria(df):
    """Mostra um resumo dos dados brutos e salva os gráficos."""
    PASTA_SAIDAS.mkdir(parents=True, exist_ok=True)
    imprimir_titulo("ANÁLISE EXPLORATÓRIA")

    print(f"\nFormato: {df.shape[0]} linhas x {df.shape[1]} colunas")

    print("\nTipos das colunas:")
    print(df.dtypes.to_string())

    mostrar_ausentes(df)

    print(f"\nLinhas duplicadas: {df.duplicated().sum()}")

    mostrar_distribuicao_alvo(df)

    # Média de Survived = proporção de sobreviventes (porque é 0 ou 1)
    print("\nTaxa de sobrevivência por sexo (%):")
    print((df.groupby("Sex")[COLUNA_ALVO].mean() * 100).round(1).to_string())
    print("\nTaxa de sobrevivência por classe (%):")
    print((df.groupby("Pclass")[COLUNA_ALVO].mean() * 100).round(1).to_string())

    grafico_distribuicao_sobreviventes(df)
    grafico_histogramas(df)
    grafico_taxa(df, "Sex", "Taxa de sobrevivência por sexo", "03_sobrevivencia_por_sexo.png", LARANJA)
    grafico_taxa(df, "Pclass", "Taxa de sobrevivência por classe", "04_sobrevivencia_por_classe.png", VERDE)
    print(f"\nGráficos salvos em: {PASTA_SAIDAS}")


def mostrar_ausentes(df):
    """Mostra quantos valores faltam em cada coluna (só as que têm falta)."""
    print("\nValores ausentes por coluna:")
    ausentes = df.isna().sum()
    ausentes = ausentes[ausentes > 0]

    if ausentes.empty:
        print("Nenhum valor ausente.")
        return

    tabela = pd.DataFrame({
        "ausentes": ausentes,
        "%": (ausentes / len(df) * 100).round(1),
    })
    print(tabela.to_string())


def mostrar_distribuicao_alvo(df):
    """Mostra quantos passageiros sobreviveram e quantos não."""
    print(f"\nDistribuição da variável alvo ({COLUNA_ALVO}):")
    quantidade = df[COLUNA_ALVO].value_counts()
    porcentagem = df[COLUNA_ALVO].value_counts(normalize=True) * 100

    tabela = pd.DataFrame({"quantidade": quantidade, "%": porcentagem.round(1)})
    tabela.index = traduzir_indice(tabela.index, COLUNA_ALVO)
    print(tabela.to_string())


def salvar_grafico(fig, arquivo):
    """Salva a figura em outputs/ e fecha, para liberar memória."""
    fig.tight_layout()
    fig.savefig(PASTA_SAIDAS / arquivo, dpi=100)
    plt.close(fig)


def grafico_distribuicao_sobreviventes(df):
    """Gráfico 1: barras com quantos sobreviveram e quantos não."""
    contagem = df[COLUNA_ALVO].value_counts().sort_index()
    contagem.index = traduzir_indice(contagem.index, COLUNA_ALVO)

    fig, ax = plt.subplots(figsize=(6, 4))
    contagem.plot(kind="bar", color=[AZUL, LARANJA], ax=ax)
    ax.set_title("Distribuição de sobreviventes")
    ax.set_xlabel("")
    ax.set_ylabel("Quantidade de passageiros")
    ax.tick_params(axis="x", rotation=0)
    salvar_grafico(fig, "01_distribuicao_sobreviventes.png")


def grafico_histogramas(df):
    """Gráfico 2: idade e valor da passagem, comparando quem sobreviveu e quem não."""
    nao_sobreviveu = df[df[COLUNA_ALVO] == 0]
    sobreviveu = df[df[COLUNA_ALVO] == 1]

    fig, (ax_idade, ax_valor) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, coluna in [(ax_idade, "Age"), (ax_valor, "Fare")]:
        nao_sobreviveu[coluna].dropna().plot(
            kind="hist", bins=30, alpha=0.6, color=AZUL, label="Não sobreviveu", ax=ax
        )
        sobreviveu[coluna].dropna().plot(
            kind="hist", bins=30, alpha=0.6, color=LARANJA, label="Sobreviveu", ax=ax
        )
        ax.set_title(traduzir_coluna(coluna))
        ax.set_ylabel("Quantidade")
        ax.legend()

    fig.suptitle("Variáveis numéricas por sobrevivência")
    salvar_grafico(fig, "02_histogramas_numericas.png")


def grafico_taxa(df, coluna, titulo, arquivo, cor):
    """Gráficos 3 e 4: % de sobreviventes em cada categoria de uma coluna."""
    taxa = df.groupby(coluna)[COLUNA_ALVO].mean() * 100
    taxa.index = traduzir_indice(taxa.index, coluna)

    fig, ax = plt.subplots(figsize=(6, 4))
    taxa.plot(kind="bar", color=cor, ax=ax)
    ax.set_title(titulo)
    ax.set_xlabel("")
    ax.set_ylabel("% de passageiros que sobreviveram")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="x", rotation=0)

    # Escreve a porcentagem em cima de cada barra
    for posicao, valor in enumerate(taxa):
        ax.text(posicao, valor + 2, f"{valor:.1f}%", ha="center")

    salvar_grafico(fig, arquivo)
