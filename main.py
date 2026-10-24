"""Projeto de Machine Learning: previsão de sobreviventes do Titanic.

Executa todas as etapas em ordem:
  1. Carrega os dados
  2. Faz a análise exploratória
  3. Limpa os dados
  4. Divide em treino e teste
  5. Treina Regressão Logística e Random Forest
  6. Avalia e compara os modelos
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.analise import analise_exploratoria
from src.avaliacao import avaliar_modelo, importancia_features
from src.config import COLUNA_ALVO, PASTA_SAIDAS, RANDOM_STATE, TAMANHO_TESTE
from src.dados import carregar_dados, limpar_dados
from src.modelos import criar_modelos


def main() -> None:
    # 1) Carregar
    df_bruto = carregar_dados()

    # 2) Análise exploratória (nos dados brutos, do jeito que vieram)
    analise_exploratoria(df_bruto)

    # 3) Limpeza
    df = limpar_dados(df_bruto)

    # 4) Separar features (X) e alvo (y), e dividir em treino/teste 80/20.
    #    O stratify mantém a mesma proporção de sobreviventes nos dois conjuntos.
    X = df.drop(columns=[COLUNA_ALVO])
    y = df[COLUNA_ALVO]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TAMANHO_TESTE, stratify=y, random_state=RANDOM_STATE
    )

    print("\n" + "=" * 60)
    print("DIVISÃO TREINO / TESTE")
    print("=" * 60)
    print(f"Treino: {len(X_treino)} linhas | sobreviventes = {y_treino.mean():.1%}")
    print(f"Teste:  {len(X_teste)} linhas | sobreviventes = {y_teste.mean():.1%}")

    # 5) e 6) Treinar e avaliar cada modelo
    print("\n" + "=" * 60)
    print("TREINO E AVALIAÇÃO DOS MODELOS")
    print("=" * 60)

    modelos = criar_modelos(X_treino)
    resultados = []
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)  # o pré-processamento é ajustado só no treino
        resultados.append(avaliar_modelo(nome, modelo, X_teste, y_teste))

    # Tabela comparativa
    tabela = pd.DataFrame(resultados).set_index("Modelo").round(4)
    print("\n" + "=" * 60)
    print("COMPARAÇÃO DOS MODELOS (conjunto de teste)")
    print("=" * 60)
    print(tabela.to_string())
    tabela.to_csv(PASTA_SAIDAS / "comparacao_modelos.csv", encoding="utf-8-sig")

    # Variáveis mais importantes segundo o Random Forest
    print("\nAs 10 variáveis mais importantes (Random Forest):")
    print(importancia_features(modelos["Random Forest"]).head(10).round(4).to_string())

    # Conclusão: o F1 é a métrica principal porque as classes são um pouco
    # desbalanceadas (só 38% sobreviveram) e ele equilibra precisão e recall.
    melhor = tabela["F1"].idxmax()
    pior = tabela["F1"].idxmin()
    print("\n" + "=" * 60)
    print("CONCLUSÃO")
    print("=" * 60)
    if melhor == pior:
        print(f"Os dois modelos empataram no F1 ({tabela.loc[melhor, 'F1']:.4f}).")
    else:
        print(f"Melhor modelo pelo F1: {melhor} (F1 = {tabela.loc[melhor, 'F1']:.4f})")
        print(
            f"Recall de {melhor}: {tabela.loc[melhor, 'Recall']:.2%} | "
            f"Recall de {pior}: {tabela.loc[pior, 'Recall']:.2%}."
        )
        print(
            f"Precisão de {melhor}: {tabela.loc[melhor, 'Precisão']:.2%} | "
            f"Precisão de {pior}: {tabela.loc[pior, 'Precisão']:.2%}."
        )
    print(f"\nTabela e gráficos salvos em: {PASTA_SAIDAS}")


if __name__ == "__main__":
    main()
