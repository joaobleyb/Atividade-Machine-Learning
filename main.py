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
from src.tela import imprimir_titulo


def main():
    # 1) Carregar o CSV
    df_bruto = carregar_dados()

    # 2) Análise exploratória, nos dados do jeito que vieram
    analise_exploratoria(df_bruto)

    # 3) Limpeza
    df = limpar_dados(df_bruto)

    # 4) Separar as perguntas (X) da resposta (y)
    X = df.drop(columns=[COLUNA_ALVO])  # tudo menos "Survived"
    y = df[COLUNA_ALVO]                 # só "Survived"

    # Divide em 80% treino e 20% teste.
    # stratify=y mantém a mesma % de sobreviventes nos dois lados.
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=TAMANHO_TESTE, stratify=y, random_state=RANDOM_STATE
    )

    imprimir_titulo("DIVISÃO TREINO / TESTE")
    print(f"Treino: {len(X_treino)} linhas | sobreviventes = {y_treino.mean():.1%}")
    print(f"Teste:  {len(X_teste)} linhas | sobreviventes = {y_teste.mean():.1%}")

    # 5) e 6) Treinar e avaliar cada modelo
    imprimir_titulo("TREINO E AVALIAÇÃO DOS MODELOS")
    modelos = criar_modelos()
    resultados = []
    for nome, modelo in modelos.items():
        modelo.fit(X_treino, y_treino)  # aprende SÓ com o treino (pré-processamento incluído)
        metricas = avaliar_modelo(nome, modelo, X_teste, y_teste)
        resultados.append(metricas)

    # Tabela comparando os dois modelos
    tabela = pd.DataFrame(resultados).set_index("Modelo").round(4)
    imprimir_titulo("COMPARAÇÃO DOS MODELOS (conjunto de teste)")
    print(tabela.to_string())
    tabela.to_csv(PASTA_SAIDAS / "comparacao_modelos.csv", encoding="utf-8-sig")

    # Colunas que mais pesaram na decisão do Random Forest
    print("\nAs 10 variáveis mais importantes (Random Forest):")
    importancias = importancia_features(modelos["Random Forest"])
    print(importancias.head(10).round(4).to_string())

    mostrar_conclusao(tabela)
    print(f"\nTabela e gráficos salvos em: {PASTA_SAIDAS}")


def mostrar_conclusao(tabela):
    """Diz qual modelo foi melhor.

    O critério é o F1, porque as classes são um pouco desbalanceadas
    (só 38% sobreviveram) e o F1 equilibra precisão e recall.
    """
    imprimir_titulo("CONCLUSÃO")
    melhor = tabela["F1"].idxmax()  # nome do modelo com o maior F1
    pior = tabela["F1"].idxmin()    # nome do modelo com o menor F1

    if melhor == pior:
        print(f"Os dois modelos empataram no F1 ({tabela.loc[melhor, 'F1']:.4f}).")
        return

    print(f"Melhor modelo pelo F1: {melhor} (F1 = {tabela.loc[melhor, 'F1']:.4f})")
    print(f"Recall de {melhor}: {tabela.loc[melhor, 'Recall']:.2%} | "
          f"Recall de {pior}: {tabela.loc[pior, 'Recall']:.2%}.")
    print(f"Precisão de {melhor}: {tabela.loc[melhor, 'Precisão']:.2%} | "
          f"Precisão de {pior}: {tabela.loc[pior, 'Precisão']:.2%}.")


if __name__ == "__main__":
    main()
