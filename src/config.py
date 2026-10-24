"""Configurações gerais do projeto (caminhos, nomes de colunas e semente aleatória)."""

from pathlib import Path

# Pasta raiz do projeto (um nível acima de src/)
RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data"
PASTA_SAIDAS = RAIZ / "outputs"
ARQUIVO_CSV = PASTA_DADOS / "titanic.csv"

# Cópia pública do train.csv da competição Titanic do Kaggle
# (usada só se a cópia local não existir)
URL_DATASET = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# Coluna que queremos prever
COLUNA_ALVO = "Survived"

# Colunas que não ajudam o modelo:
# - PassengerId: só um número de identificação
# - Name e Ticket: textos praticamente únicos para cada passageiro
# - Cabin: 77% dos valores estão vazios
COLUNAS_INUTEIS = ["PassengerId", "Name", "Ticket", "Cabin"]

# Semente fixa para os resultados serem sempre os mesmos
RANDOM_STATE = 42
TAMANHO_TESTE = 0.2
