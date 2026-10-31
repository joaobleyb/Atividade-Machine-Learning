"""Configurações do projeto: caminhos, nomes de colunas e números fixos.

Tudo que pode mudar fica aqui, em um lugar só.
"""

from pathlib import Path

# ---------- Caminhos ----------

# Pasta raiz do projeto (a pasta acima de src/)
RAIZ = Path(__file__).resolve().parent.parent

PASTA_DADOS = RAIZ / "data"
PASTA_SAIDAS = RAIZ / "outputs"
ARQUIVO_CSV = PASTA_DADOS / "titanic.csv"

# Cópia pública do train.csv da competição Titanic do Kaggle.
# Só é usada se o arquivo data/titanic.csv não existir.
URL_DATASET = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

# ---------- Colunas ----------

# A coluna que queremos prever: 1 = sobreviveu, 0 = não sobreviveu
COLUNA_ALVO = "Survived"

# Colunas que não ajudam o modelo e são apagadas:
# - PassengerId: só um número de identificação
# - Name e Ticket: textos praticamente únicos para cada passageiro
# - Cabin: 77% dos valores estão vazios
COLUNAS_INUTEIS = ["PassengerId", "Name", "Ticket", "Cabin"]

# Colunas que sobram, separadas pelo tipo de tratamento que recebem
COLUNAS_NUMERICAS = ["Age", "SibSp", "Parch", "Fare"]    # idade, parentes e valor da passagem
COLUNAS_CATEGORICAS = ["Pclass", "Sex", "Embarked"]      # classe, sexo e porto

# ---------- Treino ----------

# 20% dos passageiros ficam separados para o teste
TAMANHO_TESTE = 0.2

# "Semente" do sorteio: com o mesmo número, o resultado é sempre igual
RANDOM_STATE = 42
