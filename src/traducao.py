"""Tradução dos nomes de colunas e valores do dataset (que vem em inglês) para os gráficos."""

COLUNAS = {
    "Survived": "Sobreviveu",
    "Pclass": "Classe",
    "Sex": "Sexo",
    "Age": "Idade",
    "SibSp": "Irmãos/cônjuges a bordo",
    "Parch": "Pais/filhos a bordo",
    "Fare": "Valor da passagem",
    "Embarked": "Porto de embarque",
}

# Tradução dos valores, separada por coluna
VALORES = {
    "Survived": {"0": "Não sobreviveu", "1": "Sobreviveu"},
    "Pclass": {"1": "1ª classe", "2": "2ª classe", "3": "3ª classe"},
    "Sex": {"male": "Masculino", "female": "Feminino"},
    "Embarked": {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"},
}


def traduzir_coluna(nome: str) -> str:
    """Traduz o nome de uma coluna. Se não estiver no dicionário, devolve o original."""
    return COLUNAS.get(nome, nome)


def traduzir_valor(coluna: str, valor) -> str:
    """Traduz um valor de categoria de uma coluna. Se não houver tradução, devolve o original."""
    return VALORES.get(coluna, {}).get(str(valor), str(valor))


def traduzir_feature(nome: str) -> str:
    """Traduz o nome de uma feature gerada pelo OneHotEncoder.

    Exemplo: "Sex_female" vira "Sexo: Feminino".
    Colunas numéricas (sem "_") só têm o nome traduzido.
    """
    if "_" not in nome:
        return traduzir_coluna(nome)
    coluna, valor = nome.split("_", 1)
    return f"{traduzir_coluna(coluna)}: {traduzir_valor(coluna, valor)}"
