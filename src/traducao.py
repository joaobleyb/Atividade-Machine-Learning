"""Tradução para português dos nomes que vêm em inglês no dataset (usada nos gráficos)."""

# Nome de cada coluna
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

# Valores de cada coluna (as chaves são texto, por isso usamos str(valor) abaixo)
VALORES = {
    "Survived": {"0": "Não sobreviveu", "1": "Sobreviveu"},
    "Pclass": {"1": "1ª classe", "2": "2ª classe", "3": "3ª classe"},
    "Sex": {"male": "Masculino", "female": "Feminino"},
    "Embarked": {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"},
}


def traduzir_coluna(coluna):
    """Ex.: "Age" -> "Idade". Se não houver tradução, devolve o nome original."""
    return COLUNAS.get(coluna, coluna)


def traduzir_valor(coluna, valor):
    """Ex.: ("Sex", "male") -> "Masculino". Se não houver tradução, devolve o valor original."""
    traducoes_da_coluna = VALORES.get(coluna, {})
    return traducoes_da_coluna.get(str(valor), str(valor))


def traduzir_indice(indice, coluna):
    """Traduz todos os valores de um índice do pandas. Ex.: [0, 1] -> ["Não sobreviveu", "Sobreviveu"].

    O .map() aplica traduzir_valor em cada valor e mantém o nome do índice.
    """
    return indice.map(lambda valor: traduzir_valor(coluna, valor))


def traduzir_feature(nome):
    """Traduz o nome de uma coluna criada pelo OneHotEncoder.

    Ex.: "Sex_female" -> "Sexo: Feminino"
    Colunas numéricas não têm "_", então só o nome é traduzido. Ex.: "Age" -> "Idade"
    """
    if "_" not in nome:
        return traduzir_coluna(nome)

    coluna, valor = nome.split("_", 1)  # "Sex_female" -> "Sex" e "female"
    return f"{traduzir_coluna(coluna)}: {traduzir_valor(coluna, valor)}"
