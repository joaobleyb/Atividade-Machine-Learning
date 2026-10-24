# Previsão de Sobreviventes do Titanic

Este é o meu projeto de Machine Learning para a faculdade (Análise e Desenvolvimento de Sistemas). Escolhi um clássico: o naufrágio do **Titanic**, em 1912. A pergunta é: **olhando só para as informações de cada passageiro, dá pra prever quem sobreviveu?**

Usei Python, **pandas** para mexer nos dados e **scikit-learn** para treinar os modelos.

---

## O problema

O Titanic afundou na sua viagem de estreia e, como não tinha bote salva-vidas para todo mundo, mais da metade das pessoas a bordo morreu. Só que a sobrevivência não foi pura sorte: a regra "mulheres e crianças primeiro" e a localização das cabines de cada classe fizeram muita diferença.

A ideia do projeto é ensinar um modelo a encontrar esses padrões sozinho. Como a resposta é "sim ou não" (sobreviveu ou não sobreviveu), esse é um problema de **classificação**.

É claro que ninguém vai usar isso para salvar o Titanic. Mas o raciocínio é o mesmo usado em problemas reais, como prever se um cliente vai cancelar um serviço, se um paciente tem risco de uma doença ou se um empréstimo vai ser pago. Por isso esse dataset é tão usado para aprender.

## O dataset

O dataset vem da competição **Titanic - Machine Learning from Disaster**, do Kaggle:

https://www.kaggle.com/competitions/titanic

Como o Kaggle exige login para baixar, usei uma cópia pública do mesmo arquivo (`train.csv`), que não precisa de cadastro:

https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv

Uma cópia já está salva em `data/titanic.csv`, então o projeto roda mesmo sem internet.

> A competição também tem um `test.csv`, mas ele vem **sem** a resposta de quem sobreviveu (ela fica escondida no Kaggle para dar a nota da competição). Por isso usei só o `train.csv` e separei uma parte dele para teste.

São **891 passageiros** e **12 colunas**:

| Coluna | O que significa |
|---|---|
| `PassengerId` | número de identificação do passageiro |
| `Survived` | **se sobreviveu (1) ou não (0)**. É isso que queremos prever |
| `Pclass` | classe da passagem (1ª, 2ª ou 3ª) |
| `Name` | nome |
| `Sex` | sexo |
| `Age` | idade |
| `SibSp` | quantos irmãos ou cônjuges estavam a bordo junto |
| `Parch` | quantos pais ou filhos estavam a bordo junto |
| `Ticket` | número da passagem |
| `Fare` | valor pago pela passagem |
| `Cabin` | número da cabine |
| `Embarked` | porto onde embarcou (Southampton, Cherbourg ou Queenstown) |

## Como o projeto está organizado

```
Projeto/
├── main.py              # roda tudo, do começo ao fim
├── src/
│   ├── config.py        # caminhos, nomes de colunas e semente aleatória
│   ├── dados.py         # carregamento e limpeza dos dados
│   ├── analise.py       # análise exploratória e gráficos
│   ├── modelos.py       # Pipeline, ColumnTransformer e os modelos
│   ├── avaliacao.py     # métricas, matriz de confusão e importância das variáveis
│   └── traducao.py      # tradução dos nomes para os gráficos ficarem em português
├── data/                # o CSV do dataset
├── outputs/             # gráficos e tabela de resultados (gerados ao rodar)
├── requirements.txt
└── README.md
```

## O que foi feito em cada etapa

### 1. Análise exploratória

Antes de treinar qualquer modelo, dei uma olhada nos dados:

- **Formato:** 891 linhas e 12 colunas.
- **Valores ausentes:**
  - `Age` (idade): faltam 177 valores, cerca de 20%;
  - `Cabin` (cabine): faltam 687 valores, 77%! Quase ninguém tem essa informação;
  - `Embarked` (porto): faltam só 2.
- **Duplicados:** nenhuma linha repetida.
- **Alvo:** 61,6% dos passageiros não sobreviveram e 38,4% sobreviveram. Ou seja, um modelo que chutasse "ninguém sobrevive" já acertaria 61,6% das vezes, então a acurácia sozinha não conta a história toda.

O que mais chamou atenção nos gráficos:

- **Sexo fez muita diferença:** 74,2% das mulheres sobreviveram, contra só 18,9% dos homens.
- **A classe também:** sobreviveram 63% da 1ª classe, 47,3% da 2ª e só 24,2% da 3ª.
- Quem pagou passagens mais caras e as crianças pequenas tiveram mais chance de sobreviver.

Os gráficos ficam salvos em `outputs/`:

| Arquivo | O que mostra |
|---|---|
| `01_distribuicao_sobreviventes.png` | quantos sobreviveram e quantos não |
| `02_histogramas_numericas.png` | idade e valor da passagem, separados por sobrevivência |
| `03_sobrevivencia_por_sexo.png` | taxa de sobrevivência de homens e mulheres |
| `04_sobrevivencia_por_classe.png` | taxa de sobrevivência por classe |

### 2. Tratamento dos dados

- **Corrigi o tipo da coluna `Pclass`.** Ela vem como número (1, 2, 3), mas na verdade é uma categoria: a 3ª classe não vale "três vezes" a 1ª. Por isso converti para texto, e assim ela passa pelo OneHotEncoder como as outras categorias.
- **Removi as colunas que não ajudam:**
  - `PassengerId`, porque é só um número de identificação;
  - `Name` e `Ticket`, porque são textos praticamente únicos para cada pessoa;
  - `Cabin`, porque 77% dos valores estão vazios. Preencher tanta coisa seria praticamente inventar dados.
- **Tratei os valores ausentes que sobraram:**
  - `Age` foi preenchida com a **mediana** das idades;
  - `Embarked` foi preenchida com o **porto mais comum**.
- **Removi duplicados.** Não tinha nenhum, mas o passo ficou no código caso o dataset mude.
- **Padronizei as numéricas e codifiquei as categóricas.** Usei `StandardScaler` nas colunas numéricas (idade, valor da passagem, número de parentes) e `OneHotEncoder` nas categóricas (sexo, classe, porto).

O preenchimento dos ausentes e a padronização foram montados com `Pipeline` e `ColumnTransformer`. Assim, a mediana da idade, a média, o desvio padrão e as categorias são calculados **só com os dados de treino**. Se eu usasse a base inteira, o modelo estaria "espiando" o teste, e o resultado ficaria melhor do que seria na vida real (o famoso vazamento de dados).

### 3. Divisão treino/teste

Separei **80% para treino** (712 passageiros) e **20% para teste** (179 passageiros). Usei `stratify` para manter a mesma proporção de sobreviventes (~38%) nos dois lados, e `random_state=42` para o resultado ser sempre o mesmo.

### 4. Treino

Treinei dois algoritmos:

- **Regressão Logística:** um modelo mais simples, que no fim é uma "conta ponderada" das características do passageiro. É rápido e fácil de interpretar.
- **Random Forest:** um conjunto de várias árvores de decisão (usei 300) que votam juntas. Costuma pegar relações mais complicadas, como "homem adulto da 3ª classe" ou "criança da 2ª classe".

### 5. Avaliação

Avaliei os dois no conjunto de teste com:

- **Acurácia:** quantos o modelo acertou no total.
- **Precisão:** dos passageiros que o modelo disse que sobreviveram, quantos sobreviveram mesmo.
- **Recall:** dos passageiros que sobreviveram de verdade, quantos o modelo conseguiu encontrar.
- **F1:** um equilíbrio entre precisão e recall.
- **Matriz de confusão:** mostra exatamente onde o modelo acerta e onde erra.

## Resultados

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Regressão Logística | 0,8045 | 0,7931 | 0,6667 | 0,7244 |
| **Random Forest** | **0,8156** | **0,8103** | **0,6812** | **0,7402** |

Matrizes de confusão (de 179 passageiros no teste, 69 sobreviveram de verdade):

| | Regressão Logística | Random Forest |
|---|---|---|
| Sobreviveu e o modelo acertou | 46 | 47 |
| Sobreviveu e o modelo errou | 23 | 22 |
| Não sobreviveu, mas o modelo achou que sim | 12 | 11 |
| Não sobreviveu e o modelo acertou | 98 | 99 |

As imagens das matrizes, o gráfico de importância das variáveis e a tabela `comparacao_modelos.csv` também ficam em `outputs/`.

Segundo o Random Forest, o que mais pesa na decisão é: **valor da passagem, idade, sexo e classe**. Isso bate direitinho com a história do naufrágio e com o que eu já tinha visto na análise exploratória.

## Conclusão

O **Random Forest foi o melhor modelo**, com F1 de 0,74 e acurácia de 81,6%. Ele ganhou em todas as métricas: acertou mais sobreviventes (recall maior) e errou menos quando disse que alguém sobreviveu (precisão maior). Usei o **F1** como critério principal porque as classes são um pouco desbalanceadas e ele equilibra essas duas coisas.

Uma explicação provável é que ele consegue combinar as variáveis entre si. Por exemplo, ser mulher ajudava muito, mas ajudava bem mais na 1ª classe do que na 3ª. A Regressão Logística olha cada variável mais "separadamente".

Mas vale ser sincero: a diferença foi pequena. No teste, o Random Forest acertou **só 2 passageiros a mais** (146 contra 144, de 179). Com um conjunto de teste pequeno assim, outra divisão dos dados poderia até inverter o resultado. Os dois modelos ficaram na faixa dos 80% de acerto, bem acima dos 61,6% de quem chutasse "ninguém sobrevive".

Como próximos passos, daria pra:

- usar **validação cruzada** para comparar os modelos de forma mais confiável;
- criar variáveis novas, como o **título** que aparece no nome (Mr., Mrs., Miss, Master) ou o **tamanho da família**;
- ajustar os hiperparâmetros do Random Forest.

## Como rodar no Windows

Você precisa ter o **Python 3** instalado (eu usei o 3.14). Abra o terminal (PowerShell ou Prompt de Comando) dentro da pasta do projeto e siga os passos:

**1. Criar o ambiente virtual**

```
python -m venv venv
```

**2. Ativar o ambiente virtual**

No PowerShell:

```
.\venv\Scripts\Activate.ps1
```

No Prompt de Comando (cmd):

```
venv\Scripts\activate
```

Se der certo, vai aparecer `(venv)` no começo da linha.

> Se o PowerShell reclamar de permissão para rodar scripts, rode este comando uma vez e tente ativar de novo:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**3. Instalar as bibliotecas**

```
pip install -r requirements.txt
```

**4. Rodar o projeto**

```
python main.py
```

O terminal vai mostrar a análise, o tratamento, as métricas e a comparação dos modelos. Os gráficos e a tabela de resultados vão para a pasta `outputs/`.
