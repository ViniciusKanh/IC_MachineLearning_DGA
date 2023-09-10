import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file, names=names)

# Separar os dados de entrada e o alvo
X = df[features]
y = df[target]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Criar e treinar o modelo de árvore de decisão
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Avaliar o desempenho do modelo nos dados de teste
accuracy = model.score(X_test, y_test)
print("Acurácia do modelo:", accuracy)
