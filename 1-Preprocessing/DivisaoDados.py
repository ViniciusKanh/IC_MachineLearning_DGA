import pandas as pd
from sklearn.model_selection import train_test_split

input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file, names=names)

# Separar os dados de entrada e o alvo
X = df[features]
y = df[target]

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Exibir o tamanho dos conjuntos de treinamento e teste
print("Tamanho do conjunto de treinamento:", X_train.shape[0])
print("Tamanho do conjunto de teste:", X_test.shape[0])
