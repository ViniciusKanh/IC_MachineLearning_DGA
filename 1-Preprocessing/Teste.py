import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

data = pd.read_csv(input_file, names=names)


# Pré-processamento dos dados
# Dividir os dados em conjunto de treinamento e conjunto de teste
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar as características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Exemplo de uso dos dados pré-processados
print("Dados de treinamento:")
print(X_train)
print("Dados de teste:")
print(X_test)
print("Rótulos de treinamento:")
print(y_train)
print("Rótulos de teste:")
print(y_test)
