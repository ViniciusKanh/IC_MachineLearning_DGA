import pandas as pd
from sklearn.model_selection import train_test_split
from opfython.models import SupervisedOPF
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler,RobustScaler,StandardScaler,MaxAbsScaler,Normalizer
from tabulate import tabulate
import numpy as np


# Carregar o conjunto de dados
input_file = '0-Datasets/DataBase_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Selecionar as características e o alvo
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'
X = df[features].values
y = df[target].values

# Normalizar manualmente os dados de entrada com a técnica min-max
#min_values = np.min(X, axis=0)
#max_values = np.max(X, axis=0)
#X = (X - min_values) / (max_values - min_values)

# Normalizar manualmente os dados de entrada com a técnica Z-score
mean_values = np.mean(X, axis=0)
std_values = np.std(X, axis=0)
X = (X - mean_values) / std_values

# Codificar o alvo para que ele seja numérico
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Dividir os dados em conjuntos de treinamento e teste
train_data, test_data, train_labels, test_labels = train_test_split(X, y, test_size=0.3, random_state=42)

# Criar uma instância do classificador OPF supervisionado
opf = SupervisedOPF(distance='log_squared_euclidean')

# Treinar o classificador OPF usando os dados de treinamento
opf.fit(train_data, train_labels)

# Testar o classificador OPF com os dados de teste
predictions = opf.predict(test_data)

# Calcular a precisão, recall e f1_score do classificador
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')
precision = precision_score(test_labels, predictions, average='weighted')
recall = recall_score(test_labels, predictions, average='weighted')

# Criar a matriz de confusão
conf_matrix = confusion_matrix(test_labels, predictions)
conf_matrix = pd.DataFrame(conf_matrix, index=encoder.classes_, columns=encoder.classes_)

# Imprimir as métricas e a matriz de confusão
print(f'A acurácia do classificador OPF é: {accuracy}')
print(f'O f1_score do classificador OPF é: {f1}')
print(f'A precisão do classificador OPF é: {precision}')
print(f'O recall do classificador OPF é: {recall}')
print("\nMatriz de confusão:")
print(tabulate(conf_matrix, headers='keys', tablefmt='fancy_grid'))
