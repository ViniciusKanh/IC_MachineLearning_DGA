import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o conjunto de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Selecionar as características e o alvo
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

X = df[features].values
y = df[target].values

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar as características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Implementação do OPF manual
def opf_train(X_train, y_train):
    n_samples, n_features = X_train.shape
    n_classes = len(np.unique(y_train))
    
    # Calcular a matriz de distâncias euclidianas entre as amostras
    distances = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            distances[i, j] = np.linalg.norm(X_train[i] - X_train[j])
            distances[j, i] = distances[i, j]
    
    # Calcular os custos entre amostras de diferentes classes
    costs = np.ones((n_samples, n_samples)) * np.inf
    for i in range(n_samples):
        for j in range(n_samples):
            if y_train[i] != y_train[j]:
                costs[i, j] = distances[i, j]
    
    # Inicializar os pesos das amostras
    weights = np.zeros(n_samples)
    for i in range(n_samples):
        indices = np.where(y_train == y_train[i])[0]
        weights[i] = np.min(costs[i, indices])
    
    # Normalizar os pesos (evitando valores nulos ou infinitos)
    max_weight = np.max(weights)
    if max_weight != 0:
        weights /= max_weight
    else:
        weights[:] = 1.0
    
    return weights

def opf_test(X_test, X_train, weights):
    n_samples_test = X_test.shape[0]
    n_samples_train = X_train.shape[0]
    distances = np.zeros((n_samples_test, n_samples_train))
    
    # Calcular as distâncias entre as amostras de teste e treinamento
    for i in range(n_samples_test):
        for j in range(n_samples_train):
            distances[i, j] = np.linalg.norm(X_test[i] - X_train[j])
    
    # Classificar as amostras de teste
    y_pred = np.argmin(distances * weights, axis=1)
    
    return y_pred

# Treinar o OPF
weights = opf_train(X_train, y_train)

# Classificar as amostras de teste
y_pred = opf_test(X_test, X_train, weights)

# Calcular a acurácia
accuracy = np.mean(y_pred == y_test)
print("Acurácia do OPF manual:", accuracy)
