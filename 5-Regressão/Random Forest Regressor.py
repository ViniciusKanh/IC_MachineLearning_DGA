import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Carregar a base de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'Ano']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

data = pd.read_csv(input_file, names=names)

# Dividir os dados em conjuntos de treinamento e teste
X = data[features].values
y = data[target].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar e ajustar o modelo de regressão Random Forest
regression_model = RandomForestRegressor(n_estimators=100, random_state=42)
regression_model.fit(X_train, y_train)

# Realizar previsões no conjunto de teste
y_pred = regression_model.predict(X_test)

# Avaliar o modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Plotar resultados
plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Valor Real')
plt.ylabel('Valor Previsto')
plt.title('Comparação entre Valor Real e Valor Previsto')
plt.show()

# Exibir métricas de avaliação
print('Mean Squared Error (MSE):', mse)
print('Coefficient of Determination (R^2):', r2)
