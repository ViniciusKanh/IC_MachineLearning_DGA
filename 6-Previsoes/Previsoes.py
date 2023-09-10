import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

data = pd.read_csv(input_file, names=names)

# Pré-processamento dos dados
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Dividir os dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(scaled_data, data[target], test_size=0.3)

# Treinar um modelo Random Forest
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy}')

# Gerar novos dados usando o modelo treinado
num_samples = 100  # Número de amostras a serem geradas
generated_samples = np.random.rand(num_samples, len(features))
generated_samples = scaler.inverse_transform(generated_samples)

# Fazer previsões nos dados gerados
generated_predictions = model.predict(generated_samples)

# Criar um dataframe com os dados gerados e suas previsões
generated_df = pd.DataFrame(generated_samples, columns=features)
generated_df[target] = generated_predictions

# Salvar os novos dados em um arquivo CSV
generated_df.to_csv('novos_dados.csv', index=False)
