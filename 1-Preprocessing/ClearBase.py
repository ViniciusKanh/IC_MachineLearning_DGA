import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.decomposition import PCA

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

data = pd.read_csv(input_file, names=names)

# Remover duplicatas
data = data.drop_duplicates()

# Verificar e lidar com valores ausentes (se houver)
missing_values = data.isnull().sum()
# Se houver valores ausentes, você pode preenchê-los ou removê-los, dependendo do contexto dos dados.

# Separar os dados de entrada (features) e o alvo (target)
X = data[features]
y = data[target]

# Normalizar os dados
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# Salvar os dados normalizados em um arquivo de saída
output_file = '0-Datasets/normalized_data.csv'
output_data = pd.DataFrame(X_normalized, columns=features)
output_data.to_csv(output_file, index=False)

# Redução de dimensionalidade (PCA)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_normalized)

# Plotar gráfico de PCA
plt.scatter(X_reduced[:, 0], X_reduced[:, 1])
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('Gráfico de PCA')
plt.show()
