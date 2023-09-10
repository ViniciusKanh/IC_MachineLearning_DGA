import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
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

# Padronizar os dados
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Aplicar o t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_standardized)

# Criar um DataFrame para os componentes do t-SNE e o alvo
tsne_df = pd.DataFrame(data=X_tsne, columns=['Componente 1', 'Componente 2'])
tsne_df['Alvo'] = y.map(
    {1: 'Normal', 2: 'Falha Térmica', 3: 'Falha Elétrica'})

# Plotar gráfico de t-SNE
plt.figure(figsize=(8, 6))
targets = ['Normal', 'Falha Térmica', 'Falha Elétrica']
colors = ['blue', 'red', 'green']
for target, color in zip(targets, colors):
    indices = tsne_df['Alvo'] == target
    plt.scatter(tsne_df.loc[indices, 'Componente 1'],
                tsne_df.loc[indices, 'Componente 2'],
                c=color, label=target)
plt.xlabel('Componente 1')
plt.ylabel('Componente 2')
plt.title('Gráfico de t-SNE')
plt.legend()
plt.show()
