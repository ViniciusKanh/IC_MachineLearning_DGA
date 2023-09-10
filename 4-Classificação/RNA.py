import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt

# Carregar o conjunto de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'
df = pd.read_csv(input_file, names=names)

# Dados de exemplo
X = df[features].values
y = df[target].values

# Dividir em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalização Min-Max
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo
model = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42, solver='adam', activation='relu')
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Calcular as métricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)

# Organizar as métricas na tabela
metrics = [['Acurácia', accuracy],
           ['F1-Score', f1],
           ['Precisão', precision]]

# Imprimir as métricas
print(tabulate(metrics, headers=['Métrica', 'Valor']))

# Obter os rótulos únicos esperados
labels = np.unique(y).astype(str)

# Converter a matriz de confusão para uma lista de listas
confusion_table_list = confusion.tolist()

# Adicionar rótulos aos valores da matriz de confusão
confusion_table_list = [[''] + labels.tolist()] + list(zip(labels.tolist(), confusion_table_list))

# Imprimir a matriz de confusão
print('\nMatriz de Confusão:')
print(tabulate(confusion_table_list, headers='firstrow', tablefmt='grid'))

# Plotar o diagrama da rede neural
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
cmap = plt.get_cmap('tab10')

# Configurar a posição dos neurônios nas camadas ocultas
layer_sizes = [X_train.shape[1]] + list(model.hidden_layer_sizes) + [len(labels)]
pos_y = np.concatenate([np.linspace(0.1, 0.9, num=n) for n in layer_sizes])

# Plotar os neurônios e as conexões
for i in range(len(layer_sizes) - 1):
    layer_left = layer_sizes[i]
    layer_right = layer_sizes[i + 1]
    layer_width = (i + 1) / len(layer_sizes)
    for j in range(layer_left):
        for k in range(layer_right):
            x = [i, i + 1]
            y = [pos_y[j], pos_y[k + layer_left]]
            ax.plot(x, y, color='gray')
    ax.scatter([i] * layer_left, pos_y[:layer_left], color=cmap(i), s=100, label=f'Camada {i+1}')
    ax.scatter([i + 1] * layer_right, pos_y[layer_left:layer_left+layer_right], color=cmap(i), s=100)

# Configurar os rótulos dos neurônios
for i, size in enumerate(layer_sizes[:-1]):
    for j in range(size):
        ax.annotate(f'Neurônio {j+1}', xy=(i, pos_y[j]), xytext=(-20, 0), textcoords='offset points',
                    ha='center', va='center', color='black', fontsize=8)

# Configurar a legenda
ax.legend(loc='upper right')

# Configurar os rótulos dos eixos
ax.set_xticks(range(len(layer_sizes)))
ax.set_xticklabels(['Entrada'] + [f'Camada {i+1}' for i in range(len(layer_sizes) - 2)] + ['Saída'])
ax.set_yticks([])
ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.set_title('Arquitetura da Rede Neural')

# Exibir o gráfico
plt.show()
