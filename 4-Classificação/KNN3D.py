import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Carregar o conjunto de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'
df = pd.read_csv(input_file, names=names)

# Dados de exemplo
X = df[features].values
y = df[target].values

# Normalização Min-Max
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Instanciando o classificador KNN com distância euclidiana
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

# Ajustando o modelo aos dados normalizados
knn.fit(X_scaled[:, :3], y)

# Pontos para predição
x_min, x_max = X_scaled[:, 0].min() - 0.1, X_scaled[:, 0].max() + 0.1
y_min, y_max = X_scaled[:, 1].min() - 0.1, X_scaled[:, 1].max() + 0.1
z_min, z_max = X_scaled[:, 2].min() - 0.1, X_scaled[:, 2].max() + 0.1
xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01),
                         np.arange(z_min, z_max, 0.01))
points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

# Predição para os pontos
Z = knn.predict(points)

# Plotando os resultados em 3D
Z = Z.reshape(xx.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], c=y, s=20, edgecolor='k')

# A linha abaixo foi removida para não plotar o plano azul (superfície de decisão do KNN)
# ax.plot_surface(xx[:,:,0], yy[:,:,0], zz[:,:,0], rstride=1, cstride=1, facecolors=plt.cm.jet(Z[:,:,0]), alpha=0.4)

ax.set_xlabel('H2 (Normalizado)')
ax.set_ylabel('CH4 (Normalizado)')
ax.set_zlabel('C2H2 (Normalizado)')
ax.set_title('Classificação utilizando KNN com normalização Min-Max e distância euclidiana')

# Legenda para as classes
class1_patch = plt.scatter([], [], c='r', s=20, edgecolor='k')
class2_patch = plt.scatter([], [], c='g', s=20, edgecolor='k')
class3_patch = plt.scatter([], [], c='b', s=20, edgecolor='k')
ax.legend([class1_patch, class2_patch, class3_patch], ['Defeito 1', 'Defeito 2', 'Defeito 3'])

plt.show()
