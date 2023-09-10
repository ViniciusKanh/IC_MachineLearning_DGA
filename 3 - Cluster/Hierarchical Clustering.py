import pandas as pd
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Carregar a base de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

data = pd.read_csv(input_file, names=names)

# Remover a coluna 'Defeito' do conjunto de dados para fins de clusterização
X = data.drop(['Defeito'], axis=1)

# Aplicar o algoritmo de clustering hierárquico
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('Dendrograma')
plt.xlabel('Amostras')
plt.ylabel('Distância Euclidiana')
plt.show()

# Definir o número de clusters e aplicar o algoritmo AgglomerativeClustering
n_clusters = 3
hc = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
clusters = hc.fit_predict(X)

# Plotar os resultados da clusterização
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clusters)
plt.title('Hierarchical Clustering')
plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.show()
