# Importando as bibliotecas necessárias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'
data = pd.read_csv(input_file, names=names)

# Rotular as classes
class_labels = {1: 'Normal', 2: 'Falha Térmica', 3: 'Falha Elétrica'}
data[target] = data[target].map(class_labels)

# Verificar se a base de dados é supervisionada ou não supervisionada
if target in data.columns:
    is_supervised = True
    target_values = data[target].unique()
    num_classes = len(target_values)
    class_counts = data[target].value_counts()

    print("A base de dados é supervisionada.")
    print("Número de classes:", num_classes)
    print("Classes:", target_values)
    print("Contagem de instâncias em cada classe:")
    print(class_counts)
else:
    is_supervised = False
    print("A base de dados é não supervisionada.")

# Seleção de características
X = data[features]
y = data[target]

# Normalizar os dados para que todos estejam na mesma escala
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Selecionar as melhores características usando chi-quadrado
selector = SelectKBest(chi2, k=3)
X_new = selector.fit_transform(X_scaled, y)
print("Melhores características:", [features[i] for i in selector.get_support(indices=True)])

print("Balanceamento das classes:")
print(data[target].value_counts(normalize=True))

