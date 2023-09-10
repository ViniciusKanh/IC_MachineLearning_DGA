import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file, names=names)

# Separar os dados de entrada e o alvo
X = df[features]
y = df[target]

# Aplicar a seleção de características
selector = SelectKBest(score_func=f_classif, k=3)  # Selecionar as 3 melhores características
X_new = selector.fit_transform(X, y)

# Obter os índices das características selecionadas
selected_indices = selector.get_support(indices=True)
selected_features = [features[i] for i in selected_indices]

# Exibir as características selecionadas
print("Características selecionadas:")
print(selected_features)
