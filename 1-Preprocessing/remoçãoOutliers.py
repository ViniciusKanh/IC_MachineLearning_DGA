import pandas as pd
from scipy import stats

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
output_file = '0-Datasets/Dataset_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar os dados
data = pd.read_csv(input_file, names=names)

# Definir os limites
V = {'H2': 25, 'CH4': 43, 'C2H2': 1, 'C2H4': 20, 'C2H6': 14.8}

# Identificar outliers para o target igual a 1 usando Z-Score
z_threshold = 3  # Limite de Z-Score para considerar uma amostra como outlier
for gas in features:
    z_scores = stats.zscore(data[data[target] == 1][gas])
    data = data.drop(data[(data[target] == 1) & (z_scores > z_threshold)].index)

# Salvar os dados tratados
data.to_csv(output_file, index=False, header=False)
