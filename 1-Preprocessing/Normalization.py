import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler,RobustScaler

input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file, names=names, skiprows=1)

# Converter as colunas numéricas para tipo float
df[features] = df[features].astype(float)

# Excluir a coluna 'defeito' do processo de normalização
df_features = df[features]

# Aplicar normalização min-max
scaler_minmax = MinMaxScaler()
df_features = scaler_minmax.fit_transform(df_features)

# Aplicar padronização (z-score)
scaler_standard = StandardScaler()
df_features = scaler_standard.fit_transform(df_features)

# Atualizar o DataFrame com as colunas normalizadas
df[features] = df_features

# Exibir o DataFrame após a normalização dos dados
print("DataFrame após a normalização dos dados:\n", df.head())
