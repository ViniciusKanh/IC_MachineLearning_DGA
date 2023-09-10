import pandas as pd

input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file, names=names)

# Verificar se há valores ausentes
missing_values = df.isnull().sum()
print("Valores ausentes:\n", missing_values)

# Realizar a limpeza dos dados
# Preencher valores faltantes com a média da coluna (ou outra estratégia de sua preferência)
df = df.fillna(df.mean())

# Verificar dados inconsistentes ou discrepantes
# (Aqui você pode adicionar o código para identificar e tratar outliers, se necessário)

# Remover outliers (exemplo utilizando Z-Score)
z_scores = (df[features] - df[features].mean()) / df[features].std()
df = df[(z_scores < 3).all(axis=1)]

# Exibir o DataFrame após o pré-processamento
print("DataFrame após o pré-processamento:\n", df.head())
