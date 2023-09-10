import pandas as pd

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
output_file = '0-Datasets/Dataset_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar os dados
data = pd.read_csv(input_file, names=names)

# Definir os limites AND não OR
Vnomrmal = {'H2': 25, 'CH4': 43, 'C2H2': 1, 'C2H4': 20, 'C2H6': 14.8}
Vtermico = {'H2': 25, 'CH4': 43, 'C2H2': 1, 'C2H4': 20, 'C2H6': 14.8}
Veletrico = {'H2': 25, 'CH4': 43, 'C2H2': 1, 'C2H4': 20, 'C2H6': 14.8}

# Encontrar as linhas que precisam ser descartadas
for gas, limit in Vnomrmal.items():
    data = data.drop(data[(data[target] == 1) & (data[gas] > limit)].index)
    data = data.drop(data[(data[target] == 2) & (data[gas] < limit)].index)
    
# Salvar os dados tratados
data.to_csv(output_file, index=False, header=False)
