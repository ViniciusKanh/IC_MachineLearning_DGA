import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import random

# Carregar a base de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

data = pd.read_csv(input_file, names=names)

# Separar os dados de treinamento por defeito
defect1_data = data[data['Defeito'] == 1][features]
defect2_data = data[data['Defeito'] == 2][features]
defect3_data = data[data['Defeito'] == 3][features]

# Calcular a densidade de probabilidade para cada defeito
defect1_kde = KernelDensity().fit(defect1_data)
defect2_kde = KernelDensity().fit(defect2_data)
defect3_kde = KernelDensity().fit(defect3_data)

# Gerar novos dados com base nos padrões aprendidos
num_samples = 100  # Número de novas amostras a serem geradas
new_data = pd.DataFrame(columns=names)

for _ in range(num_samples):
    new_row = [None] * len(names)
    
    # Escolher aleatoriamente um defeito com base nas proporções na base de dados original
    defect = random.choices([1, 2, 3], weights=[len(defect1_data), len(defect2_data), len(defect3_data)])[0]
    
    # Gerar um novo dado com base no defeito escolhido
    if defect == 1:
        new_data_point = defect1_kde.sample()
    elif defect == 2:
        new_data_point = defect2_kde.sample()
    else:
        new_data_point = defect3_kde.sample()
    
    new_row[0] = defect
    new_row[1:] = new_data_point.flatten()
    
    new_data = new_data.append(pd.Series(new_row, index=names), ignore_index=True)

# Salvar os novos dados em um arquivo Excel
new_data.to_excel('novos_dados.xlsx', index=False)
