import pandas as pd
import matplotlib.pyplot as plt

def evaluate_key_gas_accuracy(data):
    data['Resultado'] = ''
    
    for index, row in data.iterrows():
        if (row['Key_Gas'] == 'Superaquecimento de alta temperatura do óleo Key gas: C2H4' or
           row['Key_Gas'] == 'Superaquecimento de alta temperatura do óleo Key gas: C2H6') and row['defeito'] == 2:
            data.at[index, 'Resultado'] = 'Acertou'
        elif (row['Key_Gas'] == 'Arco Key gas: C2H2' or
              row['Key_Gas'] == 'Arco Key gas: CH4' or
              row['Key_Gas'] == 'Arco Key gas: H2') and row['defeito'] == 3:
            data.at[index, 'Resultado'] = 'Acertou'
        else:
            data.at[index, 'Resultado'] = 'Errou'
    
    return data
input_file = '0-Datasets/DataBase_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

data = pd.read_csv(input_file, names=names)

X = data[features]
y = data[target]

# Contagem condicional
c2h4_c2h6_max_defeito_2 = data[(X.idxmax(axis=1).isin(['C2H4', 'C2H6'])) & (y == 2)].shape[0]
c2h2_ch4_h2_max_defeito_3 = data[(X.idxmax(axis=1).isin(['C2H2', 'CH4', 'H2'])) & (y == 3)].shape[0]

print(f"Contagem de linhas onde 'C2H4' ou 'C2H6' é máximo e defeito é 2: {c2h4_c2h6_max_defeito_2}")
print(f"Contagem de linhas onde 'C2H2', 'CH4' ou 'H2' é máximo e defeito é 3: {c2h2_ch4_h2_max_defeito_3}")

key_gases = {
    'H2': 'Arco Key gas: H2 ',
    'CH4': 'Arco Key gas: CH4 ',
    'C2H2': 'Arco Key gas: C2H2 ',
    'C2H4': 'Superaquecimento de alta temperatura do óleo Key gas: C2H4 ',
    'C2H6': 'Superaquecimento de alta temperatura do óleo Key gas: C2H6 '
}

data['Key_Gas'] = data[features].apply(lambda row: key_gases[row.idxmax()], axis=1)

data = evaluate_key_gas_accuracy(data)

output_file = '0-Datasets/Analise Normativa/Gas Chave/Metodo do Gas_Chave.xlsx'
data.to_excel(output_file, index=False)
