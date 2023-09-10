import pandas as pd

def calculate_gas_ratios(data):
    # Calcular as proporções dos gases
    data['H2/CH4'] = data['H2'] / data['CH4']
    data['H2/C2H2'] = data['H2'] / data['C2H2']
    data['H2/C2H4'] = data['H2'] / data['C2H4']
    data['H2/C2H6'] = data['H2'] / data['C2H6']
    
    return data

def interpret_gas_ratios(data):
    # Interpretar as proporções dos gases de acordo com o método de Doernenburg
    data['Interpretation'] = ''
    
    for index, row in data.iterrows():
        if row['H2/CH4'] >= 0.2 and row['H2/C2H2'] >= 0.2 and row['H2/C2H4'] >= 0.2 and row['H2/C2H6'] >= 0.2:
            data.at[index, 'Interpretation'] = 'Descarga parcial'
        elif row['H2/CH4'] >= 0.2 and row['H2/C2H2'] < 0.2 and row['H2/C2H4'] < 0.2 and row['H2/C2H6'] < 0.2:
            data.at[index, 'Interpretation'] = 'Sobreaquecimento localizado'
        else:
            data.at[index, 'Interpretation'] = 'Normal'
    
    return data

def evaluate_accuracy(data):
    data['Resultado'] = ''
    for index, row in data.iterrows():
        if row['Interpretation'] == 'Normal' and row['defeito'] == 1:
            data.at[index, 'Resultado'] = 'Acertou'
        elif row['Interpretation'] == 'Descarga parcial' and row['defeito'] == 3:
            data.at[index, 'Resultado'] = 'Acertou'
        elif row['Interpretation'] == 'Sobreaquecimento localizado' and row['defeito'] == 2:
            data.at[index, 'Resultado'] = 'Acertou'
        else:
            data.at[index, 'Resultado'] = 'Errou'
    
    return data

# Definir o nome do arquivo de entrada, colunas e características
input_file = '0-Datasets/Database_Tratado.data'
output_file = '0-Datasets/Analise Normativa/Doernenburg/Metodo Doernenburg.xlsx'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']

# Ler os dados do arquivo para um DataFrame
data = pd.read_csv(input_file, names=names)

# Calcular as proporções dos gases
data = calculate_gas_ratios(data)

# Interpretar as proporções dos gases
data = interpret_gas_ratios(data)

# Avaliar a acurácia da classificação
data = evaluate_accuracy(data)

# Salvar os resultados em um arquivo Excel
data.to_excel(output_file, index=False)

print("Arquivo Excel gerado com sucesso!")
