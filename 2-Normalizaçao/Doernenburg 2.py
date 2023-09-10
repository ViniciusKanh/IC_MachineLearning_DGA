import pandas as pd

def doernenburg_ratios(h2, ch4, c2h2, c2h4, c2h6):
    # Cálculo das razões Doernenburg
    r1 = ch4 / h2
    r2 = c2h2 / ch4
    r3 = c2h4 / ch4
    r4 = c2h6 / ch4
    return r1, r2, r3, r4

def doernenburg_classification(r1, r2, r3, r4, h2, ch4, c2h2, c2h4, c2h6):
    # Definição dos limites para classificação
    l1 = {
        "H2": 100,
        "CH4": 120,
        "C2H2": 350,
        "C2H4": 50,
        "C2H6": 65,
    }
    
    if all([h2 < l1["H2"], ch4 < l1["CH4"], c2h2 < l1["C2H2"], c2h4 < l1["C2H4"], c2h6 < l1["C2H6"]]):
        return "Sem Falha"
    else:
        if r1 <= 1.0:
            if r2 < 0.1:
                return "Descarga Parcial"
            elif 0.1 <= r2 <= 1.0:
                if r3 < 1.0:
                    return "Falha Térmica"
                else:
                    return "Falha de Arco"
            elif r2 > 1.0:
                return "Falha Térmica"
        else:
            if r4 > 0.4:
                return "Falha de Arco"
            else:
                return "Falha Térmica"

# Carregar os dados
input_file = '0-Datasets/Dataset_Tratado_Analise_Metodos.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Calcular as classificações usando as funções definidas
fault_types = []
for index, row in df.iterrows():
    h2, ch4, c2h2, c2h4, c2h6 = row['H2'], row['CH4'], row['C2H2'], row['C2H4'], row['C2H6']
    r1, r2, r3, r4 = doernenburg_ratios(h2, ch4, c2h2, c2h4, c2h6)
    fault_type = doernenburg_classification(r1, r2, r3, r4, h2, ch4, c2h2, c2h4, c2h6)
    fault_types.append(fault_type)

# Adicionar a coluna "interpretação" ao DataFrame
df['interpretação'] = fault_types

# Mapear os valores das colunas "interpretação" e "defeito" para valores numéricos para facilitar a comparação
map_classifications = {
    "Sem Falha": 1,  # Normal
    "Descarga Parcial": 3,  # Falha Elétrica
    "Falha de Arco": 3,  # Falha Elétrica
    "Falha Térmica": 2,  # Falha Térmica
}
df['interpretação'] = df['interpretação'].map(map_classifications)
df['defeito'] = df['defeito'].map(map_classifications)

# Calcular o percentual de acerto
acertos = df[df['interpretação'] == df['defeito']].shape[0]
total_amostras = df.shape[0]
percentual_acerto = (acertos / total_amostras) * 100

# Exibir o percentual de acerto
print(f'Percentual de acerto: {percentual_acerto:.2f}%')

# Especificar o nome do arquivo de saída
output_file = '0-Datasets/Analise Normativa/Doernenburg/Metodo Doernenburg.xlsx'
df.to_excel(output_file, index=False)
print("Dados salvos em", output_file)
