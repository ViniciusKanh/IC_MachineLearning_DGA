import pandas as pd
import numpy as np

# Carrega o arquivo CSV em um DataFrame
input_file = '0-Datasets/Dataset_Tratado_Analise_Metodos.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Calcula as razões Rogers R1, R2, R5
df['R1'] = np.divide(df['CH4'], df['H2'], out=np.zeros_like(df['CH4']), where=df['H2'] != 0)
df['R2'] = np.divide(df['C2H2'], df['C2H4'], out=np.zeros_like(df['C2H2']), where=df['C2H4'] != 0)
df['R5'] = np.divide(df['C2H4'], df['C2H6'], out=np.zeros_like(df['C2H4']), where=df['C2H6'] != 0)

# Create a function to apply the classification conditions
def classify_samples(row):
    if row['R2'] < 0.1 and 0.1 <= row['R1'] <= 1.0 and row['R5'] < 1.0:
        return 'NO FAULT' 
    if row['R2'] <=  0.1 and row['R1'] <= 0.1 and row['R5'] < 1.0:
        return 'PARTIAL DISCHARGE'
    if 0.1 <= row['R2'] <= 3.0 and 0.1 <= row['R1'] <= 1.0 and row['R5'] > 3.0:
        return 'HIGH-ENERGY ARCING'
    if row['R2'] < 0.1 and 0.1 <= row['R1'] <= 1.0 and 1.0 <= row['R5'] <= 3.0 :
        return 'LOW TEMPERATURE THERMAL OVERLOADING'
    if row['R2'] < 0.1 and row['R1'] > 1.0 and 1.0 <= row['R5'] <= 3.0 :
        return 'THERMAL < 700 C'
    if row['R2'] < 0.1 and row['R1'] > 1.0 and row['R5'] > 3.0 :
        return 'THERMAL > 700 C'
    
    # Se nenhuma das condições anteriores for satisfeita, retorna a mensagem de não classificação
    return 'Não foi possível classificar'

# Classifique as amostras usando a função classify_samples
df['classificacao'] = df.apply(classify_samples, axis=1)

# Exiba as colunas 'defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'R1', 'R2' e 'R5' no terminal
print(df[['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'R1', 'R2', 'R5', 'classificacao']])

# Calcular o percentual de acerto
acertos = df[df['defeito'] == df['classificacao']].shape[0]
total_amostras = df.shape[0]
percentual_acerto = (acertos / total_amostras) * 100

# Exibir o percentual de acerto
print(f'Percentual de acerto: {percentual_acerto:.2f}%')

# Especifique o nome do arquivo de saída
output_file = '0-Datasets/Analise Normativa/Metodo da Razão Roger/Metodo_Razão_Rogers.xlsx'

# Crie um objeto ExcelWriter
writer = pd.ExcelWriter(output_file)

# Escreva as colunas 'defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'R1', 'R2' e 'R5' no arquivo Excel
df.to_excel(writer, index=False)

# Feche o objeto ExcelWriter
writer.save()

# Exiba uma mensagem indicando que a classificação foi concluída
print(f'Classificação das amostras concluída. Os resultados foram salvos em "{output_file}".')