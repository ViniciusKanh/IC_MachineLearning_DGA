import pandas as pd

# Carrega o arquivo CSV em um DataFrame
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Calcula as razões Rogers R1, R2, R5
df['R1'] = df['CH4'] / df['H2']
df['R2'] = df['C2H2'] / df['C2H4']
df['R5'] = df['C2H4'] / df['C2H6']

# Create a function to apply the classification conditions
def classify_samples(row):
    if row['R2'] < 0.1 and row['R1'] > (0.1 and 1.0) and row['R5'] < 1:
        return 'NO FAUT' 
    
# Classifique as amostras usando a função classify_samples
df['classificacao'] = df.apply(classify_samples, axis=1)

# Especifique o nome do arquivo de saída
output_file = '0-Datasets/Analise Normativa/Metodo da Razão Roger/Metodo_Razão_Rogers.xlsx'

# Crie um objeto ExcelWriter
writer = pd.ExcelWriter(output_file)

# Escreva as colunas 'defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6' e 'classificacao' no arquivo Excel
df[['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6', 'classificacao']].to_excel(writer, index=False)

# Feche o objeto ExcelWriter
writer.save()

# Exiba uma mensagem indicando que o arquivo foi gerado
print(f'Arquivo de saída "{output_file}" gerado com sucesso.')
