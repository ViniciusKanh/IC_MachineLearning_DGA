import pandas as pd

input_file = '0-Datasets/DataBase_Tratado.data'
output_file = '0-Datasets/BaseDados_Tradados.data'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file)

# Substituir as vírgulas por pontos nas colunas numéricas
numeric_columns = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df[numeric_columns] = df[numeric_columns].apply(lambda x: x.str.replace(',', '.'))

# Salvar o novo arquivo tratado
df.to_csv(output_file, index=False)

print("Novo arquivo tratado salvo com sucesso!")
