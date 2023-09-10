import pandas as pd

# Carregar os resultados dos métodos
# Método: Razão de Roger
df_roger = pd.read_excel('0-Datasets/Analise Normativa/Metodo da Razão Roger/Metodo_Razão_Rogers.xlsx')

# Método: Doernenburg
df_doernenburg = pd.read_excel('0-Datasets/Analise Normativa/Doernenburg/Metodo Doernenburg.xlsx')

# Método: Gas Chave
df_gas_chave = pd.read_excel('0-Datasets/Analise Normativa/Gas Chave/Metodo do Gas_Chave.xlsx')

# Método: Triângulo de Duval
df_duval = pd.read_excel('0-Datasets/Analise Normativa/Metodo de Duval/Anal_Trig_Duval.xlsx')

# Juntar os DataFrames
merged_df = pd.merge(df_roger, df_doernenburg[['interpretação']], left_index=True, right_index=True)
# Check if the column 'Defeito_Classificado' exists in df_gas_chave before merging
if 'Defeito_Classificado' in df_gas_chave.columns:
    merged_df = pd.merge(merged_df, df_gas_chave[['Defeito_Classificado']], left_index=True, right_index=True)
else:
    print("Column 'Defeito_Classificado' not found in df_gas_chave.")

merged_df = pd.merge(merged_df, df_duval[['Defeito_Classificado']], left_index=True, right_index=True)

# Renomear as colunas adicionais
merged_df.rename(columns={'interpretação': 'Doernenburg', 'Defeito_Classificado_x': 'Gas_Chava', 'Defeito_Classificado_y': 'Duval'}, inplace=True)

# Especificar o nome do arquivo de saída
output_file = '0-Datasets/Analise Normativa/Analise_Completa.xlsx'

# Salvar o DataFrame combinado em um arquivo Excel
merged_df.to_excel(output_file, index=False)

print(f'Arquivo de saída "{output_file}" criado com sucesso!')