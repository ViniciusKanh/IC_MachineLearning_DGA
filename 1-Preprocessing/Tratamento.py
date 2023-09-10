import pandas as pd
import numpy as np
from sklearn.covariance import EllipticEnvelope

# Carregar a base de dados
input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
output_file = '0-Datasets/DataBase_Tratado.xlsx'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'
data = pd.read_csv(input_file, names=names)

# Remover outliers usando EllipticEnvelope
outlier_detector = EllipticEnvelope(contamination=0.05)  # Define a taxa de contaminação
outlier_detector.fit(data[features])
outlier_mask = outlier_detector.predict(data[features]) != -1
data = data[outlier_mask]

# Verificar a qualidade da base de dados
quality_report = {}

# Verificar valores ausentes
missing_values = data.isna().sum()
quality_report['Valores Ausentes'] = missing_values

# Verificar valores duplicados
duplicate_values = data.duplicated().sum()
quality_report['Valores Duplicados'] = duplicate_values

# Verificar a consistência dos dados
inconsistent_values = {}
for feature in features:
    unique_values = data[feature].unique()
    inconsistent_values[feature] = [value for value in unique_values if sum(pd.isnull(pd.to_numeric(unique_values, errors='coerce'))) > 0]

# Verificar se há valores inconsistentes em cada coluna numérica
inconsistent_columns = [feature for feature, values in inconsistent_values.items() if len(values) > 0]

quality_report['Valores Inconsistentes'] = inconsistent_values
quality_report['Colunas Inconsistentes'] = inconsistent_columns

# Verificar a contagem de valores únicos
unique_values_count = data[features].nunique()
quality_report['Contagem de Valores Únicos'] = unique_values_count

# Verificar a contagem de valores zero
zero_values_count = (data[features] == 0).sum()
quality_report['Contagem de Valores Zero'] = zero_values_count

# Exibir relatório de qualidade da base de dados
for metric, values in quality_report.items():
    print(f"{metric}:\n{values}\n")

# Estabelecer limites de alerta e alarme com base nas normas
limites = {
    2: {
        'alerta': {'H2': 1000, 'CH4': 100, 'C2H2': 100, 'C2H4': 500, 'C2H6': 100},
        'alarme': {'H2': 1500, 'CH4': 300, 'C2H2': 300, 'C2H4': 1000, 'C2H6': 200}
    },
    3: {
        'alerta': {'H2': 300, 'CH4': 50, 'C2H2': 50, 'C2H4': 200, 'C2H6': 50},
        'alarme': {'H2': 500, 'CH4': 100, 'C2H2': 100, 'C2H4': 500, 'C2H6': 100}
    }
}

# Verificar os níveis de gases e atribuir status com base nos limites de alerta e alarme
data['status'] = 'Normal'
for index, row in data.iterrows():
    defeito = int(row['defeito'])
    if defeito in limites:
        for gas in features[1:]:
            if row[gas] >= limites[defeito]['alarme'][gas]:
                data.at[index, 'status'] = 'Alarme'
                break
            elif row[gas] >= limites[defeito]['alerta'][gas]:
                data.at[index, 'status'] = 'Alerta'
    else:
        print(f"Classe de defeito inválida para a amostra {index}: {defeito}")
        
# Remove amostras duplicadas com base em todas as colunas
data = data.drop_duplicates()

# Correção de erros de medição
# Exemplo: Substituir valores negativos por zero
data[features[1:]] = data[features[1:]].apply(lambda x: np.where(x < 0, 0, x))

# Salvar dados tratados em um arquivo Excel
data.to_excel(output_file, index=False)
