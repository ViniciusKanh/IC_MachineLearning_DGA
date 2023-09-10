import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '0-Datasets/DataBase_Trafo_Gas_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'defeito'

# Carregar o arquivo CSV em um DataFrame
df = pd.read_csv(input_file, names=names)

# Estatísticas descritivas
print(df.describe())

# Correlação entre as variáveis
correlation = df[features].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm')
plt.title('Correlação entre as variáveis')
plt.show()

# Distribuição das variáveis
for feature in features:
    sns.histplot(data=df, x=feature, kde=True)
    plt.title(f'Distribuição de {feature}')
    plt.show()
