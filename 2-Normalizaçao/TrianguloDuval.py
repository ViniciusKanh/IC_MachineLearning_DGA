import pandas as pd
import matplotlib.pyplot as plt

def classify_defect(h2, ch4, c2h2, c2h4, c2h6):
    if ch4 >= 98:
        return 'Partial Discharges'
    elif ch4 < 98 and c2h4 < 20 and c2h2 < 4:
        return 'Thermal Fault, T <300 °C'
    elif c2h4 >= 20 and c2h4 < 50 and c2h2 < 4:
        return 'Thermal Fault, 300 °C < t <700 °C'
    elif c2h4 >= 50 and c2h2 < 15:
        return 'Thermal Fault, t >700 °C'
    elif c2h4 < 50 and (c2h2 >= 4 and c2h2 < 13) and (ch4 >= 40 and ch4 < 50) and (c2h4 >= 13 and c2h4 < 29):
        return 'Electrical/ Thermal Faults'
    elif c2h4 >= 50 and (c2h2 >= 15 and c2h2 < 29):
        return 'Electrical/ Thermal Faults'
    elif c2h4 < 23 and c2h2 >= 13:
        return 'Discharges Of Low Energy'
    elif c2h4 >= 23 and c2h2 >= 29:
        return 'Discharges Of High Energy'
    elif (c2h4 >= 23 and c2h4 < 40) and (c2h2 >= 13 and c2h2 < 29):
        return 'Discharges Of High Energy'
    else:
        return 'Unknown'

# Carrega o arquivo CSV em um DataFrame
input_file = '0-Datasets/Dataset_Tratado_Analise_Metodos.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

# Obter os valores dos gases do DataFrame
h2_values = df['H2'].values
ch4_values = df['CH4'].values
c2h2_values = df['C2H2'].values
c2h4_values = df['C2H4'].values
c2h6_values = df['C2H6'].values

# Classificar os defeitos para cada amostra
defects = []
for i in range(len(df)):
    defect = classify_defect(h2_values[i], ch4_values[i], c2h2_values[i], c2h4_values[i], c2h6_values[i])
    defects.append(defect)

# Adicionar os defeitos ao DataFrame
df['Defeito_Classificado'] = defects

# Imprimir o DataFrame com os defeitos classificados
print(df)

# Exportar o DataFrame para um arquivo Excel
output_file = '0-Datasets/Anal_Trig_Duval.xlsx'
df.to_excel(output_file, index=False)


# Gerar o gráfico do Triângulo de Duval
fig, ax = plt.subplots()
ax.plot([0, 100, 0], [0, 0, 100], 'k-', color='gray')  # Triangular boundary
ax.plot(df['CH4'], df['C2H2'], 'ro')  # Plotting CH4 vs C2H2
ax.plot(df['CH4'], df['C2H4'], 'go')  # Plotting CH4 vs C2H4
ax.plot(df['C2H2'], df['C2H4'], 'bo')  # Plotting C2H2 vs C2H4
ax.fill(df['CH4'], df['C2H2'], 'gray', alpha=0.3)  # Filling the triangle
ax.set_xlabel('%CH4')
ax.set_ylabel('%C2H2')
ax.set_title('Duval Triangle')
ax.set_xlim([-10, 110])
ax.set_ylim([-10, 110])
ax.set_xticks(range(0, 101, 10))
ax.set_yticks(range(0, 101, 10))
ax.grid(True)
plt.show()
