import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path

# Código 2
input_file = '0-Datasets/DataBase_Tratado.data'
names = ['defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
df = pd.read_csv(input_file, names=names)

total_concentration = df[['C2H2', 'C2H4', 'CH4']].sum(axis=1)
df['Wc2h2'] = df['C2H2'] * 100 / total_concentration
df['Wc2h4'] = df['C2H4'] * 100 / total_concentration 
df['WCh4'] = df['CH4'] * 100 / total_concentration 

# Código 1
A = np.array([[5, 2.5, 50], [0, 5 * np.sqrt(3)/2, 50], [0, 0, 1]])

# Defina um fator de escala para aumentar o tamanho do triângulo e das regiões
fator_escala = 16

p = np.array([
    [0, 0, 1],
    [0, 100 * fator_escala, 1],
    [100 * fator_escala, 0, 1],
    [0, 87 * fator_escala, 1],
    [0, 96 * fator_escala, 1],
    [0, 98 * fator_escala, 1],
    [2 * fator_escala, 98 * fator_escala, 1],
    [23 * fator_escala, 0, 1],
    [23 * fator_escala, 64 * fator_escala, 1],
    [20 * fator_escala, 76 * fator_escala, 1],
    [20 * fator_escala, 80 * fator_escala, 1],
    [40 * fator_escala, 31 * fator_escala, 1],
    [40 * fator_escala, 47 * fator_escala, 1],
    [50 * fator_escala, 35 * fator_escala, 1],
    [50 * fator_escala, 46 * fator_escala, 1],
    [50 * fator_escala, 50 * fator_escala, 1],
    [71 * fator_escala, 0, 1],
    [85 * fator_escala, 0, 1]])

v = p @ np.transpose(A)

# Coordenadas do triângulo do código 1
region_PD = v[[5, 1, 6], :]
region_T1 = v[[4, 5, 6, 10, 9], :]
region_T2 = v[[9, 10, 15, 14], :]
region_T3 = v[[13, 15, 2, 17], :]
region_D1 = v[[0, 3, 8, 7], :]
region_D2 = v[[7, 8, 12, 11, 16], :]
region_DT = v[[3, 4, 14, 13, 17, 16, 11, 12], :]

sample_points = df[['Wc2h2', 'Wc2h4', 'WCh4']].values @ np.transpose(A)

fig, ax1 = plt.subplots(figsize=(5, 5))  # Ajuste o tamanho da figura conforme necessário

ax1.fill(region_PD[:, 0] / 2, region_PD[:, 1] / 2, '#2e962d')
ax1.fill(region_T1[:, 0] / 2, region_T1[:, 1] / 2, '#bebe12')
ax1.fill(region_T2[:, 0] / 2, region_T2[:, 1] / 2, '#ff642b')
ax1.fill(region_T3[:, 0] / 2, region_T3[:, 1] / 2, '#b46414')
ax1.fill(region_D1[:, 0] / 2, region_D1[:, 1] / 2, '#10b4a7')
ax1.fill(region_D2[:, 0] / 2, region_D2[:, 1] / 2, '#121eb4')
ax1.fill(region_DT[:, 0] / 2, region_DT[:, 1] / 2, '#f217d0')

ax1.scatter(sample_points[:, 0] / 2, sample_points[:, 1] / 2, marker='x', c='r', zorder=2)
ax1.grid(linestyle='--', alpha=0.4, axis='both')

label1 = np.array([45 * fator_escala, -5 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label1[0] / 2, label1[1] / 2, '%C2H2')
label11 = np.array([95 * fator_escala, -5 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label11[0] / 2, label11[1] / 2, '0')
label12 = np.array([5 * fator_escala, -5 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label12[0] / 2, label12[1] / 2, '100')
label2 = np.array([-10 * fator_escala, 55 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label2[0] / 2, label2[1] / 2, '%CH4')
label21 = np.array([-7 * fator_escala, 5 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label21[0] / 2, label21[1] / 2, '0')
label22 = np.array([-7 * fator_escala, 95 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label22[0] / 2, label22[1] / 2, '100')
label3 = np.array([45 * fator_escala, 55 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label3[0] / 2, label3[1] / 2, '%C2H4')
label31 = np.array([5 * fator_escala, 95 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label31[0] / 2, label31[1] / 2, '0')
label22 = np.array([95 * fator_escala, 5 * fator_escala, 1]) @ np.transpose(A)
ax1.text(label22[0] / 2, label22[1] / 2, '100')

ax1.set_xlim(0, 600 * fator_escala / 2)
ax1.set_ylim(0, 550 * fator_escala / 2)

# Salvar o gráfico como arquivo Excel
df['Region'] = ''

regions = {
    'PD': region_PD,
    'T1': region_T1,
    'T2': region_T2,
    'T3': region_T3,
    'D1': region_D1,
    'D2': region_D2,
    'DT': region_DT
}

codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY
]

paths = {}

for region, vertices in regions.items():
    num_vertices = len(vertices)
    codes = [Path.MOVETO] + [Path.LINETO] * (num_vertices - 2) + [Path.CLOSEPOLY]
    path = Path(vertices[:, :2] / 2, codes)
    paths[region] = path
        

df.to_excel('regions.xlsx', index=False)
plt.show()
