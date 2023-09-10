import pandas as pd

# Função para calcular a precisão
def calculate_accuracy(predictions, labels):
    correct_predictions = sum(1 for pred, label in zip(predictions, labels) if pred == label)
    total_predictions = len(predictions)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Ler o arquivo Excel
df = pd.read_excel("0-Datasets/Analise Normativa/Analise_Completa.xlsx")

# Rótulos corretos para cada saída dos métodos de diagnóstico
labels_Razao_de_Roger = ["Não foi possível classificar", "PARTIAL DISCHARGE", "NO FAULT", "THERMAL < 700 C", "HIGH-ENERGY ARCING"]
labels_Razao_de_Doernemburg = ["Normal", "Descarga parcial"]
labels_Gas_Chava = [
    'Arco Key gas: H2 ',
    'Arco Key gas: CH4  ',
    'Arco Key gas: C2H2  ',
    'Superaquecimento de alta temperatura do óleo Key gas: C2H4 ',
    'Superaquecimento de alta temperatura do óleo Key gas: C2H6  '
]
labels_Duval = ["Thermal Fault, 300 °C < t <700 °C", "Thermal Fault, T <300 °C", "Partial Discharges", "Discharges Of High Energy"]

# Previsões para cada saída dos métodos de diagnóstico
predictions_Razao_de_Roger = df["Razão de Roger"].tolist()
predictions_Razao_de_Doernemburg = df["Razão de Doernemburg"].tolist()
predictions_Gas_Chava = df["Gas_Chava"].tolist()
predictions_Duval = df["Duval"].tolist()

# Calcular as precisões e taxas de acerto para cada saída dos métodos de diagnóstico
accuracy_Razao_de_Roger = calculate_accuracy(predictions_Razao_de_Roger, labels_Razao_de_Roger)
accuracy_Razao_de_Doernemburg = calculate_accuracy(predictions_Razao_de_Doernemburg, labels_Razao_de_Doernemburg)

# Calcular as taxas de acerto para Gas_Chava e Duval
correct_predictions_Gas_Chava = sum(1 for label in df["Rotulo"] if any(item in label for item in labels_Gas_Chava))
total_predictions_Gas_Chava = len(predictions_Gas_Chava)
accuracy_Gas_Chava = correct_predictions_Gas_Chava / total_predictions_Gas_Chava

correct_predictions_Duval = sum(1 for pred, label in zip(predictions_Duval, df["Rotulo"]) if label in labels_Duval)
total_predictions_Duval = len(predictions_Duval)
accuracy_Duval = correct_predictions_Duval / total_predictions_Duval

# Exibir os resultados
print("Precisão da Razão de Roger:", accuracy_Razao_de_Roger)
print("Precisão da Razão de Doernemburg:", accuracy_Razao_de_Doernemburg)
print("Taxa de acerto da Gas_Chava:", accuracy_Gas_Chava)
print("Taxa de acerto da Duval:", accuracy_Duval)
