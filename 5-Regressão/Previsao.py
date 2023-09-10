import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU

# Restante do código...


# Carregar a base de dados
input_file = '0-Datasets/Dataset_Tratado.data'
names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
target = 'Defeito'

data = pd.read_csv(input_file, names=names)

# Pré-processamento dos dados
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[features])

# Separar os dados de treinamento por defeito
defect1_data = scaled_data[data['Defeito'] == 1]
defect2_data = scaled_data[data['Defeito'] == 2]
defect3_data = scaled_data[data['Defeito'] == 3]

# Definir uma função para construir o gerador da GAN
def build_generator():
    model = Sequential()
    model.add(Dense(32, input_dim=len(features)))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(64))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(len(features), activation='sigmoid'))
    return model

# Construir o gerador
generator = build_generator()

# Compilar o gerador
# Compilar o gerador
generator.compile(optimizer='adam', loss='sparse_categorical_crossentropy')


# Definir uma função para gerar amostras a partir do gerador
def generate_samples(generator, num_samples):
    noise = np.random.normal(0, 1, (num_samples, len(features)))
    generated_samples = generator.predict(noise)
    return generated_samples

# Treinar o gerador
num_samples = 100  # Número de amostras a serem geradas
num_epochs = 1000  # Número de épocas de treinamento
batch_size = 64

for epoch in range(num_epochs):
    # Gerar amostras usando o gerador atual
    generated_samples = generate_samples(generator, num_samples)

    # Juntar as amostras geradas com os dados originais
    combined_data = np.concatenate([defect1_data, defect2_data, defect3_data, generated_samples], axis=0)

    # Criar rótulos para distinguir dados originais dos gerados
    labels = np.concatenate([np.ones((len(defect1_data),)), np.ones((len(defect2_data),)),
                             np.ones((len(defect3_data),)), np.zeros((num_samples,))], axis=0)

    # Embaralhar os dados
    combined_data, labels = shuffle(combined_data, labels)

    # Dividir os dados em treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(combined_data, labels, test_size=0.3)

    # Treinar o gerador
    generator.fit(X_train, y_train, epochs=1, batch_size=batch_size)

    # Avaliar o gerador no conjunto de teste
    accuracy = generator.evaluate(X_test, y_test, verbose=0)
    print(f'Epoch: {epoch+1}, Accuracy: {accuracy}')

# Gerar novos dados usando o gerador treinado
generated_data = generate_samples(generator, num_samples)

# Arredondar os valores gerados para duas casas decimais
generated_data = np.round(generated_data, 2)

# Desescalar os dados gerados
generated_data = scaler.inverse_transform(generated_data)

# Criar um dataframe com os dados gerados
generated_df = pd.DataFrame(generated_data, columns=features)

# Arredondar os valores do dataframe para duas casas decimais
generated_df = generated_df.round(2)

# Adicionar a coluna de defeito com valores aleatórios entre 1 e 3
generated_df['Defeito'] = np.random.choice([1, 2, 3], size=num_samples)

# Salvar os novos dados em um arquivo Excel
generated_df.to_excel('novos_dados.xlsx', index=False)


