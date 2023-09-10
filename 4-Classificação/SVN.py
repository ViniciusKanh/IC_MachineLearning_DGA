import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from tabulate import tabulate

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    cm = np.round(cm, 2)
    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def main():
    # Carregar o conjunto de dados
    input_file = '0-Datasets/Dataset_Tratado.data'
    names = ['Defeito', 'H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    features = ['H2', 'CH4', 'C2H2', 'C2H4', 'C2H6']
    target = 'Defeito'
    df = pd.read_csv(input_file, names=names)
    
    target_names = ['Normal', 'Falha Termica','Falha Eletrica']

    X = df[features].values
    y = df[target].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    svm = SVC(kernel='rbf', C=1)
    svm.fit(X_train, y_train)
    y_hat_test = svm.predict(X_test)

    accuracy = accuracy_score(y_test, y_hat_test) * 100
    f1 = f1_score(y_test, y_hat_test, average='macro')
    precision = precision_score(y_test, y_hat_test, average='macro')

    metrics_data = [
        ['Accuracy', accuracy],
        ['F1-Score', f1],
        ['Precision', precision]
    ]

    print("Metrics:")
    print(tabulate(metrics_data, headers=['Metric', 'Value'], tablefmt='grid'))

    cm = confusion_matrix(y_test, y_hat_test)
    print("\nConfusion Matrix:")
    print(tabulate(cm, headers=target_names, tablefmt='grid'))

    plot_confusion_matrix(cm, target_names, False, "Confusion Matrix - SVM scikit-learn")
    plot_confusion_matrix(cm, target_names, True, "Confusion Matrix - SVM scikit-learn normalized")
    plt.show()


if __name__ == "__main__":
    main()
