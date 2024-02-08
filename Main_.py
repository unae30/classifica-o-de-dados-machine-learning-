import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Carregar os dados
data = np.loadtxt('IA_TP2/spambase/spambase.data', delimiter=',')

# Separar features e labels
X = data[:, :-1]
y = data[:, -1]

# Selecionar aleatoriamente 1000 exemplos de cada classe
class_0_indices = np.where(y == 0)[0]
class_1_indices = np.where(y == 1)[0]
np.random.shuffle(class_0_indices)
np.random.shuffle(class_1_indices)
selected_indices = np.concatenate([class_0_indices[:1000], class_1_indices[:1000]])
selected_X = X[selected_indices]
selected_y = y[selected_indices]

print("Número de exemplos na classe 0 selecionados:", np.sum(selected_y == 0))
print("Número de exemplos na classe 1 selecionados:", np.sum(selected_y == 1))


# Dividir em conjunto de treinamento, validação e teste 
#(60% treinamento, 20% validação, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(selected_X, selected_y,test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val =train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Inicializar Perceptron e MLP
#max_iter = 100 -> significa que são 100 épocas
perceptron = Perceptron(max_iter=100, random_state=42)
#hidden_layer_sizes=(100,) -> significa que a rede MLP 
#terá uma camada oculta com 100 neurônios
#se quiser trocar o numero de neuronios (epocas) só 
#=-> hidden_layer_sizes=(50,) por exemplo
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, random_state=42)

# Treinar o Perceptron
perceptron.fit(X_train, y_train)

# Fazer previsões com o Perceptron
y_pred_perceptron = perceptron.predict(X_test)

# Calcular a matriz de confusão do Perceptron
conf_matrix_perceptron = confusion_matrix(y_test, y_pred_perceptron)
true_positives_perceptron = conf_matrix_perceptron[1, 1]
false_positives_perceptron = conf_matrix_perceptron[0, 1]
true_negatives_perceptron = conf_matrix_perceptron[0, 0]
false_negatives_perceptron = conf_matrix_perceptron[1, 0]

sensitivity_perceptron = true_positives_perceptron / (true_positives_perceptron + false_negatives_perceptron)
specificity_perceptron = true_negatives_perceptron / (true_negatives_perceptron + false_positives_perceptron)
precision_perceptron = true_positives_perceptron / (true_positives_perceptron + false_positives_perceptron)
accuracy_perceptron = accuracy_score(y_test, y_pred_perceptron)

print("\nPerceptron:-----------------------")
print("Sensibilidade:", sensitivity_perceptron)
print("Especificidade:", specificity_perceptron)
print("Precisão:", precision_perceptron)
print("Acurácia:", accuracy_perceptron)
print("Matriz de Confusão do Perceptron:")
print(conf_matrix_perceptron)
print("----------------------------------")
print("\n############################################\n")

# Treinar a rede MLP com uma camada oculta
#mlp.fit(X_train, y_train)

error_rates_mlp = []
for epoch in range(1, 101):
    mlp.partial_fit(X_train, y_train, 
classes=np.unique(y_train))
    y_pred_mlp = mlp.predict(X_val)
    error_rate_mlp = 1 - accuracy_score(y_val, y_pred_mlp)
    error_rates_mlp.append(error_rate_mlp)
    print(f"Época {epoch}: Taxa de erro da Rede MLP = {error_rate_mlp}")

# Fazer previsões com a rede MLP
y_pred_mlp = mlp.predict(X_test)

# Calcular a matriz de confusão da rede MLP
conf_matrix_mlp = confusion_matrix(y_test, y_pred_mlp)
true_positives_mlp = conf_matrix_mlp[1, 1]
false_positives_mlp = conf_matrix_mlp[0, 1]
true_negatives_mlp = conf_matrix_mlp[0, 0]
false_negatives_mlp = conf_matrix_mlp[1, 0]

sensitivity_mlp = true_positives_mlp / (true_positives_mlp + false_negatives_mlp)
specificity_mlp = true_negatives_mlp / (true_negatives_mlp + false_positives_mlp)
precision_mlp = true_positives_mlp / (true_positives_mlp + false_positives_mlp)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print("\nRede MLP:-----------------------")
print("Sensibilidade:", sensitivity_mlp)
print("Especificidade:", specificity_mlp)
print("Precisão:", precision_mlp)
print("Acurácia:", accuracy_mlp)
print("\nMatriz de Confusão da Rede MLP:")
print(conf_matrix_mlp)
print("----------------------------------")
