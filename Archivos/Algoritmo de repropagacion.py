import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Definir un estilo de gráfico personalizado
custom_style = {
    'axes.facecolor': 'white',  # Color de fondo
    'axes.grid': False,          # Eliminar la cuadrícula
    'axes.edgecolor': 'black',   # Color de los ejes
    'axes.linewidth': 1.2,       # Ancho de los ejes
    'xtick.color': 'black',      # Color de las marcas en el eje x
    'ytick.color': 'black',      # Color de las marcas en el eje y
    'xtick.major.size': 5,       # Tamaño de las marcas mayores en el eje x
    'ytick.major.size': 5,       # Tamaño de las marcas mayores en el eje y
    'xtick.major.width': 1.2,    # Grosor de las marcas mayores en el eje x
    'ytick.major.width': 1.2,    # Grosor de las marcas mayores en el eje y
}

plt.rcParams.update(custom_style)  # Aplicar el estilo personalizado

# Definir la función de activación (en este caso, sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Definir la derivada de la función de activación
def sigmoid_derivative(x):
    return x * (1 - x)

# Clase para la red neuronal
class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        self.biases = [np.random.randn(1, layers[i+1]) for i in range(len(layers)-1)]
        
    def feedforward(self, inputs):
        self.activations = [inputs]
        for i in range(len(self.layers)-1):
            inputs = sigmoid(np.dot(inputs, self.weights[i]) + self.biases[i])
            self.activations.append(inputs)
        return inputs
    
    def backpropagation(self, inputs, targets, learning_rate):
        output = self.feedforward(inputs)[-1]
        error = targets - output
        deltas = [error * sigmoid_derivative(output)]
        
        for i in range(len(self.layers)-2, 0, -1):
            error = deltas[-1].dot(self.weights[i].T)
            deltas.append(error * sigmoid_derivative(self.activations[i]))
        
        deltas.reverse()
        
        for i in range(len(self.layers)-1):
            self.weights[i] += self.activations[i].T.dot(deltas[i]) * learning_rate
            self.biases[i] += np.sum(deltas[i], axis=0, keepdims=True) * learning_rate

# Cargar los datos
data = pd.read_csv('concentlite.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Definir la arquitectura de la red
input_size = X.shape[1]
hidden_layers = [5, 5]  # Por ejemplo, dos capas ocultas con 5 neuronas cada una
output_size = 1
layers = [input_size] + hidden_layers + [output_size]

# Crear la red neuronal
nn = NeuralNetwork(layers)

# Entrenamiento de la red
epochs = 1000
learning_rate = 0.1

for epoch in range(epochs):
    nn.backpropagation(X, y, learning_rate)

# Clasificar los datos
predictions = np.round(nn.feedforward(X))

# Separar los datos según la clase
class_0 = X[y.flatten() == -1]
class_1 = X[y.flatten() == 1]

# Graficar los resultados con 'X' negras para una clase y cuadros rojos sin relleno para la otra clase
plt.plot(class_0[:, 0], class_0[:, 1], 'rs', markerfacecolor='none', markersize=6)
plt.plot(class_1[:, 0], class_1[:, 1], 'kx', markersize=6)

# Generar aleatoriamente 1000 puntos y graficarlos
random_points = np.random.rand(1000, 2) * 2 - 1  # Genera puntos aleatorios en el rango [-1, 1]
plt.plot(random_points[:, 0], random_points[:, 1], 'b.', alpha=0.5, markersize=5)  # alpha=0.5 para hacer los puntos más transparentes

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Clasificación de datos con Perceptrón Multicapa')
plt.show()
