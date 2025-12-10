# Converted from neural-network.ipynb

# ======================================================================
# # Deep Learning
# ======================================================================

# ======================================================================
# - Deep learning is a subset of machine learning in aritificial intelligence (AI) that has networks capable of learning unsupervised from data that is unstructured or unlabel
# - Also know as deep neural learning or deep neural network.
# ======================================================================

# ======================================================================
# # Neural Network
# ======================================================================

# ======================================================================
# - A neural network is series of algorithm that endeavors to recognize
# underlying relationship in a set of data through a process that mimics the way the human
# brain operates.
# - In this senses, neural networks refer to systems of neurons, either organic or artificial in nature.
# ======================================================================

# %%
import warnings

import numpy as np

warnings.filterwarnings("ignore")


# %%
class NeuralNetwork:
    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 4)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x / (1 + x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for iteration in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(
                training_inputs.T, error * self.sigmoid_derivative(output)
            )
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output


# In short, use this ' if name == "main" ' block to prevent (certain) code from being run when the module
# is imported. Put simply, name is a variable defined for each script that defines whether the script is
# being run as the main module or it is being run as an imported module.

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]])

    neural_network.train(training_inputs, training_outputs, 10000)

    print("Synaptic weight after training: ")
    print(neural_network.synaptic_weights)

    # A = str(input('Input 1: '))
    # B = str(input('Input 2: '))
    # C = str(input('Input 3: '))

    A = 0
    B = 1
    C = 1

    print("New situation: input data = ", A, B, C)
    print("output data: ")
    print(neural_network.think(np.array([A, B, C])))

# %%
