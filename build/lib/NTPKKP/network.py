import numpy as np


class NeuralNetwork:
    def __init__(self, learning_rate=0.01):
        self.layers = []
        self.learning_rate = learning_rate

    def add_layer(self, layer):
        self.layers.append(layer)

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, X, y, loss_func):
        output = self.forward(X)
        loss_gradient = loss_func.gradient(y, output)
        for layer in reversed(self.layers):
            loss_gradient = layer.backward(loss_gradient)

    def update_weights(self):
        for layer in self.layers:
            if hasattr(layer, 'update_weights'):
                layer.update_weights(self.learning_rate)
