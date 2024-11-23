import numpy as np


class MeanSquaredError:
    def __call__(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropyLoss:
    def __call__(self, y_true, y_pred):
        return -np.sum(y_true * np.log(y_pred))

    def gradient(self, y_true, y_pred):
        return (y_pred - y_true) / y_true.size


class HingeLoss:
    def __call__(self, y_true, y_pred):
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def gradient(self, y_true, y_pred):
        return np.where(y_true * y_pred < 1, -y_true, 0)
