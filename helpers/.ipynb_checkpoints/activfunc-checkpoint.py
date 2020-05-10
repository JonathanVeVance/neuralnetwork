import numpy as np

class ReLu:

    @staticmethod
    def forward(z):
        z[z < 0] = 0
        return z

    @staticmethod
    def backward(z):
        grad = np.where(z > 0, 1, 0)
        return grad

class sigmoid:

    @staticmethod
    def forward(z):
        y = 1 / (1 + np.exp(-z))
        return y

    @staticmethod
    def backward(z):
        y = sigmoid.forward(z)
        grad = y * (1 - y)
        return grad

activfunc_dict = {'relu': ReLu, 'sigmoid': sigmoid}
# activfunc_dict = {'relu': ReLu(), 'sigmoid': sigmoid()}

