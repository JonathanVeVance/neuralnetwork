## Be careful whether to use Jacobian or transpose of
## Jacobian for unsymmetric jacobians (eg: softmax function)

import numpy as np

class ReLu:

    @staticmethod
    def forward(z):
        z[z < 0] = 0
        return np.array(z)

    @staticmethod
    def backward(z):
        grad = np.where(z > 0, 1, 0)
        return np.diagflat(grad) # Jacobian

class sigmoid:

    @staticmethod
    def forward(z):
        y = 1 / (1 + np.exp(-z))
        return np.array(y)

    @staticmethod
    def backward(z):
        y = sigmoid.forward(z)
        grad = y * (1 - y)
        return np.diagflat(grad) # Jacobian

# print(sigmoid.forward(np.array([2])))

