
import numpy as np
from helpers.activfunc import ReLu, sigmoid
from helpers.lossfunc import sum_squares, cross_entropy
from helpers.regularizer import l2_regularizer, l1_regularizer # pass instances
from helpers.grad_descent import SGD, MiniBatchGD
from helpers.accelerator import adam, rmsprop, momentum # pass instances

activfunc_dict = {
    'relu': ReLu, 'sigmoid': sigmoid
}

lossfunc_dict = {
    'sumsquares': sum_squares, 'crossentropy': cross_entropy
}

grad_descent_dict = {
    'sgd': SGD, 'minibatchgd': MiniBatchGD
}

class neuralnetwork:

    def __init__(self):
        self.W = []
        self.Z = []
        self.Y = []
        self.Wgrad = None
        self.nlayers = 0
        self.dropouts = []
        self.activations = []
        self.prev_layer_neurons = None

    def add_layer(self, num_neurons, activation = None, dropout = None):
        if self.nlayers == 0:
            self.W.append(None)
        else:
            weights = np.random.rand(num_neurons, self.prev_layer_neurons + 1)
            self.W.append(weights)

        self.dropouts.append(dropout)
        self.activations.append(activation)
        self.prev_layer_neurons = num_neurons
        self.nlayers += 1

    def forward(self, inputs, predict = False):

        current_x = inputs
        for k in range(self.nlayers):

            current_x = np.insert(current_x, 0, 1, axis = 0)
            self.Y.append(current_x)
            z = (np.matmul(self.W[k], current_x.T)).T
            self.Z.append(z)

            if self.activations[k] == None:
                y = z

            else:
                sigma = activfunc_dict[self.activations[k]]
                y = sigma.forward(z)

            if predict == False and self.dropouts[k] != None:
                 y /= self.dropouts[k]

            current_x = y

        self.Y.append(y)
        return y

    def backward(self, op_gradient, regularizer):
        current_grad = op_gradient
        for k in range(self.nlayers, -1, -1):

            print(f"size(current grad) = {np.shape(current_grad)}")
            if self.activations[k - 1] == None:
                gradDz = current_grad

            else:
                sigma = activfunc_dict[self.activations[k - 1]]
                gradDz = np.matmul(current_grad, sigma.backward(self.Z[k - 1]))

            print(f"gradDz == {np.shape(gradDz)}")
            gradzw = np.matmul(np.ones(np.shape(self.W[k - 1])[0]), self.Y[k - 1])
            self.Wgrad[k - 1] = np.matmul(gradDz, gradzw)

            if regularizer != None:
                self.Wgrad[k - 1] += regularizer.gradient(self.W[k - 1])

            current_grad = np.matmul(gradDz, self.W[k - 1])

    def clear_outputs(self):
        self.Z = []
        self.Y = []

    def train_network(self, X_train, Y_train, loss_function, grad_descent_type = 'sgd', batch_size = None,
                      learning_rate = 0.001, regularizer = None, accelerator = None):

        input_size = np.shape(X_train)[1]
        layer1_size = np.shape(self.W[1])[1]
        self.W[0] = np.random.rand(layer1_size - 1, input_size + 1)
        self.Wgrad = np.zeros(np.shape(self.W))
        loss_function = lossfunc_dict[loss_function]
        grad_descent = grad_descent_dict[grad_descent_type]
        grad_descent(self, X_train, Y_train, loss_function, batch_size, learning_rate, regularizer, accelerator)

    def predict(self, X_test):

        Y_test_pred = []
        for x in X_test:
            y = self.forward(x, predict = True)
            Y_test_pred.append(y)

        return Y_test_pred

