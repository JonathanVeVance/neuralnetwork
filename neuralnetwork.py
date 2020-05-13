
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

class network:

    def __init__(self):
        self.W = []
        self.Z = []
        self.Y = []
        self.bias = []
        self.Wgrad = []
        self.nlayers = 0
        self.dropouts = []
        self.bias_grad = []
        self.activations = []
        self.prev_layer_neurons = None

    def add_layer(self, num_neurons, activation = None, dropout = None):
        """
        Function to add hidden and output layers
        Inputs:     num_nerons  : number of units in layer
                    activation  : activation function (default = linear unit)
                    dropout     : dropout fraction (float) (default = None, or no dropout)
        Outputs:    None
        """
        if self.nlayers == 0:
            self.W.append(None)
        else:
            weights = np.random.rand(num_neurons, self.prev_layer_neurons)
            self.W.append(weights)

        self.bias.append(np.random.rand())
        self.dropouts.append(dropout)
        self.activations.append(activation)
        self.prev_layer_neurons = num_neurons
        self.nlayers += 1

    def forward(self, inputs, predict = False):
        """
        Function to compute output of the network
        Inputs:     inputs      : input (numpy array of dimension 1xN)
                    predict     : set to True during prediction and False during training
        Outputs:    y           : output of the network
        """
        current_x = inputs
        for k in range(self.nlayers):

            z = (np.matmul(self.W[k], current_x.T)).T + self.bias[k]
            if self.activations[k] == None:
                y = z
            else:
                sigma = activfunc_dict[self.activations[k]]
                y = sigma.forward(z)

            if predict == False and self.dropouts[k] != None:
                 y /= self.dropouts[k]

            if predict == False:
                self.Y.append(current_x)
                self.Z.append(z)

            current_x = y

        if predict == False:
            self.Y.append(y)

        return y

    def backward(self, op_gradient, regularizer):
        """
        Function to backpropagate gradients
        Inputs:     op_gradient : derrivative/gradient of loss wrt output (1xN)
                    regularizer : regularizer object (L2/L1)
        Outputs:    None
        """
        gradDy = op_gradient
        for k in range(self.nlayers, 0, -1):

            if self.activations[k - 1] == None:
                gradDz = gradDy

            else:
                sigma = activfunc_dict[self.activations[k - 1]]
                gradDz = np.matmul(gradDy, sigma.backward(self.Z[k - 1]))

            gradzy = self.W[k - 1]
            gradzw = np.matmul(np.ones((np.shape(self.W[k - 1])[0], 1)), self.Y[k - 1].reshape(1,-1))

            self.bias_grad[k - 1] = np.sum(gradDz)
            self.Wgrad[k - 1] = np.matmul(np.diagflat(gradDz), gradzw)
            if regularizer != None:
                self.Wgrad[k - 1] += regularizer.gradient(self.W[k - 1])

            gradDy = np.matmul(gradDz, gradzy)

    def clear_outputs(self):
        """
        Function to clear intermediate cached values
        Inputs      : None
        Outputs     : None
        """
        self.Z = []
        self.Y = []

    def update(self, learning_rate = None, weights_update = None, bias_update = None):
        """
        Function to update weights and biases
        Inputs :    learning_rate       : learning rate (float)
                    weights_update      : update for weights (if acceleration is used)
                    bias_update         : update for biases (if acceleration is used)
        Outputa:    None
        CAUTION:    If no acceleration is used, pass learning rate and leave weights_update and bias_update
                    as None; else, leave learning_rate as None, and pass weights_update and bias_update. Don't
                    pass in all three parameters.
        """
        if weights_update is None:
            weights_update = self.Wgrad
            bias_update = self.bias_grad

        if learning_rate == None:
            learning_rate = 1

        for idx in range(len(self.bias)):
            self.W[idx] -= learning_rate * weights_update[idx]
            self.bias[idx] -= learning_rate * bias_update[idx]

    def train_network(self, X_train, Y_train, loss_function, grad_descent_type = 'sgd', batch_size = None,
                      learning_rate = 0.001, regularizer = None, accelerator = None):
        """
        Function to train the network
        Inputs:     X_train             : training set (numpy array NxP)
                    Y_train             : target values of train set (numpy array)
                    loss_function       : choose from 'sumsquares' and 'crossentropy'
                    grad_descent_type   : choose from 'sgd'(default) and 'minibatchgd'
                    batch_size          : batch size for 'minibatchgd' (will be ignored if 'sgd')
                    learning_rate       : learning rate (float)
                    regularizer         : regularizer object (L2/L1)
                    accelerator         : accelerator object (adam/momentum/rmsprop)
        Outputs     None
        """
        input_size = np.shape(X_train)[1]
        layer1_size = np.shape(self.W[1])[1]
        self.W[0] = np.random.rand(layer1_size, input_size)
        for W in self.W:
            self.Wgrad.append(np.zeros(np.shape(W)))
        self.bias_grad = np.zeros(len(self.bias))

        loss_function = lossfunc_dict[loss_function]
        grad_descent = grad_descent_dict[grad_descent_type]
        grad_descent(self, X_train, Y_train, loss_function, batch_size, learning_rate, regularizer, accelerator)

    def predict(self, X_test):
        """
        Function to predict output
        Inputs :    X_test      : numpy matrix of input features
                    Y_test_pred : numpy array of predictions    
        """
        Y_test_pred = []
        for x in X_test:
            y = self.forward(x, predict = True)
            Y_test_pred.append(y)

        return Y_test_pred

