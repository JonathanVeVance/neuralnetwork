import numpy as np

class momentum:
    def __init__(self, mu):
        """
        Inputs  :   mu:  momentum coefficient
        Outputs :   None
        """
        self.mu = mu
        self.avg_weights = []
        self.avg_bias = []

    def calc_update(self, eta, weights_grad, bias_grad):
        """
        Function to compute momentum update value
        Inputs:     eta:    learning rate
                    weights_grad : gradients wrt all weights (list of np arrays)
                    bias_grad    : gradient of all biases (list)
        Outputs:    avg_weights  : update for weights (list)
                    avg_bias     : update for biases (list)
        """
        if len(self.avg_bias) == 0:
            for idx in range(len(bias_grad)):
                self.avg_bias.append(eta * bias_grad[idx])
                self.avg_weights.append(eta * weights_grad[idx])
        else:
            for idx in range(len(bias_grad)):
                self.avg_bias[idx] = -self.mu * self.avg_bias[idx] + eta * bias_grad[idx]
                self.avg_weights[idx] = -self.mu * self.avg_weights[idx] + eta * weights_grad[idx]

        return self.avg_weights, self.avg_bias

class rmsprop:
    def __init__(self, decay_rate):
        """
        Inputs:   decay_rate  : decay rate parameter
        Outputs   None
        """
        self.bias_cache = []
        self.weights_cache = []
        self.decay_rate = decay_rate

    def calc_update(self, eta, weights_grad, bias_grad):
        """
        Function to compute rmsprop update values
        Inputs  :   eta             : learning rate
                    weights_grad    : gradients wrt all weights (list of np arrays)
                    bias_grad       : gradient of all biases (list)
        Outputs :   weights_update  : update for weights (list)
                    bias_update     : update for biases (list)
        """
        if len(self.bias_cache) == 0:
            for idx in range(len(bias_grad)):
                self.bias_cache.append((1 - self.decay_rate) * np.square(bias_grad[idx]))
                self.weights_cache.append((1 - self.decay_rate) * np.square(weights_grad[idx]))

        else:
            for idx in range(len(bias_grad)):
                self.bias_cache[idx] = self.decay_rate * self.bias_cache[idx] + (1 - self.decay_rate) * np.square(bias_grad[idx])
                self.weights_cache[idx] = self.decay_rate * self.weights_cache[idx] + (1 - self.decay_rate) * np.square(weights_grad[idx])


        bias_update = []
        weights_update = []
        for idx in range(len(bias_grad)):
            bias_update.append(eta * np.divide(bias_grad[idx], (np.sqrt(self.bias_cache[idx]) + 1e-7)))
            weights_update.append(eta * np.divide(weights_grad[idx], (np.sqrt(self.weights_cache[idx]) + 1e-7)))

        return weights_update, bias_update

class adam:
    def __init__(self, avg_decay, var_decay):
        """
        Inputs  :   avg_decay   : decay parameter for running average of gradients
                    var_decay   : decay parameter for running variance of gradients
        Outputs :   None
        """
        self.avg_bias = []
        self.var_bias = []
        self.avg_weights = []
        self.var_weights = []
        self.avg_decay = avg_decay
        self.var_decay = var_decay

    def calc_update(self, eta, weights_grad, bias_grad):
        """
        Function to compute adam update values
        Inputs  :   eta             : learning rate
                    weights_grad    : gradients wrt all weights (list of np arrays)
                    bias_grad       : gradient of all biases (list)
        Outputs :   weights_update  : update for weights (list)
                    bias_update     : update for biases (list)
        """
        if len(self.avg_bias) == 0:
            for idx in range(len(bias_grad)):
                self.avg_weights.append((1 - self.avg_decay) * weights_grad[idx])
                self.var_weights.append((1 - self.var_decay) * np.square(weights_grad[idx]))
                self.avg_bias.append((1 - self.avg_decay) * bias_grad[idx])
                self.var_bias.append((1 - self.var_decay) * np.square(bias_grad[idx]))

        else:
            for idx in range(len(bias_grad)):
                self.avg_weights[idx] = self.avg_decay * self.avg_weights[idx] + (1 - self.avg_decay) * weights_grad[idx]
                self.var_weights[idx] = self.var_decay * self.var_weights[idx] + (1 - self.var_decay) * np.square(weights_grad[idx])
                self.avg_bias[idx] = self.avg_decay * self.avg_bias[idx] + (1 - self.avg_decay) * bias_grad[idx]
                self.var_bias[idx] = self.var_decay * self.var_bias[idx] + (1 - self.var_decay) * np.square(bias_grad[idx])

        bias_update = []
        weights_update = []
        for idx in range(len(bias_grad)):
            bias_update.append(eta * np.divide(self.avg_bias[idx], (np.sqrt(self.var_bias[idx]) + 1e-7)))
            weights_update.append(eta * np.divide(self.avg_weights[idx], (np.sqrt(self.var_weights[idx]) + 1e-7)))

        return weights_update, bias_update

