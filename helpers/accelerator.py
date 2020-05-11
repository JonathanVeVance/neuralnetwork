import numpy as np

class momentum:
    def __init__(self, mu):
        self.mu = mu
        self.running_avg = 0

    def calc_update(self, eta, gradient):
        self.running_avg = self.mu * self.running_avg - eta * gradient
        return self.running_avg

class rmsprop:
    def __init__(self, decay_rate):
        self.cache = 0
        self.decay_rate = decay_rate

    def calc_update(self, eta, gradient):
        self.cache = (self.decay_rate * self.cache +
                        (1 - self.decay_rate) * np.square(gradient))
        update = (-eta) * np.divide(gradient, (np.sqrt(self.cache) + 1e-7))
        return update

class adam:
    def __init__(self, avg_decay, var_decay):
        self.avg_decay = avg_decay
        self.var_decay = var_decay
        self.running_avg = 0
        self.running_var = 0

    def calc_update(self, eta, gradient):
        self.running_avg = (self.avg_decay * self.running_avg +
                                (1 - self.avg_decay) * gradient)
        self.running_var = (self.var_decay * self.running_var +
                                (1 - self.var_decay) * np.square(gradient))
        update = (-eta) * np.divide(self.running_avg, (np.sqrt(self.running_var) + 1e-7))
        return update

