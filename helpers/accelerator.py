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
        update = (- eta * gradient) ./ (np.sqrt(self.cache) + 1e-7)
        return update
