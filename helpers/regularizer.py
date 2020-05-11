import numpy as np

class l2_regularizer:
    def __init__(self, penalty_param):
        self.penalty_param = penalty_param

    def calc_loss(self, weights):
        penalty = self.penalty_param * np.sum(np.square(weights))
        return penalty

    def gradient(self, weights):
        penalty_grad = 2 * self.penalty_param * weights
        return penalty_grad

class l1_regularizer:
    def __init__(self, penalty_param):
        self.penalty_param = penalty_param

    def calc_loss(self, weights):
        penalty = self.penalty_param * np.sum(np.abs(weights))
        return penalty

    def gradient(self, weights):
        penalty_grad = self.penalty_param * np.sign(weights)
        return penalty_grad

# a = np.random.randn(3,4)
# print(a)
# regularizer = l1_regularizer(0.01)
# print(regularizer.gradient(a))