import numpy as np

class sum_squares:
    """
    Sum of squares loss function for SINGLE output units
    """
    @staticmethod
    def calc_loss(ypred, ytrue):
        sumsq = np.sum(np.square(ypred - ytrue))
        return sumsq.reshape(1,1)

    @staticmethod
    def gradient(ypred, ytrue):
        grad = 2 * (ypred - ytrue) * ypred
        return grad.reshape(1,1)

class cross_entropy:
    """
    Cross entropy loss function for SINGLE output units
    """
    @staticmethod
    def calc_loss(ypred, ytrue):
        ce_loss = (ytrue) * np.log(ypred + 1e-7) + (1 - ytrue) * np.log(1 - ypred + 1e-7)
        return (-ce_loss).reshape(1,1)

    @staticmethod
    def gradient(ypred, ytrue):
        if ytrue == 1:
            grad = - 1 / (ypred + 1e-7)
        else:
            grad = 1 / (1 - ypred + 1e-7)
        return grad.reshape(1,1) # 1x1

# ret = (cross_entropy.calc_loss(np.array([0.42]), np.array([0.5])))
# print(ret)

