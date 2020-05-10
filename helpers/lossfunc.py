import numpy as np

class sum_squares:
    # single input (y's are 1X1 arrays)
    @staticmethod
    def calc_loss(ypred, ytrue):
        sumsq = np.sum(np.square(ypred - ytrue))
        return sumsq

    @staticmethod
    def gradient(ypred, ytrue):
        grad = 2 * (ypred - ytrue) * ypred
        return np.array([grad])

class cross_entropy:
    # single input (y's are 1X1 arrays)
    @staticmethod
    def calc_loss(ypred, ytrue):
        ce_loss = (ytrue) * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred)
        return (- ce_loss)

    @staticmethod
    def gradient(ypred, ytrue):
        if ytrue == 1:
            grad = - 1 / ypred
        else:
            grad = - 1 / (1 - ypred)
        return grad # 1X1 array

# print(cross_entropy.gradient(np.array([0.3]), np.array([1])))