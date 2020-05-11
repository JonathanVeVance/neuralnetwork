import numpy as np

class sum_squares:
    @staticmethod
    def calc_loss(ypred, ytrue):
        sumsq = np.sum(np.square(ypred - ytrue))
        return sumsq#.reshape(1,1) # total sumsq over all o/p units

    @staticmethod
    def gradient(ypred, ytrue):
        grad = 2 * (ypred - ytrue) * ypred
        return np.array([grad]) # 1xK

class cross_entropy:
    # only applicable to 1X1 ouput units
    @staticmethod
    def calc_loss(ypred, ytrue):
        ce_loss = (ytrue) * np.log(ypred) + (1 - ytrue) * np.log(1 - ypred)
        return (-ce_loss)#.reshape(1,1)

    @staticmethod
    def gradient(ypred, ytrue):
        if ytrue == 1:
            grad = - 1 / ypred
        else:
            grad = - 1 / (1 - ypred)
        return grad.reshape(1,1) # 1x1

# ret = (cross_entropy.calc_loss(np.array([0.42]), np.array([0.5])))
# print(ret)

