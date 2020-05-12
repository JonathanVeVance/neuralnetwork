import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()

df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

from neuralnetwork import network

NN = network()
NN.add_layer(10, 'relu')
NN.add_layer(5, 'relu')
NN.add_layer(1)

x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()
x_test_np = x_test.to_numpy()
y_test_np = y_test.to_numpy()

def rmse(ytrue, ypred):
    mse = np.mean(np.square(ytrue - ypred))
    return np.sqrt(mse)

from helpers.accelerator import adam, momentum, rmsprop
from helpers.regularizer import l2_regularizer
# accel = adam(0.9, 0.9)
# accel = momentum(0.5)
accel = rmsprop(0.9)
reg = l2_regularizer(1e2)

NN.train_network(x_train_np, y_train_np, learning_rate = 1e-6, grad_descent_type = 'minibatchgd',
                loss_function = 'sumsquares', regularizer = reg, accelerator = accel, batch_size = 30)

import pickle
with open('NN_boston.pickle', 'wb') as handle:
    pickle.dump(NN, handle, protocol = pickle.HIGHEST_PROTOCOL)

ypreds = NN.predict(x_test_np)
print(rmse(y_test_np, ypreds))

