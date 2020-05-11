import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
boston = load_boston()

df_x = pd.DataFrame(boston.data, columns = boston.feature_names)
df_y = pd.DataFrame(boston.target)
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.33, random_state=42)

### Using neuralnetwork API

from neuralnetwork import neuralnetwork

NN = neuralnetwork()
NN.add_layer(10, 'relu')
NN.add_layer(5, 'relu')
NN.add_layer(1)

x_train_np = x_train.to_numpy()
y_train_np = y_train.to_numpy()

NN.train_network(x_train_np, y_train_np, loss_function = 'sumsquares')
