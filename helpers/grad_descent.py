import numpy as np

def SGD(network, X_train, Y_train, lossfunction, batch_size,
        learning_rate, regularizer, accelerator):

    prev_loss = 0
    while True:
        permut = np.random.permutation(len(Y_train))
        X_train = X_train[permut]
        Y_train = Y_train[permut]
        for idx, x in enumerate(X_train):

            print('.', end = '')

            network.clear_outputs()
            ypred = network.forward(x)
            op_gradient = lossfunction.gradient(ypred, Y_train[idx, None])
            network.backward(op_gradient, regularizer) # gradients are available

            if accelerator != None:
                weights_update, bias_update = accelerator.calc_update(learning_rate, network.Wgrad, network.bias_grad)
                network.update(weights_update = weights_update, bias_update = bias_update)
            else:
                network.update(learning_rate)

        total_loss = 0
        network.clear_outputs()
        for idx, x in enumerate(X_train):
            ypred = network.forward(x)
            total_loss += lossfunction.calc_loss(ypred, Y_train[idx, None])

        if regularizer != None:
            total_loss += regularizer.calc_loss(network.W)

        if abs(prev_loss - total_loss) < 10:
            break # stopping condition

        elif prev_loss != 0 and total_loss > 3 * prev_loss:
            print(prev_loss, total_loss)
            print('Exploding cost')
            break

        print("\n\n", total_loss)
        prev_loss = total_loss

def MiniBatchGD(network, X_train, Y_train, lossfunction, batch_size,
                learning_rate, regularizer, accelerator):

    prev_loss = 0
    while True:
        permut = np.random.permutation(len(Y_train))
        X_train = X_train[permut]
        Y_train = Y_train[permut]

        for batch_num in range(len(Y_train) // batch_size):
            start_idx = batch_num * batch_size
            end_idx = min(len(Y_train), batch_num * batch_size + batch_size)

            print('.', end = '')

            Wgrad = []
            for idx in range(len(network.Wgrad)):
                Wgrad.append(np.zeros(np.shape(network.Wgrad[idx])))
            bias_grad = np.zeros(np.shape(network.bias_grad))

            for idx in range(start_idx, end_idx):

                network.clear_outputs()
                ypred = network.forward(X_train[idx, None])
                op_gradient = lossfunction.gradient(ypred, Y_train[idx, None])
                network.backward(op_gradient, regularizer)

                bias_grad += network.bias_grad
                for idx in range(len(Wgrad)):
                    Wgrad[idx] += network.Wgrad[idx]

            if accelerator != None:
                weights_update, bias_update = accelerator.calc_update(learning_rate, Wgrad, bias_grad)
                network.update(weights_update = weights_update, bias_update = bias_update)
            else:
                network.update(learning_rate = learning_rate)

        total_loss = 0
        network.clear_outputs()
        for idx, x in enumerate(X_train):
            ypred = network.forward(x)
            total_loss += lossfunction.calc_loss(ypred, Y_train[idx, None])

        if regularizer != None:
            total_loss += regularizer.calc_loss(network.W)

        if abs(prev_loss - total_loss) < 10:
            break # stopping condition

        elif prev_loss != 0 and total_loss > 3 * prev_loss:
            print('Exploding cost')
            break

        print("\n\n", total_loss)
        prev_loss = total_loss


