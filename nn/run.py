
import numpy as np

from nn.layers import Sequential, ReLU, Dense, Sigmoid
from nn.train import train, minibatch_iterator, MSE
from nn.data import make_linear, make_sinus

N_LABELS = 3
LEARNING_RATE = 0.01
N_EPOCHS = 25
BATCH_SIZE = 100
DATA = 'sinus'

def main():
    np.random.seed(42)

    if DATA == 'linear':
        X_train, y_train, X_val, y_val = make_linear(
            x_min=-5,
            x_max=5,
            n_samples=10000,
            n_features=11,
            n_labels=N_LABELS,
            a=3,
            b=4,
            sigma=0.5,
            test_size=0.2)
    else:
        X_train, y_train, X_val, y_val = make_sinus(
            x_min=-5,
            x_max=5,
            n_samples=10000,
            n_features=11,
            n_labels=N_LABELS,
            a=3,
            b=4,
            amplitude=10,
            phase=1,
            sigma=0.5,
            test_size=0.2)

    model = Sequential([Dense(X_train.shape[1], 100),
                        ReLU(),
                        Dense(100, 200),
                        Sigmoid(),
                        Dense(200, N_LABELS)])
    loss = MSE()

    for epoch in range(N_EPOCHS):

        if BATCH_SIZE:
            for x_batch, y_batch in minibatch_iterator(X_train,
                                                       y_train,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=True):
                train(model, x_batch, y_batch, learning_rate=LEARNING_RATE)
        else:
            train(model, X_train, y_train, learning_rate=LEARNING_RATE)
        
        train_loss = np.mean(loss(model(X_train), y_train))
        val_loss = np.mean(loss(model(X_val), y_val))
        
        print("Epoch", epoch)
        print("Train MSE:", train_loss)
        print("Val MSE:", val_loss)


if __name__ == '__main__':
    main()