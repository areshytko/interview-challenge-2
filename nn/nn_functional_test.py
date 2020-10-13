
import unittest


from nn.layers import Sequential, ReLU, Dense, Sigmoid
from nn.train import train, minibatch_iterator, MSE
from nn.data import make_linear


class MultiLayerFunctionalTest(unittest.TestCase):

    def setUp(self):
        X_train, y_train, X_val, y_val = make_linear(
            x_min=-5,
            x_max=5,
            n_samples=100,
            n_features=11,
            n_labels=1,
            a=3,
            b=4,
            sigma=0.5,
            test_size=0.2)
        
        self.X_train = X_train
        self.y_train = y_train
        
        self.model = Sequential([Dense(X_train.shape[1], 100),
                        ReLU(),
                        Dense(100, 200),
                        Sigmoid(),
                        Dense(200, 1)])

    def test_smoke(self):
        train(self.model, self.X_train, self.y_train, learning_rate=0.01)


if __name__ == '__main__':
    unittest.main()
