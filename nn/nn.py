
from typing import List

import numpy as np

from nn.layers import Sequential, ReLU, Dense, Sigmoid, Layer
from nn.train import train, minibatch_iterator, MSE


def layers_factory(name: str,
                   n_input: int,
                   n_hidden: int) -> List[Layer[]:

    result = [Dense(n_input, n_hidden)]

    if name == 'linear':
        pass
    elif name == 'relu':
        result.append(ReLU())
    elif name == 'sigmoid':
        result.append(Sigmoid())
    else:
        raise ValueError(f"Unknown laye name: {name}")

    return result


class NeuralNetwork:

    def __init__(self, n_input, n_hidden, hidden_types, n_output):
        """
        Parameters
        ----------
        n_input : int
            Number of input neurons
        n_hidden: list of int
            Each item in the list is the size of the hidden layer
            at the corresponding index.
        hidden_types: list of str
            Each item in the list describes the activation function in
            the corresponding hidden layer.
            Supported values are:
                - 'relu':    ReLu activation
                - 'sigmoid': Sigmoid activation
                - 'linear':  Linear activation
            The length of this list should be the same as the length of the n_hidden list.
        n_output : int
            Number of output neurons
        """

        assert isinstance(n_hidden, int) == isinstance(hidden_types, str)

        if isinstance(n_hidden, list):
            assert len(n_hidden) == len(hidden_types)

        if isinstance(n_hidden, int):
            layers = layers_factory(name=hidden_types, n_input=n_input, n_hidden=n_hidden)
        else:
            layers = []
            n_in = n_input
            for name, n_out in zip(hidden_types, n_hidden):
                layers += layers_factory(name=name,
                                         n_input=n_in,
                                         n_hidden=n_out)
                n_in = n_out
        
        self.model = Sequential(layers)

    def train(self, x, y, learning_rate, n_epochs):
        """
        Train the neural network by backpropagation,
        using (full-batch) stochastic gradient descent.
        Parameters
        ----------
        x : np.array<np.float32>
            The dimension should be (N, M), where:
                - N is the number of training examples in the dataset
                - M is the number of features for each training example
            M should be equal to the value of n_input passed to the class constructor.
        y : np.array<np.float32>
            The dimension should be (N, M), where:
                - N is the number of training examples in the dataset
                - M is the number of outputs for each training example
            The value of N should be the same for both x and y.
            M should be equal to the value of n_output passed to the class constructor.
        learning_rate : float
            Learning rate to use
        n_epochs: int
            Number of epochs to train
        """

        for epoch in n_epochs:
            loss = train(self.model, x, y, learning_rate=learning_rate)
            print("Epoch", epoch)
            print("Train MSE:", loss)
