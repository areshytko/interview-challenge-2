
from typing import Generator, Tuple

import numpy as np
from tqdm import trange

from nn.layers import Layer

class MSE:

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        y_pred : np.array<np.float32>
            The predicted outputs of a neural network.
            The dimension should be (N, M), where:
                - N is the number of training examples in the dataset
                - M is the number of outputs for each training example
        y_true : np.array<np.float32>
            The true labels.
            The dimension should be (N, M), where:
                - N is the number of training examples in the dataset
                - M is the number of outputs for each training example
        """
        assert y_pred.shape == y_true.shape, f"Non-matching shape: {y_pred.shape}, {y_true.shape}"
        diff = y_true - y_pred
        diff_squared = diff * diff
        total_loss = diff_squared.sum(axis=0) / 2.0
        mean_loss = total_loss / y_true.shape[0]
        return mean_loss
    
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        return y_pred - y_true


def train(model: Layer,
          X: np.ndarray,
          y: np.ndarray,
          learning_rate: float) -> float:

    logits = model(X)
    
    mse = MSE()
    loss = mse(logits, y)
    
    loss_grad = mse.backward(logits, y)
    model.backward(loss_grad)
    
    model.step(learning_rate)
    
    return np.mean(loss)


def minibatch_iterator(inputs: np.ndarray,
                       labels: np.ndarray,
                       batch_size: int,
                       shuffle: bool = False) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    
    assert len(inputs) == len(labels)
    
    if shuffle:
        indices = np.random.permutation(len(inputs))
    
    for start_idx in trange(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)

        yield inputs[excerpt], labels[excerpt]
