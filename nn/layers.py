
from typing import List
import abc

import numpy as np


def norm_scale(grad: np.ndarray) -> np.ndarray:
    """
    L2 norm for gradient scaling
    """
    norm = np.sqrt(np.sum(grad * grad))
    return grad / norm


class Layer(metaclass=abc.ABCMeta):
    """
    Base abstract class for all layers.
    Provides forward pass caching functionality
    """

    def __init__(self):
        self._inputs = None

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        self._inputs = inputs
        return self.forward(inputs)

    @abc.abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass
    
    @abc.abstractmethod
    def backward(self, gradient_signal: np.ndarray) -> np.ndarray:
        pass
    
    def step(self, learning_rate: float) -> None:
        pass


class Sequential(Layer):
    """
    Aggregation for multiple layers.
    See Composer Design Pattern.
    """
    
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        result = inputs
        for layer in self.layers:
            result = layer(result)
        
        return result
    
    def backward(self, gradient_signal: np.ndarray) -> np.ndarray:
        result = gradient_signal
        for layer in reversed(self.layers):
            result = norm_scale(result)
            result = layer.backward(result)

        return result
    
    def step(self, learning_rate: float) -> None:
        for layer in self.layers:
            layer.step(learning_rate)


class Dense(Layer):

    def __init__(self,
                 input_units: int,
                 output_units: int,
                 weight_initializer: str = 'xavier'):
        super().__init__()
        
        if weight_initializer.lower() == 'xavier':
            weight_norm_coef = 1
        elif weight_initializer.lower() == 'he':
            weight_norm_coef = 2
        else:
            raise ValueError(f"Unknown weight_initializer: {weight_initializer}")

        weight_norm = weight_norm_coef / input_units

        self.weights = np.random.randn(input_units, output_units) * weight_norm
        self.biases = np.zeros(output_units)
        self._weights_grad = None
        self._biases_grad = None
        
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.matmul(inputs, self.weights) + self.biases
      
    def backward(self, gradient_signal: np.ndarray) -> np.ndarray:

        inputs_grad = np.dot(gradient_signal, self.weights.T)
        self._weights_grad = np.dot(self._inputs.T, gradient_signal)
        self._biases_grad = np.sum(gradient_signal, axis = 0)
        
        return inputs_grad
    
    def step(self, learning_rate: float) -> None:
        self.weights -= learning_rate * self._weights_grad
        self.biases -= learning_rate * self._biases_grad


class ReLU(Layer):
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.maximum(0, inputs)

    def backward(self, gradient_signal: np.ndarray) -> np.ndarray:
        grad = self._inputs > 0
        return gradient_signal * grad


class Sigmoid(Layer):

    def __call__(self, inputs: np.ndarray) -> np.ndarray:
        self._inputs = self.forward(inputs)
        return self._inputs
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.where(inputs >= 0, 
                        1 / (1 + np.exp(-1 * inputs)), 
                        np.exp(inputs) / (1 + np.exp(inputs)))

    def backward(self, gradient_signal: np.ndarray) -> np.ndarray:
        grad = self._inputs * (1 - self._inputs)
        return gradient_signal * grad
