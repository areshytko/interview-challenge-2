
from typing import Tuple, Union

import numpy as np


def make_linear(x_min: float,
                x_max: float,
                n_samples: int,
                n_features: int,
                n_labels: int,
                a: float,
                b: float,
                sigma: float,
                test_size: float = 0.0,
                seed = 42) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates linear synthetic regression dataset
    """
    np.random.seed(seed)

    X = np.random.rand(n_samples, n_features) * (x_max - x_min) + x_min
    a = a * np.random.rand(n_features, n_labels)
    y = np.dot(X, a) + b + np.random.randn(n_samples, n_labels) * sigma
    
    if 0 < test_size:
        test_size = int(n_samples * test_size)
        X_test, y_test = X[0:test_size, :], y[0:test_size, :]
        X_train, y_train = X[test_size:, :], y[test_size:, :]
        return X_train, y_train, X_test, y_test
    
    return X, y


def make_sinus(x_min: float,
               x_max: float,
               n_samples: int,
               n_features: int,
               n_labels: int,
               a: float,
               b: float,
               amplitude: float,
               phase: float,
               sigma: float,
               test_size: float = 0.0,
               seed = 42) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates synthetic dataset with sinusoidal structure with linear noisy trend
    """
    np.random.seed(seed)

    X = np.random.rand(n_samples, n_features) * (x_max - x_min) + x_min
    a = a * np.random.rand(n_features, n_labels)
    noisy_trend = np.dot(X, a) + b + np.random.randn(n_samples, n_labels) * sigma

    amplitude = amplitude * np.random.rand(n_features, n_labels)
    y = np.dot(np.sin(X + phase), amplitude) + noisy_trend
    
    if 0 < test_size:
        test_size = int(n_samples * test_size)
        X_test, y_test = X[0:test_size, :], y[0:test_size, :]
        X_train, y_train = X[test_size:, :], y[test_size:, :]
        return X_train, y_train, X_test, y_test
    
    return X, y

