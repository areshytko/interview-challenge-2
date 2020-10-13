
import unittest

import numpy as np

from nn.layers import Dense


class DenseTest(unittest.TestCase):
    
    def test_dimensions(self):
        model = Dense(2, 1)
        actual = model(np.array([[1, 2]]))
        self.assertEqual((1, 1), actual.shape)

        model = Dense(1, 1)
        actual = model(np.array([[1]]))
        self.assertEqual((1, 1), actual.shape)

        model = Dense(4, 3)
        actual = model(np.random.randn(5, 4))
        self.assertEqual((5, 3), actual.shape)
    
    def test_backward_zero_gradient(self):
        
        model = Dense(4, 3)
        model(np.zeros((1, 4)))
        
        weights = np.copy(model.weights)

        model.backward(np.ones((1, 3)))
        model.step(1)
        new_weights = model.weights

        self.assertEqual(weights.tolist(), new_weights.tolist())
    
    def test_backward_non_zero_gradient(self):
        
        model = Dense(4, 3)
        model(np.random.randn(1, 4))
        
        weights = np.copy(model.weights)

        model.backward(np.ones((1, 3)))
        model.step(1)
        new_weights = model.weights

        self.assertNotEqual(weights.tolist(), new_weights.tolist())
    
    def test_backward_signal_is_zero(self):

        model = Dense(4, 3)
        model(np.random.randn(1, 4))
        actual = model.backward(np.zeros((1, 3)))

        self.assertEqual(np.zeros((1, 4)).tolist(), actual.tolist())


if __name__ == '__main__':
    unittest.main()
