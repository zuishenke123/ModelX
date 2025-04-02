# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.kron using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([[1, 2], [3, 4]], dtype=np.float32), np.array([[0, 5], [6, 7]], dtype=np.float32)),
        # Small 2x2 matrices
        (np.array([1, 2, 3], dtype=np.float32), np.array([0, 4], dtype=np.float32)),  # Vectors
        (np.eye(3, dtype=np.float32), np.array([[2, 0], [0, 2]], dtype=np.float32)),  # Identity matrix and 2x2 matrix
        (np.random.rand(3, 3), np.random.rand(3, 3)),  # Random 3x3 matrices
        (np.array([1], dtype=np.float32), np.array([7, 8, 9], dtype=np.float32)),  # Scalar and vector
        (np.ones((2, 2), dtype=np.float32), np.full((2, 2), 2, dtype=np.float32)),  # Ones matrix and constant matrix
        (np.array([], dtype=np.float32).reshape(0, 2), np.array([1, 2], dtype=np.float32)),  # Empty array and vector
        (np.array([[1, 2], [3, 4]], dtype=np.float32), np.eye(2, dtype=np.float32)),  # Matrix and identity
        (np.full((1, 1), 3, dtype=np.float32), np.array([[1, 2], [3, 4]], dtype=np.float32)),
        # Scalar matrix and matrix
        (np.array([[1, 2, 3]], dtype=np.float32), np.array([[4], [5], [6]], dtype=np.float32))
        # Row vector and column vector
    ]
    return test_cases


# Function to test torch.kron with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.kron on multiple test cases:")
    ret = []
    for i, (a, b) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_a = torch.from_numpy(a)
        tensor_b = torch.from_numpy(b)

        # Apply torch.kron
        kron_results = torch.kron(tensor_a, tensor_b)
        ret.append(kron_results.numpy())

    return ret
