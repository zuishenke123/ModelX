# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.add using NumPy
def generate_test_cases():
    # Generate a list of numpy arrays with varying dimensions and values
    test_cases = [
        (np.random.randn(5, 5), np.random.randn(5, 5)),  # same-sized arrays
        (np.random.randn(1, 5), np.random.randn(5, 5)),  # broadcasting smaller array
        (np.random.randn(5, 5), 0.5),                    # adding scalar to array
        (np.random.randn(5, 5), np.full((5, 5), 3)),     # array and constant-filled array
        (np.array([1, 2, 3]), np.array([4, 5, 6])),      # one-dimensional arrays
        (np.random.randn(2, 3), np.random.randn(2, 1)),  # broadcasting column vector
        (np.random.randn(3, 1), np.random.randn(1, 3)),  # broadcasting row vector
        (np.random.randn(5), np.random.randn(5, 5)),     # broadcasting row vector across matrix
        (np.random.randn(5, 5, 5), np.random.randn(5, 5)), # broadcasting 2D array across 3D array
        (np.full((2, 2), 1), np.full((2, 2), 2))         # constant-filled arrays
    ]
    return test_cases


# Function to test torch.add with generated test cases
def test_operator(test_cases):
    import torch

    print("Testing torch.add on multiple test cases:")
    ret = []

    for i, (a_np, b_np) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors just before operation
        a = torch.from_numpy(a_np)
        b = torch.from_numpy(b_np) if isinstance(b_np, np.ndarray) else b_np

        # Perform addition using torch.add
        result = torch.add(a, b)
        ret.append(result.numpy())
    return ret