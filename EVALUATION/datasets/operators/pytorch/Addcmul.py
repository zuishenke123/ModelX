# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.addcmul using NumPy
def generate_test_cases():
    # Generate a list of test cases with compatible dimensions for element-wise multiplication
    test_cases = [
        (np.random.rand(5, 5), np.random.rand(5, 5), np.random.rand(5, 5)),  # same-sized arrays
        (np.random.rand(10, 5), np.random.rand(10, 5), np.random.rand(10, 5)), # different larger sizes
        (np.random.rand(1, 5), np.random.rand(5, 5), np.random.rand(5, 5)),  # broadcasting the first tensor
        (np.random.rand(5, 1), np.random.rand(5, 5), np.random.rand(5, 5)),  # broadcasting the second tensor
        (np.random.rand(5, 5), np.random.rand(1, 5), np.random.rand(5, 5)),  # broadcasting the third tensor
        (np.zeros((5, 5)), np.ones((5, 5)), np.ones((5, 5))),                # Zero elements
        (np.ones((5, 5)), np.zeros((5, 5)), np.zeros((5, 5))),               # Zero multiplication
        (np.linspace(1, 5, 25).reshape(5, 5), np.linspace(5, 1, 25).reshape(5, 5), np.full((5, 5), 2)), # Linearly spaced values
        (np.random.rand(5, 5), np.random.rand(5, 5), np.random.rand(1, 5)),  # broadcasting in multiplication
        (np.random.rand(3, 3, 5), np.random.rand(3, 3, 5), np.random.rand(3, 3, 5))  # 3D arrays
    ]
    return test_cases


# Function to test torch.addcmul with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.addcmul on multiple test cases:")
    ret = []

    for i, (t, t1, t2) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(t)
        tensor1 = torch.from_numpy(t1)
        tensor2 = torch.from_numpy(t2)

        # Apply torch.addcmul with a scalar value
        value = 0.5
        result = torch.addcmul(tensor, value, tensor1, tensor2)
        ret.append(result.numpy())

    return ret
