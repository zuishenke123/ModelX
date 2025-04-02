# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.addcdiv using NumPy
def generate_test_cases():
    # Generate a list of test cases with different characteristics
    test_cases = [
        (np.random.rand(5, 5), np.random.rand(5, 5), np.random.rand(5, 5) + 0.1),  # Random positive values
        (np.random.randn(5, 5), np.random.randn(5, 5), np.random.randn(5, 5) + 0.1), # Random values, shift to avoid zero in denominator
        (np.full((5, 5), 4), np.full((5, 5), 2), np.full((5, 5), 0.5)),   # Constant values
        (np.zeros((5, 5)), np.ones((5, 5)), np.ones((5, 5))),            # Zero numerator
        (np.linspace(1, 5, 25).reshape(5, 5), np.linspace(5, 1, 25).reshape(5, 5), np.ones((5, 5))),  # Linearly spaced values
        (np.random.rand(5, 5), np.random.rand(5, 5) + 0.01, np.random.rand(5, 5)), # Small denominators
        (np.random.rand(1, 5), np.random.rand(5, 5), np.random.rand(5, 5)),  # Broadcasting first tensor
        (np.random.rand(5, 1), np.random.rand(5, 5), np.random.rand(5, 5)),  # Broadcasting second tensor
        (np.random.rand(5, 5), np.random.rand(5, 5), np.random.rand(1, 5)),  # Broadcasting third tensor
        (np.random.rand(3, 3, 5), np.random.rand(3, 3, 5), np.random.rand(3, 3, 5))  # 3D arrays
    ]
    return test_cases


# Function to test torch.addcdiv with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.addcdiv on multiple test cases:")
    ret = []
    for i, (t, t1, t2) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(t)
        tensor1 = torch.from_numpy(t1)
        tensor2 = torch.from_numpy(t2)

        # Apply torch.addcdiv with a scalar value
        value = 0.5
        result = torch.addcdiv(tensor, value, tensor1, tensor2)
        ret.append(result.numpy())

    return ret
