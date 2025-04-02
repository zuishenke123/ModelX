# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.logit using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([0.1, 0.5, 0.9], dtype=np.float32),  # Simple probabilities
        np.linspace(0.01, 0.99, num=10, dtype=np.float32),  # Linearly spaced probabilities close to 0 and 1
        np.random.rand(10),  # Random probabilities
        np.array([0.001, 0.999], dtype=np.float32),  # Values very close to 0 and 1
        np.full((10,), 0.5, dtype=np.float32),  # Constant probability
        np.random.uniform(0.1, 0.9, size=(5, 5)),  # 2D array of probabilities
        np.array([1e-6, 1-1e-6], dtype=np.float32),  # Extreme probabilities close to 0 and 1
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), 0.25, dtype=np.float32),  # 2D array with constant value
        np.array([0.05, 0.95], dtype=np.float32)  # Probabilities near the boundaries
    ]
    return test_cases


# Function to test torch.logit with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.logit on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.logit with a small eps for numerical stability
        logit_results = torch.logit(tensor, eps=1e-6)
        ret.append(logit_results.numpy())

    return ret
