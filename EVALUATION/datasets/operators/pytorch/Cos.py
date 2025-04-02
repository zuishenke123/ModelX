# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.cos using NumPy
def generate_test_cases():
    # Generate a list of test cases with various characteristics
    test_cases = [
        np.array([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], dtype=np.float32),  # Standard angles
        np.linspace(-2*np.pi, 2*np.pi, num=10, dtype=np.float32),  # Linearly spaced values around the unit circle
        np.random.uniform(-3*np.pi, 3*np.pi, size=(10,)),          # Random values within multiple periods
        np.zeros(10, dtype=np.float32),                           # Zero values
        np.full((10,), np.pi, dtype=np.float32),                  # Constant pi values
        np.random.uniform(-np.pi, np.pi, size=(5, 5)),            # 2D array of random angles within one period
        np.array([np.inf, -np.inf, np.nan], dtype=np.float32),    # Infinity and NaN values
        np.array([], dtype=np.float32),                           # Empty array
        np.full((10, 10), 0.5*np.pi, dtype=np.float32),           # 2D array with pi/2 values
        np.full((10, 10), 1.5*np.pi, dtype=np.float32)            # 2D array with 3*pi/2 values
    ]
    return test_cases


# Function to test torch.cos with generated test cases
def test_torch_cos(test_cases):
    import torch
    print("Testing torch.cos on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.cos
        cos_results = torch.cos(tensor)
        ret.append(cos_results.numpy())

    return ret

