# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.clip using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        np.array([1.5, -2.5, 3.0, -4.1, 5.5], dtype=np.float32),  # Mixed positive and negative floats
        np.random.randn(10),  # Random normal values
        np.linspace(-10, 10, num=10, dtype=np.float32),  # Linearly spaced values
        np.zeros(10, dtype=np.float32),  # Zero values
        np.full((10,), 20.0, dtype=np.float32),  # Constant high values
        np.full((10,), -20.0, dtype=np.float32),  # Constant low values
        np.random.uniform(-15, 15, size=(5, 5)),  # 2D array of random floats
        np.array([], dtype=np.float32),  # Empty array
        np.full((10, 10), -0.1, dtype=np.float32)  # 2D array with negative values
    ]
    return test_cases


# Function to test torch.clip with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.clip on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.clip with min=-1.0 and max=1.0
        clipped_results = torch.clip(tensor, -1.0, 1.0)
        ret.append(clipped_results.numpy())

    return ret
