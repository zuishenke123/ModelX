# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.asin using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying conditions
    test_cases = [
        np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32),  # Edge and midpoint values
        np.linspace(-1, 1, num=10, dtype=np.float32),             # Linearly spaced values within domain
        np.random.uniform(-1, 1, size=(10,)),                     # Random values within the domain
        np.array([-1.1, 0.0, 1.1], dtype=np.float32),             # Values out of domain including valid ones
        np.full((10,), 1.0, dtype=np.float32),                    # Constant array of edge value
        np.zeros((10, 10), dtype=np.float32),                     # 2D array of zeros
        np.full((5, 5), -1.0, dtype=np.float32),                  # 2D array of negative edge value
        np.random.uniform(-0.5, 0.5, size=(5, 5)),                # 2D array of small values
        np.random.uniform(-2, 2, size=(10, 10)),                  # 2D array including values out of domain
        np.array([], dtype=np.float32)                            # Empty array
    ]
    return test_cases


# Function to test torch.asin with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.asin on multiple test cases:")
    ret = []

    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.asin
        asin_results = torch.asin(tensor.clamp(-1, 1))  # Clamping to ensure validity within function domain
        ret.append(asin_results.numpy())
    return ret
