# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.abs using NumPy
def generate_test_cases():
    # Generate a list of numpy arrays with different characteristics
    test_cases = [
        np.array([-1, -2, -3], dtype=int),            # Simple negative integers
        np.array([1.5, -2.5, 3.5], dtype=float),      # Floating point numbers
        np.array([[-1, 2], [-3, 4]], dtype=int),      # 2D array with mixed sign
        np.array([0], dtype=int),                     # Zero
        np.array([[-1.2, 2.3], [0, -0.1]], dtype=float) # Mixed 2D array with zero and floating points
    ]
    return test_cases


# Function to test torch.abs with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.abs on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        tensor = torch.from_numpy(case)
        result = torch.abs(tensor)
        print(f"Test Case {i+1}:")
        print("Input tensor:", case)
        print("Output tensor (absolute values):", result.numpy())
        ret.append(result.numpy())

    return ret
