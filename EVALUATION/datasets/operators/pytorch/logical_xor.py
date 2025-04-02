# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.logical_xor using NumPy
def generate_test_cases():
    # Generate a list of test cases with varying characteristics
    test_cases = [
        (np.array([True, False, True, False], dtype=bool), np.array([True, True, False, False], dtype=bool)),  # Basic cases
        (np.zeros((10,), dtype=bool), np.ones((10,), dtype=bool)),  # All False vs all True
        (np.full((5, 5), True, dtype=bool), np.full((5, 5), False, dtype=bool)),  # 2D array all True vs all False
        (np.random.choice([True, False], size=(10, 10)), np.random.choice([True, False], size=(10, 10))),  # Random Boolean 2D arrays
        (np.array([], dtype=bool), np.array([], dtype=bool)),  # Empty arrays
        (np.tile([True, False], 10), np.tile([False, True], 10)),  # Repeated pattern arrays
        (np.eye(5, dtype=bool), np.ones((5, 5), dtype=bool)),  # Identity matrix vs all True
        (np.tri(5, dtype=bool), np.tri(5, k=-1, dtype=bool)),  # Lower triangular vs strictly lower triangular
        (np.full((10,), True, dtype=bool), np.full((10,), True, dtype=bool)),  # All True for both arrays
        (np.full((10,), False, dtype=bool), np.full((10,), False, dtype=bool))  # All False for both arrays
    ]
    return test_cases


# Function to test torch.logical_xor with generated test cases
def test_torch_logical_xor(test_cases):
    import torch
    print("Testing torch.logical_xor on multiple test cases:")
    ret = []
    for i, (a, b) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor_a = torch.from_numpy(a)
        tensor_b = torch.from_numpy(b)

        # Apply torch.logical_xor
        logical_xor_results = torch.logical_xor(tensor_a, tensor_b)
        ret.append(logical_xor_results.numpy())

    return ret
