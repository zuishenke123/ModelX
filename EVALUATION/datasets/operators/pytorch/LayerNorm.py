# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for LayerNorm
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of multi-dimensional data
    # This function simulates typical scenarios for LayerNorm, which is used to normalize data across the last certain dimensions
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 8   # Number of channels
        H = 28  # Height of the feature map
        W = 28  # Width of the feature map

        # Random data for each element in the batch, representing a feature map with multiple channels
        data = np.random.randn(N, C, H, W).astype(np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test LayerNorm with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a LayerNorm layer with the normalization applied across the last three dimensions (C, H, W)
    layer_norm = nn.LayerNorm([8, 28, 28])
    ret = []

    print("Testing LayerNorm on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply LayerNorm
        normalized_data = layer_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=(1, 2, 3)).detach().numpy())

    return ret
