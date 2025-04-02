# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for AdaptiveAvgPool2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data
    # This function simulates typical scenarios for AdaptiveAvgPool2d, which is used to output fixed-size tensors by adapting the average pooling operation to the input size in two dimensions
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels
        H = 64  # Height of the feature map
        W = 64  # Width of the feature map

        # Random data for each element in the batch, representing 2D feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases

# Function to test AdaptiveAvgPool1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an AdaptiveAvgPool2d layer targeting a fixed output size (e.g., 10x10)
    adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))
    ret = []

    print("Testing AdaptiveAvgPool2d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply AdaptiveAvgPool2d
        pooled_data = adaptive_pool(data_tensor)
        ret.append(pooled_data[0, 0].numpy())

    return ret
