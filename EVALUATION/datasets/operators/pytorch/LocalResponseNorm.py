# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for LocalResponseNorm
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of multi-dimensional data
    # This function simulates typical scenarios for LocalResponseNorm, which is used to normalize data across local input regions
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 20  # Number of channels
        H = 28  # Height of the feature map
        W = 28  # Width of the feature map

        # Random data for each element in the batch, representing a feature map with multiple channels
        data = np.random.randn(N, C, H, W).astype(
            np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test LocalResponseNorm with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a LocalResponseNorm layer with specific parameters
    # Size defines the amount of neighbouring channels used for normalization
    local_resp_norm = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0)
    ret = []

    print("Testing LocalResponseNorm on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply LocalResponseNorm
        normalized_data = local_resp_norm(data_tensor)
        ret.append(normalized_data[0, :2, :2, :2].numpy())

    return ret
