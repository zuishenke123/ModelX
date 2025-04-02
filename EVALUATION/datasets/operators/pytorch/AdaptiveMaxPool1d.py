# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for AdaptiveMaxPool1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data This function
    # simulates typical scenarios for AdaptiveMaxPool1d, which is used to output fixed-size tensors by adapting the
    # pooling operation
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 4   # Number of channels
        L = 100 # Original length of each sequence

        # Random data for each element in the batch, representing 1D sequences
        data = np.random.randn(N, C, L).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test AdaptiveMaxPool1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an AdaptiveMaxPool1d layer targeting a fixed output size
    # Example targets an output size of 20
    adaptive_pool = nn.AdaptiveMaxPool1d(20)
    ret = []

    print("Testing AdaptiveMaxPool1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply AdaptiveMaxPool1d
        pooled_data = adaptive_pool(data_tensor)
        ret.append(pooled_data[0, 0].numpy())

    return ret
