# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MaxPool1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data This function
    # simulates typical scenarios for MaxPool1d, which is used to downsample 1D data by applying a maximum filter
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5   # Number of channels
        L = 100 # Length of each sequence

        # Random data for each element in the batch, representing 1D sequences
        data = np.random.randn(N, C, L).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test MaxPool1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a MaxPool1d layer with specific parameters
    # Example uses a pool size of 3 with stride 2
    max_pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
    ret = []

    print("Testing MaxPool1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply MaxPool1d
        pooled_data = max_pool(data_tensor)
        ret.append(pooled_data[0, 0, :10].numpy())

    return ret
