# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MaxUnpool1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data This function
    # simulates typical scenarios for MaxUnpool1d, which is used to perform the inverse operation of MaxPool1d
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 4   # Number of channels
        L = 50  # Length of each sequence after pooling

        # Random data for each element in the batch, representing 1D sequences
        data = np.random.randn(N, C, L).astype(np.float32)

        # Create indices to simulate the outputs of MaxPool1d with a kernel size of 2 and stride 2
        indices = np.random.randint(0, L, (N, C, L))

        test_cases.append((data, indices))
    return test_cases


# Function to test MaxUnpool1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a MaxPool1d layer to simulate pooling
    max_pool = nn.MaxPool1d(kernel_size=2, stride=2, return_indices=True)
    # Define a MaxUnpool1d layer with specific parameters to reverse the operation
    max_unpool = nn.MaxUnpool1d(kernel_size=2, stride=2)
    ret = []

    print("Testing MaxUnpool1d on multiple test cases:")
    # Iterate over each test case
    for i, (data, indices) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        # First, simulate the pooling operation to get realistic indices
        pooled_data, pool_indices = max_pool(data_tensor)

        # Apply MaxUnpool1d using the indices from the simulated pooling
        unpooled_data = max_unpool(pooled_data, pool_indices)
        ret.append(unpooled_data[0, 0, :10].numpy())

    return ret
