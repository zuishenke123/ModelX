# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for InstanceNorm1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data
    # This function simulates typical scenarios for InstanceNorm1d, which is used to normalize features individually for each sample
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5   # Number of channels or features
        L = 50  # Length of each feature

        # Random data for each element in the batch, representing 1D features across multiple channels
        data = np.random.randn(N, C, L).astype(np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test InstanceNorm1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an InstanceNorm1d layer with the number of features (channels) set to match the test data
    instance_norm = nn.InstanceNorm1d(num_features=5, affine=True)
    ret = []

    print("Testing InstanceNorm1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply InstanceNorm1d
        normalized_data = instance_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=2).detach().numpy())

    return ret
