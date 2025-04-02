# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for BatchNorm1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data
    # This function simulates typical scenarios for BatchNorm1d, which is used to normalize features across the batch
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        D = 20  # Number of features

        # Random data for each element in the batch
        data = np.random.randn(N, D).astype(np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test BatchNorm1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a BatchNorm1d layer with the number of features set to match the test data
    batch_norm = nn.BatchNorm1d(momentum=0.9, eps=1e-05, num_features=20)
    ret = []

    print("Testing BatchNorm1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Enable training mode to update running estimates of mean and variance
        batch_norm.eval()

        # Apply BatchNorm1d
        normalized_data = batch_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=0).detach().numpy())

    return ret
