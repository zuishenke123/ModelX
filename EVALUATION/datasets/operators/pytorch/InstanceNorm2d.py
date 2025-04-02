# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for InstanceNorm2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data This function
    # simulates typical scenarios for InstanceNorm2d, which is used to normalize feature maps individually for each
    # sample
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 8   # Number of channels
        H = 28  # Height of the feature map
        W = 28  # Width of the feature map

        # Random data for each element in the batch, representing 2D feature maps across multiple channels
        data = np.random.randn(N, C, H, W).astype(np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test InstanceNorm2d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an InstanceNorm2d layer with the number of features (channels) set to match the test data, including
    # learnable affine parameters
    instance_norm = nn.InstanceNorm2d(num_features=8, affine=True)
    ret = []

    print("Testing InstanceNorm1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply InstanceNorm1d
        normalized_data = instance_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=(2, 3)).detach().numpy())

    return ret
