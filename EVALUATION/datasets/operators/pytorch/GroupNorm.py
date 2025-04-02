# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for GroupNorm
def generate_test_cases_GroupNorm(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of feature data This function
    # simulates typical scenarios for GroupNorm, which is used to normalize features across defined groups of channels
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 32  # Number of channels
        H = 28  # Height of the feature map
        W = 28  # Width of the feature map

        # Random data for each element in the batch, representing feature maps with a certain number of channels
        data = np.random.randn(N, C, H, W).astype(np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test GroupNorm with generated test cases
def test_GroupNorm(test_cases):
    import torch
    import torch.nn as nn

    # Define a GroupNorm layer with a specific number of groups and channels
    # Here, dividing the 32 channels into 8 groups
    group_norm = nn.GroupNorm(num_groups=8, num_channels=32)
    ret = []

    print("Testing GroupNorm on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply GroupNorm
        normalized_data = group_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=(2, 3)).detach().numpy())

    return ret