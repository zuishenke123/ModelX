# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for InstanceNorm3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data
    # This function simulates typical scenarios for InstanceNorm3d, which is used to normalize volumetric feature maps individually for each sample
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 4   # Number of channels
        D = 16  # Depth of the feature map
        H = 16  # Height of the feature map
        W = 16  # Width of the feature map

        # Random data for each element in the batch, representing 3D feature maps across multiple channels
        data = np.random.randn(N, C, D, H, W).astype(np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases



# Function to test InstanceNorm3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an InstanceNorm3d layer with the number of features (channels) set to match the test data, including
    # learnable affine parameters
    instance_norm = nn.InstanceNorm3d(num_features=4, affine=True)
    ret = []

    print("Testing InstanceNorm3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply InstanceNorm1d
        normalized_data = instance_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=(2, 3, 4)).detach().numpy())

    return ret
