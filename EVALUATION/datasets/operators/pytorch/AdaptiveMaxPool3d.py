# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for AdaptiveMaxPool3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data This function
    # simulates typical scenarios for AdaptiveMaxPool3d, which is used to output fixed-size tensors by adapting the
    # pooling operation to the input size in three dimensions
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 2   # Number of channels
        D = 30  # Depth of the volumetric data
        H = 30  # Height
        W = 30  # Width

        # Random data for each element in the batch, representing 3D volumetric feature maps
        data = np.random.randn(N, C, D, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test AdaptiveMaxPool3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an AdaptiveMaxPool3d layer targeting a fixed output size (e.g., 5x5x5)
    adaptive_pool = nn.AdaptiveMaxPool3d((5, 5, 5))
    ret = []

    print("Testing AdaptiveMaxPool3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply AdaptiveMaxPool3d
        pooled_data = adaptive_pool(data_tensor)
        ret.append(pooled_data[0, 0, pooled_data.shape[2]//2].numpy())

    return ret
