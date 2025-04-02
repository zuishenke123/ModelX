# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for AvgPool3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data This function
    # simulates typical scenarios for AvgPool3d, which is used to downsample 3D data by applying an average filter
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels
        D = 16  # Depth of the volumetric data
        H = 16  # Height
        W = 16  # Width

        # Random data for each element in the batch, representing 3D volumetric feature maps
        data = np.random.randn(N, C, D, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test AvgPool1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an AvgPool3d layer with specific parameters
    # Example uses a pool size of 2x2x2 with stride 2 and padding 1
    avg_pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=1)
    ret = []

    print("Testing AvgPool3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply AvgPool3d
        pooled_data = avg_pool(data_tensor)
        ret.append(pooled_data[0, 0, pooled_data.shape[2]//2, :5, :5].numpy())

    return ret
