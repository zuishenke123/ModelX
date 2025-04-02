# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MaxPool3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data This function
    # simulates typical scenarios for MaxPool3d, which is used to downsample 3D data by applying a maximum filter
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


# Function to test MaxPool3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a MaxPool3d layer with specific parameters
    # Example uses a pool size of 2x2x2 with stride 2
    max_pool = nn.MaxPool3d(kernel_size=2, stride=2)
    ret = []

    print("Testing MaxPool3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply MaxPool3d
        pooled_data = max_pool(data_tensor)
        ret.append(pooled_data[0, 0, 0, :5, :5].numpy())

    return ret
