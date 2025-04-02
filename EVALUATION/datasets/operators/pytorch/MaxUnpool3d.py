# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MaxUnpool3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data This function
    # simulates typical scenarios for MaxUnpool3d, which is used to perform the inverse operation of MaxPool3d
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 2   # Number of channels
        D = 10  # Depth of the volumetric data after pooling
        H = 10  # Height
        W = 10  # Width

        # Random data for each element in the batch, representing 3D volumetric feature maps
        data = np.random.randn(N, C, D, H, W).astype(np.float32)

        # Create indices to simulate the outputs of MaxPool3d with a kernel size of 2x2x2 and stride 2
        indices = np.random.randint(0, D * H * W, (N, C, D, H, W))

        test_cases.append((data, indices))
    return test_cases


# Function to test MaxUnpool3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a MaxPool3d layer to simulate pooling
    max_pool = nn.MaxPool3d(kernel_size=2, stride=2, return_indices=True)
    # Define a MaxUnpool3d layer with specific parameters to reverse the operation
    max_unpool = nn.MaxUnpool3d(kernel_size=2, stride=2)
    ret = []

    print("Testing MaxUnpool3d on multiple test cases:")
    # Iterate over each test case
    for i, (data, indices) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)
        indices_tensor = torch.tensor(indices, dtype=torch.long)

        # First, simulate the pooling operation to get realistic indices
        pooled_data, pool_indices = max_pool(data_tensor)

        # Apply MaxUnpool3d using the indices from the simulated pooling
        unpooled_data = max_unpool(pooled_data, pool_indices)
        ret.append(unpooled_data[0, 0, :10, :10].numpy())

    return ret
