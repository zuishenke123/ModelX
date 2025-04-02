# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ReplicationPad3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data This function
    # simulates typical scenarios for ReplicationPad3d, which is used to pad 3D volumetric data by replicating the
    # edge values
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels
        D = 12  # Depth of the volumetric data
        H = 12  # Height
        W = 12  # Width

        # Random data for each element in the batch, representing 3D volumetric feature maps
        data = np.random.randn(N, C, D, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ReplicationPad3d with generated test cases
def test_ReplicationPad3d(test_cases):
    import torch
    import torch.nn as nn

    # Define a ReplicationPad3d layer with padding applied to each dimension Example padding adds 1 voxel on each
    # side in the depth, 2 voxels on each side in the height, and 3 voxels on each side in the width
    replication_pad = nn.ReplicationPad3d((3, 3, 2, 2, 1, 1))
    ret = []

    print("Testing ReplicationPad3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ReplicationPad3d
        padded_data = replication_pad(data_tensor)
        ret.append(padded_data[0, 0, padded_data.shape[2]//2, :5, :5].numpy())

    return ret
