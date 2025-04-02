# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ReflectionPad3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data This function
    # simulates typical scenarios for ReflectionPad3d, which is used to pad 3D volumetric data by reflecting them at
    # the borders
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 2   # Number of channels
        D = 10  # Depth of the volumetric data
        H = 15  # Height
        W = 15  # Width

        # Random data for each element in the batch, representing 3D volumetric feature maps
        data = np.random.randn(N, C, D, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ReflectionPad3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ReflectionPad3d layer with padding applied symmetrically around each dimension
    # Example padding adds 1 voxel on each side of the depth, 2 voxels on each side of the height,
    # and 2 voxels on each side of the width
    reflection_pad = nn.ReflectionPad3d((2, 2, 2, 2, 1, 1))
    ret = []

    print("Testing ReflectionPad3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ReflectionPad3d
        padded_data = reflection_pad(data_tensor)
        ret.append(padded_data[0, 0, 0, :5, :5].numpy())

    return ret
