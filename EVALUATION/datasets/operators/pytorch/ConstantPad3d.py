# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ConstantPad3d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 3-dimensional data
    # This function simulates typical scenarios for ConstantPad3d, which is used to pad 3D volumetric data with a constant value
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels
        D = 10  # Depth of the volumetric data
        H = 20  # Height
        W = 20  # Width

        # Random data for each element in the batch, representing 3D volumetric feature maps
        data = np.random.randn(N, C, D, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ConstantPad3d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ConstantPad3d layer with padding values and a constant value to pad
    # Example padding adds 1 voxel on all sides in the depth dimension, 2 voxels on all sides in the height dimension,
    # and 3 voxels on all sides in the width dimension with a constant value of 0
    constant_pad = nn.ConstantPad3d((3, 3, 2, 2, 1, 1), value=0)
    ret = []

    print("Testing ConstantPad3d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ConstantPad3d
        padded_data = constant_pad(data_tensor)
        ret.append(padded_data[0, 0, 0, :5, :5].numpy())

    return ret
