# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for UpsamplingBilinear2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of multi-channel data
    # This function simulates typical scenarios for UpsamplingBilinear2d, which is used to increase
    # the spatial dimensions of the input tensor using bilinear interpolation
    test_cases = []
    for _ in range(num_cases):
        N = 5  # Batch size
        C = 3   # Number of channels
        H = 8   # Original height of the feature map
        W = 8   # Original width of the feature map

        # Random data for each element in the batch, representing multi-channel feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test UpsamplingBilinear2d with generated test cases
def test_operator(test_cases, scale_factor):
    import torch
    import torch.nn as nn

    # Define an UpsamplingBilinear2d layer with a given scale factor
    upsampling_bilinear = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
    ret = []

    print("Testing UpsamplingBilinear2d on multiple test cases:")
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply UpsamplingBilinear2d
        upsampled_data = upsampling_bilinear(data_tensor)
        ret.append(upsampled_data[0, 0, :2, :2].numpy())

    return ret
