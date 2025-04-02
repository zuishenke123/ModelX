# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for PixelUnshuffle
def generate_test_cases_PixelUnshuffle(num_cases=10, downscale_factor=2):
    # Generate multiple test cases, each with its own randomly generated batch of multi-channel data
    # This function simulates typical scenarios for PixelUnshuffle, which is used to downscale the spatial dimensions
    # by rearranging elements from the spatial dimensions into the channel dimension
    test_cases = []
    for _ in range(num_cases):
        N = 5  # Batch size
        H = 16  # Height of the feature map
        W = 16  # Width of the feature map
        C = 4   # Number of channels, which must be divisible by (downscale_factor^2)

        # Random data for each element in the batch, representing multi-channel feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test PixelUnshuffle with generated test cases
def test_PixelUnshuffle(test_cases, downscale_factor=2):
    import torch
    import torch.nn as nn

    # Define a PixelUnshuffle layer
    pixel_unshuffle = nn.PixelUnshuffle(downscale_factor)
    ret = []

    print("Testing PixelUnshuffle on multiple test cases:")
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply PixelUnshuffle
        unshuffled_data = pixel_unshuffle(data_tensor)
        ret.append(unshuffled_data[0, :4, 0, 0].numpy())

    return ret
