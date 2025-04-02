# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for PixelShuffle
def generate_test_cases(num_cases=10, upscale_factor=4):
    # Generate multiple test cases, each with its own randomly generated batch of multi-channel data
    # This function simulates typical scenarios for PixelShuffle, which is used to upscale the spatial dimensions
    test_cases = []
    for _ in range(num_cases):
        N = 5  # Batch size
        H = 8   # Height of the feature map
        W = 8   # Width of the feature map
        C = upscale_factor ** 2 * 4  # Number of channels must be a multiple of (upscale_factor^2), here using 4
        # times for example

        # Random data for each element in the batch, representing multi-channel feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test PixelShuffle with generated test cases
def test_PixelShuffle(test_cases, upscale_factor=4):
    import torch
    import torch.nn as nn

    # Define a PixelShuffle layer
    pixel_shuffle = nn.PixelShuffle(upscale_factor=4)
    ret = []

    print("Testing PixelShuffle on multiple test cases:")
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply PixelShuffle
        shuffled_data = pixel_shuffle(data_tensor)
        ret.append(shuffled_data[0, 0, :upscale_factor, :upscale_factor].numpy())

    return ret
