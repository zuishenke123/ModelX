# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ZeroPad2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data
    # This function simulates typical scenarios for ZeroPad2d, which is used to pad the edges of 2D feature maps with zeros
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels (e.g., RGB)
        H = 28  # Height of the feature map
        W = 28  # Width of the feature map

        # Random data for each element in the batch, representing feature maps that might be images
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ZeroPad2d with generated test cases
def test_ZeroPad2d(test_cases):
    import torch
    import torch.nn as nn

    # Define a ZeroPad2d layer with padding applied to the left, right, top, and bottom
    # Example padding adds 1 pixel on the left and right, and 2 pixels on the top and bottom
    zero_pad = nn.ZeroPad2d((1, 1, 2, 2))
    ret = []

    print("Testing ZeroPad2d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ZeroPad2d
        padded_data = zero_pad(data_tensor)
        ret.append(padded_data[0, 0, :5, :5].numpy())

    return ret
