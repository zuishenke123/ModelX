# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ConstantPad2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data This function
    # simulates typical scenarios for ConstantPad2d, which is used to pad 2D feature maps with a constant value
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels (e.g., RGB for images)
        H = 24  # Height of the feature map
        W = 24  # Width of the feature map

        # Random data for each element in the batch, representing 2D feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ConstantPad2d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ConstantPad2d layer with padding values and a constant value to pad Example padding adds 2 pixels on
    # the left, 2 pixels on the right, 3 pixels on the top, and 1 pixel on the bottom with a constant value of -1
    constant_pad = nn.ConstantPad2d((2, 2, 3, 1), value=-1.0)
    ret = []

    print("Testing ConstantPad2d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ConstantPad2d
        padded_data = constant_pad(data_tensor)
        ret.append(padded_data[0, 0, :5, :5].numpy())

    return ret
