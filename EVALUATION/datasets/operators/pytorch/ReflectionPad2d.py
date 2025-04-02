# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ReflectionPad2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data This function
    # simulates typical scenarios for ReflectionPad2d, which is used to pad 2D feature maps by reflecting them at the
    # borders
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels (e.g., RGB for images)
        H = 32  # Height of the feature map
        W = 32  # Width of the feature map

        # Random data for each element in the batch, representing 2D feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ReflectionPad2d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ReflectionPad2d layer with padding applied symmetrically
    # Example padding adds 3 pixels on each side and 2 pixels on the top and bottom
    reflection_pad = nn.ReflectionPad2d((3, 3, 2, 2))
    ret = []

    print("Testing ReflectionPad2d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ReflectionPad2d
        padded_data = reflection_pad(data_tensor)
        ret.append(padded_data[0, 0, :5, :5].numpy())

    return ret
