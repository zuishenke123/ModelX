# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ReplicationPad2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data This function
    # simulates typical scenarios for ReplicationPad2d, which is used to pad 2D feature maps by replicating the edge
    # values
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 4   # Number of channels
        H = 20  # Height of the feature map
        W = 20  # Width of the feature map

        # Random data for each element in the batch, representing 2D feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ReplicationPad2d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ReplicationPad2d layer with padding applied to both horizontal and vertical edges
    # Example padding adds 2 pixels on the left and right, and 3 pixels on the top and bottom
    replication_pad = nn.ReplicationPad2d((2, 2, 3, 3))
    ret = []
    print("Testing ReplicationPad2d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ReplicationPad2d
        padded_data = replication_pad(data_tensor)
        ret.append(padded_data[0, 0, :5, :5].numpy())

    return ret
