# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for BatchNorm2d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 2-dimensional data This function
    # simulates typical scenarios for BatchNorm2d, which is used to normalize feature maps in convolutional neural
    # networks
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 20  # Number of channels
        H = 28  # Height of the feature map
        W = 28  # Width of the feature map

        # Random data for each element in the batch, simulating feature maps from a convolutional layer
        data = np.random.randn(N, C, H, W).astype(
            np.float32) * 10 + 5  # Generate data with non-zero mean and larger variance

        test_cases.append(data)
    return test_cases


# Function to test BatchNorm2d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a BatchNorm2d layer with the number of features (channels) set to match the test data
    batch_norm = nn.BatchNorm2d(momentum=0.9, eps=1e-05, num_features=20)
    ret = []
    print("Testing BatchNorm2d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Enable training mode to update running estimates of mean and variance
        batch_norm.eval()

        # Apply BatchNorm2d
        normalized_data = batch_norm(data_tensor)
        ret.append(torch.mean(normalized_data, dim=(0, 2, 3)).detach().numpy())

    return ret
