# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ChannelShuffle
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of multi-channel data This function
    # simulates typical scenarios for ChannelShuffle, which is used to permute the order of channels in the input tensor
    test_cases = []
    for _ in range(num_cases):
        N = 5  # Batch size
        C = 16  # Number of channels, chosen to be a multiple of g (groups) for shuffling
        H = 32  # Height of the feature map
        W = 32  # Width of the feature map

        # Random data for each element in the batch, representing multi-channel feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ChannelShuffle with generated test cases
def test_operator(test_cases, groups=4):
    import torch
    import torch.nn as nn

    # Define a ChannelShuffle layer
    channel_shuffle = nn.ChannelShuffle(groups)
    ret = []

    print("Testing ChannelShuffle on multiple test cases:")
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ChannelShuffle
        shuffled_data = channel_shuffle(data_tensor)
        ret.append(shuffled_data[0, :4, 0, 0].numpy())

    return ret

