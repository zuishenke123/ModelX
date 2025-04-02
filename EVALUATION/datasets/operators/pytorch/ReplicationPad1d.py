# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ReplicationPad1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data This function
    # simulates typical scenarios for ReplicationPad1d, which is used to pad sequences by replicating the edge values
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 3   # Number of channels
        L = 40  # Length of each sequence

        # Random data for each element in the batch, representing 1D sequences
        data = np.random.randn(N, C, L).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ReplicationPad1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ReplicationPad1d layer with padding applied to both ends of the sequence
    # Example padding adds 4 elements on the left and 2 elements on the right
    replication_pad = nn.ReplicationPad1d((4, 2))
    ret = []

    print("Testing ReplicationPad1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ReplicationPad1d
        padded_data = replication_pad(data_tensor)
        ret.append(padded_data[0, 0, :10].numpy())

    return ret
