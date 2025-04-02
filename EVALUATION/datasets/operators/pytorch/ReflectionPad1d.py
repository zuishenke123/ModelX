# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ReflectionPad1d
def generate_test_cases_ReflectionPad1d(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data This function
    # simulates typical scenarios for ReflectionPad1d, which is used to pad sequences by reflecting them at the borders
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 4   # Number of channels
        L = 30  # Length of each sequence

        # Random data for each element in the batch, representing 1D sequences across multiple channels
        data = np.random.randn(N, C, L).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ReflectionPad1d with generated test cases
def test_ReflectionPad1d(test_cases):
    import torch
    import torch.nn as nn

    # Define a ReflectionPad1d layer with padding applied to both ends of the sequence
    # Example padding adds 3 elements on the left and 2 elements on the right
    reflection_pad = nn.ReflectionPad1d((3, 2))
    ret = []
    print("Testing ReflectionPad1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ReflectionPad1d
        padded_data = reflection_pad(data_tensor)
        ret.append(padded_data[0, 0, :10].numpy())

    return ret
