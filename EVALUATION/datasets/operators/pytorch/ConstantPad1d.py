# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for ConstantPad1d
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of 1-dimensional data This function
    # simulates typical scenarios for ConstantPad1d, which is used to pad 1D sequences with a constant value
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5   # Number of channels
        L = 50  # Length of each sequence

        # Random data for each element in the batch, representing 1D sequences
        data = np.random.randn(N, C, L).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test ConstantPad1d with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a ConstantPad1d layer with padding values and a constant value to pad
    # Example padding adds 3 elements on the left and 2 elements on the right with a constant value of -1
    constant_pad = nn.ConstantPad1d((3, 2), value=-1.0)
    ret = []

    print("Testing ConstantPad1d on multiple test cases:")
    # Iterate over each test case
    for i, data in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        data_tensor = torch.from_numpy(data)

        # Apply ConstantPad1d
        padded_data = constant_pad(data_tensor)
        ret.append(padded_data[0, 0, :10].numpy())

    return ret

