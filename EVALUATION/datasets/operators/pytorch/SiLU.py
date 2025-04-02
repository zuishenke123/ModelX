# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for SiLU (Sigmoid Linear Unit, also known as Swish)
def generate_test_cases():
    test_cases = []
    for _ in range(10):
        # Randomly generate input data with shape (5, 4) and round to 4 decimal places
        input_array = np.random.randn(5, 4).round(4)
        test_cases.append(input_array)
    return test_cases


# Function to test SiLU with generated test cases
def test_SiLU(test_cases):
    import torch
    import torch.nn as nn

    # Define a SiLU layer
    silu = nn.SiLU()
    ret = []
    for i, input_array in enumerate(test_cases):
        # Convert numpy array to torch tensor
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # Compute SiLU
        output_tensor = silu(input_tensor)
        ret.append(output_tensor.detach().numpy())

    return ret
