# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for GLU
def generate_test_cases():
    test_cases = []
    for _ in range(10):
        # Randomly generate input data with shape (5, 8) and round to 4 decimal places
        # The input size should be divisible by 2 along the dimension specified in the GLU
        input_array = np.random.randn(5, 8).round(4)
        test_cases.append(input_array)
    return test_cases


# Function to test GLU with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a GLU layer
    glu = nn.GLU(dim=1)  # GLU has a dim parameter, default is -1
    ret = []

    for i, input_array in enumerate(test_cases):
        # Convert numpy array to torch tensor
        input_tensor = torch.tensor(input_array, dtype=torch.float32)

        # Compute GLU
        output_tensor = glu(input_tensor)
        ret.append(output_tensor.detach().numpy())
    return ret