# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    # Create several numpy arrays
    array1 = np.random.randn(5, 5).astype(np.float32)
    array2 = np.random.randn(5).astype(np.float32)
    array3 = np.random.randn(10).astype(np.float32)

    # Combine arrays into a list
    arrays = [array1, array2, array3]
    return arrays

# Function to test ParameterList with generated test cases
def test_operator(arrays):
    import torch
    import torch.nn as nn
    # Convert numpy arrays to torch tensors and then to parameters
    param_list = nn.ParameterList([nn.Parameter(torch.from_numpy(array)) for array in arrays])
    ret = []

    print("Testing ParameterList with various parameters:")
    # Iterate through each parameter in the ParameterList
    for i, param in enumerate(param_list):
        ret.append(param.detach().numpy())
    return ret