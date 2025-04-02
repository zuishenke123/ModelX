# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Linear
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input
    # Inputs are in the shape typical for Linear layers, for example (batch_size, features)
    test_cases = [np.random.randn(10, 20).astype(np.float32) for _ in
                  range(num_cases)]  # 10 batches of 20 features each
    return test_cases


# Function to test Linear with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a Linear layer
    # Assuming an input feature of 20 and transforming it to an output feature of 10
    linear = nn.Linear(in_features=20, out_features=10)
    ret = []

    # Iterate over each test case
    for i, input_data in enumerate(test_cases):
        # Convert numpy array to torch tensor
        input_tensor = torch.from_numpy(input_data)

        # Apply Linear transformation
        output_tensor = linear(input_tensor)
        ret.append(output_tensor[0].numpy())

    return ret
