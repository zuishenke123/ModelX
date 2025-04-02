# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Identity
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with a randomly generated input
    # Inputs are in the shape typical for layers that might use Identity, for example (batch_size, features)
    test_cases = [np.random.randn(10, 20).astype(np.float32) for _ in range(num_cases)]  # 10 batches of 20 features each
    return test_cases


# Function to test Identity with generated test cases
def test_Identity(test_cases):
    import torch
    import torch.nn as nn

    # Define an Identity layer
    identity = nn.Identity()
    ret = []

    # Iterate over each test case
    for i, input_data in enumerate(test_cases):
        # Convert numpy array to torch tensor
        input_tensor = torch.from_numpy(input_data)

        # Apply Identity transformation
        output_tensor = identity(input_tensor)
        ret.append(output_tensor[0].numpy())

    return ret
