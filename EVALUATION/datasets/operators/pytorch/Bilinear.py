# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Bilinear
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with two sets of randomly generated input
    # Inputs are in the shape typical for Bilinear layers, for example (batch_size, features)
    test_cases = [(np.random.randn(10, 20).astype(np.float32),  # 10 instances, 20 features each for input1
                   np.random.randn(10, 30).astype(np.float32)) # 10 instances, 30 features each for input2
                  for _ in range(num_cases)]
    return test_cases


# Function to test Bilinear with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a Bilinear layer
    # Assuming in1_features=20, in2_features=30, and transforming it to an output feature of 40
    bilinear = nn.Bilinear(in1_features=20, in2_features=30, out_features=40)
    ret = []

    print("Testing Bilinear layer on multiple test cases:")
    # Iterate over each test case
    for i, (input1_data, input2_data) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        input1_tensor = torch.from_numpy(input1_data)
        input2_tensor = torch.from_numpy(input2_data)

        # Apply Bilinear transformation
        output_tensor = bilinear(input1_tensor, input2_tensor)
        ret.append(output_tensor[0].detach().numpy())

    return ret




