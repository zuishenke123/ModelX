# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for AlphaDropout
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input
    # AlphaDropout is often used with SELU activation functions, typically requiring normalized inputs
    test_cases = [np.random.randn(10, 3, 24, 24).astype(np.float32) for _ in range(num_cases)]  # 10 distinct sets
    return test_cases


# Function to test AlphaDropout with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define an AlphaDropout layer with a dropout probability
    alpha_dropout = nn.AlphaDropout(p=0.5)  # 50% probability to drop elements
    ret = []

    print("Testing AlphaDropout on multiple test cases:")
    # Iterate over each test case
    for i, input_data in enumerate(test_cases):
        # Convert numpy array to torch tensor
        input_tensor = torch.from_numpy(input_data)

        # Enable training mode to activate dropout behavior
        alpha_dropout.eval()

        # Apply AlphaDropout
        output_tensor = alpha_dropout(input_tensor)
        ret.append(output_tensor[0, 0, :5, :5].numpy())

    return ret
