# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MSELoss
def generate_test_cases_MSELoss(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input and target
    # Inputs and targets are in shapes typical for loss functions, e.g., (batch_size, features)
    test_cases = [
        (np.random.randn(10, 20).astype(np.float32),  # 10 instances, 20 features each for input
         np.random.randn(10, 20).astype(np.float32)) # 10 instances, 20 features each for target
        for _ in range(num_cases)
    ]
    return test_cases


# Function to test MSELoss with generated test cases
def test_MSELoss(test_cases):
    import torch
    import torch.nn as nn

    # Define an MSELoss layer
    mse_loss = nn.MSELoss()
    ret = []
    print("Testing MSELoss on multiple test cases:")
    # Iterate over each test case
    for i, (input_data, target_data) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        input_tensor = torch.from_numpy(input_data)
        target_tensor = torch.from_numpy(target_data)

        # Compute MSE Loss
        loss = mse_loss(input_tensor, target_tensor)
        ret.append(loss.item())

    return ret
