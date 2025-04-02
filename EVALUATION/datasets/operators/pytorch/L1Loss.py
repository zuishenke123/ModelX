# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for L1Loss
def generate_test_cases_L1Loss(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input and target
    # Inputs and targets are in shapes typical for loss functions, e.g., (batch_size, features)
    test_cases = [
        (np.random.randn(10, 20).astype(np.float32),  # 10 instances, 20 features each for input
         np.random.randn(10, 20).astype(np.float32)) # 10 instances, 20 features each for target
        for _ in range(num_cases)
    ]
    return test_cases


# Function to test L1Loss with generated test cases
def test_L1Loss(test_cases):
    import torch
    import torch.nn as nn

    # Define an L1Loss layer
    l1_loss = nn.L1Loss()

    print("Testing L1Loss on multiple test cases:")
    # Iterate over each test case
    for i, (input_data, target_data) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        input_tensor = torch.from_numpy(input_data)
        target_tensor = torch.from_numpy(target_data)

        # Compute L1 Loss
        loss = l1_loss(input_tensor, target_tensor)