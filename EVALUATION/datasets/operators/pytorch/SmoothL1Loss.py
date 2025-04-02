# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for SmoothL1Loss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated inputs and targets Inputs and targets
    # simulate typical scenarios for SmoothL1Loss, which is often used in regression tasks, especially in areas like
    # computer vision for bounding box regression
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        D = 4   # Dimension of outputs, e.g., coordinates of a bounding box

        # Random predictions and targets for each element in the batch
        predictions = np.random.randn(N, D).astype(np.float32)
        targets = predictions + np.random.randn(N, D).astype(np.float32) * 0.1  # Small noise around the predictions

        test_cases.append((predictions, targets))
    return test_cases


# Function to test SmoothL1Loss with generated test cases
def test_SmoothL1Loss(test_cases):
    import torch
    import torch.nn as nn

    # Define a SmoothL1Loss layer
    smooth_l1_loss = nn.SmoothL1Loss()
    ret = []

    print("Testing SmoothL1Loss on multiple test cases:")
    # Iterate over each test case
    for i, (predictions, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        predictions_tensor = torch.from_numpy(predictions)
        targets_tensor = torch.from_numpy(targets)

        # Compute SmoothL1 Loss
        loss = smooth_l1_loss(predictions_tensor, targets_tensor)
        ret.append(loss.item())

    return ret