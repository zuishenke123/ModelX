# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for GaussianNLLLoss
def generate_test_cases_GaussianNLLLoss(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input (mean predictions), targets,
    # and variances GaussianNLLLoss is typically used for regression tasks where predictions include both mean and
    # variance of the predicted values
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 1   # Number of outputs, typically 1 for regression predictions

        # Random predictions for the mean of each element in the batch
        mean_preds = np.random.randn(N, C).astype(np.float32)

        # Random positive values for variance of each prediction
        variances = np.abs(np.random.randn(N, C).astype(np.float32)) + 0.1  # Ensure non-zero variance

        # Random target values, which could be considered as drawn from a Gaussian distribution centered at mean_preds
        targets = mean_preds + np.random.randn(N, C).astype(np.float32) * np.sqrt(variances)

        test_cases.append((mean_preds, variances, targets))
    return test_cases


# Function to test GaussianNLLLoss with generated test cases
def test_GaussianNLLLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a GaussianNLLLoss layer
    gaussian_nll_loss = nn.GaussianNLLLoss()
    ret = []

    print("Testing GaussianNLLLoss on multiple test cases:")
    # Iterate over each test case
    for i, (mean_preds, variances, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        mean_preds_tensor = torch.from_numpy(mean_preds)
        variances_tensor = torch.from_numpy(variances)
        targets_tensor = torch.from_numpy(targets)

        # Compute Gaussian NLL Loss
        loss = gaussian_nll_loss(mean_preds_tensor, targets_tensor, variances_tensor)
        ret.append(loss.item())

    return ret

