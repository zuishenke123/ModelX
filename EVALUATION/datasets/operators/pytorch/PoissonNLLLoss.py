# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for PoissonNLLLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input (predictions) and targets
    # Inputs and targets simulate rates for the Poisson distribution, typically non-negative
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 1  # Number of outputs, typically 1 for rate predictions

        # Random positive predictions for each element in the batch (rate of the Poisson distribution)
        rate_preds = np.random.rand(N, C).astype(np.float32) * 10  # Scale to get a broader range of rates

        # Random target values from a Poisson distribution based on the predicted rates
        targets = np.random.poisson(rate_preds).astype(np.float32)

        test_cases.append((rate_preds, targets))
    return test_cases


# Function to test PoissonNLLLoss with generated test cases
def test_PoissonNLLLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a PoissonNLLLoss layer
    poisson_nll_loss = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
    ret = []
    # Iterate over each test case
    for i, (rate_preds, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        rate_preds_tensor = torch.from_numpy(rate_preds)
        targets_tensor = torch.from_numpy(targets)

        # Compute Poisson NLL Loss
        loss = poisson_nll_loss(rate_preds_tensor, targets_tensor)
        ret.append(loss.item())

    return ret
