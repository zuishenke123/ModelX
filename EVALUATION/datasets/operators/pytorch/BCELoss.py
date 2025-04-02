# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for BCELoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input (probabilities) and targets
    # Inputs should simulate the typical format for BCELoss, which involves probabilities of classes
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 1   # Number of outputs, typically 1 for binary classification

        # Random probabilities for each element in the batch
        # These are sigmoid outputs (values between 0 and 1)
        predictions = np.random.rand(N, C).astype(np.float32)

        # Random binary targets for each batch element
        targets = np.random.randint(0, 2, size=(N, C)).astype(np.float32)

        test_cases.append((predictions, targets))
    return test_cases


# Function to test BCELoss with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a BCELoss layer
    bce_loss = nn.BCELoss()
    ret = []
    print("Testing BCELoss on multiple test cases:")
    # Iterate over each test case
    for i, (predictions, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        predictions_tensor = torch.from_numpy(predictions)
        targets_tensor = torch.from_numpy(targets)

        # Compute BCE Loss
        loss = bce_loss(predictions_tensor, targets_tensor)
        ret.append(loss.item())

    return ret