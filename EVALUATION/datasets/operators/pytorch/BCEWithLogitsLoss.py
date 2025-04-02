# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for BCEWithLogitsLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input (logits) and targets Inputs should
    # simulate the typical format for BCEWithLogitsLoss, which involves logits before the sigmoid activation
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 1   # Number of outputs, typically 1 for binary classification

        # Random logits for each element in the batch, can range widely as these are pre-sigmoid outputs
        logits = np.random.randn(N, C).astype(np.float32)

        # Random binary targets for each batch element
        targets = np.random.randint(0, 2, size=(N, C)).astype(np.float32)

        test_cases.append((logits, targets))
    return test_cases


# Function to test BCEWithLogitsLoss with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a BCEWithLogitsLoss layer
    bce_with_logits_loss = nn.BCEWithLogitsLoss()
    ret = []

    print("Testing BCEWithLogitsLoss on multiple test cases:")
    # Iterate over each test case
    for i, (logits, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        logits_tensor = torch.from_numpy(logits)
        targets_tensor = torch.from_numpy(targets)

        # Compute BCEWithLogits Loss
        loss = bce_with_logits_loss(logits_tensor, targets_tensor)
        ret.append(loss.item())

    return ret
