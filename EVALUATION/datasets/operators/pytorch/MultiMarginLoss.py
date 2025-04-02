# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MultiMarginLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated inputs and targets
    # Inputs simulate typical scenarios for MultiMarginLoss, which is often used in classification tasks
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5   # Number of classes

        # Random scores for each element in the batch, representing raw outputs for each class
        scores = np.random.randn(N, C).astype(np.float32)

        # Random target class indices for each batch element
        targets = np.random.randint(0, C, size=(N,)).astype(np.int64)

        test_cases.append((scores, targets))
    return test_cases


# Function to test MultiMarginLoss with generated test cases
def test_MultiMarginLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a MultiMarginLoss layer with default settings
    multi_margin_loss = nn.MultiMarginLoss()
    ret = []

    print("Testing MultiMarginLoss on multiple test cases:")
    # Iterate over each test case
    for i, (scores, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        scores_tensor = torch.from_numpy(scores)
        targets_tensor = torch.from_numpy(targets)

        # Compute MultiMargin Loss
        loss = multi_margin_loss(scores_tensor, targets_tensor)
        ret.append(loss.item())

    return ret
