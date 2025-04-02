# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for SoftMarginLoss
def generate_test_cases_SoftMarginLoss(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated inputs and binary labels Inputs simulate
    # typical scenarios for SoftMarginLoss, which is often used in binary classification tasks with labels -1 or 1
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        D = 1  # Dimension of outputs, typically 1 for binary classification tasks

        # Random predictions for each element in the batch
        predictions = np.random.randn(N, D).astype(np.float32)

        # Random binary labels for each batch element, where labels are -1 or 1
        labels = np.random.choice([-1, 1], size=(N, D)).astype(np.float32)

        test_cases.append((predictions, labels))
    return test_cases


# Function to test SoftMarginLoss with generated test cases
def test_SoftMarginLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a SoftMarginLoss layer
    soft_margin_loss = nn.SoftMarginLoss()
    ret = []

    print("Testing SoftMarginLoss on multiple test cases:")
    # Iterate over each test case
    for i, (predictions, labels) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        predictions_tensor = torch.from_numpy(predictions)
        labels_tensor = torch.from_numpy(labels)

        # Compute SoftMargin Loss
        loss = soft_margin_loss(predictions_tensor, labels_tensor)
        ret.append(loss.item())

    return ret
