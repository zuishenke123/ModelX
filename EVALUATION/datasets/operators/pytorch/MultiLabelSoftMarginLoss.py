# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MultiLabelSoftMarginLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated inputs (logits) and multi-label targets
    # Inputs should simulate typical scenarios for MultiLabelSoftMarginLoss, involving logits before the sigmoid activation for multi-label tasks
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5   # Number of classes or labels

        # Random logits for each element in the batch, representing raw outputs for each class
        logits = np.random.randn(N, C).astype(np.float32)

        # Random binary targets for each batch element and each class
        targets = np.random.randint(0, 2, size=(N, C)).astype(np.float32)

        test_cases.append((logits, targets))
    return test_cases


# Function to test MultiLabelSoftMarginLoss with generated test cases
def test_MultiLabelSoftMarginLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a MultiLabelSoftMarginLoss layer
    multi_label_soft_margin_loss = nn.MultiLabelSoftMarginLoss()
    ret = []

    print("Testing MultiLabelSoftMarginLoss on multiple test cases:")
    # Iterate over each test case
    for i, (logits, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        logits_tensor = torch.from_numpy(logits)
        targets_tensor = torch.from_numpy(targets)

        # Compute MultiLabelSoftMargin Loss
        loss = multi_label_soft_margin_loss(logits_tensor, targets_tensor)
        ret.append(loss.item())

    return ret
