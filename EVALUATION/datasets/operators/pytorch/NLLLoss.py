# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for NLLLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input (log probabilities) and targets
    # Inputs should simulate the typical format for NLLLoss, which involves log probabilities of classes
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5  # Number of classes

        # Random log probabilities for each element in the batch
        log_probs = np.random.rand(N, C).astype(np.float32)
        log_probs = np.log(log_probs / log_probs.sum(axis=1, keepdims=True))  # Normalize to get valid log probabilities

        targets = np.random.randint(0, C, size=(N,)).astype(
            np.int64)  # Random target class indices for each batch element

        test_cases.append((log_probs, targets))
    return test_cases


# Function to test NLLLoss with generated test cases
def test_NLLLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define an NLLLoss layer
    nll_loss = nn.NLLLoss()
    ret = []
    print("Testing NLLLoss on multiple test cases:")
    # Iterate over each test case
    for i, (log_probs, targets) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        log_probs_tensor = torch.from_numpy(log_probs)
        targets_tensor = torch.from_numpy(targets)

        # Compute NLLLoss
        loss = nll_loss(log_probs_tensor, targets_tensor)
        ret.append(loss.item())

    return ret
