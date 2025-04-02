# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for KLDivLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input (log probabilities) and target
    # probabilities Inputs and targets simulate typical scenarios where KLDivLoss is used, e.g., comparing two
    # probability distributions
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        C = 5   # Number of classes or outputs

        # Random log probabilities for input (must be log probabilities for KLDivLoss)
        log_probs = np.random.rand(N, C).astype(np.float32)
        log_probs = np.log(log_probs / np.sum(log_probs, axis=1, keepdims=True))  # Normalize and log

        # Random probabilities for target, ensuring they sum to 1
        target_probs = np.random.rand(N, C).astype(np.float32)
        target_probs = target_probs / np.sum(target_probs, axis=1, keepdims=True)

        test_cases.append((log_probs, target_probs))
    return test_cases


# Function to test KLDivLoss with generated test cases
def test_KLDivLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a KLDivLoss layer
    kl_div_loss = nn.KLDivLoss(reduction='batchmean')
    ret = []

    print("Testing KLDivLoss on multiple test cases:")
    # Iterate over each test case
    for i, (log_probs, target_probs) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        log_probs_tensor = torch.from_numpy(log_probs)
        target_probs_tensor = torch.from_numpy(target_probs)

        # Compute KLDiv Loss
        loss = kl_div_loss(log_probs_tensor, target_probs_tensor)
        ret.append(loss.item())

    return ret

