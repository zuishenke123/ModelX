# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for CTCLoss
def generate_test_cases_CTCLoss(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated input, target, input_lengths,
    # and target_lengths Inputs should simulate the typical format for CTC, which involves log probabilities of
    # sequences
    test_cases = []
    for _ in range(num_cases):
        T = 50  # Length of the sequence
        C = 20  # Number of classes (including the blank label for CTC)
        N = 16  # Batch size

        # Random log probabilities for each element in the sequence
        log_probs = np.log(np.random.dirichlet(np.ones(C), T).astype(np.float32).transpose(1, 0))
        log_probs = np.expand_dims(log_probs, 1).repeat(N, axis=1)  # Replicate for each batch

        targets = np.random.randint(1, C, size=(N, 10), dtype=np.int32)  # Random target sequences
        input_lengths = np.full((N,), T, dtype=np.int32)  # Length of the input sequence
        target_lengths = np.random.randint(1, 10, size=(N,), dtype=np.int32)  # Length of the target sequence

        test_cases.append((log_probs, targets, input_lengths, target_lengths))
    return test_cases


# Function to test CTCLoss with generated test cases
def test_CTCLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a CTCLoss layer, assume zero-infinity loss to handle potential infs or nans
    ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)
    ret = []

    # Iterate over each test case
    for i, (log_probs, targets, input_lengths, target_lengths) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        log_probs_tensor = torch.from_numpy(log_probs).log_softmax(2).detach().requires_grad_()
        targets_tensor = torch.from_numpy(targets)
        input_lengths_tensor = torch.from_numpy(input_lengths)
        target_lengths_tensor = torch.from_numpy(target_lengths)

        # Compute CTCLoss
        loss = ctc_loss(log_probs_tensor, targets_tensor, input_lengths_tensor, target_lengths_tensor)
        ret.append(loss.item())

    return ret
