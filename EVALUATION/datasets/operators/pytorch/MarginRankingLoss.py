# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for MarginRankingLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated inputs and targets
    # Inputs simulate typical scenarios for MarginRankingLoss, involving pairs of items and a binary target indicating which item should rank higher
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size

        # Random values for two sets of predictions
        x1 = np.random.randn(N, 1).astype(np.float32)
        x2 = np.random.randn(N, 1).astype(np.float32)

        # Binary targets indicating whether x1 should be ranked higher than x2
        # 1 indicates x1 should be higher, -1 indicates x2 should be higher
        y = np.random.choice([-1, 1], size=(N, 1)).astype(np.float32)

        test_cases.append((x1, x2, y))
    return test_cases


# Function to test MarginRankingLoss with generated test cases
def test_operator(test_cases):
    import torch
    import torch.nn as nn

    # Define a MarginRankingLoss layer with a margin
    margin_ranking_loss = nn.MarginRankingLoss(margin=1.0)
    ret =[]

    print("Testing MarginRankingLoss on multiple test cases:")
    # Iterate over each test case
    for i, (x1, x2, y) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        x1_tensor = torch.from_numpy(x1)
        x2_tensor = torch.from_numpy(x2)
        y_tensor = torch.from_numpy(y)

        # Compute MarginRanking Loss
        loss = margin_ranking_loss(x1_tensor, x2_tensor, y_tensor)
        ret.append(loss.item())
        
    return ret
