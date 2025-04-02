# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for TripletMarginLoss
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated anchor, positive, and negative samples This
    # function simulates typical scenarios for TripletMarginLoss, which is used to learn embeddings by ensuring that
    # an anchor is closer to a positive sample than to a negative sample by a margin
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        D = 128  # Dimension of embeddings

        # Random embeddings for anchor, positive, and negative samples
        anchors = np.random.randn(N, D).astype(np.float32)
        positives = np.random.randn(N, D).astype(np.float32)
        negatives = np.random.randn(N, D).astype(np.float32)

        # Normalize embeddings to prevent overly large values which can destabilize learning
        anchors /= np.linalg.norm(anchors, axis=1, keepdims=True)
        positives /= np.linalg.norm(positives, axis=1, keepdims=True)
        negatives /= np.linalg.norm(negatives, axis=1, keepdims=True)

        test_cases.append((anchors, positives, negatives))
    return test_cases


# Function to test TripletMarginLoss with generated test cases
def test_TripletMarginLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a TripletMarginLoss layer with a margin
    triplet_margin_loss = nn.TripletMarginLoss(margin=1.0)
    ret = []

    print("Testing TripletMarginLoss on multiple test cases:")
    # Iterate over each test case
    for i, (anchors, positives, negatives) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        anchors_tensor = torch.from_numpy(anchors)
        positives_tensor = torch.from_numpy(positives)
        negatives_tensor = torch.from_numpy(negatives)

        # Compute Triplet Margin Loss
        loss = triplet_margin_loss(anchors_tensor, positives_tensor, negatives_tensor)
        ret.append(loss.item())

    return ret