# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Embedding
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated batch of indices
    # This function simulates typical scenarios for Embedding, which maps indices into embedding vectors
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Number of indices in each test case
        # Randomly generate indices based on the embedding size
        # Assume embedding size is between 1 and 100 for testing purposes
        indices = np.random.randint(0, 100, size=(N,)).astype(np.int64)
        test_cases.append(indices)
    return test_cases


# Function to test Embedding with generated test cases
def test_Embedding(test_cases, num_embeddings=100):
    import torch
    import torch.nn as nn

    # Define Embedding layer parameters
    num_embeddings = 100  # Number of unique embeddings
    embedding_dim = 50  # Dimension of each embedding vector
    padding_idx = 0
    scale_grad_by_freq = True
    norm_type = 2.0

    # Define an Embedding layer with the given parameters
    embedding_layer = nn.Embedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        padding_idx=padding_idx,
        scale_grad_by_freq=scale_grad_by_freq,
        norm_type=norm_type
    )

    ret = []

    print("Testing Embedding on multiple test cases:")
    for i, indices in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        indices_tensor = torch.from_numpy(indices)

        # Apply Embedding
        embedded_data = embedding_layer(indices_tensor)
        ret.append(embedded_data[:5].numpy())

    return ret

