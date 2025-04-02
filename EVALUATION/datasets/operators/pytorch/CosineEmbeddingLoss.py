# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for CosineEmbeddingLoss
def generate_test_cases_CosineEmbeddingLoss(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated inputs and a label indicating if the
    # embeddings should be similar (1) or dissimilar (-1)
    test_cases = []
    for _ in range(num_cases):
        N = 10  # Batch size
        D = 100  # Dimension of embeddings, typically large for high-dimensional vector comparisons

        # Random embeddings for two sets of inputs
        embeddings1 = np.random.randn(N, D).astype(np.float32)
        embeddings2 = np.random.randn(N, D).astype(np.float32)

        # Normalize embeddings to ensure they are on the unit sphere, as cosine similarity expects
        embeddings1 /= np.linalg.norm(embeddings1, axis=1, keepdims=True)
        embeddings2 /= np.linalg.norm(embeddings2, axis=1, keepdims=True)

        # Random labels indicating whether pairs should be similar (1) or dissimilar (-1)
        labels = np.random.choice([-1, 1], size=(N,)).astype(np.float32)

        test_cases.append((embeddings1, embeddings2, labels))
    return test_cases


# Function to test CosineEmbeddingLoss with generated test cases
def test_CosineEmbeddingLoss(test_cases):
    import torch
    import torch.nn as nn

    # Define a CosineEmbeddingLoss layer
    cosine_embedding_loss = nn.CosineEmbeddingLoss()
    ret = []
    print("Testing CosineEmbeddingLoss on multiple test cases:")
    # Iterate over each test case
    for i, (embeddings1, embeddings2, labels) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        embeddings1_tensor = torch.from_numpy(embeddings1)
        embeddings2_tensor = torch.from_numpy(embeddings2)
        labels_tensor = torch.from_numpy(labels)

        # Compute Cosine Embedding Loss
        loss = cosine_embedding_loss(embeddings1_tensor, embeddings2_tensor, labels_tensor)
        ret.append(loss.item())

    return ret
