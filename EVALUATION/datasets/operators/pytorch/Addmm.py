# -*- coding: utf-8 -*-
import numpy as np


def generate_test_cases():
    test_cases = [
        (np.random.rand(5, 10).astype(np.float32), np.random.rand(5, 5).astype(np.float32), np.random.rand(5, 10).astype(np.float32)),
        (np.random.rand(10, 3).astype(np.float32), np.random.rand(10, 5).astype(np.float32), np.random.rand(5, 3).astype(np.float32)),
        (np.zeros((5, 5), dtype=np.float32), np.ones((5, 5), dtype=np.float32), np.full((5, 5), 3, dtype=np.float32)),
        (np.full((3, 3), 2, dtype=np.float32), np.eye(3, dtype=np.float32), np.random.rand(3, 3).astype(np.float32)),
        (np.random.rand(1, 10).astype(np.float32), np.random.rand(5, 1).astype(np.float32), np.random.rand(1, 10).astype(np.float32)),
        (np.random.rand(5, 5).astype(np.float32), np.random.rand(5, 5).astype(np.float32), np.random.rand(5, 5).astype(np.float32)),
        (np.random.rand(2, 6).astype(np.float32), np.random.rand(2, 4).astype(np.float32), np.random.rand(4, 6).astype(np.float32)),
        (np.random.rand(6, 8).astype(np.float32), np.random.rand(6, 7).astype(np.float32), np.random.rand(7, 8).astype(np.float32)),
        (np.random.rand(8, 5).astype(np.float32), np.random.rand(8, 9).astype(np.float32), np.random.rand(9, 5).astype(np.float32))
    ]
    return test_cases


# Function to test torch.addmm with generated test cases
def test_operator(test_cases):
    import torch
    print("Testing torch.addmm on multiple test cases:")
    ret = []

    for i, (m, mat1, mat2) in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        M = torch.from_numpy(m)
        mat1 = torch.from_numpy(mat1)
        mat2 = torch.from_numpy(mat2)

        # Parameters for addmm
        beta = 0.5
        alpha = 2.0

        # Perform matrix multiplication and addition
        result = torch.addmm(M, mat1, mat2, beta=beta, alpha=alpha)
        ret.append(result.numpy())
    return ret


