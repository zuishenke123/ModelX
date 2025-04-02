# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Transformer
def generate_test_cases(num_cases=10):
    # Generate multiple test cases, each with its own randomly generated data for src (source) and tgt (target)
    test_cases = []
    for _ in range(num_cases):
        N = 5  # Batch size
        T = 10  # Target sequence length
        S = 12  # Source sequence length
        E = 512  # Feature dimension
        # Generating data for the source and target sequences using numpy
        src = np.random.randn(S, N, E).astype(np.float32)
        tgt = np.random.randn(T, N, E).astype(np.float32)
        # Generate random boolean masks using numpy
        src_mask = np.tril(np.ones((S, S))).astype(np.bool_)
        tgt_mask = np.tril(np.ones((T, T))).astype(np.bool_)
        src_key_padding_mask = np.random.randint(0, 2, (N, S)).astype(np.bool_)
        tgt_key_padding_mask = np.random.randint(0, 2, (N, T)).astype(np.bool_)

        test_cases.append((src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask))
    return test_cases


# Function to test Transformer with generated test cases
def test_Transformer(test_cases):
    import torch
    import torch.nn as nn

    # Define a Transformer model
    transformer_model = nn.Transformer(d_model=512, nhead=8, num_encoder_layers=3, num_decoder_layers=3)
    ret = []

    print("Testing Transformer on multiple test cases:")
    for i, (src, tgt, src_mask, tgt_mask, src_key_padding_mask, tgt_key_padding_mask) in enumerate(test_cases):
        # Convert numpy arrays to torch tensors
        src_tensor = torch.from_numpy(src)
        tgt_tensor = torch.from_numpy(tgt)
        src_mask_tensor = torch.from_numpy(src_mask)
        tgt_mask_tensor = torch.from_numpy(tgt_mask)
        src_key_padding_mask_tensor = torch.from_numpy(src_key_padding_mask)
        tgt_key_padding_mask_tensor = torch.from_numpy(tgt_key_padding_mask)

        # Apply Transformer
        output = transformer_model(
            src_tensor, tgt_tensor,
            src_mask=src_mask_tensor, tgt_mask=tgt_mask_tensor,
            src_key_padding_mask=src_key_padding_mask_tensor, tgt_key_padding_mask=tgt_key_padding_mask_tensor
        )
        ret.append(output.detach().numpy())

    return ret

