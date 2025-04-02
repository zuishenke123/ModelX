# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for Upsample
def generate_test_cases_Upsample(num_cases=10, scale_factor=2):
    # Generate multiple test cases, each with its own randomly generated batch of multi-channel data
    # This function simulates typical scenarios for Upsample, which is used to increase the spatial dimensions
    # of the input tensor using different modes of interpolation
    test_cases = []
    for _ in range(num_cases):
        N = 5  # Batch size
        C = 3   # Number of channels
        H = 8   # Height of the feature map
        W = 8   # Width of the feature map

        # Random data for each element in the batch, representing multi-channel feature maps
        data = np.random.randn(N, C, H, W).astype(np.float32)

        test_cases.append(data)
    return test_cases


# Function to test Upsample with generated test cases
def test_Upsample(test_cases, scale_factor=2):
    import torch
    import torch.nn as nn

    # Example testing different modes of upsample
    modes = ['nearest', 'linear', 'bilinear', 'bicubic', 'trilinear']
    results = {}

    print("Testing Upsample on multiple test cases with different modes:")
    ret = []
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        for i, data in enumerate(test_cases):

            # Convert numpy arrays to torch tensors
            data_tensor = torch.from_numpy(data)

            # Define Upsample using a given mode and scale factor
            if 'linear' in mode:  # Adjust dimensionality for 'linear' modes
                if data_tensor.dim() == 4 and mode == 'linear':
                    continue  # Skip as linear is meant for 3D tensors
                upsample = nn.Upsample(scale_factor=scale_factor, mode=mode,
                                       align_corners=True if 'linear' in mode else None)
            else:
                upsample = nn.Upsample(scale_factor=scale_factor, mode=mode)

            # Apply Upsample
            upsampled_data = upsample(data_tensor)

            if mode not in results:
                results[mode] = []
            results[mode].append(upsampled_data.shape)
            ret.append(upsampled_data[0, 0, :2, :2].numpy())

    return ret
