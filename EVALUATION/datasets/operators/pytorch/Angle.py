# -*- coding: utf-8 -*-
import numpy as np


# Function to generate test cases for torch.angle using NumPy
def generate_test_cases():
    # Generate a list of numpy arrays representing complex numbers
    test_cases = [
        np.array([1+1j, -1-1j, 1-1j, -1+1j], dtype=np.complex64),  # Quadrant angles
        np.array([1+0j, 0+1j, -1+0j, 0-1j], dtype=np.complex64),   # Axis aligned angles
        np.array([1+1j, 0+0j, -1-1j], dtype=np.complex64),         # Including zero
        np.array([np.exp(1j * np.pi/4) * i for i in range(1, 5)], dtype=np.complex64)  # Increasing magnitudes
    ]
    return test_cases


def test_operator(test_cases):
    import torch
    print("Testing torch.angle on multiple test cases:")
    ret = []
    for i, case in enumerate(test_cases):
        # Convert NumPy arrays to PyTorch tensors
        tensor = torch.from_numpy(case)

        # Apply torch.angle
        angles = torch.angle(tensor)
        ret.append(angles.numpy())

    return ret

