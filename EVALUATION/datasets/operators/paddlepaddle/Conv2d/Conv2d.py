import torch
import numpy as np


def generate_test_cases():
    np.random.seed(42)
    test_cases = []
    weights = np.random.randn(6, 3, 3, 3).astype(np.float32)
    biases = np.random.randn(6).astype(np.float32)
    for _ in range(10):
        input_array = np.random.randn(1, 3, 32, 32).astype(np.float32)
        test_cases.append((input_array, weights, biases))
    return test_cases


def test_operator(test_cases):
    conv = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3,
        stride=1, padding=1)
    ret = []
    for input_array, weights, biases in test_cases:
        input_tensor = torch.tensor(data=input_array)
        weights_tensor = torch.tensor(data=weights)
        biases_tensor = torch.tensor(data=biases)
        conv.weight.set_value(weights_tensor)
        conv.bias.set_value(biases_tensor)
        output_tensor = conv(input_tensor)
        ret.append(output_tensor.numpy())
    return ret
