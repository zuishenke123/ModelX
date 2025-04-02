import numpy as np
import paddle


def generate_test_cases():
    np.random.seed(42)  # 设置随机种子以保证每次生成的数据是一致的
    test_cases = []
    # 生成固定的卷积权重和偏置
    # 对于 nn.Conv1D，权重的形状为 [out_channels, in_channels, kernel_width]
    weights = np.random.randn(5, 3, 7).astype(np.float32)  # 例如：5个输出通道，3个输入通道，长度为7的卷积核
    biases = np.random.randn(5).astype(np.float32)  # 偏置的形状为 [out_channels]
    for _ in range(10):
        # 输入数据的形状为 [batch_size, in_channels, width]
        # 假设 width 为 100
        input_array = np.random.randn(1, 3, 100).astype(np.float32)
        test_cases.append((input_array, weights, biases))
    return test_cases


def test_operator(test_cases):
    # 定义卷积层
    conv = paddle.nn.Conv1D(in_channels=3, out_channels=5, kernel_size=7, stride=1, padding=3)

    ret = []
    for input_array, weights, biases in test_cases:
        # 将numpy数组转换为paddle张量
        input_tensor = paddle.to_tensor(input_array)
        weights_tensor = paddle.to_tensor(weights)
        biases_tensor = paddle.to_tensor(biases)

        # 设置卷积层的权重和偏置
        conv.weight.set_value(weights_tensor)
        conv.bias.set_value(biases_tensor)

        # 应用卷积层
        output_tensor = conv(input_tensor)
        ret.append(output_tensor.numpy())  # 将输出转换回numpy数组
    return ret

