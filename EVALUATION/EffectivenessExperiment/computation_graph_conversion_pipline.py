import pickle
import os
import re
import shutil
from functools import partial
import torch.nn as nn
import torch
from torchvision.models.densenet import _load_state_dict

# vision Models
from EVALUATION.datasets.models.vision.sourceModels.densenet import DenseNet as TorchDenseNet
from EVALUATION.datasets.models.vision.sourceModels.resnet import ResNet as TorchResNet, BasicBlock, Bottleneck
from EVALUATION.datasets.models.vision.sourceModels.vgg import VGG as TorchVgg, make_layers, cfgs
from EVALUATION.datasets.models.vision.sourceModels.squeezenet import SqueezeNet as TorchSqueezeNet
from EVALUATION.datasets.models.vision.sourceModels.shufflenetv2 import ShuffleNetV2 as TorchShufflenet
from EVALUATION.datasets.models.vision.sourceModels.alexnet import AlexNet as TorchAlexnet
from EVALUATION.datasets.models.vision.sourceModels.inception import Inception3 as TorchInception3


# Initialize Result Path
input_tensor = torch.randn(64, 3, 224, 224)
output_folder = os.path.join(os.getcwd(), "computingGraphModels")

if not os.path.exists(output_folder):
    os.makedirs(output_folder)


def GenerateOnnxAndPd(model, model_name):
    model.eval()
    # Specify the name of the storage folder
    if not os.path.exists(os.path.join(output_folder, f"{model_name}_IR_and_pd")):
        os.makedirs(os.path.join(output_folder, f"{model_name}_IR_and_pd"))

    try:
        import torch
    except Exception as e:
        raise ValueError(f"import torch failed:{e}")

    onnx_file = os.path.join(output_folder, f"{model_name}_IR_and_pd", f"testTorch{model_name}.onnx")
    try:
        torch.onnx.export(model, input_tensor, onnx_file, verbose=True)
    except Exception as e:
        raise ValueError(f"Failed to export ONNX model: {e}")

    pd_model_dir = os.path.join(output_folder, f"{model_name}_IR_and_pd")
    # Use x2paddle to convert ONNX models to PaddlePaddle
    try:
        conversion_command = f"x2paddle --framework=onnx --model={onnx_file} --save_dir={pd_model_dir}"
        conversion_result = os.system(conversion_command)
        print(f"x2paddle conversion command returned: {conversion_result}")
    except Exception as e:
        raise ValueError(f"Failed to convert ONNX model with x2paddle: {e}")

    print(f"Model conversion completed. ONNX and PaddlePaddle models are saved in '{output_folder}'.")


def generate_pd_and_convert_npy(weights, model, model_name):
    if weights is not None:
        _load_state_dict(model=model, weights=weights, progress=True)
    # Convert Weight in PaddlePaddle
    paddle_state_dict = {}
    for key, value in weights.get_state_dict(progress=True).items():
        key = key.replace('running_mean', '_mean').replace('running_var', '_variance')
        key = re.sub(r'norm\.(\d+)\.', r'norm\1.', key)
        key = re.sub(r'conv\.(\d+)\.', r'conv\1.', key)
        paddle_state_dict[key] = value.detach().numpy()

    npy_dir = os.path.join(output_folder, f"{model_name}_IR_and_pd")
    if os.path.exists(npy_dir):
        shutil.rmtree(npy_dir)
    os.makedirs(npy_dir)

    with open(os.path.join(npy_dir, 'paddleModel.npy'), 'wb') as f:
        pickle.dump(paddle_state_dict, f)

    # Cross-framework Conversion converted with ONNX
    GenerateOnnxAndPd(model, model_name)


def testAlexnet(num_classes=1000):
    from torchvision.models.alexnet import AlexNet_Weights, alexnet
    weights = AlexNet_Weights.DEFAULT

    model = TorchAlexnet(num_classes=num_classes)
    model_name = "alexnet"

    generate_pd_and_convert_npy(weights, model, model_name)

def testDensenet121(num_classes=1000):
    from torchvision.models.densenet import DenseNet121_Weights
    weights = DenseNet121_Weights.DEFAULT
    model = TorchDenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    model_name = "densenet121"

    generate_pd_and_convert_npy(weights, model, model_name)

def testDensenet161(num_classes=1000):
    from torchvision.models.densenet import DenseNet161_Weights
    weights = DenseNet161_Weights.DEFAULT
    model = TorchDenseNet(48, (6, 12, 36, 24), 96, num_classes=num_classes)
    model_name = "densenet161"

    generate_pd_and_convert_npy(weights, model, model_name)


def testDensenet169(num_classes=1000):
    from torchvision.models.densenet import DenseNet169_Weights
    weights = DenseNet169_Weights.DEFAULT
    model = TorchDenseNet(32, (6, 12, 32, 32), 64, num_classes=num_classes)
    model_name = "densenet169"

    generate_pd_and_convert_npy(weights, model, model_name)


def testDensenet201(num_classes=1000):
    from torchvision.models.densenet import DenseNet201_Weights
    weights = DenseNet201_Weights.DEFAULT
    model = TorchDenseNet(32, (6, 12, 48, 32), 64, num_classes=num_classes)
    model_name = "densenet201"

    generate_pd_and_convert_npy(weights, model, model_name)


def testResNet18(num_classes=1000):
    from torchvision.models.resnet import ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    model = TorchResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    model_name = "resnet18"

    generate_pd_and_convert_npy(weights, model, model_name)

def testResNet34(num_classes=1000):
    from torchvision.models.resnet import ResNet34_Weights
    weights = ResNet34_Weights.DEFAULT
    model = TorchResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    model_name = "resnet34"

    generate_pd_and_convert_npy(weights, model, model_name)


def testResNet50(num_classes=1000):
    from torchvision.models.resnet import ResNet50_Weights
    weights = ResNet50_Weights.DEFAULT
    model = TorchResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model_name = "resnet50"

    generate_pd_and_convert_npy(weights, model, model_name)

def testResNet101(num_classes=1000):
    from torchvision.models.resnet import ResNet101_Weights
    weights = ResNet101_Weights.DEFAULT
    model = TorchResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    model_name = "resnet101"

    generate_pd_and_convert_npy(weights, model, model_name)


def testVgg11(num_classes=1000):
    from torchvision.models.vgg import VGG11_Weights
    weights = VGG11_Weights.DEFAULT
    model = TorchVgg(make_layers(cfgs["A"], batch_norm=False), num_classes=num_classes)
    model_name = "vgg11"

    generate_pd_and_convert_npy(weights, model, model_name)


def testVgg13(num_classes=1000):
    from torchvision.models.vgg import VGG13_Weights
    weights = VGG13_Weights.DEFAULT
    model = TorchVgg(make_layers(cfgs["B"], batch_norm=False), num_classes=num_classes)
    model_name = "vgg13"

    generate_pd_and_convert_npy(weights, model, model_name)


def testVgg16(num_classes=1000):
    from torchvision.models.vgg import VGG16_Weights
    weights = VGG16_Weights.DEFAULT
    model = TorchVgg(make_layers(cfgs["D"], batch_norm=False), num_classes=num_classes)
    model_name = "vgg16"

    generate_pd_and_convert_npy(weights, model, model_name)


def testVgg19(num_classes=1000):
    from torchvision.models.vgg import VGG19_Weights
    weights = VGG19_Weights.DEFAULT
    model = TorchVgg(make_layers(cfgs["E"], batch_norm=False), num_classes=num_classes)
    model_name = "vgg19"

    generate_pd_and_convert_npy(weights, model, model_name)


def testSqueezenet1_0(num_classes=1000):
    from torchvision.models.squeezenet import SqueezeNet1_0_Weights
    weights = SqueezeNet1_0_Weights.DEFAULT
    model = TorchSqueezeNet(version="1_0", num_classes=num_classes)
    model_name = "squeezenet1_0"

    generate_pd_and_convert_npy(weights, model, model_name)


def testSqueezenet1_1(num_classes=1000):
    from torchvision.models.squeezenet import SqueezeNet1_1_Weights
    weights = SqueezeNet1_1_Weights.DEFAULT
    model = TorchSqueezeNet(version="1_1", num_classes=num_classes)
    model_name = "squeezenet1_1"

    generate_pd_and_convert_npy(weights, model, model_name)


def testShufflenet_v2_0_5(num_classes=1000):
    from torchvision.models.shufflenetv2 import ShuffleNet_V2_X0_5_Weights
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT
    model = TorchShufflenet(stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024],
                            num_classes=num_classes)
    model_name = "shufflenet_v2_0_5"

    generate_pd_and_convert_npy(weights, model, model_name)


def testShufflenet_v2_1_0(num_classes=1000):
    from torchvision.models.shufflenetv2 import ShuffleNet_V2_X1_0_Weights
    weights = ShuffleNet_V2_X1_0_Weights.DEFAULT
    model = TorchShufflenet(stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024],
                            num_classes=num_classes)
    model_name = "shufflenet_v2_1_0"
    generate_pd_and_convert_npy(weights, model, model_name)


def testShufflenet_v2_1_5(num_classes=1000):
    from torchvision.models.shufflenetv2 import ShuffleNet_V2_X1_5_Weights
    weights = ShuffleNet_V2_X1_5_Weights.DEFAULT
    model = TorchShufflenet(stages_repeats=[4, 8, 4], stages_out_channels=[24, 176, 352, 704, 1024],
                            num_classes=num_classes)
    model_name = "shufflenet_v2_1_5"

    generate_pd_and_convert_npy(weights, model, model_name)


def testShufflenet_v2_2_0(num_classes=1000):
    from torchvision.models.shufflenetv2 import ShuffleNet_V2_X2_0_Weights
    weights = ShuffleNet_V2_X2_0_Weights.DEFAULT
    model = TorchShufflenet(stages_repeats=[4, 8, 4], stages_out_channels=[24, 244, 488, 976, 2048],
                            num_classes=num_classes)
    model_name = "shufflenet_v2_2_0"

    generate_pd_and_convert_npy(weights, model, model_name)


def testInceptionV3(num_classes=1000):
    from torchvision.models.inception import Inception_V3_Weights
    weights = Inception_V3_Weights.DEFAULT
    model = TorchInception3(num_classes=num_classes)
    model_name = "Inception"

    generate_pd_and_convert_npy(weights, model, model_name)


if __name__ == '__main__':
    # # generate densenet in Paddle based on computing Graph
    testAlexnet()  # sucess!
    # testDensenet161()  # failed!
    # testDensenet169()  # failed!
    # testDensenet201()  # failed!
    # # # generate resnet in Paddle based on computing Graph
    # testResNet18()  # sucess!
    # testResNet50()  # sucess!
    # # # generate vgg in Paddle based on computing Graph
    # testVgg11()  # sucess!
    # testVgg13()  # sucess!
    # testVgg16()  # sucess!
    # testVgg19()  # sucess!
    # # # generate squeezenet in Paddle based on computing Graph
    # testSqueezenet1_0()  # sucess!
    # testSqueezenet1_1()  # sucess!
    # # # generate squeezenet in Paddle based on computing Graph
    # testShufflenet_v2_0_5()  # failed!
    # testShufflenet_v2_1_0()  # failed!
    # testShufflenet_v2_1_5()  # failed!
    # testShufflenet_v2_2_0()  # failed!
    # # generate inception in Paddle based on computing Graph
    # testInceptionV3()  # sucess!
    pass
