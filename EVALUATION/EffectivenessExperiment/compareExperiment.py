import importlib.util
import re
import sys

from paddle.vision.transforms import Compose, Normalize, ToTensor, Resize
from sklearn.metrics import precision_recall_fscore_support
from datasets import load_dataset
# vision Models
from EVALUATION.datasets.models.vision.targetModels.alexnet.alexnet import AlexNet as PaddleAlexNet
from EVALUATION.datasets.models.vision.targetModels.densenet.densenet import DenseNet as PaddleDenseNet
from EVALUATION.datasets.models.vision.targetModels.inception.inception import Inception3 as PaddleInception3
from EVALUATION.datasets.models.vision.targetModels.resnet.resnet import ResNet as PaddleResNet, Bottleneck, BasicBlock
from EVALUATION.datasets.models.vision.targetModels.shufflenetv2.shufflenetv2 import ShuffleNetV2 as PaddleShuffleNetV2
from EVALUATION.datasets.models.vision.targetModels.squeezenet.squeezenet import SqueezeNet as PaddleSqueezeNet
from EVALUATION.datasets.models.vision.targetModels.vgg.vgg import VGG as PaddleVGG, make_layers, cfgs


import os
import paddle
from paddle.io import Dataset, DataLoader
from PIL import Image
import numpy as np

transform = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_dir = f"./"
num_classes = 1000


class smallImageNetDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.transform = transform
        self.dataset = load_dataset("mrm8488/image_net1_k-val", split="train")

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['label']
        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.dataset)


def eval_visionModel(model, mode, batch_size=64):
    test_dataset = smallImageNetDataset()
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, drop_last=True)
    y_true_list = []
    y_pred_list = []
    for batch_id, data in enumerate(test_loader()):
        images, labels = data
        if mode == "dynamic":
            # eval vision model
            model.eval()
            logits = model(images)
            predicted = paddle.argmax(paddle.to_tensor(logits, dtype='float64'), axis=1)
            y_true_list.append(labels.numpy())
            y_pred_list.append(predicted.numpy())
        elif mode == "static":
            # numpylabels = labels.copy().numpy()
            paddle.enable_static()
            #place = paddle.CPUPlace()
            place = paddle.CUDAPlace(0)
            executor = paddle.static.Executor(place)
            logits = executor.run(model[0], feed={model[1][0]: images}, fetch_list=model[2])[0]
            predicted = np.argmax(logits, axis=1)

            y_true_list.append(labels)
            y_pred_list.append(predicted)
        else:
            raise ValueError("not support such mode.")

    y_true_all = np.concatenate(y_true_list, axis=0)
    y_pred_all = np.concatenate(y_pred_list, axis=0)

    # accuracy = 100 * np.sum(np.array(y_true_all) == np.array(y_pred_all)) / len(y_true_all)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true_all, y_pred_all, average='macro')

    # print(
    #     f'Accuracy: {accuracy:.4f}, Precision (Macro Avg.): {precision_macro:.4f},'
    #     f' Recall (Macro Avg.): {recall_macro:.4f}')
    print(
        f' Precision (Macro Avg.): {precision:.4f},'
        f' Recall (Macro Avg.): {recall:.4f},'
        f' F1 (Macro Avg.): {f1:.4f},'
    )


def compareMethod(modelname, model):
    weight_file = os.path.join(os.getcwd(), "computingGraphModels", f"{modelname}_IR_and_pd", "paddleModel.npy")
    paddleModelWeight = np.load(weight_file, allow_pickle=True)
    print(f"============================{modelname}=================================")
    try:
        # eval model based on LID-CMC
        # load weight to model
        state_dict = model.state_dict()
        for name, param in state_dict.items():
            if name in paddleModelWeight:
                param_tensor = paddle.to_tensor(paddleModelWeight[name])
                if len(param_tensor.shape) == 2 and ('fc' in name or 'classifier' in name):
                    # Transpose weights for fully connected layers
                    param_tensor = param_tensor.T

                param.set_value(param_tensor)
            else:
                print(f"Missing {name} in loaded weights.")

        # eval model from different model conversion methods
        print("eval metrics of model based on LID-CMC:")
        eval_visionModel(model, "dynamic")
        print("======================================================")
    except:
        print(f"{modelname} not be supported in model conversion based on LID-CMC!")

    try:
        # eval model based on ONNX
        paddle.enable_static()

        place = paddle.CUDAPlace(0)
        exe = paddle.static.Executor(place)
        path_to_model = os.path.join(base_dir, "computingGraphModels", f"{modelname}_IR_and_pd", "inference_model",
                                     "model")
        try:
            model, feed_target_names, fetch_targets = paddle.static.load_inference_model(
                path_to_model, exe
            )
        except:
            raise ValueError(f" X2Paddle not support to convert {modelname} from ONNX to Paddle!")

        # eval model from different model conversion methods
        print("eval metrics of model based on ONNX:")
        eval_visionModel((model, feed_target_names, fetch_targets), "static")
        print("======================================================")
    except:
        print(f"{modelname} not be supported in model conversion based on ONNX!")


def testAlexNet():
    model = PaddleAlexNet(num_classes=num_classes)
    compareMethod(f"alexnet", model)

def testDenseNet121():
    model = PaddleDenseNet(32, (6, 12, 24, 16), 64, num_classes=num_classes)
    compareMethod(f"densenet121", model)

def testDenseNet161():
    model = PaddleDenseNet(48, (6, 12, 36, 24), 96, num_classes=num_classes)
    compareMethod(f"densenet161", model)


def testDenseNet169():
    model = PaddleDenseNet(32, (6, 12, 32, 32), 64, num_classes=num_classes)
    compareMethod(f"densenet169", model)


def testDenseNet201():
    model = PaddleDenseNet(32, (6, 12, 48, 32), 64, num_classes=num_classes)
    compareMethod(f"densenet201", model)


def testInception():
    model = PaddleInception3(num_classes=num_classes)
    compareMethod(f"Inceptionv3", model)


def testResNet18():
    model = PaddleResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    compareMethod(f"resnet18", model)

def testResNet34():
    model = PaddleResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
    compareMethod(f"resnet34", model)

def testResNet50():
    model = PaddleResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
    compareMethod(f"resnet50", model)

def testResNet101():
    model = PaddleResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)
    compareMethod(f"resnet101", model)

def testShuffleNetV2_0_5():
    model = PaddleShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 48, 96, 192, 1024],
                               num_classes=num_classes)
    compareMethod(f"shufflenet_v2_0_5", model)


def testShuffleNetV2_1_0():
    model = PaddleShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 116, 232, 464, 1024],
                               num_classes=num_classes)
    compareMethod(f"shufflenet_v2_1_0", model)


def testShuffleNetV2_1_5():
    model = PaddleShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 176, 352, 704, 1024],
                               num_classes=num_classes)
    compareMethod(f"shufflenet_v2_1_5", model)


def testShuffleNetV2_2_0():
    model = PaddleShuffleNetV2(stages_repeats=[4, 8, 4], stages_out_channels=[24, 244, 488, 976, 2048],
                               num_classes=num_classes)
    compareMethod(f"shufflenet_v2_2_0", model)


def testSqueezeNet1_0():
    model = PaddleSqueezeNet(version="1_0", num_classes=num_classes)
    compareMethod(f"squeezenet1_0", model)


def testSqueezeNet1_1():
    model = PaddleSqueezeNet(version="1_1", num_classes=num_classes)
    compareMethod(f"squeezenet1_1", model)


def testVGG11():
    model = PaddleVGG(make_layers(cfgs["A"], batch_norm=False), num_classes=num_classes)
    compareMethod(f"vgg11", model)


def testVGG13():
    model = PaddleVGG(make_layers(cfgs["B"], batch_norm=False), num_classes=num_classes)
    compareMethod(f"vgg13", model)


def testVGG16():
    model = PaddleVGG(make_layers(cfgs["D"], batch_norm=False), num_classes=num_classes)
    compareMethod(f"vgg16", model)


def testVGG19():
    model = PaddleVGG(make_layers(cfgs["E"], batch_norm=False), num_classes=num_classes)
    compareMethod(f"vgg19", model)

def testInception3():
    model = PaddleInception3(num_classes=num_classes)
    compareMethod(f"inception3", model)


# paddle.device.set_device("gpu:0")
if __name__ == "__main__":
    testAlexNet()
