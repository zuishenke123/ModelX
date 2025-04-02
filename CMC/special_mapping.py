import os, json


def get_api_mapping(sourceDLClass="pytorch", targetDLClass="paddlepaddle"):
    json_file = os.path.join(os.path.dirname(__file__), f"{sourceDLClass}_to_{targetDLClass}.json")
    try:
        with open(json_file, "r") as file:
            API_MAPPING = json.load(file)
            return API_MAPPING
    except json.JSONDecodeError as e:
        raise ValueError(f"Error parsing JSON: {e}. Please check if the file contains valid JSON.")


SUPPORT_PACKAGE_LIST = {
    "pytorch": [
        "torch",
        "torchvision",

    ],
    "paddlepaddle": [
        "paddle",
        "paddle.vision"
    ]
}

TENSOR_MAPPING = {
    "pytorch": {}
}

SPECIALMODULE_MAPPING = {
    "pytorch": {
        "nn": {
            "tensorflow": "layer"
        },
        "Module": {
            "paddlepaddle": "Layer"
        },
        "utils.data": {
            "paddlepaddle": "io"
        },
        "optim": {
            "paddlepaddle": "optimizer"
        }
    },
    "paddlepaddle": {
        "nn": {
            "tensorflow": "layer"
        },
        "Layer": {
            "pytorch": "Module"
        },
        "io": {
            "pytorch": "utils.data"
        },
        "optimizer": {
            "pytorch": "optim"
        }
    }
}

# Different frameworks correspond to different main Modules name
FrameworkPackage = {
    "pytorch": ["torch", "torchvision"],
    "tensorflow": ["tensorflow.keras"],
    "paddlepaddle": ["paddle"],
}
omitSuffixCall = [
    "contiguous"
]

dataTypeMapping = {
    "int": "int32",
    "long": "int64",
    "float": "float32",
    "double": "float64",
    "short": "int8",
    "bool": "bool"
}



addOp = {
    "pytorch": "torch.add",
    "paddlepaddle": "paddle.add"
}

subOp = {
    "pytorch": "torch.sub",
    "paddlepaddle": "paddle.sub"
}

divOp = {
    "pytorch": "torch.div",
    "paddlepaddle": "paddle.div"
}

mulOp = {
    "pytorch": "torch.mul",
    "paddlepaddle": "paddle.mul"
}
