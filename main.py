import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__) + "/..")

from CMC.converter import Converter


def main():
    parser = argparse.ArgumentParser(
        prog="LID-CMC", description="CMC tool entry point"
    )
    parser.add_argument(
        "--in_dir",
        # default="./test.py",
        # default="./EVALUATION/datasets/models/pytorch/vision/sourceModels/densenet.py",
        default="./EVALUATION/datasets/models/pytorch/vision/sourceModels/inception.py",
        # default="./EVALUATION/datasets/models/pytorch/vision/sourceModels/googlenet.py",
        type=str,
        help="the Source DL Model file or directory.",
    )
    parser.add_argument(
        "--source_framework",
        default="pytorch",
        # default="paddlepaddle",
        type=str,
        help="the Source framework class of DL Model.",
    )
    parser.add_argument(
        "--out_dir", default=None, type=str, help="the Target DL Model file or directory."
    )
    parser.add_argument(
        "--log_dir", default="./logs/", type=str, help="the log file or directory."
    )
    parser.add_argument(
        "--target_framework",
        default="paddlepaddle",
        # default="pytorch",
        type=str,
        help="the Target framework class of DL Model.",
    )
    parser.add_argument(
        "--show_unsupport",
        default=True,
        type=bool,
        help="show these APIs which are not supported to convert now",
    )
    args = parser.parse_args()

    assert args.in_dir is not None, "User must specify --in_dir "
    converter = Converter(args.in_dir, args.out_dir, args.log_dir, args.show_unsupport, args.source_framework, args.target_framework)
    converter.run()


if __name__ == '__main__':
    main()
