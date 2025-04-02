import argparse, os, sys
import importlib.util
import csv
import os
import multiprocessing
from multiprocessing import Process, Queue
import time
import numpy as np

sys.path.append(os.path.dirname(__file__) + "/..")

from CMC.converter import Converter

# load operator datasets
from EVALUATION.datasets import operators as ops

operator_dataset = os.path.dirname(ops.__file__)


def evaluate_operator_MAE_and_RMSE(sourceOperator_output, targetOperator_output):
    source_array = sourceOperator_output
    target_array = targetOperator_output

    assert len(source_array) == len(target_array), "Arrays must have the same length"

    total_mae = 0
    total_mse = 0
    count = 0

    for src, tgt in zip(sourceOperator_output, targetOperator_output):
        if isinstance(src, (list, tuple, str, dict, set, np.ndarray)) and isinstance(tgt, (
        list, tuple, str, dict, set, np.ndarray)):
            if len(src) != len(tgt):
                continue

        # Convert src and tgt to NumPy arrays, compatible with Boolean types
        src_array = np.array(src, dtype=float)
        tgt_array = np.array(tgt, dtype=float)

        total_mae += np.sum(np.abs(src_array - tgt_array))
        total_mse += np.sum((src_array - tgt_array) ** 2)
        if isinstance(src, (list, tuple, str, dict, set, np.ndarray)):
            count += len(src)
        else:
            count += 1

    if count > 0:
        mae = total_mae / count
        mse = total_mse / count
        rmse = np.sqrt(mse)
    else:
        mae, mse, rmse = None, None, None  # Return None if there is no valid data

    return mae, rmse


def get_operator_test_cases(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Check if there is a 'testOperator' function in the module
    if hasattr(module, 'generate_test_cases'):
        generate_test_cases = getattr(module, 'generate_test_cases')

        if callable(generate_test_cases):
            # Call the 'testOperator' function
            print(f"Calling generate_test_cases from module {module_name}")
            return generate_test_cases()
    else:
        print(f"No generate_test_cases function in module {module_name}")
        return None


def get_operator_output(queue, module_name, file_path, test_cases):
    try:
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Check if there is a 'testOperator' function in the module
        if hasattr(module, 'test_operator'):
            testOperator = getattr(module, 'test_operator')

            if callable(testOperator):
                # Call the 'testOperator' function
                print(f"Calling testOperator from module {module_name}")
                queue.put(testOperator(test_cases))
        else:
            queue.put(None)
            print(f"No testOperator function in module {module_name}")
    except Exception as e:
        queue.put(None)
        print(f"Error loading or executing module {module_name}: {e}")


def test_operator(fileName):
    # Build the full file path
    file_path = os.path.join(operator_dataset, fileName)
    queue = multiprocessing.Queue()
    # Dynamic import module
    module_name = fileName[:-3]
    test_cases = get_operator_test_cases(module_name, file_path)
    # Prepare to run source framework (e.g., PyTorch) in a separate process
    queue1 = Queue()
    p1 = Process(target=get_operator_output,
                 args=(queue1, module_name, file_path, test_cases))
    p1.start()

    if not os.path.exists(os.path.join(os.getcwd(), "Results")):
        os.makedirs(os.path.join(os.getcwd(), "Results"))
        print("Created the 'Results' directory.")
    else:
        print("The 'Results' directory already exists.")


    start_time = time.perf_counter()
    # Convert source model
    converter = Converter(in_dir=file_path, out_dir=os.path.join(os.getcwd(), "Results"), show_unsupport=True)
    converter.run()

    end_time = time.perf_counter()


    execution_time = (end_time - start_time) * 1000

    print(f"Conversion time: {execution_time:.3f} ms")

    # Prepare to run target framework (e.g., Paddle) in a separate process
    file_path_converted = os.path.join(os.getcwd(), "Results", module_name, fileName)
    queue2 = Queue()
    p2 = Process(target=get_operator_output,
                 args=(queue2, module_name, file_path_converted, test_cases))
    p2.start()

    from queue import Empty
    try:
        target_framework_ret = queue2.get()
        source_framework_ret = queue1.get()
    except Empty:
        raise EnvironmentError("Failed to retrieve data from queues")

    # Wait for both processes to finish
    p1.join()
    p2.join()


    assert source_framework_ret is not None and target_framework_ret is not None, "operator must be not None"

    MAE, RMSE = evaluate_operator_MAE_and_RMSE(source_framework_ret, target_framework_ret)
    print(f"{module_name} - MAE: {MAE}, RMSE: {RMSE}")
    return MAE, RMSE


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="EquivalenceExperiment", description="test equivalence of LID-CMC in converting pytorch"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--all', action='store_true', help="Test all pytorch test cases in the dataset.")
    group.add_argument('--single_operator', metavar='OPERATOR', help="Tests the test case of one of the selected pytorch.")

    args = parser.parse_args()

    if args.all:
        with open(os.path.join(operator_dataset, 'test.csv'), 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['operator', 'MAE', 'RMSE'])
            for filename in os.listdir(operator_dataset):
                if filename.endswith('.py') and filename != '__init__.py':
                    try:
                        mae, rmse = test_operator(filename)
                        writer.writerow([filename[:-3], mae, rmse])
                    except Exception as e:
                        print(f"{filename[:-3]}, test failed !, {e}")
                        pass
                    continue
    elif args.single_operator:
            found = False
            for filename in os.listdir(operator_dataset):

                if filename.endswith('.py') and filename.lower() == f"{args.single_operator}.py".lower():
                    mae, rmse = test_operator(filename)
                    found = True
                    # print(f"{args.single_operator}: MAE: {mae}, RMSE: {rmse}")
                    break

            if not found:
                raise ValueError(f"Unsupported operator: {args.single_operator}")
