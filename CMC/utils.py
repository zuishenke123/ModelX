import ast
import collections
import importlib
import inspect
import os
import pkgutil
import platform
import re, logging
import subprocess
import textwrap
import types
from typing import Dict, List

import astor
import numpy as np
import pandas as pd

from CMC.special_mapping import FrameworkPackage, SPECIALMODULE_MAPPING, addOp, subOp, divOp, mulOp


def log_info(logger, msg, file=None, line=None):
    logger.setLevel(logging.INFO)
    if file:
        if line:
            msg = "[{}:{}] {}".format(file, line, msg)
        else:
            msg = "[{}] {}".format(file, msg)
    else:
        msg = "{}".format(msg)
    logger.info(msg)


def log_warning(logger, msg, file=None, line=None):
    logger.setLevel(logging.WARNING)
    if file:
        if line:
            msg = "[{}:{}] {}".format(file, line, msg)
        else:
            msg = "[{}] {}".format(file, msg)
    else:
        msg = "{}".format(msg)
    logger.warning(msg)


class UniqueNameGenerator:
    def __init__(self):
        self.ids = collections.defaultdict(int)

    def __call__(self, key):
        counter = self.ids[key]
        self.ids[key] += 1
        return "_".join([key, str(counter)])


Generator = UniqueNameGenerator()


def get_unique_name(key):
    return Generator(key)


class CodeFileGenerator(object):
    _instance = None
    _file_initialized = False

    def __init__(self, fileName=None, *args, **kwargs):
        if fileName:
            self.file_name = fileName
            self.api_written = set()
            self._initialize_file()

    def _initialize_file(self):
        if not self._file_initialized and self.file_name:
            header = (
                "# This file is generated automatically, please don't edit it!\n"
                "import paddle\n\n"
            )
            self._write_to_file(header, mode="w")
            self._file_initialized = True

    def _write_to_file(self, content, mode="a"):
        try:
            with open(self.file_name, mode) as file:
                file.write(content)
        except IOError as e:
            print(f"Error writing to file {self.file_name}: {e}")

    def __new__(cls, file_name=None):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def write_code(self, code, torch_api):
        if torch_api not in self.api_written:
            self._write_to_file(code)
            self.api_written.add(torch_api)


def HandleAliasApi(api, DLClass):
    # alias ast node content
    patterns = {}
    if DLClass == "pytorch":
        patterns = {
            r"nn\\.modules\\.\\w+\\.": r"nn\\.\\w+\\.",
            r"sampler\\.\\w+\\.": r".\\w+\\.",
            r"data\\.sampler\\.\\w+\\.": r"data\\.\\w+\\.",
            r"torch\\.special\\.\\w+\\.": r"torch\\.\\w+\\."
        }

    # 遍历列表，使用re.sub方法移除匹配的子字符串
    for pattern, re_pattern in patterns.items():
        api = re.sub(pattern, re_pattern, api)

    return api


def traverseChildNodes(node):
    '''
    Traverse the tree of nodes
    '''
    for field in node._fields:
        try:
            yield field, getattr(node, field)
        except AttributeError:
            pass


def check_api_exists(full_api):
    parts = full_api.split('.')
    module_path, attr_name = '.'.join(parts[:-1]), parts[-1]
    try:
        module = importlib.import_module(module_path)
        return hasattr(module, attr_name)
    except (ModuleNotFoundError, ValueError):
        return False


def verify_import_component(base, component):
    try:
        module = __import__(base, fromlist=[component])
        return hasattr(module, component)
    except ImportError:
        return False


def get_framework_modules(targetModule="paddlepaddle"):
    try:
        import pkgutil
        if targetModule == "paddlepaddle":
            import paddle  # 将导入放在函数内部
            modules = ["paddle"]
            for importer, modname, ispkg in pkgutil.iter_modules(paddle.__path__, "paddle."):
                modules.append(modname)
        elif targetModule == "pytorch":
            import torch  # 将导入放在函数内部
            modules = ["torch"]
            for importer, modname, ispkg in pkgutil.iter_modules(torch.__path__, "torch."):
                modules.append(modname)
        else:
            raise EnvironmentError("Unsupport Target Framework!")
        return modules

    except ImportError:
        # 如果 PaddlePaddle 未安装，可以返回一个错误信息或空列表
        print(f"{targetModule} is not installed. Unable to retrieve {targetModule} modules.")
        return []


def get_operators(parent_package):
    assert parent_package is not None, "The query package cannot be None"
    callables = []
    try:
        module = importlib.import_module(parent_package)
    except ImportError as e:
        print(f"# Unable to import the module: {parent_package}: {e}")
        return callables

    # Iterate through module attributes to find callable objects
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, (types.FunctionType, type)):
            callables.append(f"{parent_package}.{attr_name}")

    # Recursively iterate through submodules
    if hasattr(module, '__path__'):
        for _, submod_name, is_pkg in pkgutil.iter_modules(module.__path__):
            full_submod_name = f"{parent_package}.{submod_name}"
            callables.extend(get_operators(full_submod_name))

    return callables


def get_modules_and_operators(parent_package):
    assert parent_package is not None, "The query package cannot be None"
    module_list = []
    try:
        if parent_package.endswith("__main__"):
            return []
        module = importlib.import_module(parent_package)
        module_list.append(parent_package)  # Add the module itself

        # Add callable methods under the module
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if inspect.isfunction(attr) or inspect.isclass(attr):
                module_list.append(f"{parent_package}.{attr_name}")

        # Recursively traverse submodules
        if hasattr(module, '__path__'):
            for _, submod_name, is_pkg in pkgutil.iter_modules(module.__path__):
                full_submod_name = f"{parent_package}.{submod_name}"
                module_list.extend(get_modules_and_operators(full_submod_name))

        return module_list
    except ImportError as e:
        print(f"无法导入模块 {parent_package}: {e}")
        return []


def search_mapping_api(aliasName, logger, sourceDLClass="pytorch", targetDLClass="paddlepaddle"):
    # best match for pytorch api in paddlepaddle
    parts = aliasName.split('.')
    if len(parts) == 1:
        if sourceDLClass == "paddlepaddle":
            return "torch"
        elif sourceDLClass == "pytorch":
            return "paddle"
    if parts[0] in FrameworkPackage[sourceDLClass]:
        try:
            from difflib import get_close_matches
            modules = get_framework_modules(targetDLClass)
            parts[0] = FrameworkPackage[targetDLClass][0]
            torch_base_parts = parts[:-1]  # package path
            torch_component = parts[-1]  # pytorch component
            target_module = None  # Initialize as an empty package
            # Generate a progressively reduced list of PyTorch base paths
            for i, part in enumerate(torch_base_parts):
                if part in SPECIALMODULE_MAPPING[sourceDLClass] and targetDLClass in \
                        SPECIALMODULE_MAPPING[sourceDLClass][part]:
                    torch_base_parts[i] = SPECIALMODULE_MAPPING[sourceDLClass][part][targetDLClass]
            potential_bases = ['.'.join(torch_base_parts[:i]) for i in range(len(torch_base_parts), 0, -1)]
            for base in potential_bases:
                # 检查该路径是否在 PaddlePaddle 的子模块列表中
                if base in modules:
                    # 验证组件是否存在于找到的 PaddlePaddle 路径下
                    if verify_import_component(base, torch_component):
                        target_module = f"{base}.{torch_component}"
                        break
                    else:
                        attributes = get_modules_and_operators(base)
                        alias_lower = ".".join(parts).lower()
                        for callable in attributes:
                            if alias_lower in callable.lower():
                                # 如果找到了大小写不敏感的匹配，返回成功
                                target_module = callable
                                break
                        if target_module is None:
                            # 如果大小写不敏感的匹配也失败了，进一步寻找最接近的组件
                            closest_match = get_close_matches(aliasName, attributes)
                            if len(closest_match) == 0:
                                target_module = None
                            else:
                                target_module = closest_match[0]
                        break
            if target_module:
                log_info(logger,
                         f"{sourceDLClass} module: {aliasName} approximately matches to {targetDLClass} module: {target_module}. Please check.")
            else:
                log_info(logger, "converting failed {} module: {}.".format(sourceDLClass, aliasName))
            return target_module
        except ImportError:
            exit("difflib module is not available. please install difflib!")


def get_operator_parameters(operator_name, logger, sourceDLClass="pytorch", targetDLClass="paddlepaddle"):
    try:
        framework = None
        if targetDLClass == "paddlepaddle":
            import paddle as framework
        elif targetDLClass == "pytorch":
            import torch as framework

        if framework is None:
            log_info(logger, "Unsupported deep learning framework.")
            return None, None

        # 通过分割字符串，安全地解析操作符
        module_parts = operator_name.split('.')
        operator = framework
        for part in module_parts:
            operator = getattr(operator, part, None)
            if operator is None:
                log_info(logger, f"Operator {operator_name} not found in {targetDLClass}.")
                return None, None

    except ImportError as e:
        log_info(logger, f"Failed to import the framework: {str(e)}")
        return None, None

        # 检查操作符是否可调用
    if callable(operator):
        try:
            sig = inspect.signature(operator)
            parameters = sig.parameters
            positional_params = []
            keyword_params = []

            for name, param in parameters.items():
                if param.default == inspect.Parameter.empty:
                    if param.kind == inspect.Parameter.VAR_POSITIONAL:
                        positional_params.append("*" + name)
                    else:
                        positional_params.append(name)
                else:
                    keyword_params.append((name, param.default))
            if positional_params == ['*args', 'kwargs']:
                try:
                    positional_params, keyword_params = get_parent_signature_params(operator)
                except ValueError as e:
                    return None, None
            return positional_params, keyword_params
        except ValueError as e:
            return extract_parameters_from_doc(operator, logger)
    else:
        log_info(logger, f"{operator_name} is not a callable operator.")
        return None, None


def get_parent_signature_params(cls):
    """获取直接父类的签名参数，并区分位置参数和关键字参数，跳过 *args 和 **kwargs。"""

    positional_params = []
    keyword_params = []

    # 检查是否有父类
    if len(cls.__bases__) > 0:
        parent_class = cls.__bases__[0]
        try:
            # 获取父类的签名
            parent_sig = inspect.signature(parent_class)
            for name, param in parent_sig.parameters.items():
                # 跳过 *args 和 **kwargs
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    continue

                # 区分无默认值和有默认值的参数
                if param.default == inspect.Parameter.empty:
                    positional_params.append(name)
                else:
                    keyword_params.append((name, param.default))
        except ValueError:
            pass  # 如果无法获取父类签名，跳过

    return positional_params, keyword_params

import re


def extract_parameters_from_doc(func, logger):
    try:
        docstring = func.__doc__
        if not docstring:
            log_info(logger, "No docstring found.")
            return None, None

        # General regex pattern to match different formats of parameter sections
        arg_pattern = re.compile(r"(Args|Arguments):\s+(.*?)(?=\n\S|\Z)", re.S)
        kwarg_pattern = re.compile(r"(Keyword args|Kwargs):\s+(.*?)(?=\n\S|\Z)", re.S)

        positional_params = []
        keyword_params = []

        # Extract positional arguments (Args or Arguments)
        args_match = arg_pattern.search(docstring)
        if args_match:
            args_block = args_match.group(2)
            args_lines = re.finditer(r"\s*(\w+)\s*\((.*?)\):", args_block)
            for match in args_lines:
                param_name = match.group(1)
                positional_params.append(param_name)

        # Extract keyword arguments (Keyword args or Kwargs)
        kwargs_match = kwarg_pattern.search(docstring)
        if kwargs_match:
            kwargs_block = kwargs_match.group(2)
            kwargs_lines = re.finditer(r"\s*(\w+)\s*\((.*?)\):\s*(.*?)\s*(?=^\s*\w+\s*\(|\Z)", kwargs_block,
                                       re.S | re.M)
            for match in kwargs_lines:
                param_name = match.group(1)
                param_info = match.group(3)

                # Extract default value if specified
                default_match = re.search(r"Default:\s*`([^`]+)`", param_info)
                if default_match:
                    default_value_str = default_match.group(1).strip()
                    if default_value_str.lower() == "none":
                        default_value = None
                    elif default_value_str.lower() == "true":
                        default_value = True
                    elif default_value_str.lower() == "false":
                        default_value = False
                    else:
                        default_value = default_value_str
                else:
                    default_value = None
                keyword_params.append((param_name, default_value))

        # Handle asterisk (*) for keyword-only arguments if present
        asterisk_match = re.search(r"\*\s*(\w+)\s*\((.*?)\)", docstring)
        if asterisk_match:
            param_name = asterisk_match.group(1)
            keyword_params.append((param_name, None))  # Assume None as default for optional parameters

        return positional_params, keyword_params

    except Exception as e:
        log_info(logger, f"Failed to extract parameters from docstring: {str(e)}.")
        return None, None


def replace_pattern(text, pattern, replacement):
    """
     Function for replacing text that matches a specific pattern.
    :param text: The original text to be processed.
    :param pattern: The regular expression pattern used to match the parts that need to be replaced.
    :param replacement: The text to replace the matched pattern with.
    :return: The text after replacement.
    """
    # Use a loop to continuously replace the text until the text no longer changes
    while True:
        new_text = re.sub(pattern, replacement, text)
        if new_text == text:  # 如果文本未发生变化，终止循环
            break
        text = new_text
    return text


def tensor_replacement(match):
    # Get the matched variable name part (excluding '_val')
    original_name = match.group(1)
    if original_name == 'self':
        return 'self'
    elif original_name.startswith('t') and original_name[1:].isdigit():
        return f'tensor{original_name[1:]}'
    else:
        # For other cases, directly return the original name (removing '_val')
        return original_name


def funcMapping(targetDLClass, operator):
    if targetDLClass == "paddlepaddle":
        interfaces = get_modules_and_operators("paddle._C_ops")
        for interface in interfaces:
            if interface.endswith(operator):
                # Indicate the presence of a higher-level interface and return the actual interface
                return interface
        return None


def get_entity_from_string(path):
    """
    Get a class or function from a Python module using a string path
    """
    # Split off the final class or function name
    module_path, entity_name = path.split('.')[:-1], path.split('.')[-1]
    module_path = '.'.join(module_path)

    try:
        # import module
        module = importlib.import_module(module_path)
        entity = getattr(module, entity_name)
        return entity
    except ImportError as e:
        return f"Import error: {str(e)}"
    except AttributeError as e:
        return f"Attribute error: {str(e)}"


def get_method_source_code_from_string(path, method_name, method_source_code):
    try:
        if method_name in method_source_code:
            return method_source_code[method_name]
        else:
            entity = get_entity_from_string(path)

            if inspect.isclass(entity):
                method = getattr(entity, method_name, None)
                if method:
                    source = inspect.getsource(method)
                    method_source_code[method_name] = source
                    return source
                else:
                    return "Method not found in class."

            elif inspect.isfunction(entity) and method_name == entity.__name__:
                source = inspect.getsource(entity)
                method_source_code[method_name] = source
                return source
            else:
                return "The specified path does not lead to a class or matched function."
    except Exception as e:
        return f"Error extracting source code of {path}: {str(e)}"


def insert_code(method_source_code, locs, insert_codes, param, value=None, target_api=None, type=False):
    if type:
        method_source_code = remove_single_line_comments(method_source_code)
        method_source_code = re.sub(r'\s+', ' ', method_source_code)
    modified_method_source_code = method_source_code
    for loc in locs:
        # Special case: insertion at the end of the function
        if loc == "bottom":
            insert_codes = [insert_codes] if isinstance(insert_codes, str) else insert_codes
            for insert_code in insert_codes:
                if type:
                    insert_code = remove_single_line_comments(insert_code)
                    insert_code = re.sub(r'\s+', ' ', insert_code)
                # Prepare the code for replacement and insertion
                if value is not None:
                    value = tuple_str_to_list_str(value)
                    new_code_line = re.sub(r'\b' + re.escape(param) + r'\b', str(value), insert_code)
                    # Construct new code lines, using spaces and special characters to ensure only the full variable name is replaced
                    # Handle replacement for variable names at the beginning of a line
                    if new_code_line.startswith(param + " "):
                        new_code_line = value + new_code_line[len(param):]
                    # Handle replacement for variable names at the end of a line
                    if new_code_line.endswith(" " + param):
                        new_code_line = new_code_line[:-len(param)] + value
                else:
                    new_code_line = insert_code
                modified_method_source_code = add_code_to_function_end(method_source_code, new_code_line)
            continue
        if type:
            loc = remove_single_line_comments(loc)
            loc = re.sub(r'\s+', ' ', loc)
        start_index = 0
        while True:
            # Find the location of loc
            start_index = method_source_code.find(loc, start_index)
            if start_index == -1:
                break

            insertion_point = start_index + len(loc)
            # Determine the indentation for inserting the code
            newline_index = method_source_code.rfind('\n', 0,
                                                     start_index) + 1  # Find the position of the previous newline character
            indentation_line = method_source_code[
                               newline_index:start_index]  # Get the string from the newline character to the found position

            indent_space = re.match(r'^[ \t]*', indentation_line).group() * 2

            insert_codes = [insert_codes] if isinstance(insert_codes, str) else insert_codes
            for insert_code in insert_codes:
                if type:
                    insert_code = remove_single_line_comments(insert_code)
                    insert_code = re.sub(r'\s+', ' ', insert_code)
                # Prepare the code for replacement and insertion
                if value is not None:
                    value = tuple_str_to_list_str(value)
                    new_code_line = re.sub(r'\b' + re.escape(param) + r'\b', str(value), insert_code)
                    # Construct new code lines, using spaces and special characters to ensure only the full variable name is replaced
                    # Handle replacement for variable names at the beginning of a line
                    if new_code_line.startswith(param + " "):
                        new_code_line = value + new_code_line[len(param):]
                    # Handle replacement for variable names at the end of a line
                    if new_code_line.endswith(" " + param):
                        new_code_line = new_code_line[:-len(param)] + value
                else:
                    new_code_line = insert_code

                # Add indentation and insert the new code line
                modified_method_source_code = (modified_method_source_code[:insertion_point] +
                                               '\n' + indent_space + new_code_line +
                                               modified_method_source_code[insertion_point:])
                insertion_point += len('\n' + indent_space + new_code_line)

            start_index = insertion_point  # Update start_index to avoid duplicate insertion

    return modified_method_source_code


class FunctionAdder(ast.NodeTransformer):
    def __init__(self, code_to_insert):
        self.code_to_insert = ast.parse(code_to_insert).body  # 解析要插入的代码行并获取第一条语句

    def visit_FunctionDef(self, node):
        node.body.extend(self.code_to_insert)
        return node


def adjust_code(code):
    # Split the code into lines
    lines = code.split('\n')
    # Find the first non-empty line and its indentation
    first_line_indent = len(lines[0]) - len(lines[0].lstrip())
    # Remove the same amount of indentation from all lines
    adjusted_lines = [line[first_line_indent:] if len(line) > first_line_indent else line for line in lines]
    # Join the lines back into a single string
    adjusted_code = '\n'.join(adjusted_lines)
    return adjusted_code


def add_code_to_function_end(source_code, code_to_insert):
    tree = ast.parse(adjust_code(source_code))
    modifier = FunctionAdder(code_to_insert)
    modified_tree = modifier.visit(tree)
    return astor.to_source(modified_tree)


def modify_code(method_source_code, locs, modify_codes, param, value=None, target_api=None, type=False):
    if type:
        method_source_code = remove_single_line_comments(method_source_code)
        method_source_code = re.sub(r'\s+', ' ', method_source_code)
    target_function_code = method_source_code
    for loc in locs:
        if type:
            loc = remove_single_line_comments(loc)
            loc = re.sub(r'\s+', ' ', loc)
        start_index = 0
        while True:
            # Find the location of loc
            start_index = target_function_code.find(loc, start_index)
            if start_index == -1:
                break

            # Determine the indentation level
            newline_index = target_function_code.rfind('\n', 0, start_index) + 1
            indentation = start_index - newline_index
            indent_space = ' ' * indentation

            end_index = start_index + len(loc)
            # Replace the found matching segment
            for modify_code in modify_codes:
                for old, new in modify_code.items():
                    if type:
                        old = remove_single_line_comments(old)
                        new = remove_single_line_comments(new)
                        old = re.sub(r'\s+', ' ', old)
                        new = re.sub(r'\s+', ' ', new)
                    if value is not None:
                        value = tuple_str_to_list_str(value)
                        new_code_line = re.sub(r'\b' + re.escape(param) + r'\b', str(value), new)
                        if new_code_line.startswith(param + " "):
                            new_code_line = value + new_code_line[len(param):]
                        if new_code_line.endswith(" " + param):
                            new_code_line = new_code_line[:-len(param)] + value
                    else:
                        new_code_line = new
                    # Ensure the new code maintains correct indentation
                    new_with_indent = new_code_line.replace('\n', '\n' + indent_space)
                    target_function_code = target_function_code[:start_index] + \
                                           target_function_code[start_index:end_index].replace(old, new_with_indent,
                                                                                               1) + \
                                           target_function_code[end_index:]
            start_index = end_index  # Update the search start position to avoid duplicate replacements

    return target_function_code


def remove_multiline_comments(code):
    """
    Remove triple-quote comments from Python code
    """
    # Use a regular expression to match triple-quote comments, including everything inside them
    pattern = r'\"\"\"(.*?)\"\"\"'
    # Use non-greedy mode (.*?) to ensure matching the shortest pair of triple quotes
    # The re.DOTALL parameter allows '.' to match any character, including newline characters
    cleaned_code = re.sub(pattern, '', code, flags=re.DOTALL)
    return cleaned_code


def python_replace_and_persist_modified_methods(target_api, modified_forward_codes, out_dir,
                                                targetDLClass="paddlepaddle"):
    imports, defs = extract_imports_and_defs_from_module(target_api, targetDLClass)
    code = ""
    if targetDLClass == "paddlepaddle":
        code = f"""import paddle \n"""
    elif targetDLClass == "pytorch":
        code = f"""import torch \n"""

    for importModule in imports:
        code += importModule + '\n'

    code += '\n\n'

    for function in defs:
        for _, method_code in modified_forward_codes.items():
            # why code.find()?
            method_code = remove_multiline_comments(method_code)
            if method_code.find(function.name) != -1 or code.find(function.name) != -1:
                code += astor.to_source(function).strip("\n") + '\n\n'
                break
    entity = get_entity_from_string(target_api)
    # If it's a class, get the source code of the methods
    if inspect.isclass(entity):
        code += f"""class Modified_{target_api.split(".")[-1]}({target_api}):\n"""
        for _, method_code in modified_forward_codes.items():
            # Add each line of the method's code, ensuring the indentation is correct
            method_code_indented = "\n".join([f"\t{line}" for line in method_code.split("\n")])
            code += method_code_indented + "\n\n"
    else:
        for _, method_code in modified_forward_codes.items():
            # Add each line of the method's code, ensuring the indentation is correct
            code += method_code.split("\n")[0].replace(target_api.split(".")[-1],
                                                       f"Modified_{target_api.split('.')[-1]}") + "\n"
            method_code_indented = "\n".join([f"\t{line}" for line in method_code.split("\n")[1:]])
            code += method_code_indented + "\n\n"

    # Ensure trailing extra blank lines are removed
    code = code.rstrip() + "\n\n"

    modified_methods_file = f"modified_{target_api.split('.')[-1]}.py"

    # If the file already exists, delete it.
    if os.path.exists(os.path.join(out_dir, modified_methods_file)):
        os.remove(os.path.join(out_dir, modified_methods_file))

    with open(os.path.join(out_dir, modified_methods_file), 'w') as file:
        file.write(code)

    return modified_methods_file, f"Modified_{target_api.split('.')[-1]}"


def c_replace_and_persist_modified_methods(backend, target_api, method_source_codes, out_dir,
                                           targetDLClass="paddlepaddle"):
    # import headers:
    code = f""""""
    for header in method_source_codes["headers"]:
        code += header + '\n'

    code += '\n\n'

    # set namespace
    for method_name, namespaces in method_source_codes["namespace"].items():
        for ns in namespaces:
            code += f'namespace {ns} {{\n'

        # set functionContext
        for method in method_source_codes["funcContext"][method_name]:
            code += method + f"\n\n"

        # inject modified Code
        code += method_source_codes[method_name] + f"\n\n"

        for ns in reversed(namespaces):
            code += f'}} // namespace {ns}\n'
        code += '\n'

    os_name = platform.system().lower()
    if "cpu" in backend.lower():
        if "windows" in os_name:
            modified_methods_file = f"modified_{target_api.split('.')[-1]}.cpp"
        else:
            modified_methods_file = f"modified_{target_api.split('.')[-1]}.cc"
    elif "gpu" in backend.lower() or "cuda" in backend.lower():
        modified_methods_file = f"modified_{target_api.split('.')[-1]}.cu"
    else:
        raise EnvironmentError("not supported such backend.")

    # If the file already exists, delete it.
    if os.path.exists(os.path.join(out_dir, modified_methods_file)):
        os.remove(os.path.join(out_dir, modified_methods_file))

    with open(os.path.join(out_dir, modified_methods_file), 'w') as file:
        file.write(code)

    # compile modified_methods_file
    generate_cmake_lists(backend, modified_methods_file, f"modified_{target_api.split('.')[-1]}", out_dir,
                         targetDLClass)

    so_file = compile_so(out_dir)

    # delete CMakeLists.txt and xxxx.cu
    os.remove(os.path.join(out_dir, 'CMakeLists.txt'))
    os.remove(os.path.join(out_dir, modified_methods_file))

    return so_file


def extract_imports_and_defs_from_module(module_path, targetDLClass="paddlepaddle"):
    """
    Extract import statements and function definitions from the specified module.
    """
    # Split the string to get the module path and class name

    module_path, class_name = module_path.rsplit('.', 1)

    # Dynamically import the target module
    module = importlib.import_module(module_path)

    target_class = getattr(module, class_name, None)
    if target_class is None:
        raise ImportError(f"{class_name} not found in {module_path}")

    module_file = inspect.getfile(target_class)
    with open(module_file, 'r', encoding='utf-8') as file:
        source_code = file.read()

    parsed = ast.parse(source_code)

    imports = [node for node in parsed.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    imports = convert_imports_to_absolute(imports, module_file, None, targetDLClass=targetDLClass)
    defs = [node for node in parsed.body if isinstance(node, ast.FunctionDef)]

    return imports, defs


def convert_imports_to_absolute(imports, module_file, project_root=None, targetDLClass="paddlepaddle"):
    """
    Convert relative imports to absolute imports.

    :param imports: List of extracted import statement nodes.
    :param module_file: Path of the file containing the operator.
    :param project_root: Path of the project root directory, used to determine the package path of the module.
    :returns: List of code strings with converted import statements.
    """
    # Determine the full package path of the module
    if targetDLClass == "paddlepaddle":
        targetDLClass = "paddle"
    elif targetDLClass == "pytorch":
        targetDLClass = "torch"
    relative_path = os.path.relpath(module_file, project_root if project_root else find_project_root(module_file, top_package_name=targetDLClass))
    package_path = os.path.splitext(relative_path)[0].replace(os.sep, '.')

    converted_imports = []
    for node in imports:
        if isinstance(node, ast.ImportFrom):
            # 生成绝对导入路径
            if node.level > 0:
                # 根据相对级别调整包路径
                parts = package_path.split('.')
                base_path = '.'.join(parts[:-node.level])
                module_path = (base_path + '.' + node.module) if node.module else base_path
                new_import = f"from {module_path} import " + ', '.join([f"{alias.name} as {alias.asname}" if alias.asname else alias.name for alias in node.names])
                converted_imports.append(new_import)
            else:
                converted_imports.append(ast.unparse(node))
        else:
            converted_imports.append(ast.unparse(node))

    return converted_imports


def find_project_root(module_file, top_package_name="paddle"):
    """
    Infer the project root directory by locating the specified top-level package name in the path.

    :param module_file: The full path of the module file.
    :param top_package_name: The top-level package name used in the project.
    :return: The inferred project root directory path.
    """
    # Split the module file path into parts
    parts = module_file.split(os.sep)

    # Attempt to find the index of the top-level package name in the split path parts
    if top_package_name in parts:
        top_index = parts.index(top_package_name)

        # The project root directory is the result of concatenating all path parts before the top-level package name
        project_root = os.sep.join(parts[:top_index])
        return project_root
    else:
        raise ValueError(f"Top package name '{top_package_name}' not found in the module file path.")


def detect_backend(backend):
    if "cpu" in backend:
        return "CPU"
    elif any(element in backend.lower() for element in ["gpu", "cuda"]):
        return "GPU"
    elif any(element in backend.lower() for element in ["tpu"]):
        return "TPU"


def generate_cmake_lists(backend, source_file, executable_name, out_dir, targetDLClass="paddlepaddle"):
    """
        Dynamically Generate the content for CMakeLists.txt.

        Parameters:
        - cmake_min_version: Minimum version of CMake.
        - cxx_standard: C++ standard version used.
        - paddle_include_dirs: Directories containing PaddlePaddle header files.
        - paddle_library_dirs: Directories containing PaddlePaddle library files.
        - source_files: List of source files.
        - executable_name: Name of the generated executable file.
        - paddle_libraries: List of PaddlePaddle libraries to link.
        """

    if "paddle" in targetDLClass.lower():
        with open(os.path.join(os.getcwd(), os.path.join("CMC", "paddle_CMakeLists_template.txt")), 'r',
                  encoding="utf-8") as file:
            template = file.read()
        try:
            import paddle
        except Exception as e:
            raise EnvironmentError("unsupported PaddlePaddle.")
        if "GPU" in backend:
            template = template.replace("project()", f"project({executable_name} LANGUAGES CUDA CXX)")
            template = template.replace("set(CMAKE_{}_STANDARD 14)", "set(CMAKE_CUDA_STANDARD 14)")
        else:
            template = template.replace("project()", f"project({executable_name})")
            template = template.replace("set(CMAKE_{}_STANDARD 14)", "set(CMAKE_CXX_STANDARD 14)")
        template = template.replace("set(CONDA_PREFIX {})", f"set(CONDA_PREFIX \"{os.getenv('CONDA_PREFIX')}\")")
        template = template.replace("set(PADDLE_INCLUDE_DIR {})",
                                    f"set(PADDLE_INCLUDE_DIR \"{paddle.sysconfig.get_include()}\")")
        template = template.replace("set(PADDLE_LIB_DIR {})", f"set(PADDLE_LIB_DIR \"{paddle.sysconfig.get_lib()}\")")
        if "windows" in platform.system().lower():
            template = template.replace("set(PADDLE_LIBRARIES {})",
                                        f"set(PADDLE_LIBRARIES \"{os.path.join(os.path.dirname(paddle.__file__), 'base', 'libpaddle.pyd')}\")")
        else:
            template = template.replace("set(PADDLE_LIBRARIES {})",
                                        f"set(PADDLE_LIBRARIES \"{os.path.join(os.path.dirname(paddle.__file__), 'fluid', 'libpaddle.so')}\")")
        if "GPU" in backend:
            template = template.replace("add_library({} SHARED {})",
                                        f"add_library({executable_name} SHARED {source_file})")
            cuda_flags = "--expt-relaxed-constexpr"
            insertion_point = f"add_library({executable_name} SHARED {source_file})"
            if insertion_point in template:
                # 在 'add_library' 调用后添加 CUDA 编译选项
                template = template.replace(insertion_point,
                                            f"{insertion_point}\nset(CMAKE_CUDA_FLAGS {cuda_flags})")
            template = template.replace("set_target_properties({} PROPERTIES COMPILE_FLAGS '-fPIC')",
                                        f"set_target_properties({executable_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)\ntarget_compile_definitions({executable_name} PRIVATE PADDLE_WITH_CUDA)")

        else:
            template = template.replace("add_library({} SHARED {})",
                                        f"add_library({executable_name} SHARED {source_file})")
            template = template.replace("set_target_properties({}", f"set_target_properties({executable_name}")
        template = template.replace("target_link_libraries({}", f"target_link_libraries({executable_name}")
    elif "torch" in targetDLClass.lower():
        with open(os.path.join(os.getcwd(), os.path.join("CMC", "pytorch_CMakeLists_template.txt")), 'r',
                  encoding="utf-8") as file:
            template = file.read()
        try:
            import torch
            from torch.utils.cpp_extension import include_paths
        except Exception as e:
            raise EnvironmentError("unsupported PyTorch.")
        if "GPU" in backend:
            template = template.replace("project()", f"project({executable_name} LANGUAGES CUDA CXX)")
            template = template.replace("set(CMAKE_{}_STANDARD 14)", "set(CMAKE_CUDA_STANDARD 14)")
        else:
            template = template.replace("project()", f"project({executable_name})")
            template = template.replace("set(CMAKE_{}_STANDARD 14)", "set(CMAKE_CXX_STANDARD 14)")
        template = template.replace("set(CONDA_PREFIX {})", f"set(CONDA_PREFIX \"{os.getenv('CONDA_PREFIX')}\")")
        template = template.replace("set(TORCH_INCLUDE_DIR {})",
                                    f"set(TORCH_INCLUDE_DIR \"{include_paths()[0]}\")")
        template = template.replace("set(TORCH_LIB_DIR {})", f"set(TORCH_LIB_DIR \"{os.path.join(os.path.dirname(torch.__file__), 'lib')}\")")
        if "windows" in platform.system().lower():
            template = template.replace("set(TORCH_LIBRARIES {})",
                                        f"set(TORCH_LIBRARIES \"{os.path.join(os.path.dirname(torch.__file__), 'lib', 'torch.dll')}\")")
        else:
            template = template.replace("set(TORCH_LIBRARIES {})",
                                        f"set(TORCH_LIBRARIES \"{os.path.join(os.path.dirname(torch.__file__), 'lib', 'libtorch.so')}\")")
        if "GPU" in backend:
            template = template.replace("add_library({} SHARED {})",
                                        f"add_library({executable_name} SHARED {source_file})")
            cuda_flags = "--expt-relaxed-constexpr"
            insertion_point = f"add_library({executable_name} SHARED {source_file})"
            if insertion_point in template:
                # 在 'add_library' 调用后添加 CUDA 编译选项
                template = template.replace(insertion_point,
                                            f"{insertion_point}\nset(CMAKE_CUDA_FLAGS {cuda_flags})")
            template = template.replace("set_target_properties({} PROPERTIES COMPILE_FLAGS '-fPIC')",
                                        f"set_target_properties({executable_name} ROPERTIES COMPILE_FLAGS '-fPIC')")

        else:
            template = template.replace("add_library({} SHARED {})",
                                        f"add_library({executable_name} SHARED {source_file})")
            template = template.replace("set_target_properties({}", f"set_target_properties({executable_name}")
        template = template.replace("target_link_libraries({}", f"target_link_libraries({executable_name}")

    else:
        raise ValueError("Only the target DL Framework for paddle is supported")
    # write new CMakeLists.txt
    with open(os.path.join(out_dir, "CMakeLists.txt"), 'w') as file:
        file.write(template)


def compile_so(out_dir):
    build_dir = os.path.join(out_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    os.chdir(build_dir)

    # Copy the current environment variables
    env = os.environ.copy()
    # If needed, you can modify env here to add or modify environment variables

    # Run CMake configuration
    cmake_command = ["cmake", ".."]
    result = subprocess.run(cmake_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode != 0:
        raise Exception("CMake configuration failed:\n" + result.stderr.decode())

    # Build the project
    make_command = ["make"]
    result = subprocess.run(make_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    if result.returncode != 0:
        raise Exception("Compilation failed:\n" + result.stderr.decode())

    # Find the generated .so file
    so_files = [f for f in os.listdir(build_dir) if f.endswith('.so')]
    if not so_files:
        raise FileNotFoundError("No .so file found in the build directory.")

    # Return the full path of the .so file
    return os.path.join(build_dir, so_files[0])


def get_filename(filepath):
    base_name = os.path.basename(filepath)

    filename, _ = os.path.splitext(base_name)

    return filename


def tuple_str_to_list_str(s):
    try:
        result = ast.literal_eval(s)
        if isinstance(result, tuple):
            return str(list(result))
        else:
            return result
    except:
        return s


def remove_outer_parentheses(code_str):
    """Remove outer parentheses from a code string if they enclose the entire string."""
    code_str = code_str.strip()
    if code_str.startswith('(') and code_str.endswith(')'):
        # Strip one layer of parentheses
        inner_str = code_str[1:-1].strip()
        # Check if removing parentheses breaks the expression (counting matching pairs)
        if inner_str.count('(') == inner_str.count(')'):
            # Try to parse it to ensure it's still a valid Python expression
            try:
                ast.parse(inner_str)
                return inner_str
            except SyntaxError:
                return code_str  # Return original if parsing fails
    return code_str


def remove_single_line_comments(code):
    # Use a regular expression to remove all comments starting with // up to the end of the line
    # Regular expression explanation: '//.*?$' matches the string from // to the end of the line, with re.MULTILINE mode handling multiple lines
    cleaned_code = re.sub(r'//.*?$', '', code, flags=re.MULTILINE)
    return cleaned_code

def paddle_param_change(kwargs, params_change):
    new_kwargs = {}
    for k in list(kwargs.keys()):
        if isinstance(kwargs[k], str) and "self." in kwargs[k]:
            self_args = kwargs[k].split("self.")
            new_args = [self_args[0]]
            for self_arg in self_args[1:]:
                if self_arg.startswith("bias"):
                    self_arg.replace("bias", "_bias_attr")
                elif self_arg.startswith("weight"):
                    self_arg.replace("weight", "_weight_attr")
                elif self_arg.startswith("eps"):
                    self_arg = self_arg.replace("eps", "_epsilon")
                elif self_arg.startswith("stride"):
                    self_arg = self_arg.replace("stride", "_stride")
                elif self_arg.startswith("kernel_size"):
                    self_arg = self_arg.replace("kernel_size", "_kernel_size")
                elif self_arg.startswith("padding"):
                    self_arg = self_arg.replace("padding", "_padding")
                elif self_arg.startswith("normalized_shape"):
                    self_arg = f"_{self_arg}"
                new_args.append(self_arg)
            kwargs[k] = "self.".join(new_args)

        if k in params_change:
            if params_change[k]:
                # rename/copy in new_kwargs
                if isinstance(params_change[k], list):
                    if k == "affine" or k == "elementwise_affine":
                        if kwargs[k] == "(False)":
                            for param in params_change[k]:
                                new_kwargs[param] = kwargs[k]
                        else:
                            kwargs.pop(k)
                    else:
                        for v in params_change[k]:
                            if v in new_kwargs and isinstance(new_kwargs[v], list):
                                new_kwargs[v].append(kwargs[k])
                            elif v in new_kwargs:
                                new_kwargs[v] = [new_kwargs[v], kwargs[k]]
                            else:
                                new_kwargs[v] = kwargs[k]
                else:
                    # torch_to_paddle: bias-> bias_attr
                    if k == "bias":
                        if kwargs[k] == "(False)":
                            if isinstance(params_change[k], str):
                                new_kwargs[params_change[k]] = False
                            else:
                                for param in params_change[k]:
                                    new_kwargs[param] = False
                        elif kwargs[k] == "(True)":
                            kwargs.pop(k)
                        else:
                            new_kwargs[params_change[k]] = kwargs[k]
                    # torch_to_paddle: dim-> axis
                    elif k == "dim":
                        if kwargs[k] is None:
                            kwargs.pop(k)
                        else:
                            new_kwargs[params_change[k]] = kwargs[k]
                    elif k == "bidirectional":
                        if kwargs[k] == "(True)":
                            new_kwargs[params_change[k]] = 'bidirect'
                        else:
                            kwargs.pop(k)
                    #  parameter in torchvision should be converted
                    elif k == "version":
                        new_kwargs[k] = kwargs.pop(k).replace(".", "_")
                    elif k == "weights" and kwargs.pop(k) is not None:
                        new_kwargs[params_change[k]] = True
                    else:
                        new_kwargs[params_change[k]] = kwargs[k]
            else:
                # remove in new_kwargs
                kwargs.pop(k)
        else:
            # copy to new_kwargs
            new_kwargs[k] = kwargs.pop(k)

            # common process for some args
            if k in [
                "layout",
                "device",
                "memory_format",
                "inplace",
                "generator",
                "non_blocking",
                "async",
                "proj_size",
                "init_weights",
                "memory_efficient",
                "batch_first",
                "process",
                "gain",
                "mask_check",
            ]:
                new_kwargs.pop(k)
                continue

            if k == "momentum":
                new_kwargs[k] = "1.0 - " + new_kwargs.pop(k)
                continue
    return new_kwargs


def pytorch_param_change(kwargs, params_change):
    new_kwargs = {}
    for k in list(kwargs.keys()):
        if k in params_change:
            if params_change[k]:
                # rename/copy in new_kwargs
                if isinstance(params_change[k], list):
                    for v in params_change[k]:
                        if v in new_kwargs and isinstance(new_kwargs[v], list):
                            new_kwargs[v].append(kwargs[k])
                        elif v in new_kwargs:
                            new_kwargs[v] = [new_kwargs[v], kwargs[k]]
                        else:
                            new_kwargs[v] = kwargs[k]
                else:
                    # torch_to_paddle: bias-> bias_attr
                    if params_change[k] not in new_kwargs and (k == "weight_attr" or k == "bias_attr" or k == "bias_ih_attr" or k == "bias_hh_attr"):
                        if kwargs[k] == "(False)":
                            if isinstance(params_change[k], str):
                                new_kwargs[params_change[k]] = False
                            else:
                                for param in params_change[k]:
                                    new_kwargs[param] = False
                        else:
                            new_kwargs[k] = kwargs.pop(k)
                    # paddle_to_torch: axis -> dim
                    elif k == "axis":
                        if kwargs[k] is None:
                            kwargs.pop(k)
                        else:
                            new_kwargs[params_change[k]] = kwargs[k]
                    elif k == "direction":
                        if kwargs[k] == "bidirect":
                            new_kwargs[params_change[k]] = True
                        else:
                            kwargs.pop(k)
                    #  parameter in torchvision should be converted
                    elif k == "version":
                        new_kwargs[k] = kwargs.pop(k).replace(".", "_")
                    elif k == "weights" and kwargs.pop(k) is not None:
                        new_kwargs[params_change[k]] = True
                    else:
                        new_kwargs[params_change[k]] = kwargs[k]
            else:
                # remove in new_kwargs
                kwargs.pop(k)
        else:
            # copy to new_kwargs
            new_kwargs[k] = kwargs.pop(k)

            # common process for some args
            if k in [
               "name",
               "data_format",
               "mode",
               "output_size"
            ]:
                new_kwargs.pop(k)
                continue

            if k == "momentum":
                new_kwargs[k] = "1.0 - " + new_kwargs.pop(k)
                continue

    return new_kwargs

def select_paddle_operator(op, a, b, targetDLClass):
    if op == '+':
        return r"{}(paddle.to_tensor({},dtype='float32'), paddle.to_tensor({},dtype='float32'))".format(
            addOp[targetDLClass], a, b)
    elif op == '-':
        return r"{}(paddle.to_tensor({},dtype='float32'), paddle.to_tensor({},dtype='float32'))".format(
            subOp[targetDLClass], a, b)
    elif op == '*':
        return r"{}(paddle.to_tensor({},dtype='float32'), paddle.to_tensor({},dtype='float32'))".format(
            mulOp[targetDLClass], a, b)
    elif op == '/':
        return r"{}(paddle.to_tensor({},dtype='float32'), paddle.to_tensor({},dtype='float32'))".format(
            divOp[targetDLClass], a, b)
    else:
        raise ValueError("Operator {} not supported".format(op['value']))


def select_pytorch_operator(op, a, b, targetDLClass):
    if op == '+':
        return r"{}(torch.tensor({}, dtype=torch.float32), torch.tensor({}, dtype=torch.float32))".format(
            addOp[targetDLClass], a, b)
    elif op == '-':
        return r"{}(torch.tensor({}, dtype=torch.float32), torch.tensor({}, dtype=torch.float32))".format(
            subOp[targetDLClass], a, b)
    elif op == '*':
        return r"{}(torch.tensor({}, dtype=torch.float32), torch.tensor({}, dtype=torch.float32))".format(
            mulOp[targetDLClass], a, b)
    elif op == '/':
        return r"{}(torch.tensor({}, dtype=torch.float32), ptorch.tensor({}, dtype=torch.float32))".format(
            divOp[targetDLClass], a, b)
    else:
        raise ValueError("Operator {} not supported".format(op['value']))