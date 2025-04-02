import collections, sys, datetime
import logging, os, shutil
import ast, astor

from CMC.codeGeneration import commonGenerateHelper
from CMC.astAnalysis import astTree
from CMC.utils import log_info, get_filename
from CMC.special_mapping import SUPPORT_PACKAGE_LIST


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith("."):
            yield f


def mark_unsupport(code, sourceDLClass):
    lines = code.split("\n")
    mark_next_line = False
    in_str = False
    for i, line in enumerate(lines):
        # torch.* in __doc__
        # torch.* in str
        if line.count('"""') % 2 != 0:
            in_str = not in_str

        tmp_line = re.sub(r"[\'\"]{1}[^\'\"]+[\'\"]{1}", "", line)
        if in_str:
            continue

        if "Class Method:" in line or "Tensor Attribute:" in line:
            mark_next_line = True
            continue
        else:
            # func decorator_list: @
            if mark_next_line and line != "@":
                lines[i] = ">>>" + line
                mark_next_line = False
                continue

        for package in SUPPORT_PACKAGE_LIST[sourceDLClass]:
            if tmp_line.startswith("%s." % package):
                lines[i] = "# " + line
                break

            if re.match(r".*[^A-Za-z_]{1}%s\." % package, tmp_line):
                lines[i] = ">>>" + line

    return "\n".join(lines)


import re


def format_sequential(code):
    pattern = re.compile(r'([ \t]*)([\w\.]*\s*=\s*)?.*?Sequential\((?:[^()]|\([^)]*\))+\)')

    formatted_code = ""
    last_idx = 0  # Track the last index processed

    for seq_block in re.finditer(pattern, code):
        start, end = seq_block.span()
        seq_code = seq_block.group(0)
        indent_space = seq_block.group(1)  # Capture leading spaces or tabs for indentation

        # Add the previous code (unformatted section)
        formatted_code += code[last_idx:start]

        # Format the internal code
        inner_code_start = seq_code.find('(') + 1
        inner_code_end = seq_code.rfind(')')
        inner_code = seq_code[inner_code_start:inner_code_end].strip()

        # 处理inner_code中不必要的换行
        inner_code = re.sub(r'\s*\n\s*', ' ', inner_code)
        # Remove extra spaces
        inner_code = re.sub(r'\s+', ' ', inner_code)

        inner_parts = re.split(r',\s*(?![^()]*\))', inner_code)

        # Add indentation and line breaks to each segmented part, with an additional 4 spaces of indentation for
        # internal code
        formatted_inner = (",\n" + indent_space + '    ').join(inner_parts)

        # Reassemble the entire expression
        formatted_seq_code = seq_code[
                             :inner_code_start] + '\n' + indent_space + '    ' + formatted_inner + '\n' + indent_space + seq_code[
                                                                                                                         inner_code_end:]

        # Remove all spaces following a dot
        formatted_seq_code = re.sub(r'\.\s+', '.', formatted_seq_code)

        formatted_code += formatted_seq_code

        last_idx = end  # Update the last processed index

    # Add the final part of the code (if any)
    formatted_code += code[last_idx:]

    return formatted_code


class Converter:
    def __init__(self, in_dir, out_dir=None, log_dir=None, show_unsupport=False, sourceDLClass="PyTorch",
                 targetDLClass="PaddlePaddle"):
        self.source_library_api_count = 0
        self.success_api_count = 0
        self.possible_modules = collections.defaultdict(dict)
        self.unsupport_map = collections.defaultdict(int)
        self.show_unsupport = show_unsupport
        self.log_dir = log_dir
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.sourceDLClass = sourceDLClass
        self.targetDLClass = targetDLClass

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

        self.logger = logging.getLogger(name=get_unique_name("{}to{}Converter".format(sourceDLClass, targetDLClass)))
        self.logger.propagate = False
        if log_dir == "disable":
            logging.disable(logging.CRITICAL + 1)
        else:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            self.logger.addHandler(logging.StreamHandler())
            self.logger.addHandler(logging.FileHandler(os.path.join(log_dir,
                                                                    f"log_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{get_filename(self.in_dir)}", ),
                                                       mode="w"))

    def run(self):
        in_dir = os.path.abspath(self.in_dir)
        global transpose_num
        if self.out_dir is None:
            self.out_dir = self.in_dir.replace("sourceModels", "targetModels")
            # Remove the '.py' file extension if the path ends with '.py'
            if self.out_dir.endswith(".py"):
                self.out_dir = self.out_dir[:-3]
            if not os.path.exists(self.out_dir):
                os.makedirs(self.out_dir)
        else:
            if os.path.isfile(self.out_dir):
                # If it's a file, get the absolute path of the directory where the file is located
                self.out_dir = os.path.abspath(os.path.dirname(self.out_dir))
            elif os.path.isdir(self.out_dir):
                # If it's already a directory, directly get the absolute path of the directory
                self.out_dir = os.path.abspath(self.out_dir)
            else:
                raise ValueError("The input 'out_dir' must be a file or directory!")

        if self.out_dir.endswith(".py"):
            commonGenerateHelper(os.path.join(os.path.dirname(self.out_dir), "info", "codeGenerationInfo.py"))
        else:
            commonGenerateHelper(os.path.join(self.out_dir, "info", "codeGenerationInfo.py"))

        assert self.out_dir != in_dir, "--out_dir must be different from --in_dir"

        self.convert_files(self.in_dir, self.out_dir)
        failed_api_count = 0
        unsupport_map = sorted(
            self.unsupport_map.items(), key=lambda x: x[1], reverse=True
        )
        for k, v in unsupport_map:
            failed_api_count += v
        if self.show_unsupport:
            log_info(self.logger, "\n===========================================")
            log_info(self.logger, "Not Support API List:")
            log_info(self.logger, "===========================================")
            log_info(
                self.logger,
                "These APIs are not supported to be converted !\n",
            )
            for k, v in unsupport_map:
                log_info(self.logger, "{}: {}".format(k, v))

        success_api_count = self.source_library_api_count - failed_api_count
        log_info(self.logger, "\n===========================================")
        log_info(self.logger, "Convert Summary:")
        log_info(self.logger, "===========================================")
        # Calculate the success rate
        if self.source_library_api_count > 0:
            success_rate = (success_api_count / self.source_library_api_count) * 100
        else:
            success_rate = 0

        # Formatting the message
        log_info(self.logger, f"Total APIs in source library: {self.source_library_api_count}\n")
        log_info(self.logger, f"APIs successfully converted: {success_api_count}\n")
        log_info(self.logger, f"APIs failed to convert: {failed_api_count}\n")
        log_info(self.logger, f"Success rate: {success_rate:.2f}%")

        log_info(self.logger, "out_dir :{}".format(str(self.out_dir)))
        return success_api_count, failed_api_count

    def convert_files(self, in_dir, out_dir, exclude_dir_list=None, *args, **kwargs):
        """
        :param in_dir: Source framework model-related files
        :param out_dir: Target framework model-related files
        :param exclude_dir_list: Model files to be excluded from conversion
        :return: None
        """

        if not os.path.exists(in_dir):
            raise ValueError("The input 'in_dir' must be an existing file or directory!")

        if not exclude_dir_list:
            exclude_dir_list = []

        if os.path.isfile(in_dir):
            if in_dir not in exclude_dir_list and not any(
                    in_dir.startswith(exclude_dir + "/") for exclude_dir in exclude_dir_list):
                if os.path.exists(out_dir):
                    shutil.rmtree(out_dir)
                os.makedirs(out_dir)
                self.out_dir = out_dir
                self.convert_file(in_dir, os.path.join(out_dir, os.path.basename(in_dir)))

        elif os.path.isdir(in_dir):
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for item in os.listdir(in_dir):
                old_path = os.path.join(in_dir, item)
                if not os.path.exists(os.path.join(out_dir, os.path.basename(in_dir))):
                    os.makedirs(os.path.join(out_dir, os.path.basename(in_dir)))
                new_path = os.path.join(out_dir, os.path.basename(in_dir), item)
                if any(old_path == exclude_dir or old_path.startswith(exclude_dir + "/") for exclude_dir in
                       exclude_dir_list):
                    continue

                if os.path.isdir(old_path) or os.path.isfile(old_path):
                    self.convert_file(old_path, new_path)

    def convert_file(self, old_path, new_path):
        sys.path.append(self.out_dir)
        if old_path.endswith(".py"):
            log_info(self.logger, "Start convert {} --> {}".format(old_path, new_path))
            with open(old_path, "r", encoding="UTF-8") as f:
                code = f.read()
                root = ast.parse(code)

            # give ast of the file
            fileAst = astTree(root, old_path, self.out_dir, self.possible_modules, self.unsupport_map, self.logger, self.sourceDLClass, self.targetDLClass)
            fileAst.analyze_ast_node()
            self.source_library_api_count += fileAst.source_library_api_count
            code = astor.to_source(root)
            code = format_sequential(code)
            code = mark_unsupport(code, self.sourceDLClass)

            with open(new_path, "w", encoding="UTF-8") as file:
                file.write(code)
            log_info(
                self.logger, "Finish convert {} --> {}\n".format(old_path, new_path)
            )
        else:
            log_info(
                self.logger,
                "No need to convert, just Copy {} --> {}\n".format(old_path, new_path),
            )
            try:
                shutil.copyfile(old_path, new_path)
            except Exception as e:
                pass
