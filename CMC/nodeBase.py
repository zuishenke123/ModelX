import os, collections, ast
import re
import sys
import textwrap

import astor

from CMC.utils import CodeFileGenerator, get_operator_parameters, log_info, \
    get_method_source_code_from_string, \
    python_replace_and_persist_modified_methods, c_replace_and_persist_modified_methods, \
    insert_code, get_unique_name, modify_code, detect_backend, search_mapping_api

global transpose_num
transpose_num = 0


class astNodes(ast.NodeTransformer):
    def __init__(self, root, node_stack, import_map=None):
        self.root = root
        self.node_stack = node_stack
        self.black_list = []
        self.import_map = import_map

    def visit(self, node):
        self.node_stack.append(node)
        method = 'Handle' + node.__class__.__name__
        # Common node types are handled by a generic method
        if isinstance(node, (ast.While, ast.For, ast.Try, ast.With, ast.If)):
            method = 'HandleCommonNodes'
        visitor = getattr(self, method, self.generic_visit)
        node = visitor(node)
        self.node_stack.pop()
        return node

    @property
    def parent_node(self):
        return self.node_stack[-2] if len(self.node_stack) > 1 else None

    def get_full_attr(self, node):
        # ast.func --> ast.Attribute: value.attr
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + "." + node.attr

        elif isinstance(node, ast.Name):

            node_str = astor.to_source(node).strip("\n")
            return node.id
        elif isinstance(
                node,
                (ast.Call, ast.Compare, ast.BinOp, ast.BoolOp, ast.UnaryOp, ast.Subscript, ast.Assert),
        ):
            """
                1. torch.abs(x).transpose(1, 0) -> 'DLClass'
                2. (x == y).transpose(1, 0) -> 'DLClass'
                3. (x + y).transpose(1, 0) -> 'DLClass'
                4. (!X).transpose(1, 0) -> ‘DLClass’
                5. x[0].transpose(1, 0) -> 'DLClass'
                6. (-x).transpose(1, 0) -> 'DLClass'
            """
            node_str = astor.to_source(node).strip("\n")
            for item in self.black_list:
                # (array(1.) + array(2.)).abs() ...
                if re.match(".*[^A-Za-z_]{1}%s\(" % item, node_str):
                    return "unDLClass"
                # np.array(1.).abs() ...
                if re.match("%s\." % item, node_str):
                    return "unDLClass"
                # array(1.).abs() ...
                if re.match("%s\(" % item, node_str):
                    return "unDLClass"
            # getattr() Dynamic attribute or method
            if "getattr" in node_str:
                return "PotentialUnDLClass"
            return "DLClass"
        else:
            return "unDLClass"

    def find_node_in_scope_parts(self, scope_node, target_node, parts):
        for part_name in parts:
            part = getattr(scope_node, part_name, None)
            if part:
                for index, node in enumerate(part):
                    if node == target_node:
                        return scope_node, part_name, index
        return None


class nodeContext:
    def __init__(self, file, logger):
        self.scope_stack = []
        self.scope_insert_lines = collections.defaultdict(dict)
        self.logger = logger
        self.file = file
        self.file_name = os.path.basename(file)

    def insert_scope(self):
        for scope_node in self.scope_insert_lines:
            for body in self.scope_insert_lines[scope_node]:
                insert_lines = self.scope_insert_lines[scope_node][body]
                insert_lines = sorted(
                    insert_lines.items(), key=lambda x: x[0], reverse=True
                )
                for index, lines in insert_lines:
                    log_info(
                        self.logger,
                        "insert extra {} lines".format(len(lines)),
                        self.file_name,
                    )
                    for line in lines[::-1]:
                        getattr(scope_node, body).insert(index, line)

    def record_scope(self, scope_info, nodes_to_insert):
        """
        Records a list of nodes to be inserted into a specific scope.

        Args:
            scope_info (tuple): A tuple containing information about the scope.
            nodes_to_insert (list of ast.Node): A list of AST nodes to insert into the scope.
        """

        if not isinstance(nodes_to_insert, list):
            nodes_to_insert = [nodes_to_insert]


        if len(nodes_to_insert) == 0:
            return

        scope_node, body, index = scope_info


        if scope_node not in self.scope_insert_lines:
            self.scope_insert_lines[scope_node] = collections.defaultdict(dict)


        existing_nodes = {ast.dump(ele) for ele in self.scope_insert_lines[scope_node][body].get(index, [])}


        unique_nodes = [node for node in nodes_to_insert if ast.dump(node) not in existing_nodes]


        self.scope_insert_lines[scope_node][body].setdefault(index, []).extend(unique_nodes)


class astBaseNodes(astNodes, nodeContext):
    def __init__(self, root, file, possible_modules, unsupport_map, logger, sourceDLClass="pytorch", targetDLClass="paddlepaddle"):
        self.node_stack = []
        self.sourceDLClass = sourceDLClass
        self.targetDLClass = targetDLClass
        self.converting_api_count = 0
        self.success_api_count = 0
        self.filed_api_count = 0
        self.unsupport_api_map = unsupport_map
        self.classInstance = {}
        self.functionBase = None
        super(astBaseNodes, self).__init__(root, self.node_stack, possible_modules)
        nodeContext.__init__(self, file, logger)

    def transform(self):
        self.visit(self.root)
        self.insert_scope()

    def get_full_api_from_node(self, node, file=None):
        if file is None:
            file = self.file
        full_attr = self.get_full_attr(node)
        if not full_attr:
            return None

        parts = full_attr.split(".")
        module_name = parts[0]

        if module_name in self.import_map[file]:
            new_module = self.import_map[file][module_name]
            parts[0] = new_module
            return ".".join(parts)

        return None

    def scope_body_index(self, level=-1):
        scope_node = self.scope_stack[level]

        # reverse find scope_node in node_stack
        lower = -1 * (len(self.node_stack) + 1)
        for i in range(-1, lower, -1):
            if self.node_stack[i] == scope_node:
                for index, node in enumerate(scope_node.body):
                    if node == self.node_stack[i + 1]:
                        return scope_node, "body", index

                if getattr(scope_node, "orelse", None):
                    for index, node in enumerate(scope_node.orelse):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "orelse", index

                if getattr(scope_node, "decorator_list", None):
                    for index, node in enumerate(scope_node.decorator_list):
                        if node == self.node_stack[i + 1]:
                            return scope_node, "decorator_list", index

        return self.scope_body_index(-2)

    def insert_before_target(self, node, new_node):
        """Insert a new_node before the target node."""
        # Get the position of the target node in the parent's child list
        target_index = self.parent_node.body.index(node)
        # Insert a new node before the target node
        self.parent_node.body.insert(target_index, new_node)

    def insert_multi_node(self, node_list):
        if len(node_list) == 0:
            return
        import_nodes = []
        other_nodes = []
        for node in node_list:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.append(node)
            elif "sys.path" in astor.to_source(node):
                import_nodes.append(node)
            else:
                other_nodes.append(node)

        self.record_scope((self.root, "body", 0), import_nodes)
        if len(other_nodes) > 0:
            self.record_scope(self.scope_body_index(), other_nodes)


class BaseMatcher:
    def __init__(self, transformer, source_api, api_mapping, logger, lr = None, out_dir=None,
                 sourceDLClass="pytorch", targetDLClass="paddlepaddle"):
        self.transformer = transformer
        self.source_api = source_api
        self.sourceDLClass = sourceDLClass
        self.targetDLClass = targetDLClass
        self.target_api = None
        self.api_mapping = api_mapping
        self.logger = logger
        self.args = None
        self.kwargs = None
        self.init_val = None
        self.persisted_target_apis = []
        self.out_dir = out_dir
        self.lr = lr

    def get_aux_dir(self):
        return os.path.dirname(CodeFileGenerator().file_name)

    def get_target_api(self):
        # existing target_api
        if self.target_api:
            return self.target_api
        # special tar_api
        if self.api_mapping.get("target_api"):
            return self.api_mapping["target_api"]
        # finding the most similar operator
        try:
            if self.targetDLClass == "paddlepaddle":
                import paddle
            elif self.targetDLClass == "pytorch":
                import torch
        except ImportError as e:
            print(f"import filed: {e}")
            sys.exit(1)
        self.target_api = search_mapping_api(self.source_api, self.logger, self.sourceDLClass, self.targetDLClass)
        return self.target_api

    def set_target_api(self, target_api):
        self.target_api = target_api

    def parse_args(self, args):
        new_args = []
        for node in args:
            ele = astor.to_source(node).strip('\n')
            new_args.append(ele)

        return new_args

    def parse_kwargs(self, kwargs):
        unsupport_args = self.api_mapping.get("unsupport_args") or []

        new_kwargs = {}
        for node in kwargs:
            k = node.arg
            if k in unsupport_args:
                return None
            v = astor.to_source(node.value).strip("\n")
            new_kwargs[k] = v

        return new_kwargs

    def parse_args_and_kwargs(self, api, args, kwargs, tensorCall=False):
        # get arg_list
        if api is None:
            return args + kwargs
        self.args, self.kwargs = get_operator_parameters(api, self.logger, sourceDLClass=self.sourceDLClass, targetDLClass=self.targetDLClass)

        unsupport_args = self.api_mapping.get("unsupport_args", [])

        if tensorCall and self.args is not None and self.kwargs is not None:
            for target in ["x", "self"]:
                self.args = [arg for arg in self.args if arg not in ['x', 'self']]

                self.kwargs = [(key, value) for key, value in self.kwargs if key not in ['x', 'self']]

        # more args, not match torch class method, indicate it is not torch Class
        if api.startswith("paddle.nn.initializer"):
            self.init_val = astor.to_source(args[0]).strip('\n')
            args = args[1:]
        if  api.startswith("torch.nn.init"):
            self.args =self.args[1:]
        elif self.source_api == 'torch.ones' or self.source_api == 'torch.zeros':
            args = [ast.List(elts=args, ctx=ast.Load())]
        elif self.args and ("perm" in self.args or "repeat_times" in self.args or "shape" in self.args):
            # x.transpose()
            # x.repeat()
            # x.shape()
            # torch.randn()
            if tensorCall or self.source_api == 'torch.randn':
                if ("perm" in self.args[0] or "repeat_times" in self.args or "shape" in self.args) and len(args) != 1:
                    args = [ast.List(elts=args, ctx=ast.Load())]
            else:
                if len(args[1:]) != 1:
                    args = [args[0], ast.List(elts=args[1:], ctx=ast.Load())]

        # operator is a property object
        elif self.args is None and self.kwargs is None:
            if args:
                self.args = []
                for node in args:
                    if isinstance(node, ast.Constant):
                        if isinstance(node.value, str):
                            arg_value = "'" + node.value + "'"
                        else:
                            arg_value = node.value
                    else:
                        arg_value = astor.to_source(node).strip("\n")
                    self.args.append(arg_value)
            return "Property"
        elif 'torch.Tensor.view' in self.source_api or 'torch.view' in self.source_api:
            args = [astor.to_source(arg).strip('\n') for arg in args]
            return {'shape': args}
        elif 'torch.optim' in self.source_api and len(args) and "parameters()" in astor.to_source(args[0]).strip():
            modelparam = args.pop()
            kwargs.append(ast.keyword(arg='params', value=modelparam))
        elif self.targetDLClass == "paddlepaddle" and len(args) + len(kwargs) > len(self.args) + len(self.kwargs):
            if self.args and self.args[0] == "shape":
                args = [ast.Tuple(elts=args, ctx=ast.Load())]
            elif api.startswith('paddle.nn.Sequential'):
                return {"layers": args}
            else:
                old_kwargs = args + kwargs
                new_kwargs = {}
                idx = 0
                for arg in self.args:
                    if isinstance(old_kwargs[idx], ast.keyword):
                        new_kwargs[arg] = astor.to_source(old_kwargs[idx].value).strip("\n")
                    else:
                        if '*' in arg:
                            new_kwargs[arg] = [astor.to_source(old_kwarg).strip("\n") for old_kwarg in old_kwargs[idx:]]
                            return new_kwargs
                        else:
                            new_kwargs[arg] = astor.to_source(old_kwargs[idx]).strip("\n")
                    idx = idx + 1

                old_kwargs = old_kwargs[idx:]
                for kwarg in self.kwargs:
                    flag = False
                    for arg in old_kwargs:
                        if isinstance(arg, ast.keyword) and kwarg[0] == arg.arg:
                            new_kwargs[kwarg[0]] = astor.to_source(arg.value).strip("\n")
                            old_kwargs.remove(arg)
                    if not flag:
                        arg = old_kwargs[0]
                        if not isinstance(arg, ast.keyword):
                            new_kwargs[kwarg[0]] = astor.to_source(arg).strip("\n")
                            old_kwargs.pop(0)
                        else:
                            break

                for old_kwarg in old_kwargs:
                    new_kwargs[old_kwarg.arg] = astor.to_source(old_kwarg.value).strip("\n")
                return new_kwargs


        new_kwargs = {}
        for i, node in enumerate(args):
            # not support 'torch.rot90(tensor, *config)'
            if isinstance(node, ast.Starred):
                return None
            if i < len(self.args):
                arg_name = self.args[i]
            else:
                arg_name = self.kwargs[i - len(self.args)][0]
            # not support some API args
            if arg_name in unsupport_args:
                return None
            if isinstance(node, ast.Constant):
                if isinstance(node.value, str):
                    arg_value = "'" + node.value + "'"
                else:
                    arg_value = node.value
            else:
                arg_value = astor.to_source(node).strip("\n")
            # have comma indicates a tuple
            new_kwargs[arg_name] = arg_value

        for node in kwargs:
            arg_name = node.arg
            # not support some API args
            #
            if (arg_name is None and node.value.id != "kwargs") or arg_name in unsupport_args:
                return None
            if isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, str):
                    arg_value = "'" + node.value.value + "'"
                else:
                    arg_value = node.value.value
            else:
                arg_value = astor.to_source(node.value).strip("\n")
            new_kwargs[arg_name] = arg_value

        return new_kwargs

    def args_to_str(self, args):
        str_list = []
        for ele in args:
            str_list.append("   {}".format(ele))

        return ",\n".join(str_list)

    def kwargs_to_str(self, kwargs):
        str_list = []
        if isinstance(kwargs, str):
            if "Property" in kwargs:
                for tmp in self.args:
                    str_list.append("[{}]".format(tmp))
                return "".join(str_list)
        else:
            for k, v in kwargs.items():
                if self.args and k in self.args:  # adjust arg
                    if '*' in k and isinstance(v, list):
                        for tmp in v:
                            str_list.append("{}".format(tmp))
                    else:
                        str_list.append("{}={}".format(k, v))
                elif k is None:
                    str_list.append("**{}".format(v))
                else:
                    str_list.append("{}={}".format(k, v))

            return ", ".join(str_list)

    def parse_func(self, func):
        new_func = astor.to_source(func).strip("\n")
        self.paddleClass = new_func[0: new_func.rfind(".")]
        # The operator target API name has been set
        if self.get_target_api():
            new_paddle_api = re.sub(
                "paddle.Tensor|paddle.nn.Layer|paddle.optimizer.Optimizer",
                self.paddleClass,
                self.get_target_api(),
            )
            self.set_target_api(new_paddle_api)

        return new_func

    def args_or_kwargs_to_str(self, args_or_kwargs):
        str_list = []

        # Determine whether the parameter is a positional argument list or a keyword argument dictionary
        if isinstance(args_or_kwargs, dict):
            # Handle keyword arguments
            for k, v in args_or_kwargs.items():
                str_list.append("{}={}".format(k, v))
        elif isinstance(args_or_kwargs, (list, tuple)):
            # Handle positional arguments
            for arg in args_or_kwargs:
                str_list.append("{}".format(arg))
        else:
            raise ValueError("Unsupported argument type")

        return ", ".join(str_list)

    def get_full_attr(self, node):
        if isinstance(node, ast.Attribute):
            return self.get_full_attr(node.value) + "." + node.attr
        elif isinstance(node, ast.Name):
            return node.id
        else:
            return "None"

    def set_default_kwargs(self, kwargs):
        """
        process the redundant parameters of Target framework and set the default values
        and return the new parameter list in the form of a dictionary.
        """
        if "default_kwargs" in self.api_mapping:
            default_kwargs = self.api_mapping["default_kwargs"]
            for k in default_kwargs:
                if k not in kwargs:
                    kwargs[k] = default_kwargs[k]

        return kwargs

    def generate_aux_code(self):
        # Implement the logic to generate helper code
        return None

    def write_aux_code(self):
        aux_code = self.generate_aux_code()
        if aux_code:
            code_generator = CodeFileGenerator()
            log_info(
                self.logger,
                "When convert {}, write auxiliary code to file: {}".format(
                    self.source_api, code_generator.file_name
                ),
            )
            code_generator.write_code(aux_code, self.source_api)

            CODE_TEMPLATE = textwrap.dedent(
                """
                import sys
                sys.path.append('{}')
                import paddle_aux
                """
            )
            code = CODE_TEMPLATE.format(
                self.get_aux_dir(),
            )
            log_info(
                self.logger, "add 'import paddle_aux'", self.transformer.file_name
            )
            self.transformer.insert_multi_node(ast.parse(code).body)

    @staticmethod
    def generate_code(kwargs):

        return None

    def get_paddle_nodes(self, args, kwargs, tensorCall=False):
        new_kwargs = self.parse_args_and_kwargs(self.get_target_api(), args, kwargs, tensorCall)
        new_code = self.generate_code(new_kwargs)
        if new_code:
            return ast.parse(new_code).body

        return None

    def get_paddle_class_nodes(self, func, args, kwargs, tensorCall=False, callobj=None):
        if callobj is not None:
            self.init_val = callobj
        new_kwargs = self.parse_args_and_kwargs(self.get_target_api(), args, kwargs, tensorCall)
        self.parse_func(func)
        # NonTorchClass means This API usage not match torch.Tensor/Module/Optimizer, so it is not a torch Class
        if new_kwargs == "NonDLClass":
            return "NonDLClass"
        elif new_kwargs == "Property" and not ("size" in self.source_api):
            return ast.parse(self.get_target_api()).body
        elif new_kwargs is not None:
            new_code = self.generate_code(new_kwargs)
            if new_code == "NNonDLClass":
                return "NonDLClass"
            elif new_code == "unchange":
                return "NonDLClass"
            elif isinstance(new_code, str):
                return ast.parse(new_code).body
            else:
                return new_code

        return None

    def get_paddle_class_attribute_nodes(self, node):
        return None

    def handleVariantCode(self, new_kwargs, res, targetDLClass):
        # Determine the target API
        self.target_api = self.get_target_api()
        method_source_codes = {}
        try:
            if targetDLClass == "paddlepaddle":
                import paddle
                # paddle.set_device('cpu')
                # rebuild target kernel function
                current_device = detect_backend(paddle.get_device())
            elif targetDLClass == "pytorch":
                import torch
                # paddle.set_device('gpu')
                # rebuild target kernel function
                current_device = detect_backend(torch.device("cuda" if torch.cuda.is_available() else "cpu").type)
            else:
                raise EnvironmentError("Unsupported targetDLClass!")
        except ImportError:
            raise ValueError("import paddle failed.")
        if not self.target_api in self.persisted_target_apis:
            for param in self.api_mapping['variantCode']:
                if param["param"] in new_kwargs:
                    if new_kwargs[param["param"]] is not None and (("value" not in param) or (param["value"] in new_kwargs[param["param"]])):
                        for sequence in param["operatorSequence"]:
                            if sequence["level"] == "python underlying library":
                                if not "python" in method_source_codes:
                                    method_source_codes["python"] = {}
                                method_source_code = get_method_source_code_from_string(self.target_api,
                                                                                        sequence["function"],
                                                                                        method_source_codes["python"])
                                modified_source_code = method_source_code
                                for modifyOp in reversed(sequence["sequence"]):
                                    if "add" in modifyOp["class"]:
                                        modified_source_code = insert_code(modified_source_code, modifyOp["loc"],
                                                                           modifyOp["code"], param["param"],
                                                                           new_kwargs[param["param"]],
                                                                           target_api=self.target_api)
                                    elif "modify" in modifyOp["class"]:
                                        modified_source_code = modify_code(modified_source_code, modifyOp["loc"],
                                                                           modifyOp["code"],param["param"],
                                                                           new_kwargs[param["param"]],
                                                                           target_api=self.target_api)
                                    else:
                                        raise ValueError("Unknown operation in Python underlying library")
                                method_source_codes["python"][sequence["function"]] = modified_source_code
                            elif sequence["level"] == "C underlying library":
                                if not "c" in method_source_codes:
                                    method_source_codes["c"] = {}
                                # import headers
                                if "headers" not in method_source_codes["c"]:
                                    method_source_codes["c"]["headers"] = []
                                for header in sequence["backend"][current_device]["headers"]:
                                    if header not in method_source_codes["c"]["headers"]:
                                        method_source_codes["c"]["headers"].append(header)

                                if sequence["function"][current_device] in method_source_codes["c"]:
                                    method_source_code = method_source_codes["c"][sequence["function"][current_device]]
                                else:
                                    method_source_codes["c"][sequence["function"][current_device]] = ""
                                    method_source_code = sequence["backend"][current_device]["sourceCode"]

                                modified_source_code = method_source_code
                                for modifyOp in reversed(sequence["backend"][current_device]["sequence"]):
                                    if "add" in modifyOp["class"]:
                                        modified_source_code = insert_code(modified_source_code, modifyOp["loc"],
                                                                           modifyOp["code"], param["param"],
                                                                           new_kwargs[param["param"]],
                                                                           target_api=self.target_api, type=True)
                                    elif "modify" in modifyOp["class"]:
                                        modified_source_code = modify_code(modified_source_code, modifyOp["loc"],
                                                                           modifyOp["code"],param["param"],
                                                                           new_kwargs[param["param"]],
                                                                           target_api=self.target_api, type=True)
                                    else:
                                        raise ValueError("Unknown operation in C underlying library")
                                method_source_codes["c"][sequence["function"][current_device]] = modified_source_code
                                if "namespace" not in method_source_codes["c"]:
                                    method_source_codes["c"]["namespace"] = {}
                                method_source_codes["c"]["namespace"][sequence["function"][current_device]] = sequence["namespace"][current_device]
                                if "funcContext" not in method_source_codes["c"]:
                                    method_source_codes["c"]["funcContext"] = {}
                                method_source_codes["c"]["funcContext"][sequence["function"][current_device]] = sequence["backend"][current_device]["funcContext"]
                    # Processed variant parameters
                    new_kwargs.pop(param["param"])
            # # unmodified difference code
            if method_source_codes == {}:
                return new_kwargs, res

            if "c" in method_source_codes:
                # one operator ,one write
                lib_file = c_replace_and_persist_modified_methods(current_device, self.target_api, method_source_codes["c"],
                                                                  self.out_dir, self.targetDLClass)
                if self.target_api not in self.persisted_target_apis:
                    self.persisted_target_apis.append(self.target_api)

                if res != "":
                    API_TEMPLATE = textwrap.dedent(
                        """
                        {}
                        import ctypes
                        lib = ctypes.CDLL("{}")
                        """
                    )
                    res = API_TEMPLATE.format(res, lib_file)
                else:
                    API_TEMPLATE = textwrap.dedent(
                        """
                        import ctypes
                        lib = ctypes.CDLL("{}")
                        """
                    )
                    res = API_TEMPLATE.format(lib_file)
            if "python" in method_source_codes:
                # one operator ,one write
                file, self.target_api = python_replace_and_persist_modified_methods(self.target_api,
                                                                                    method_source_codes["python"],
                                                                                    self.out_dir, self.targetDLClass)
                if self.target_api not in self.persisted_target_apis:
                    self.persisted_target_apis.append(self.target_api)

                if res != "":
                    API_TEMPLATE = textwrap.dedent(
                        """
                        {}
                        from .{} import {}
                        """
                    )
                    res = API_TEMPLATE.format(res, os.path.splitext(file)[0], self.target_api)
                else:
                    res = "from .{} import {}\n".format(os.path.splitext(file)[0], self.target_api)
            return new_kwargs, res

    def handleCodeGeneration(self, new_kwargs, res, targetDLClass):
        if isinstance(new_kwargs, str) and "Property" in new_kwargs:
            if self.args:
                return "{}{}".format(self.get_target_api(), self.kwargs_to_str(new_kwargs))
            else:
                return "{}".format(self.get_target_api())

        new_kwargs = self.set_default_kwargs(new_kwargs)
        if targetDLClass == "paddlepaddle":
            return self.paddle_handleCodeGeneration(new_kwargs, res)
        elif targetDLClass == "pytorch":
            return self.pytorch_handleCodeGeneration(new_kwargs, res)

    def paddle_handleCodeGeneration(self, new_kwargs, res):
        if "x" in new_kwargs and isinstance(new_kwargs["x"], str):
            if self.source_api == "torch.all" or self.source_api == "torch.any":
                new_kwargs["x"] = "paddle.to_tensor(" + new_kwargs["x"] + ", dtype='bool')"
            elif self.source_api == "torch.angle" or self.source_api == "torch.conj":
                new_kwargs["x"] = "paddle.to_tensor(" + new_kwargs["x"] + ")"
            elif "bitwise" in self.source_api:
                new_kwargs["x"] = "paddle.to_tensor(" + new_kwargs["x"] + ", dtype='int32')"

        if "y" in new_kwargs and isinstance(new_kwargs["y"], str):
            if self.source_api == "torch.all" or self.source_api == "torch.any":
                new_kwargs["y"] = "paddle.to_tensor(" + new_kwargs["y"] + ", dtype='bool')"
            elif self.source_api == "torch.angle" or self.source_api == "torch.conj":
                new_kwargs["y"] = "paddle.to_tensor(" + new_kwargs["y"] + ")"
            elif "bitwise" in self.source_api:
                new_kwargs["y"] = "paddle.to_tensor(" + new_kwargs["y"] + ", dtype='int32')"

        # dtype_v = None
        # if "dtype" in new_kwargs:
        #     dtype_v = new_kwargs.pop("dtype")

        pin_memory_v = False
        if "pin_memory" in new_kwargs:
            pin_memory_v = eval(new_kwargs.pop("pin_memory"))

        stop_gradient_v = None
        if "requires_grad" in new_kwargs:
            stop_gradient_v = "not " + new_kwargs.pop("requires_grad").strip("()")

        rounding_mode = None
        if "rounding_mode" in new_kwargs:
            rounding_mode = new_kwargs.pop("rounding_mode") + "()"

        out_v = None
        if "out" in new_kwargs:
            out_v = new_kwargs.pop("out")

        # special parameters
        if any(api in self.source_api for api in
               ["torch.swapaxes", "torch.transpose", "torch.swapdims"]) and "perm" in new_kwargs:
            stripped_str = new_kwargs.pop("perm").strip("[]")
            attrs = stripped_str.split(", ")
            global transpose_num, code
            transpose_num = transpose_num + 1
            dim1, dim2 = attrs[0], attrs[1]
            if 'x' in new_kwargs:
                API_TEMPLATE = textwrap.dedent(
                    """
                    perm{} = list(range(len({}.shape)))
                    perm{}[{}], perm{}[{}] = perm{}[{}], perm{}[{}] 
                    {}({}, perm=perm{})
                    """
                )
                code = API_TEMPLATE.format(transpose_num, new_kwargs['x'],
                                           transpose_num, dim1, transpose_num, dim2, transpose_num,
                                           dim2, transpose_num, dim1, self.get_target_api(),
                                           new_kwargs['x'], transpose_num)
            else:
                API_TEMPLATE = textwrap.dedent(
                    """
                    perm{} = list(range(len({}.shape)))
                    perm{}[{}], perm{}[{}] = perm{}[{}], perm{}[{}] 
                    {}(perm=perm{})
                    """
                )
                code = API_TEMPLATE.format(transpose_num, self.init_val,
                                           transpose_num, dim1, transpose_num, dim2, transpose_num,
                                           dim2, transpose_num, dim1, self.get_target_api(),
                                           transpose_num)
            return code
        if self.init_val and ".zero_" == self.source_api:
            if len(new_kwargs):
                res += "{}={}({},{})".format(self.init_val, self.get_target_api(), self.init_val,
                                             self.kwargs_to_str(new_kwargs))
            else:
                res += "{}={}({})".format(self.init_val, self.get_target_api(), self.init_val)
        else:
            res += "{}({})".format(self.get_target_api(), self.kwargs_to_str(new_kwargs))

        # if dtype_v:
        #     res += ".astype('{}')".format(dtype_v)

        if pin_memory_v:
            res += ".pin_memory()"

        if rounding_mode:
            res += f".{rounding_mode}"
        elif stop_gradient_v and out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                x = {}
                x.stop_gradient = {}
                paddle.assign(x, output={})
                """
            )
            code = API_TEMPLATE.format(res, stop_gradient_v, out_v)
        elif stop_gradient_v and not out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(out, res, out, stop_gradient_v, out)
        elif not stop_gradient_v and out_v:
            API_TEMPLATE = textwrap.dedent(
                """
                paddle.assign({}, output={})
                """
            )
            code = API_TEMPLATE.format(res, out_v)
        else:
            code = "{}".format(res)

        return code

    def pytorch_handleCodeGeneration(self, new_kwargs, res):
        requires_grad_v = None
        if "requires_grad" in new_kwargs:
            requires_grad_v = "not " + new_kwargs.pop("requires_grad").strip("()")

        rounding_mode = None
        if "rounding_mode" in new_kwargs:
            rounding_mode = new_kwargs.pop("rounding_mode") + "()"
        res += "{}({})".format(self.get_target_api(), self.kwargs_to_str(new_kwargs))
        if rounding_mode:
            res += f".{rounding_mode}"
        elif requires_grad_v:
            API_TEMPLATE = textwrap.dedent(
                """
                {} = {}
                {}.stop_gradient = {}
                {}
                """
            )
            out = get_unique_name("out")
            code = API_TEMPLATE.format(out, res, out, requires_grad_v, out)
        else:
            code = "{}".format(res)

        return code