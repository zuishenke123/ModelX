import ast
import os

from CMC.nodeBase import astBaseNodes
from CMC.utils import log_info, HandleAliasApi, traverseChildNodes, log_warning, remove_outer_parentheses
from CMC.special_mapping import SUPPORT_PACKAGE_LIST, FrameworkPackage, TENSOR_MAPPING, dataTypeMapping, \
    omitSuffixCall, get_api_mapping
import CMC.Matchers.importMatcher
from CMC.Matchers.basicMatcher import *



class astTree(ast.AST):
    def __init__(self, root, file, out_dir, possible_modules, unsupport_map, logger, sourceDLClass, targetDLClass, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = root
        self.file = file
        self.source_library_api_count = 0
        self.unsupport_map = unsupport_map
        self.import_map = possible_modules
        self.logger = logger
        self.out_dir = out_dir
        self.sourceDLClass = sourceDLClass
        self.targetDLClass = targetDLClass

    def analyze_ast_node(self):
        """
        # The file has been processed into an astTree. At this point, different parts of the code need to be handled.
        # :return: None for now
        """

        # analyze the import information
        import_nodes = astImportNodes(self.root, self.file, self.import_map, self.unsupport_map, self.logger, self.sourceDLClass, self.targetDLClass)
        import_nodes.transform()

        # analyze the requires_grad information
        grad_nodes = astGradApiNodes(self.root, self.file, self.import_map, self.unsupport_map, self.logger, self.sourceDLClass, self.targetDLClass)
        grad_nodes.transform()

        # analyze the basic api information
        basic_nodes = astBasicNodes(self.root, self.file, self.out_dir, self.import_map, self.unsupport_map,
                                    self.logger, self.sourceDLClass, self.targetDLClass)
        basic_nodes.transform()

        self.source_library_api_count = import_nodes.converting_api_count + grad_nodes.converting_api_count + basic_nodes.converting_api_count


class astImportNodes(astBaseNodes):
    def __init__(self, root, file, possible_modules, unsupport_map, logger, sourceDLClass="pytorch",
                 targetDLClass="paddlepaddle"):
        super(astImportNodes, self).__init__(root, file, possible_modules, unsupport_map, logger, sourceDLClass,
                                             targetDLClass)
        self.import_framework = False

        self.import_map[self.file]["other_packages"] = []

    def HandleImport(self, node):
        """
        handle import Ast Node of model
        three branches:
        1. import xxx
        2. import xxx.xx
        3. import xxx.xx as xx
        """
        newImportNodes = []
        for idx, package in enumerate(node.names):
            framework_related = False
            for frameworkPackage in SUPPORT_PACKAGE_LIST[self.sourceDLClass]:
                if re.search(r"{}\.|^{}$".format(re.escape(frameworkPackage), re.escape(frameworkPackage)),
                             package.name):
                    self.import_framework = True
                    framework_related = True
                    self.converting_api_count += 1
                    import_statement = f"import {package.name}" + \
                                       (f" as {package.asname}" if package.asname else "")
                    log_info(self.logger, f"remove '{import_statement}'", self.file_name, node.lineno)
                    import_key = package.asname if package.asname else package.name
                    self.import_map[self.file][import_key] = package.name
                    source_name = package.name
                    try:
                        package.name = eval(
                            f'CMC.Matchers.importMatcher.conversion_replace_Import')(
                            package.name, self.sourceDLClass, self.targetDLClass, self.logger)
                        self.success_api_count += 1
                    except Exception as e:
                        log_warning(self.logger,
                                 f"conversion_replace_Import is not supported")
                        # Exception handling logic, correctly referencing the exception variable e
                        if source_name in self.unsupport_api_map:
                            self.unsupport_api_map[source_name] += 1
                        else:
                            self.unsupport_api_map[source_name] = 1
                    break

            # unspported Import
            if package.name is None:
                return None

            # framework-agnostic
            if not framework_related:
                if package.asname:
                    self.import_map[self.file]["other_packages"].append(package.asname)
                else:
                    self.import_map[self.file]["other_packages"].append(package.name)
                newImportNodes.append(package)
            if package.name is None or package.asname is None:
                node.names.pop(idx)
        if newImportNodes:  # reserve framework-agnostic packages
            node.names = newImportNodes
        if not len(node.names):
            return None
        return node

    def HandleImportFrom(self, node):
        """
        handle importFrom Ast Node of model
        two branches:
        1. from xxx import xx
        2. from xxx.xx import xx as xx
        3. from ../. import xx (node.module is None)
        """
        if node.module and any(
                frameworkPackage in node.module for frameworkPackage in SUPPORT_PACKAGE_LIST[self.sourceDLClass]):
            targetImportFrom = dict()
            for i, package in enumerate(node.names):
                import_statement = "remove 'from {} import {}{}' ".format(
                    node.module, package.name, f" as {package.asname}" if package.asname else "")
                log_info(self.logger, import_statement, self.file_name, node.lineno)

                alias_key = package.asname if package.asname else package.name
                self.import_map[self.file][alias_key] = ".".join([node.module, package.name])

                # Assume that there is already a safer method to replace the use of eval
                full_package = eval(
                    f'CMC.Matchers.importMatcher.conversion_replace_Import')(
                    ".".join([node.module, package.name]), self.sourceDLClass, self.targetDLClass, self.logger)
                # Unsupported ImportForm
                if full_package is None:
                    return None
                target_Module = ".".join(full_package.split('.')[:-1])
                target_component = full_package.split('.')[-1]

                # Check if the key is in the dictionary
                if target_Module not in targetImportFrom:
                    targetImportFrom[target_Module] = []

                # Construct a temporary dictionary to check if it already exists in the list
                temp_dict = {"name": target_component, "asname": package.asname}
                if temp_dict not in targetImportFrom[target_Module]:
                    targetImportFrom[target_Module].append(temp_dict)

            for module, components in targetImportFrom.items():
                names = []
                new_nodes = []
                for component_info in components:
                    names.append(ast.alias(asname=component_info["asname"], name=component_info["name"]))
                new_node = ast.ImportFrom(module=module, names=names, level=node.level)
                new_nodes.append(new_node)
            return new_node

        # IF node.module is Path or Other_packages
        import_path = os.path.join(os.path.dirname(self.file), *[".."] * (node.level - 1),
                                   node.module.replace(".", "/") if node.module else "")
        # IF node.module is Other_packages
        if not (os.path.exists(import_path) or os.path.exists(import_path + ".py")):
            for package in node.names:
                self.import_map[self.file]["other_packages"].append(
                    package.asname if package.asname else package.name)
                self.import_map[self.file]["other_packages"].append(
                    ".".join([node.module, package.name]))
        return node

    def HandleName(self, node):
        """
        handle attribute information of nodes in the AST
        eg，two branches:
        1. nn.Module -> torch.nn.Module
        2. Module -> torch.nn.Module
        """

        framework_api = self.get_full_api_from_node(node, self.file)
        if framework_api:
            # if torch_api in ALIAS_MAPPING: torch_api = ALIAS_MAPPING[torch_api] Use regular expressions with
            # torch_api to perform matching and replacement, obtaining the actual package invocation code
            framework_api = HandleAliasApi(framework_api, self.sourceDLClass)
            return ast.parse(framework_api).body[0].value
        return node

    def HandleModule(self, node):
        """
        e.g. add import paddle
        """
        self.generic_visit(node)

        # if self.import_framework:
        #     importTargetFramework = FrameworkPackage[self.targetDLClass]
        #     log_info(self.logger, "add 'import {}' in first line".format(importTargetFramework), self.file_name)
        #     self.record_scope((self.root, "body", 0), ast.parse("import {}".format(importTargetFramework)).body)


class astBasicNodes(astBaseNodes):
    def __init__(self, root, file, out_dir, possible_modules, unsupport_map, logger, sourceDLClass, targetDLClass):
        super(astBasicNodes, self).__init__(root, file, possible_modules, unsupport_map, logger)
        self.classbase = None
        self.out_dir = out_dir
        self.current_optimizer_lr = None
        self.sourceDLClass = sourceDLClass
        self.targetDLClass = targetDLClass
        self.API_MAPPING = get_api_mapping(self.sourceDLClass, self.targetDLClass)

    def HandleAttribute(self, node):
        """
        handle attribute information of nodes in the AST
        e.g. five  branches:
        1. torch.abs(x).transpose(1, 0)
        2. (x == y).transpose(1, 0)
        3. (x + y).transpose(1, 0)
        4. (-x).transpose(1, 0)
        5. x[0].transpose(1, 0)
        6. x.transpose(1, 0)
        """
        if isinstance(node.value, (ast.Call, ast.BinOp, ast.UnaryOp, ast.Subscript, ast.Assert)) or \
                isinstance(node.value, ast.Attribute) and node.value.attr == "T":
            self.generic_visit(node)

        # should be handled by visit_Call
        if isinstance(self.parent_node, ast.Call) and node == self.parent_node.func:
            return node

        full_attr = self.get_full_attr(node)
        if "device" in full_attr:
            self.converting_api_count += 1
            if self.targetDLClass == "paddlepaddle":
                node_list = ast.parse("paddle.get_device()".strip("\n")).body
                new_node = node_list[-1]
                # ast.Expr, which contain ast.Call or ast.Name
                if isinstance(new_node, ast.Expr):
                    new_node = new_node.value
                return new_node
        # handleSourceApi
        for package in FrameworkPackage[self.sourceDLClass]:
            if full_attr.startswith("%s." % package):
                source_api = full_attr
                self.converting_api_count += 1
                log_info(self.logger,
                         "[Start] convert {} to {} for {}".format(self.sourceDLClass, self.targetDLClass, full_attr),
                         self.file_name,
                         node.lineno, )
                matcher = self.get_api_mather(source_api)  # get conversion matcher
                if matcher:
                    target_api = matcher.get_target_api()  # api converting rules
                    if target_api == "delete":
                        if isinstance(self.parent_node, ast.Expr):
                            self.success_api_count += 1
                            log_info(self.logger, "[Success]remove {} ".format(source_api), self.file_name,
                                     node.lineno, )
                            return None
                    elif target_api:
                        new_node = ast.parse(target_api).body[0].value
                        self.success_api_count += 1
                        log_info(self.logger,
                                 "[Success] convert {} to {} Successfully".format(source_api, self.targetDLClass),
                                 self.file_name,
                                 node.lineno,
                                 )
                        return new_node
                if source_api in self.unsupport_api_map:
                    self.unsupport_api_map[source_api] += 1
                else:
                    self.unsupport_api_map[source_api] = 1
                log_warning(self.logger,
                            "[Not Support] convert {} to {} is not supported currently".format(source_api,
                                                                                               self.targetDLClass),
                            self.file_name,
                            node.lineno,
                            )
                # unsupported
                if "jit" in source_api:
                    return None
                return node

        # Class attribute
        # x.ndims
        # Extra processing branch for No DL Class Attribute
        if "NonDLClass" not in full_attr or "PotentialUnDLClass" not in full_attr:
            attr_list = full_attr.split(".")
            if attr_list[0] in self.classInstance and self.classInstance[attr_list[0]].startswith("{}.Tensor".format(
                    self.sourceDLClass if self.sourceDLClass == 'tensorflow' else FrameworkPackage[
                        self.sourceDLClass])):
                source_api = ".".join(["{}.Tensor".format(self.sourceDLClass if self.sourceDLClass == 'tensorflow' else \
                                                              FrameworkPackage[self.sourceDLClass]), attr_list[-1]])
                if source_api in TENSOR_MAPPING[self.sourceDLClass]:
                    self.converting_api_count += 1
                    log_info(self.logger,
                             "Start convert Tensor Attribute: {} to {} ".format(source_api, self.targetDLClass),
                             self.file_name,
                             node.lineno,
                             )
                    return self.trans_tensor_attribute(node, source_api)
        if "self." in full_attr and (self.functionName and "__init__" != self.functionName):
            substrings = ['in_channels', 'out_channels', 'stride', 'padding', 'padding_mode', 'kernel_size']
            flag = 0
            if "self.bias" == full_attr:
                full_attr.replace("bias", "_bias_attr")
                flag = 1
            elif "weight" == full_attr:
                full_attr.replace("weight", "_weight_attr")
                flag = 1
            elif any("self." + substring == full_attr for substring in substrings):
                for substring in substrings:
                    if substring in full_attr:
                        # Perform operations related to the found substring
                        full_attr = full_attr.replace(substring, f"_{substring}")
                        flag = 1
                        break  # If only the first match needs to be processed, break the loop here
            if flag:
                node_list = ast.parse(full_attr.strip("\n")).body
                new_node = node_list[-1]
                # ast.Expr, which contain ast.Call or ast.Name
                if isinstance(new_node, ast.Expr):
                    new_node = new_node.value
                return new_node
        # NonTorchClass or x.transpose(1, 0)
        # not need to convert x, transpose have converted in visit_Call
        return node

    def trans_tensor_attribute(self, node, source_tensor_api):
        if source_tensor_api in TENSOR_MAPPING[self.sourceDLClass]:
            matcher = eval(TENSOR_MAPPING[source_tensor_api])(self, source_tensor_api, self.logger)
            if matcher:
                new_node = matcher.get_paddle_class_attribute_nodes(node)
                if new_node == "delete":
                    if isinstance(self.parent_node, ast.Expr):
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Success]remove {} ".format(source_tensor_api),
                            self.file_name,
                            node.lineno,
                        )
                        return None
                elif new_node == "unchange":
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "[Success]convert Tensor Attribute: {} to Paddle, just remain the same".format(
                            source_tensor_api
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return node
                elif new_node:
                    new_node = new_node[-1]
                    if isinstance(new_node, ast.Expr):
                        new_node = new_node.value

                    if isinstance(
                            new_node,
                            (
                                    ast.Call,
                                    ast.Attribute,
                                    ast.Name,
                                    ast.Constant,
                                    ast.Compare,
                                    ast.BinOp,
                                    ast.UnaryOp,
                                    ast.Tuple,
                                    ast.Assert,
                            ),
                    ):
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Success]convert Tensor Attribute: {} to Paddle".format(
                                source_tensor_api
                            ),
                            self.file_name,
                            node.lineno,
                        )
                        return new_node

        annotate_node = ast.parse(
            "'Tensor Attribute: {}, not convert, please check whether it is torch.Tensor.* and convert manually'".format(
                source_tensor_api
            )
        ).body[0]
        self.record_scope(self.scope_body_index(), annotate_node)
        if source_tensor_api in self.unsupport_api_map:
            self.unsupport_api_map[source_tensor_api] += 1
        else:
            self.unsupport_api_map[source_tensor_api] = 1
        log_warning(
            self.logger,
            "[Not Support] convert Tensor Attribute: {} to Paddle is not supported currently".format(
                source_tensor_api
            ),
            self.file_name,
            node.lineno,
        )
        return node

    def HandleCall(self, node):
        def trans_class_method(node, source_api, callobj=None):
            self.converting_api_count += 1
            matcher = self.get_api_mather(source_api)
            if matcher:
                node_list = matcher.get_paddle_class_nodes(
                    node.func, node.args, node.keywords, True, callobj
                )
                # record optimizer lr
                for keyword in node.keywords:
                    if keyword.arg == 'lr':
                        # Get the current learning rate
                        self.current_optimizer_lr = remove_outer_parentheses(astor.to_source(keyword.value).strip())
                        break
                if node_list == "NonDLClass":
                    # This API usage indicate that is not a torch.Tensor
                    self.success_api_count -= 1
                    log_info(
                        self.logger,
                        " Misidentify Class Method: {}, so just remain the same".format(
                            torch_api
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return node
                elif node_list:
                    new_node = node_list[-1]
                    # ast.Expr which contain ast.Call or ast.Name
                    if isinstance(new_node, ast.Expr):
                        new_node = new_node.value

                    if isinstance(
                            new_node,
                            (
                                    ast.Call,
                                    ast.Name,
                                    ast.Constant,
                                    ast.Attribute,
                                    ast.Subscript,
                                    ast.BinOp,
                                    ast.Assert,
                                    ast.Tuple,
                                    ast.Assign,
                            ),
                    ):
                        self.insert_multi_node(node_list[0:-1])
                        self.success_api_count += 1
                        log_info(
                            self.logger,
                            "[Success]convert Class Method: {} to Paddle".format(source_api),
                            self.file_name,
                            node.lineno,
                        )
                        return new_node

            source_api = "*" + source_api[source_api.rfind("."):]
            annotate_node = ast.parse(
                "'Class Method: {}, not convert, please check whether it is {}.Tensor.*/Optimizer.*/nn.Module.*, "
                "and convert manually'".format(
                    source_api, FrameworkPackage[self.sourceDLClass]
                )
            ).body[0]
            self.record_scope(self.scope_body_index(), annotate_node)
            if source_api in self.unsupport_api_map:
                self.unsupport_api_map[source_api] += 1
            else:
                self.unsupport_api_map[source_api] = 1
            log_warning(
                self.logger,
                "[Not Support] convert Class Method: {} to Paddle is not supported currently".format(
                    source_api
                ),
                self.file_name,
                node.lineno,
            )
            return node

        full_attr = self.get_full_attr(node.func)
        if self.functionBase and full_attr in self.functionBase:
            full_attr = self.functionBase[full_attr]
        if node.keywords:
            for idx, keyword in enumerate(node.keywords):
                if keyword.arg and 'src_key_padding_mask' in keyword.arg:
                    node.keywords.pop(idx)
        Module = None
        # handle partial()

        if full_attr == 'partial':
            source_api = self.get_full_attr(node.args[0])
            while True:
                if source_api in self.classInstance:
                    source_api = self.classInstance[source_api]
                else:
                    break
            if isinstance(node.args[0], ast.Name) or isinstance(node.args[0], ast.Attribute):
                # Multiple Inheritance
                if source_api in self.import_map[self.file]:
                    full_attr = self.import_map[self.file][source_api]
                else:
                    for importModule in self.import_map[self.file].values():
                        if isinstance(importModule, str) and importModule.endswith(".*"):
                            Module = importModule.replace("*", '')
                            full_attr = importModule.replace("*", source_api)
                if full_attr == 'partial':
                    full_attr = source_api
            else:
                full_attr = source_api

            # Use Postorder traversal
            self.generic_visit(node)

            # torch operator
            for package in FrameworkPackage[self.sourceDLClass]:
                if full_attr.startswith("%s." % package):
                    self.converting_api_count += 1
                    source_api = full_attr
                    support = True
                    matcher = self.get_api_mather(source_api)
                    if not matcher:
                        support = False
                    # such as torch.max(*args, **kwargs)
                    if isinstance(node.args, ast.Starred):
                        support = False
                    for k_node in node.keywords:
                        if k_node.arg is None and k_node.value == "kwargs":
                            support = False
                    if support:
                        node.args.pop(0)
                        new_node = matcher.get_paddle_nodes(node.args, node.keywords)
                        if new_node:
                            new_node = new_node[-1]
                            # ast.Expr, which contain ast.Call or ast.Name
                            if isinstance(new_node, ast.Expr):
                                new_node = new_node.value
                            # mapping partial()
                            new_full_attr = self.get_full_attr(new_node.func)
                            if Module is not None:
                                new_full_attr.replace(Module, "")
                            node.args = [ast.parse(new_full_attr).body[0].value]
                            if len(new_node.args):
                                node.args += new_node.args
                            node.keywords = new_node.keywords
                    else:
                        if source_api in self.unsupport_api_map:
                            self.unsupport_api_map[source_api] += 1
                        else:
                            self.unsupport_api_map[source_api] = 1
                        log_warning(self.logger,
                                    "[Not Support] convert {} to {} is not supported currently".format(source_api,
                                                                                                       self.targetDLClass),
                                    self.file_name,
                                    node.lineno,
                                    )

            return node

        # Use Postorder traversal
        self.generic_visit(node)

        # handle suffix Call
        for suffixCall in omitSuffixCall:
            if suffixCall in full_attr:
                return node.func.value

        # Normal Package Call
        # include torch or third_party
        # e.g. torch.add(x, y) or torch.add(torch.tensor(x), y)
        for package in FrameworkPackage[self.sourceDLClass]:
            # special_case: self.xxx:
            if full_attr.startswith("%s." % package):
                # 检查是否是 `paddle.ParamAttr` 的调用
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'ParamAttr':
                    # 遍历 `paddle.ParamAttr` 的参数，寻找 `initializer`
                    for keyword in node.keywords:
                        if keyword.arg == 'initializer':
                            # 返回 initializer 的值节点，移除 `paddle.ParamAttr` 本身
                            return keyword.value
                source_api = full_attr
                self.converting_api_count += 1
                log_info(self.logger,
                         "[Start] convert {} to {} for {}.".format(self.sourceDLClass, self.targetDLClass, full_attr),
                         self.file_name,
                         node.lineno, )
                support = True

                matcher = self.get_api_mather(source_api)
                # record optimizer lr
                for keyword in node.keywords:
                    if keyword.arg == 'lr':
                        # Get the current learning rate
                        self.current_optimizer_lr = remove_outer_parentheses(astor.to_source(keyword.value).strip())
                        break
                if not matcher:
                    support = False
                # such as torch.max(*args, **kwargs)
                if isinstance(node.args, ast.Starred):
                    support = False
                    log_warning(self.logger,
                             "[Failed] Parameters in {} contain {}.".format(full_attr, "*args, **kwargs"),
                             self.file_name,
                             node.lineno, )
                for k_node in node.keywords:
                    if k_node.arg is None and k_node.value == "kwargs":
                        support = False
                        log_warning(self.logger,
                                 "[Failed] Parameters in {} contain {}.".format(full_attr, "*args, **kwargs"),
                                 self.file_name,
                                 node.lineno, )
                if support:
                    node_list = matcher.get_paddle_nodes(node.args, node.keywords)
                    if node_list:
                        new_node = node_list[-1]
                        # ast.Expr, which contain ast.Call or ast.Name
                        if isinstance(new_node, ast.Expr):
                            new_node = new_node.value

                        if isinstance(
                                new_node,
                                (
                                        ast.Call,
                                        ast.Name,
                                        ast.Constant,
                                        ast.Attribute,
                                        ast.Compare,
                                        ast.BinOp,
                                        ast.UnaryOp,
                                        ast.Tuple,
                                        ast.Assert,
                                        ast.Assign,
                                        ast.Subscript,
                                        ast.Lambda
                                ),
                        ):
                            self.insert_multi_node(node_list[0:-1])
                            self.success_api_count += 1
                            log_info(
                                self.logger,
                                "[Success]convert {} to Paddle ".format(source_api),
                                self.file_name,
                                node.lineno,
                            )
                            return new_node
                else:
                    if source_api in self.unsupport_api_map:
                        self.unsupport_api_map[source_api] += 1
                    else:
                        self.unsupport_api_map[source_api] = 1
                    log_warning(self.logger,
                             "[Not Support] convert {} to {} is not supported currently".format(source_api,
                                                                                                self.targetDLClass),
                             self.file_name,
                             node.lineno,
                             )
        if "unDLClass" not in full_attr:
            is_tensor_api = False
            is_module_api = False
            is_optim_api = False
            #  x.reshape
            #  self.weight.reshape
            #  x.T.reshape
            # when > 2, need to more strict
            attr_list = full_attr.split(".")
            if ".".join(attr_list[:-1]) in self.import_map[self.file]['other_packages']:
                return node
            if len(attr_list) > 2:
                if "self." in full_attr:
                    is_tensor_api = True
                    is_module_api = True
                if ".T." in full_attr:
                    is_tensor_api = True
                if ".data." in full_attr:
                    is_tensor_api = True
            elif len(attr_list) == 2:
                if "self." in full_attr:
                    is_module_api = True
                    is_optim_api = True
                else:
                    is_tensor_api = True
                    is_module_api = True
                    is_optim_api = True
            API_MAPPING = get_api_mapping(self.sourceDLClass, self.targetDLClass)
            if is_tensor_api:
                # Handle forced data type conversion using paddle.cast
                dataType = full_attr.split(".")[-1]
                if dataType in ["int", "float", "bool", "double", "long", "short"]:
                    self.converting_api_count += 1
                    code_line = ast.unparse(node).strip("\n").split(".")[:-1]
                    code_line.append(f"cast(dtype=\"{dataTypeMapping[dataType]}\")")
                    node_list = ast.parse(".".join(code_line).strip("\n")).body
                    new_node = node_list[-1]
                    # ast.Expr, which contain ast.Call or ast.Name
                    if isinstance(new_node, ast.Expr):
                        new_node = new_node.value
                    return new_node
                else:
                    torch_api = ".".join(["torch", attr_list[-1]])
                    if torch_api in API_MAPPING:
                        log_info(
                            self.logger,
                            "Start convert Tensor Class Method: {} to Paddle --> ".format(
                                torch_api
                            ),
                            self.file_name,
                            node.lineno,
                        )

                        new_node = trans_class_method(node, torch_api,
                                                      astor.to_source(node.func.value).strip("\n") if len(
                                                          attr_list) == 2 and attr_list[0] == "DLClass" else
                                                      ".".join(attr_list[:-1]))
                        self.success_api_count += 1
                        code_line = astor.to_source(new_node).strip('\n')
                        code_line = code_line.split(".")
                        if "Tensor" in code_line:
                            code_line[0:2] = [astor.to_source(node.func.value).strip('\n')]
                        elif not (".zero_" in torch_api):
                            code_line[0] = astor.to_source(node.func.value).strip('\n')
                        node_list = ast.parse(".".join(code_line).strip('\n')).body
                        new_node = node_list[-1]
                        # ast.Expr, which contain ast.Call or ast.Name
                        if isinstance(new_node, ast.Expr):
                            new_node = new_node.value
                        return new_node

            if is_module_api:
                torch_api = ".".join(["torch.nn", attr_list[-1]])
                if torch_api in API_MAPPING:
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "Start convert Layer Class Method: {} to Paddle --> ".format(
                            torch_api
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return trans_class_method(node, torch_api, ".".join(attr_list[:-1]))

            if is_optim_api:
                torch_api = ".".join(["torch.optim", attr_list[-1]])
                if torch_api in API_MAPPING:
                    self.success_api_count += 1
                    log_info(
                        self.logger,
                        "Start convert Optimizer Class Method: {} to Paddle --> ".format(
                            torch_api
                        ),
                        self.file_name,
                        node.lineno,
                    )
                    return trans_class_method(node, torch_api, ".".join(attr_list[:-1]))

        # NonTorchClass
        return node

    def HandleFunctionDef(self, node):
        self.scope_stack.append(node)
        self.functionBase = {}
        self.functionName = node.name
        for idx, default in enumerate(reversed(node.args.defaults)):
            num = len(node.args.args)
            self.functionBase[node.args.args[num - idx - 1].arg] = astor.to_source(default).strip("\n")
        self.generic_visit(node)
        self.functionBase = None
        self.FunctionName = None
        self.scope_stack.pop()
        return node

    def HandleClassDef(self, node):
        self.scope_stack.append(node)
        self.classbase = [base for base in node.bases]
        if len(node.bases):
            self.classInstance[node.name] = astor.to_source(node.bases[0]).strip("\n")
        self.generic_visit(node)
        self.classbase = None
        self.scope_stack.pop()
        return node

    def HandleExpr(self, node):
        for field, old_value in traverseChildNodes(node):
            new_node = self.visit(old_value)
            if new_node is None:
                return None
            else:
                setattr(node, field, new_node)
        return node

    def HandleCommonNodes(self, node):
        self.scope_stack.append(node)
        self.generic_visit(node)
        self.scope_stack.pop()
        return node

    def HandleTryFinally(self, node):
        self.scope_stack.append(node)
        node = self.generic_visit(node)
        self.scope_stack.pop()
        return node

    # def HandleAssign(self, node):
    #     """
    #     handle 'requires_grad' and 'stop_gradient' in the AST
    #     """
    #     self.scope_stack.append(node)
    #     node = self.generic_visit(node)
    #     self.scope_stack.pop()
    #     return node

    def HandleModule(self, node):
        """
        e.g. add import paddle
        """
        self.scope_stack.append(node)
        self.generic_visit(node)

        if self.import_map:
            importTargetFramework = FrameworkPackage[self.targetDLClass][0]
            log_info(self.logger, "add 'import {}' in first line".format(importTargetFramework), self.file_name)
            self.record_scope((self.root, "body", 0), ast.parse("import {}".format(importTargetFramework)).body)
        self.scope_stack.pop()

    def get_api_mather(self, source_api):
        if source_api in self.API_MAPPING.keys():
            api_mapping = self.API_MAPPING[source_api]
            if "disable" in api_mapping and eval(api_mapping["disable"]):
                return None

            if "Matcher" in api_mapping:
                matcher = api_mapping["Matcher"]
                if self.current_optimizer_lr and "LrschedMatcher" == matcher:
                    return eval(matcher)(self, source_api, api_mapping, self.logger, lr=self.current_optimizer_lr, out_dir=self.out_dir, sourceDLClass=self.sourceDLClass, targetDLClass=self.targetDLClass)
                else:
                    return eval(matcher)(self, source_api, api_mapping, self.logger, out_dir=self.out_dir, sourceDLClass=self.sourceDLClass, targetDLClass=self.targetDLClass)
        return None


class astGradApiNodes(astBaseNodes):
    def __init__(self, root, file, possible_modules, unsupport_map, logger, sourceDLClass, targetDLClass):
        self.insert_nodes_list = []
        super(astGradApiNodes, self).__init__(root, file, possible_modules, unsupport_map, logger, sourceDLClass, targetDLClass)

    @property
    def parent_node(self):
        return self.node_stack[-2]

    def HandleAssign(self, node):
        """
        handle 'requires_grad' and 'stop_gradient' in the AST
        """

        if isinstance(node.targets[0], ast.Attribute):
            log_info(self.logger,
                     "[Start] convert {} to {} for {}".format(self.sourceDLClass, self.targetDLClass,
                                                              node.targets[0].attr),
                     self.file_name,
                     node.lineno, )
            if node.targets[0].attr == "requires_grad":
                log_info(self.logger,
                         "[Success] convert {} to {} Successfully".format(node.targets[0].attr, self.targetDLClass),
                         self.file_name,
                         node.lineno,
                         )
                node.targets[0].attr = "stop_gradient"
                node = ast.Assign(
                    targets=[node.targets[0]],
                    value=ast.UnaryOp(ast.Not(), operand=node.value),
                )
                return node
        elif isinstance(node.targets[0], ast.Tuple):
            flag = False
            for j in range(len(node.targets[0].elts)):
                if isinstance(node.targets[0].elts[j], ast.Attribute) and (
                        node.targets[0].elts[j].attr == "requires_grad"
                ):
                    flag = True
                    new_node = ast.Name(id="temp", ctx=ast.Load())
                    node.targets[0].elts[j].attr = "stop_gradient"
                    assign_node = ast.Assign(
                        targets=[node.targets[0].elts[j]],
                        value=ast.UnaryOp(ast.Not(), operand=new_node),
                    )
                    node.targets[0].elts[j] = new_node
                    index = self.parent_node.body.index(node)
                    self.insert_nodes_list.append(
                        (self.parent_node, index, assign_node)
                    )
            if flag:
                return None
        return node

    def insert_assign_node(self):
        for parent_node, index, node in self.insert_nodes_list[::-1]:
            parent_node.body.insert(index + 1, node)

    def transform(self):
        self.visit(self.root)
        self.insert_assign_node()
