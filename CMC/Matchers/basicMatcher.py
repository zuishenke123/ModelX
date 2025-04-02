import ast, astor
import textwrap
import re

from CMC.nodeBase import BaseMatcher
from CMC.special_mapping import addOp, subOp, divOp, mulOp
from CMC.utils import (get_unique_name, replace_pattern, tensor_replacement, funcMapping,select_paddle_operator, select_pytorch_operator,
                       paddle_param_change, pytorch_param_change)


class SequentialMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs, tensorCall=False):
        # nn.Sequential(OrderedDict([...]) / nn.Sequential(OrderedDict(blocks))
        if (
                len(args) == 1
                and isinstance(args[0], ast.Call)
                and self.get_full_attr(args[0].func).endswith("OrderedDict")
        ):
            new_args = self.parse_args(args[0].args)
            new_args = ["*{}".format(new_args[0])]
        # nn.Sequential(module1, module2, ...)
        else:
            new_args = self.parse_args(args)
        code = "paddle.nn.Sequential(\n{}\n)".format(self.args_to_str(new_args))
        return ast.parse(code).body


class AssertMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs, tensorCall=False):
        code = "assert {}, {}".format(ast.unparse(args[0]).strip("\n"), ast.unparse(args[1]).strip("\n"))
        return ast.parse(code).body


class ViewMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        shape_vars = kwargs['shape']
        # shape_vars = shape_vars[1:-1]
        params = []
        shape_vars = re.split(r',\s*', shape_vars.strip('[]'))
        for shape_var in shape_vars:
            if "*" in shape_var:
                code = "paddle.Tensor.reshape({})".format(shape_var)
                return ast.parse(code).body
            else:
                params.append(shape_var)
        code = "paddle.Tensor.reshape([{}])".format(",".join(params))
        return ast.parse(code).body


class LrschedMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        # recording the optimizer and correct learning_rate
        optimizer = kwargs["learning_rate"]
        kwargs["learning_rate"] = self.lr

        params_change = {}
        if "params_change" in self.api_mapping:
            params_change = self.api_mapping["params_change"]
        if kwargs and kwargs != 'Property':
            new_kwargs = handlingParam(kwargs, params_change, self.targetDLClass)
        else:
            new_kwargs = {}
        new_kwargs = self.set_default_kwargs(new_kwargs)

        res = ""
        if 'variantCode' in self.api_mapping:
            new_kwargs, res = self.handleVariantCode(new_kwargs, res)

        API_TEMPLATE = textwrap.dedent(
            """
            tmp_scheduler = {}({})
            {}._learning_rate = tmp_scheduler
            tmp_scheduler
            """
        )
        code = API_TEMPLATE.format(self.get_target_api(), self.kwargs_to_str(new_kwargs), optimizer)

        return code


class InitializerMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        params_change = {}
        if "params_change" in self.api_mapping:
            params_change = self.api_mapping["params_change"]
        if kwargs and kwargs != 'Property':
            new_kwargs = handlingParam(kwargs, params_change, self.targetDLClass)
        else:
            new_kwargs = {}
        new_kwargs = self.set_default_kwargs(new_kwargs)

        res = ""
        if 'variantCode' in self.api_mapping:
            new_kwargs, res = self.handleVariantCode(new_kwargs, res, self.targetDLClass)
        if self.targetDLClass == "pytorch":
            if res == "":
                API_TEMPLATE = textwrap.dedent(
                    """
                    lambda tensor: {}(tensor, {})
                    """
                )
                code = API_TEMPLATE.format(self.get_target_api(), self.kwargs_to_str(new_kwargs))
            else:
                API_TEMPLATE = textwrap.dedent(
                    """
                    {}
                    lambda tensor: {}(tensor, {})
                    """
                )
                code = API_TEMPLATE.format(res, self.get_target_api(), self.kwargs_to_str(new_kwargs))
        elif self.targetDLClass == "paddlepaddle":
            if res == "":
                API_TEMPLATE = textwrap.dedent(
                    """
                    tmp_initializer = {}({})
                    """
                )
                code = API_TEMPLATE.format(self.get_target_api(), self.kwargs_to_str(new_kwargs))
            else:
                API_TEMPLATE = textwrap.dedent(
                    """
                    {}
                    tmp_initializer = {}({})
                    """
                )
                code = API_TEMPLATE.format(res, self.get_target_api(), self.kwargs_to_str(new_kwargs))

            API_TEMPLATE = textwrap.dedent(
                """
                {}
                {}.set_value(paddle.create_parameter(shape={}.shape, dtype={}.dtype, default_initializer=tmp_initializer))
                """
            )
            code = API_TEMPLATE.format(code, self.init_val, self.init_val, self.init_val)

        return code


class tensorParamMatcher(BaseMatcher):
    def get_paddle_nodes(self, args, kwargs, tensorCall=False):
        # 使用astor.to_source转换args[0]，然后去除尾随换行符，避免在代码中引入额外的换行
        initial_tensor_code = astor.to_source(args[0]).strip()

        # 构造paddle.create_parameter的参数字符串
        new_args = {
            "shape": "initial_tensor.shape",
            "dtype": "initial_tensor.dtype",
            "default_initializer": "paddle.nn.initializer.Assign(initial_tensor)"
        }
        args_str = self.kwargs_to_str(new_args)

        # 生成最终的代码字符串，确保不会引入不必要的缩进
        code = f"initial_tensor = {initial_tensor_code}\npaddle.create_parameter({args_str})"

        # 将生成的代码字符串解析为AST节点并返回
        return ast.parse(code).body


def handlingParam(kwargs, params_change, targetDLClass):
    new_kwargs = {}
    # Operator -> Property
    if isinstance(kwargs, str) and "Property" in kwargs:
        return kwargs
    if targetDLClass == "paddlepaddle":
        return paddle_param_change(kwargs, params_change)
    elif targetDLClass == "pytorch":
        return pytorch_param_change(kwargs, params_change)


class DirectMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        params_change = {}
        unsupport_params = []
        if "params_change" in self.api_mapping:
            params_change = self.api_mapping["params_change"]
        if "unsupport_params" in self.api_mapping:
            for param in self.api_mapping["unsupport_params"]:
                kwargs.pop(param, None)
        if kwargs:
            new_kwargs = handlingParam(kwargs, params_change, self.targetDLClass)
        else:
            new_kwargs = {}

        new_kwargs = self.set_default_kwargs(new_kwargs)
        res = ""

        return self.handleCodeGeneration(new_kwargs, res, self.targetDLClass)


class FuncParamMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        params_change = {}
        if "params_change" in self.api_mapping:
            params_change = self.api_mapping["params_change"]
        if "params_unsupport" in self.api_mapping:
                for param in self.api_mapping["params_unsupport"]:
                    kwargs.pop(param, None)
        if "params_deprecation" in self.api_mapping:
            params_deprecation = self.api_mapping["params_deprecation"]
            kwargs = {key: value for key, value in kwargs.items() if key not in params_deprecation}
        if kwargs:
            new_kwargs = handlingParam(kwargs, params_change, self.targetDLClass)
        else:
            new_kwargs = {}

        new_kwargs = self.set_default_kwargs(new_kwargs)
        res = ""
        if 'variantCode' in self.api_mapping:
            new_kwargs, res = self.handleVariantCode(new_kwargs, res, self.targetDLClass)

        return self.handleCodeGeneration(new_kwargs, res, self.targetDLClass)

class OCSMatcher(BaseMatcher):
    def generate_code(self, kwargs):
        pattern = re.compile(r"""
            (?P<func_call>                                      # 开始捕获函数调用
                [a-zA-Z_][\w:]*                                 # 匹配函数名称（可能包含命名空间）
                (?:\s*::\s*|\.)?                                # 可选的命名空间分隔符(::)或点(.)，适配不同的调用方式
                [a-zA-Z_][\w:]*                                 # 匹配函数名称
            )\s*\(\s*                                           # 匹配左括号和可选的空格
            (?P<args>                                           # 开始捕获参数
                (?:
                    [^()]*?                                     # 非括号内容，非贪婪
                    |                                           # 或
                    \((?:                                       # 嵌套的括号开始
                        [^()]*                                  # 非括号内容
                        |                                       # 或
                        \( [^()]* \)                            # 更深层嵌套的括号（限制为一层嵌套）
                    )*?\)                                       # 嵌套的括号结束
                )*?                                             # 重复以上模式，匹配所有参数
            )                                                   # 结束捕获参数
            \)\s*         |                                        # 匹配右括号和可选的空格
           (?P<variable>[a-zA-Z_][\w]*) |                          # 变量
           (?P<operator>\+|\-|\*|\/)                               # 运算符
           ()
        """, re.VERBOSE)

        result_sequence = [{'type': "function_call", 'value': self.api_mapping["operatorSequence"][0]["function"]}]

        for sequences in self.api_mapping["operatorSequence"]:
            func = sequences["function"]
            params = sequences["params"]
            operation_sequence = result_sequence

            for i, op in enumerate(operation_sequence):
                if op["value"] != sequences["function"] and (op["type"] != "function_call" or not
                sum(1 for op in operation_sequence if op["type"] == "function_call") == 1):
                    continue
                # 获取参数映射关系
                func_args_mapping = {}
                if 'args' in op:
                    if len(op["args"]) != len(params):
                        pass
                    else:
                        for j, arg in enumerate(op['args']):
                            func_args_mapping[params[j]] = arg["value"]
                else:
                    for param in params:
                        func_args_mapping[param] = param
                new_ops = [{"value": "return"}]  # 用于存储基于当前op解析得到的新操作
                for idx, (k, sequence) in enumerate(sequences["ReturnSequence"].items()):
                    index_of_matching_op = next((index for index, op in enumerate(new_ops) if op.get('value') == k),
                                                None)
                    if index_of_matching_op is not None:
                        stmt = sequence
                        # 匹配形如variable.method<generic>()的字符串, 保留关键变量
                        stmt = replace_pattern(stmt, r"(\w+)\.\w+<[^>]*>\(\)", r"\1")

                        matches = pattern.finditer(stmt)
                        tmp_ops = []
                        for match in matches:
                            if match.group('func_call'):
                                # 处理函数调用，提取函数名和参数列表
                                func_call = match.group('func_call')
                                args = match.group('args')
                                # 获取参数列表
                                args_matches = [arg.strip() for arg in args.split(",")]
                                # 排除非用户参数
                                for arg in ["device_type()"]:
                                    if arg in args_matches:
                                        args_matches.remove(arg)
                                func_args = []
                                # args_matches = re.finditer(r'\b(?P<arg_variable>[a-zA-Z_][\w]*)\b', args)
                                for arg_match in args_matches:
                                    # 遍历func_args_mapping，替换参数中的变量
                                    for key, value in func_args_mapping.items():
                                        arg_match = arg_match.replace(key, value)
                                    if arg_match == "*this":
                                        subargs = []
                                        for param in params:
                                            if not param in args_matches:
                                                subargs.append(param)
                                        func_args.append(
                                            {'type': 'arg', 'value': arg_match, 'context': 'function_argument',
                                             'belongs_to': func_call, "sub_args": subargs})
                                    else:
                                        func_args.append(
                                            {'type': 'arg', 'value': arg_match, 'context': 'function_argument',
                                             'belongs_to': func_call})
                                tmp_ops.append({'type': 'function_call', 'value': func_call, 'args': func_args})
                            elif match.group('variable'):
                                # 处理独立的变量
                                variable = match.group('variable')
                                # 检查是否已作为参数处理过
                                if not any(op['type'] == 'variable' and op['value'] == variable and op.get(
                                        'context') == 'function_argument' for op in tmp_ops):
                                    variable = func_args_mapping.get(variable, variable)
                                    tmp_ops.append(
                                        {'type': 'variable', 'value': variable, 'context': 'standalone'})
                            elif match.group('operator'):
                                # 处理运算符
                                tmp_ops.append({'type': 'operator', 'value': match.group('operator')})
                        try:
                            new_ops[index_of_matching_op: index_of_matching_op + 1] = tmp_ops
                        except ValueError:
                            pass
                operation_sequence[i: i + 1] = new_ops

            result_sequence = operation_sequence

        result_sequence = [
            {**op, 'value': re.sub(r'(\w+)_val$', tensor_replacement, op['value'])}
            if op['type'] == 'variable' else op
            for op in result_sequence
        ]
        # 动态生成部分
        sourceDLCLass_param = self.api_mapping["operatorSequence"][0]["params"]
        param_mapping = {}
        idx = 0
        if len(kwargs) != len(sourceDLCLass_param):
            # 调用形式 torch.addcdiv(tensor, value, tansor1, tensor2)
            arg = kwargs.pop(0)
            param_mapping["self"] = arg.id
        for arg in kwargs:
            if isinstance(arg, ast.keyword):
                if arg.arg in sourceDLCLass_param:
                    param_mapping[arg.arg] = arg.value
                else:
                    param_mapping[sourceDLCLass_param[idx]] = arg.value
                    idx = idx + 1
            else:
                param_mapping[sourceDLCLass_param[idx]] = arg.id
                idx = idx + 1

        return self.sequenceToCode(result_sequence, param_mapping)

    def sequenceToCode(self, result_sequence, param_mapping):

        def apply_operator(op, a, b):
            if self.targetDLClass == "paddlepaddle":
                return select_paddle_operator(op, a, b, self.targetDLClass)
            elif self.targetDLClass == "pytorch":
                return select_pytorch_operator(op, a, b, self.targetDLClass)
            else:
                raise EnvironmentError("{} not supported".format(self.targetDLClass))

        # Two stacks: one for operands and one for pytorch
        value_stack = []
        operator_stack = []
        # Priority mapping
        precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
        for i, op in enumerate(result_sequence):
            if op['type'] == "operator":
                # 处理栈中的操作符，确保遵守优先级
                while operator_stack and precedence[operator_stack[-1]['value']] >= precedence[op['value']]:
                    operator = operator_stack.pop()
                    b = value_stack.pop()
                    a = value_stack.pop()
                    result_code = apply_operator(operator['value'], a, b)
                    value_stack.append(result_code)
                operator_stack.append(op)
            elif op['type'] == 'variable':
                value_stack.append(param_mapping[op['value']])
            elif op['type'] == 'function':
                value_stack.append(funcMapping(self.targetDLClass, op))
            # 清空操作符栈，完成剩余的计算
        while operator_stack:
            op = operator_stack.pop()
            b = value_stack.pop()
            a = value_stack.pop()
            result_code = apply_operator(op['value'], a, b)
            value_stack.append(result_code)

        return value_stack.pop()
