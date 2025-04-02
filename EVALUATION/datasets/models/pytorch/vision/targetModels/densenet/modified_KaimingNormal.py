import paddle 
import math
from paddle import _C_ops
from paddle.fluid import core, framework, unique_name
from paddle.fluid.framework import _current_expected_place, in_dygraph_mode
from paddle.nn.initializer.initializer import Initializer, calculate_gain


class Modified_KaimingNormal(paddle.nn.initializer.KaimingNormal):
	    def forward(self, var, block=None):
	        """Initialize the input tensor with MSRA initialization.
	
	        Args:
	            var(Tensor): Tensor that needs to be initialized.
	            block(Block, optional): The block in which initialization ops
	                   should be added. Used in static graph only, default None.
	
	        Returns:
	            The initialization op
	        """
	        block = self._check_block(block)
	
	        assert isinstance(var, framework.Variable)
	        assert isinstance(block, framework.Block)
	        f_in, f_out = self._compute_fans(var)
	
	        # If fan_in is passed, use it
	        fan_in = f_in if self._fan_in is None else f_out
	
	        if self._seed == 0:
	            self._seed = block.program.random_seed
	
	        # to be compatible of fp16 initalizers
	        if var.dtype == core.VarDesc.VarType.FP16 or (
	            var.dtype == core.VarDesc.VarType.BF16 and not self._uniform
	        ):
	            out_dtype = core.VarDesc.VarType.FP32
	            out_var = block.create_var(
	                name=unique_name.generate(
	                    ".".join(['masra_init', var.name, 'tmp'])
	                ),
	                shape=var.shape,
	                dtype=out_dtype,
	                type=core.VarDesc.VarType.LOD_TENSOR,
	                persistable=False,
	            )
	        else:
	            out_dtype = var.dtype
	            out_var = var
	
	        if in_dygraph_mode():
	            if self._uniform:
	                gain = calculate_gain(self._nonlinearity, self._negative_slope)
	                limit = gain * math.sqrt(3.0 / float(fan_in))
	                out_var = _C_ops.uniform(
	                    var.shape,
	                    out_dtype,
	                    -limit,
	                    limit,
	                    self._seed,
	                    _current_expected_place(),
	                )
	            else:
	                gain = calculate_gain(self._nonlinearity, self._negative_slope)
	                std = gain / math.sqrt(float(fan_in))
	                place = _current_expected_place()
	                out_var = _C_ops.gaussian(
	                    out_var.shape, 0.0, std, self._seed, out_dtype, place
	                )
	
	            if var.dtype == core.VarDesc.VarType.FP16 or (
	                var.dtype == core.VarDesc.VarType.BF16 and not self._uniform
	            ):
	                var_tmp = _C_ops.cast(out_var, var.dtype)
	                var_tmp._share_underline_tensor_to(var)
	            else:
	                out_var._share_underline_tensor_to(var)
	            return None
	        else:
	            if self._uniform:
	                gain = calculate_gain(self._nonlinearity, self._negative_slope)
	                limit = gain * math.sqrt(3.0 / float(fan_in))
	                op = block.append_op(
	                    type="uniform_random",
	                    inputs={},
	                    outputs={"Out": out_var},
	                    attrs={
	                        "shape": out_var.shape,
	                        "dtype": int(out_dtype),
	                        "min": -limit,
	                        "max": limit,
	                        "seed": self._seed,
	                    },
	                    stop_gradient=True,
	                )
	
	            else:
	                gain = calculate_gain(self._nonlinearity, self._negative_slope)
	                std = gain / math.sqrt(float(fan_in))
	                op = block.append_op(
	                    type="gaussian_random",
	                    outputs={"Out": out_var},
	                    attrs={
	                        "shape": out_var.shape,
	                        "dtype": int(out_dtype),
	                        "mean": 0.0,
	                        "std": std,
	                        "seed": self._seed,
	                    },
	                    stop_gradient=True,
	                )
	
	            if var.dtype == core.VarDesc.VarType.FP16 or (
	                var.dtype == core.VarDesc.VarType.BF16 and not self._uniform
	            ):
	                block.append_op(
	                    type="cast",
	                    inputs={"X": out_var},
	                    outputs={"Out": var},
	                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
	                )
	
	            var.op = op
	            return op

