import paddle 
from paddle import _C_ops, pir
from paddle.base import core, framework, unique_name
from paddle.base.data_feeder import check_variable_and_dtype
from paddle.base.framework import _current_expected_place, in_dygraph_mode, in_pir_mode
from paddle.nn.initializer.initializer import Initializer


class Modified_TruncatedNormal(paddle.nn.initializer.TruncatedNormal):
	    def forward(self, var, block=None):
	        """Initialize the input tensor with TruncatedNormal distribution.
	
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
	
	        if self._seed == 0:
	            self._seed = block.program.random_seed
	
	        # to be compatible of fp16 initalizers
	        if var.dtype in [core.VarDesc.VarType.FP16, core.VarDesc.VarType.BF16]:
	            out_dtype = core.VarDesc.VarType.FP32
	            out_var = block.create_var(
	                name=unique_name.generate(
	                    ".".join(['truncated_gaussian_random', var.name, 'tmp'])
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
	            out_var = _C_ops.truncated_gaussian_random(
	                var.shape,
	                self._mean,
	                self._std_dev,
	                self._seed,
	                out_dtype,
	                _current_expected_place(),
	            )
	                        upper_bound = self._mean + 2 * self._std_dev
	                        out_var = paddle.clip(out_var, max = upper_bound)
	                        lower_bound = self._mean + -2 * self._std_dev
	                        out_var = paddle.clip(out_var, min = lower_bound)
	            if var.dtype in [
	                core.VarDesc.VarType.FP16,
	                core.VarDesc.VarType.BF16,
	            ]:
	                var_tmp = _C_ops.cast(out_var, var.dtype)
	                var_tmp._share_underline_tensor_to(var)
	            else:
	                out_var._share_underline_tensor_to(var)
	            return None
	
	        else:
	            op = block.append_op(
	                type="truncated_gaussian_random",
	                outputs={"Out": out_var},
	                attrs={
	                    "shape": var.shape,
	                    "dtype": out_dtype,
	                    "mean": self._mean,
	                    "std": self._std_dev,
	                    "seed": self._seed,
	                },
	                stop_gradient=True,
	            )
	
	            if var.dtype in [
	                core.VarDesc.VarType.FP16,
	                core.VarDesc.VarType.BF16,
	            ]:
	                block.append_op(
	                    type="cast",
	                    inputs={"X": out_var},
	                    outputs={"Out": var},
	                    attrs={"in_dtype": out_var.dtype, "out_dtype": var.dtype},
	                )
	            var.op = op
	            return op

