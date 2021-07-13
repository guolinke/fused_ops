try:
    import torch
    import fused_bias_gelu_cuda
    from .fused_bias_gelu import BiasTorchGeLUFunction, BiasFastGeLUFunction
    from .fused_bias_gelu import bias_torch_gelu, bias_fast_gelu
    del torch
    del fused_bias_gelu_cuda
    del fused_bias_gelu
except ImportError as err:
    print("cannot import kernels, please install the package")
