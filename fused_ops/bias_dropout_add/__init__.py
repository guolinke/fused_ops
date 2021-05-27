try:
    import torch
    import fused_bias_dropout_add_cuda
    from .fused_bias_dropout_add import BiasDropoutAddFunction
    del torch
    del fused_bias_dropout_add_cuda
    del fused_bias_dropout_add
except ImportError as err:
    print("cannot import kernels, please install the package")
