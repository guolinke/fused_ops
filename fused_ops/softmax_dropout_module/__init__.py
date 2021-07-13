try:
    import torch
    import fused_softmax_dropout_fast_cuda
    from .fused_softmax_dropout import softmax_dropout
    del torch
    del fused_softmax_dropout_fast_cuda
    del fused_softmax_dropout
except ImportError as err:
    print("cannot import kernels, please install the package")
