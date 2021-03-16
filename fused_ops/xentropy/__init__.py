try:
    import torch
    import fused_xentropy_cuda, bf16_fused_xentropy_cuda
    from .softmax_xentropy import SoftmaxCrossEntropyLoss
    from .bf16_softmax_xentropy import BF16SoftmaxCrossEntropyLoss
    del torch
    del fused_xentropy_cuda
    del bf16_fused_xentropy_cuda
    del softmax_xentropy
except ImportError as err:
    print("cannot import kernels, please install the package")
