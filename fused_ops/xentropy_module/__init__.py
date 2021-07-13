try:
    import torch
    import fused_xentropy_cuda
    from .softmax_xentropy import SoftmaxCrossEntropyLoss, softmax_cross_entropy_loss
    del torch
    del fused_xentropy_cuda
    del softmax_xentropy
except ImportError as err:
    print("cannot import kernels, please install the package")
