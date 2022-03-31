try:
    import torch
    import fused_rmsnorm_fast_cuda
    import fused_rmsnorm_backward_gamma_cuda
    from .fused_rms_norm import FusedRMSNorm
    del torch
    del fused_rmsnorm_fast_cuda
    del fused_rmsnorm_backward_gamma_cuda
except ImportError as err:
    print(err)
    print("cannot import rmsnorm kernel, please install the package")