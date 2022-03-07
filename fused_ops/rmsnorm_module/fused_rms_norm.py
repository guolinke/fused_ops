import math
import torch
import numbers
from torch.nn.parameter import Parameter
from torch.nn import init
from torch.nn import functional as F
import fused_rmsnorm_fast_cuda
import fused_rmsnorm_backward_gamma_cuda

class FusedRMSNormFastFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, weight, normalized_shape, eps):
    ctx.normalized_shape = normalized_shape
    ctx.eps = eps
    input_ = input.contiguous()
    weight_ = weight.contiguous()
    output, invvar = fused_rmsnorm_fast_cuda.forward(
        input_, ctx.normalized_shape, weight_, ctx.eps)
    ctx.save_for_backward(input_, weight_, invvar)
    return output

  @staticmethod
  def backward(ctx, grad_output):
    input_, weight_, invvar = ctx.saved_tensors
    grad_input = grad_weight = grad_bias = None
    grad_input = fused_rmsnorm_fast_cuda.backward(
        grad_output.contiguous(), invvar,
        input_, ctx.normalized_shape,
        weight_, ctx.eps)
    grad_weight = fused_rmsnorm_backward_gamma_cuda.backward_gamma(
        grad_output.contiguous(), invvar,
        input_, ctx.normalized_shape,
        weight_, ctx.eps)
    return grad_input, grad_weight, None, None

class FusedRMSNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(FusedRMSNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        assert elementwise_affine
        self.weight = Parameter(torch.Tensor(*normalized_shape))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, input):
        return FusedRMSNormFastFunction.apply(
            input, self.weight, self.normalized_shape, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine=True'.format(**self.__dict__)
