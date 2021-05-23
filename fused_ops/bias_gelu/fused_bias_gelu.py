import torch
import fused_bias_gelu_cuda

class BiasTorchGeLUFunction(torch.autograd.Function):
    r"""Fused bias and GELU operator.
    
    Shape:
        - Input: :math:`(*, D)`
        - Bias: :math:`(D)`
        - Output: :math:`(*, D)`
    """
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return fused_bias_gelu_cuda.forward_torch(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = fused_bias_gelu_cuda.backward_torch(input, bias, grad_output)
        if len(grad_input.size()) > 1:
            sizes = grad_input.size()
            grad_input_bias = grad_input.view(-1, sizes[-1]).sum(dim=0)
        else:
            grad_input_bias = grad_input
        return grad_input, grad_input_bias

class BiasTanhGeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return fused_bias_gelu_cuda.forward_tanh(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = fused_bias_gelu_cuda.backward_tanh(input, bias, grad_output)
        if len(grad_input.size()) > 1:
            sizes = grad_input.size()
            grad_input_bias = grad_input.view(-1, sizes[-1]).sum(dim=0)
        else:
            grad_input_bias = grad_input
        return grad_input, grad_input_bias