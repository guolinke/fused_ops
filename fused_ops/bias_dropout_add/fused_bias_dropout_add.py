import torch
import fused_bias_dropout_add_cuda

class BiasDropoutAddFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias, residual, is_training, dropout_prob):
        result, mask = fused_bias_dropout_add_cuda.forward(input, bias, residual, is_training, dropout_prob, None)
        ctx.save_for_backward(mask)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        mask, = ctx.saved_tensors
        grad_input = fused_bias_dropout_add_cuda.backward(grad_output, mask)
        if len(grad_input.size()) > 1:
            sizes = grad_input.size()
            grad_input_bias = grad_input.view(-1, sizes[-1]).sum(dim=0)
        else:
            grad_input_bias = grad_input
        return grad_input, grad_input_bias, grad_output, None, None