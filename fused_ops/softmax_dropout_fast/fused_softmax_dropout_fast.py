import torch
import fused_softmax_dropout_fast_cuda

class SoftmaxDropoutFast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, heads, inputs, dropout_prob):
        # don't use ctx.save_for_backward to save dropout_prob and heads
        # allocating space for a tensor is time-consuming
        dropout_results, dropout_mask, softmax_results = fused_softmax_dropout_fast_cuda.forward(is_training,
            heads, inputs, dropout_prob, None)
        if is_training:
            ctx.heads = heads
            ctx.dropout_prob = dropout_prob
            ctx.save_for_backward(softmax_results, dropout_mask)
        return dropout_results

    @staticmethod
    def backward(ctx, grad_output):
        softmax_results, dropout_mask = ctx.saved_tensors
        heads = ctx.heads
        dropout_prob = ctx.dropout_prob
        grad_input = fused_softmax_dropout_fast_cuda.backward(heads, grad_output, softmax_results,
            dropout_mask, dropout_prob)
        return None, None, grad_input, None