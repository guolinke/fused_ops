import torch
import sys 
sys.path.append('..')
import fused_softmax_dropout_cuda
import fused_softmax_dropout_fast_cuda
from fused_ops.softmax_dropout_fast import SoftmaxDropoutFast

class SoftmaxDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, is_training, heads, inputs, dropout_prob):
        # don't use ctx.save_for_backward to save dropout_prob and heads
        # allocating space for a tensor is time-consuming
        dropout_results, dropout_mask, softmax_results = fused_softmax_dropout_cuda.forward(is_training,
            heads, inputs, dropout_prob)
        if is_training:
            ctx.heads = heads
            ctx.dropout_prob = dropout_prob
            ctx.save_for_backward(softmax_results, dropout_mask)
            return dropout_results
        else:
            return softmax_results

    @staticmethod
    def backward(ctx, grad_output):
        softmax_results, dropout_mask = ctx.saved_tensors
        heads = ctx.heads
        dropout_prob = ctx.dropout_prob
        grad_input = fused_softmax_dropout_cuda.backward(heads, grad_output, softmax_results,
            dropout_mask, dropout_prob)
        return None, None, grad_input, None

softmax_dropout_func = SoftmaxDropout.apply
softmax_dropout_fast_func = SoftmaxDropoutFast.apply

def test_speed(name, bsz, seq_len, n, f):
    torch.manual_seed(0)
    x = torch.rand((bsz, seq_len, seq_len)).cuda().half()
    grad = torch.rand((bsz, seq_len, seq_len)).cuda().half()
    x.requires_grad_(True)
    forward_time = 0
    backward_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(n):
        x.grad = None
        start.record()
        y = f(x)
        end.record()
        torch.cuda.synchronize()
        forward_time += start.elapsed_time(end)
        
        start.record()
        y.backward(grad)
        end.record()
        torch.cuda.synchronize()
        backward_time += start.elapsed_time(end)

    print(name, 'forward', forward_time / n, 'ms, backward', backward_time / n, 'ms')

test_err(10, 512)
test_speed('old', 16 * 12, 512, 10000, lambda x: softmax_dropout_func(True, 12, x, 0.1))
test_speed('new', 16 * 12, 512, 10000, lambda x: softmax_dropout_fast_func(True, 12, x, 0.1))