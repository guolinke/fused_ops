import torch
import torch.nn.functional as F
import sys 
sys.path.append('..')
import fused_bias_gelu_cuda
from fused_ops.bias_gelu import BiasFastGeLUFunction, BiasTorchGeLUFunction

torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)
torch._C._jit_override_can_fuse_on_cpu(True)
torch._C._jit_override_can_fuse_on_gpu(True)

@torch.jit.script
def bias_gelu(bias, y):
    x = bias + y
    return  x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.jit.script
def bias_gelu_back(g, bias, y):
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff*g

class JITBiasGeLUFunction(torch.autograd.Function):
    r"""Fused bias and GELU operator.
    
    Shape:
        - Input: :math:`(*, D)`
        - Bias: :math:`(D)`
        - Output: :math:`(*, D)`
    """
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(bias, input)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        grad_input = bias_gelu_back(grad_output, bias, input)
        if len(grad_input.size()) > 1:
            sizes = grad_input.size()
            grad_input_bias = grad_input.view(-1, sizes[-1]).sum(dim=0)
        else:
            grad_input_bias = grad_input
        return grad_input, grad_input_bias

jit_bias_gelu_func = JITBiasGeLUFunction.apply
bias_fast_gelu_func = BiasFastGeLUFunction.apply
bias_torch_gelu_func = BiasTorchGeLUFunction.apply

def test_speed(name, bsz, dim, seq_len, n, f):
    torch.manual_seed(0)
    bias = torch.rand(dim).cuda().half()
    x = torch.rand((seq_len, bsz, dim)).cuda().half()
    grad = torch.rand((seq_len, bsz, dim)).cuda().half()
    bias.requires_grad_(True)
    x.requires_grad_(True)
    forward_time = 0
    backward_time = 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for i in range(n):
        bias.grad = None
        x.grad = None
        start.record()
        y = f(x, bias)
        end.record()
        torch.cuda.synchronize()
        forward_time += start.elapsed_time(end)
        start.record()
        y.backward(grad)
        end.record()
        torch.cuda.synchronize()
        backward_time += start.elapsed_time(end)

    print(name, 'forward', forward_time / n, 'ms, backward', backward_time / n, 'ms')

test_speed('old', 16, 3072, 512, 10000, lambda x, b: F.gelu(x + b))
test_speed('jit', 16, 3072, 512, 10000, jit_bias_gelu_func)
test_speed('fast', 16, 3072, 512, 10000, bias_fast_gelu_func)
test_speed('accurate(torch\'s algorithm)', 16, 3072, 512, 10000, bias_torch_gelu_func)